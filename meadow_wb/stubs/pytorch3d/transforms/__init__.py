"""Minimal pure-torch implementation of the pytorch3d.transforms API used by SAM 3D Objects.

References:
- https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/transform3d.py
- https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py

We only implement what the inference path needs: Transform3d (compose, transform_points,
inverse, rotate, translate, scale), and the small handful of quaternion / matrix helpers
imported in inference_pipeline_pointmap, inference_utils, layout_post_optimization_utils,
data/dataset/tdfy/transforms_3d.py, data/dataset/tdfy/pose_target.py, and notebook/inference.py.
"""

from __future__ import annotations

from typing import Optional, Union

import torch


# ---------------------------------------------------------------------------
# Quaternion helpers (real_part-first convention, [w, x, y, z]).
# Reproduces pytorch3d's behaviour byte-for-byte.
# ---------------------------------------------------------------------------


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.clamp(x, min=0.0))


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )
    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))
    out = quat_candidates[
        torch.nn.functional.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    return _standardize_quaternion(out)


def _standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def quaternion_raw_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)


def quaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _standardize_quaternion(quaternion_raw_multiply(a, b))


def quaternion_invert(quaternion: torch.Tensor) -> torch.Tensor:
    scaling = torch.tensor([1, -1, -1, -1], device=quaternion.device, dtype=quaternion.dtype)
    return quaternion * scaling


def quaternion_apply(quaternion: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    if point.size(-1) != 3:
        raise ValueError(f"Points are not in 3D, {point.shape}.")
    real_parts = point.new_zeros(point.shape[:-1] + (1,))
    point_as_quaternion = torch.cat((real_parts, point), -1)
    out = quaternion_raw_multiply(
        quaternion_raw_multiply(quaternion, point_as_quaternion),
        quaternion_invert(quaternion),
    )
    return out[..., 1:]


def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))


def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles


def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e) for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])


def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)
    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("axis must be one of X, Y, Z")
    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


# ---------------------------------------------------------------------------
# Transform3d: 4x4 homogeneous transform composition. Matches pytorch3d API.
# ---------------------------------------------------------------------------


class Transform3d:
    def __init__(
        self,
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device, None] = "cpu",
        matrix: Optional[torch.Tensor] = None,
    ):
        if matrix is None:
            self._matrix = torch.eye(4, dtype=dtype, device=device).view(1, 4, 4)
        else:
            if matrix.ndim not in (2, 3):
                raise ValueError("Transform3d matrix must be 2D or 3D")
            if matrix.shape[-2] != 4 or matrix.shape[-1] != 4:
                raise ValueError("Transform3d matrix must be 4x4")
            self._matrix = matrix.view(-1, 4, 4) if matrix.ndim == 2 else matrix
        self.device = self._matrix.device
        self.dtype = self._matrix.dtype

    def get_matrix(self) -> torch.Tensor:
        return self._matrix

    def to(self, device, dtype=None):
        new = Transform3d(matrix=self._matrix.to(device=device, dtype=dtype or self.dtype))
        return new

    def compose(self, *others: "Transform3d") -> "Transform3d":
        mat = self._matrix
        for o in others:
            # pytorch3d uses row-vector convention: out = x @ M1 @ M2 ...
            mat = mat @ o._matrix
        return Transform3d(matrix=mat)

    def inverse(self) -> "Transform3d":
        return Transform3d(matrix=torch.inverse(self._matrix))

    # Builders ---------------------------------------------------------------

    def translate(self, *tx, **kwargs) -> "Transform3d":
        return self.compose(Translate(*tx, **kwargs))

    def scale(self, *sx, **kwargs) -> "Transform3d":
        return self.compose(Scale(*sx, **kwargs))

    def rotate(self, R: torch.Tensor, **kwargs) -> "Transform3d":
        return self.compose(Rotate(R, **kwargs))

    def rotate_axis_angle(self, angle, axis="X", degrees=True, **kwargs) -> "Transform3d":
        return self.compose(RotateAxisAngle(angle, axis=axis, degrees=degrees, **kwargs))

    # Application ------------------------------------------------------------

    def transform_points(self, points: torch.Tensor, eps: Optional[float] = None) -> torch.Tensor:
        # points: (..., N, 3); we use pytorch3d's row-vector convention.
        points_batch = points.clone()
        if points_batch.dim() == 2:
            points_batch = points_batch[None]
        N, P, _ = points_batch.shape
        ones = torch.ones((N, P, 1), dtype=points.dtype, device=points.device)
        points_h = torch.cat([points_batch, ones], dim=-1)
        # broadcast matrix
        M = self._matrix
        if M.shape[0] == 1 and N != 1:
            M = M.expand(N, 4, 4)
        out_h = torch.bmm(points_h, M)
        denom = out_h[..., 3:4]
        if eps is not None:
            denom_sign = torch.sign(denom) + (denom == 0).to(denom.dtype)
            denom = denom_sign * torch.clamp(denom.abs(), min=eps)
        out = out_h[..., :3] / denom
        if points.dim() == 2:
            out = out[0]
        return out

    def transform_normals(self, normals: torch.Tensor) -> torch.Tensor:
        # Normals transform with the inverse-transpose of the linear part.
        if normals.dim() == 2:
            normals_b = normals[None]
        else:
            normals_b = normals
        M = self._matrix
        if M.shape[0] == 1 and normals_b.shape[0] != 1:
            M = M.expand(normals_b.shape[0], 4, 4)
        Mi = torch.inverse(M[:, :3, :3]).transpose(-1, -2)
        out = torch.bmm(normals_b, Mi)
        if normals.dim() == 2:
            out = out[0]
        return out


class Translate(Transform3d):
    def __init__(self, x, y=None, z=None, dtype=torch.float32, device="cpu"):
        if isinstance(x, torch.Tensor):
            xyz = x
            if xyz.dim() == 1:
                xyz = xyz[None]
            dtype = xyz.dtype
            device = xyz.device
        else:
            xyz = torch.tensor([[x, y, z]], dtype=dtype, device=device)
        N = xyz.shape[0]
        mat = torch.eye(4, dtype=dtype, device=device)[None].repeat(N, 1, 1)
        # row-vector convention: translation goes in last row
        mat[:, 3, 0] = xyz[:, 0]
        mat[:, 3, 1] = xyz[:, 1]
        mat[:, 3, 2] = xyz[:, 2]
        super().__init__(matrix=mat)


class Scale(Transform3d):
    def __init__(self, x, y=None, z=None, dtype=torch.float32, device="cpu"):
        if isinstance(x, torch.Tensor):
            if x.dim() == 0:
                xyz = torch.stack([x, x, x])[None]
            elif x.dim() == 1 and x.numel() == 3:
                xyz = x[None]
            elif x.dim() == 2:
                xyz = x
            else:
                xyz = x.repeat(3).view(-1, 3) if x.numel() == 1 else x.view(-1, 3)
            dtype = xyz.dtype
            device = xyz.device
        else:
            if y is None:
                y = x
            if z is None:
                z = x
            xyz = torch.tensor([[x, y, z]], dtype=dtype, device=device)
        N = xyz.shape[0]
        mat = torch.zeros((N, 4, 4), dtype=dtype, device=device)
        mat[:, 0, 0] = xyz[:, 0]
        mat[:, 1, 1] = xyz[:, 1]
        mat[:, 2, 2] = xyz[:, 2]
        mat[:, 3, 3] = 1.0
        super().__init__(matrix=mat)


class Rotate(Transform3d):
    def __init__(self, R: torch.Tensor, dtype=torch.float32, device=None, orthogonal_tol=1e-5):
        if device is None:
            device = R.device
        if R.dim() == 2:
            R = R[None]
        N = R.shape[0]
        mat = torch.zeros((N, 4, 4), dtype=R.dtype, device=R.device)
        # row-vector convention: rotation matrix lives in upper-left 3x3
        # but pytorch3d stores R^T there; transform_points does p @ M.
        mat[:, :3, :3] = R.transpose(-1, -2)
        mat[:, 3, 3] = 1.0
        super().__init__(matrix=mat)


class RotateAxisAngle(Transform3d):
    def __init__(self, angle, axis="X", degrees=True, dtype=torch.float32, device="cpu"):
        if not isinstance(angle, torch.Tensor):
            angle = torch.tensor([angle], dtype=dtype, device=device)
        if degrees:
            angle = angle * (torch.pi / 180.0)
        R = _axis_angle_rotation(axis, angle)
        super().__init__(matrix=Rotate(R).get_matrix())


__all__ = [
    "Transform3d",
    "Translate",
    "Scale",
    "Rotate",
    "RotateAxisAngle",
    "quaternion_to_matrix",
    "matrix_to_quaternion",
    "quaternion_multiply",
    "quaternion_invert",
    "quaternion_apply",
    "axis_angle_to_quaternion",
    "axis_angle_to_matrix",
    "quaternion_to_axis_angle",
    "matrix_to_axis_angle",
    "euler_angles_to_matrix",
]
