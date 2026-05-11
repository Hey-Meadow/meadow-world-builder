"""Stubs for pytorch3d.renderer. We supply a real `look_at_view_transform` (used to build
the camera-convention rotation in inference_pipeline_pointmap.camera_to_pytorch3d_camera)
and importable-but-unusable stubs for the rendering machinery (mesh rasterizer / cameras /
textures). The Gaussian-only inference path doesn't touch rendering."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import torch


def look_at_view_transform(
    dist: Union[float, torch.Tensor] = 1.0,
    elev: Union[float, torch.Tensor] = 0.0,
    azim: Union[float, torch.Tensor] = 0.0,
    degrees: bool = True,
    eye: Optional[Union[torch.Tensor, np.ndarray]] = None,
    at: Optional[Union[torch.Tensor, np.ndarray]] = None,
    up: Optional[Union[torch.Tensor, np.ndarray]] = None,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (R, T) following the pytorch3d convention: world->camera, with rows = camera axes
    in world coordinates (i.e. p_camera = p_world @ R + T).

    Only the eye/at/up branch (used by SAM 3D Objects via
    `look_at_view_transform(eye=..., at=..., up=..., device=...)`) is exercised. We implement it
    correctly. The dist/elev/azim branch is implemented for completeness but not byte-for-byte
    matched against pytorch3d (close enough for camera framing).
    """

    def _to_tensor(x, dtype=torch.float32):
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=dtype)
        return torch.as_tensor(x, dtype=dtype, device=device)

    if eye is not None:
        eye_t = _to_tensor(eye)
        if eye_t.dim() == 1:
            eye_t = eye_t[None]
        if at is None:
            at_t = torch.zeros_like(eye_t)
        else:
            at_t = _to_tensor(at)
            if at_t.dim() == 1:
                at_t = at_t[None]
        if up is None:
            up_t = torch.tensor([[0.0, 1.0, 0.0]], dtype=eye_t.dtype, device=device)
        else:
            up_t = _to_tensor(up)
            if up_t.dim() == 1:
                up_t = up_t[None]
        # Broadcast
        N = max(eye_t.shape[0], at_t.shape[0], up_t.shape[0])
        eye_t = eye_t.expand(N, 3)
        at_t = at_t.expand(N, 3)
        up_t = up_t.expand(N, 3)
    else:
        # dist/elev/azim path
        dist_t = _to_tensor(dist)
        elev_t = _to_tensor(elev)
        azim_t = _to_tensor(azim)
        if dist_t.dim() == 0:
            dist_t = dist_t[None]
        if elev_t.dim() == 0:
            elev_t = elev_t[None]
        if azim_t.dim() == 0:
            azim_t = azim_t[None]
        if degrees:
            elev_t = elev_t * (torch.pi / 180.0)
            azim_t = azim_t * (torch.pi / 180.0)
        x = dist_t * torch.cos(elev_t) * torch.sin(azim_t)
        y = dist_t * torch.sin(elev_t)
        z = dist_t * torch.cos(elev_t) * torch.cos(azim_t)
        eye_t = torch.stack([x, y, z], dim=-1)
        if at is None:
            at_t = torch.zeros_like(eye_t)
        else:
            at_t = _to_tensor(at)
            if at_t.dim() == 1:
                at_t = at_t[None]
        up_t = torch.tensor([[0.0, 1.0, 0.0]], dtype=eye_t.dtype, device=device).expand_as(eye_t)

    # Camera basis: z = (eye - at) (looking from eye to at, but pytorch3d uses
    # -z forward, so z axis points eye->at? Actually pytorch3d: camera looks down +z
    # in NDC from camera coordinates after the world->view rotation, where view is right-handed
    # with +x right, +y up, +z forward. So forward = at - eye.)
    z_axis = at_t - eye_t
    z_axis = z_axis / (z_axis.norm(dim=-1, keepdim=True) + 1e-8)
    x_axis = torch.linalg.cross(up_t, z_axis, dim=-1)
    x_axis = x_axis / (x_axis.norm(dim=-1, keepdim=True) + 1e-8)
    y_axis = torch.linalg.cross(z_axis, x_axis, dim=-1)
    R = torch.stack([x_axis, y_axis, z_axis], dim=-1)  # (N, 3, 3) columns are axes
    # pytorch3d uses row-vector convention so they actually return R such that
    # p_world @ R + T = p_camera. The rows of R are the camera axes in world frame.
    # Our column-stack above has columns = axes, so we transpose.
    R = R.transpose(-1, -2)
    T = -torch.einsum("nij,nj->ni", R.transpose(-1, -2), eye_t)
    return R, T


# ---------------------------------------------------------------------------
# Stubs for unused-but-imported names. We make them harmless classes that
# raise when actually used.
# ---------------------------------------------------------------------------


class _NotImplementedClass:
    _name = "<unknown>"

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            f"pytorch3d.renderer.{self._name} is not available in the Mac stub. "
            "This is the rendering / mesh-postprocessing path; use the Gaussian-only branch."
        )


def _make(name):
    return type(name, (_NotImplementedClass,), {"_name": name})


PerspectiveCameras = _make("PerspectiveCameras")
FoVPerspectiveCameras = _make("FoVPerspectiveCameras")
FoVOrthographicCameras = _make("FoVOrthographicCameras")
OrthographicCameras = _make("OrthographicCameras")
RasterizationSettings = _make("RasterizationSettings")
MeshRasterizer = _make("MeshRasterizer")
MeshRenderer = _make("MeshRenderer")
SoftPhongShader = _make("SoftPhongShader")
HardPhongShader = _make("HardPhongShader")
PointLights = _make("PointLights")
AmbientLights = _make("AmbientLights")
TexturesAtlas = _make("TexturesAtlas")
TexturesUV = _make("TexturesUV")
TexturesVertex = _make("TexturesVertex")


# Sub-module aliases
class _MeshSubmodule:
    class textures:
        TexturesVertex = TexturesVertex
        TexturesAtlas = TexturesAtlas
        TexturesUV = TexturesUV


mesh = _MeshSubmodule
cameras = type("cameras_module", (), {
    "CamerasBase": _make("CamerasBase"),
    "PerspectiveCameras": PerspectiveCameras,
    "FoVPerspectiveCameras": FoVPerspectiveCameras,
})
camera_utils = type("camera_utils_module", (), {
    "camera_to_eye_at_up": lambda *a, **k: (_ for _ in ()).throw(
        NotImplementedError("camera_to_eye_at_up not in stub")
    ),
})


__all__ = [
    "look_at_view_transform",
    "PerspectiveCameras",
    "FoVPerspectiveCameras",
    "FoVOrthographicCameras",
    "OrthographicCameras",
    "RasterizationSettings",
    "MeshRasterizer",
    "MeshRenderer",
    "SoftPhongShader",
    "HardPhongShader",
    "PointLights",
    "AmbientLights",
    "TexturesAtlas",
    "TexturesUV",
    "TexturesVertex",
    "mesh",
    "cameras",
    "camera_utils",
]
