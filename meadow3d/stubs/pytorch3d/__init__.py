# Mac-compatible stub for pytorch3d (CUDA-only on Linux).
# Provides minimal API surface needed by SAM 3D Objects inference path.
# Authored 2026-05 by OBJ-ENV agent for the MLX port.
#
# What is implemented (real torch math):
#   - pytorch3d.transforms.{Transform3d, quaternion_to_matrix, matrix_to_quaternion,
#     quaternion_multiply, quaternion_invert, axis_angle_to_*, euler_angles_to_matrix}
#   - pytorch3d.renderer.look_at_view_transform
#
# What is stubbed (raises NotImplementedError on use, but imports succeed):
#   - pytorch3d.structures.{Meshes, Pointclouds}
#   - pytorch3d.renderer.{PerspectiveCameras, RasterizationSettings, MeshRasterizer,
#     TexturesVertex, FoVPerspectiveCameras, FoVOrthographicCameras, ...}
#   - pytorch3d.vis.plotly_vis
#
# These are only invoked by visualization and mesh-postprocessing paths.
# The Gaussian-only inference path should never need them.

__version__ = "0.7.7+stub"
