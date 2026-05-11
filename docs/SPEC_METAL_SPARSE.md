# SPEC_METAL_SPARSE.md — Agent OBJ-METAL-SPARSE (SubMConv3d Metal kernel)

## Goal
Replace `spconv.pytorch.SubMConv3d` (the only critical CUDA hot path) with a Metal kernel callable from MLX.

## Why this matters
Per `mlx_port/docs/CUDA_DEPS.md`:
- ~800 SubMConv3d calls per inference (50 ODE steps × 8 calls per step × 2 stages)
- **Submanifold rule**: input/output coords are IDENTICAL → coord→row hash table can be **built once per slat sample, reused all 50 steps**
- This is the single biggest performance lever

## Background: SubMConv3d (submanifold sparse 3D convolution)
- Input: sparse tensor with N voxels, each at integer 3D coord, with feature vector (N, C_in)
- Output: sparse tensor with SAME N voxels at SAME coords, but features (N, C_out)
- For each voxel `v`, gather neighbors within k×k×k window (k=3) AT EXACT COORDS that exist in input set
- Compute output feature = sum over found neighbors of (neighbor_feature @ kernel_weight[offset])

### Standard implementation strategy
1. **Hash table**: map (x,y,z) tuple → row index in feature buffer
2. **For each voxel and each kernel offset (-1..+1)³ = 27 offsets**:
   - Look up (x+dx, y+dy, z+dz) in hash table
   - If found, add `feature[neighbor_row] @ weight[offset_idx]` to output
3. With k=3, 27 offsets per voxel per layer

### Submanifold optimization
Since output coords = input coords, the neighbor lookup table is the same across all SubMConv3d calls within one inference. Build once, reuse 800× per inference.

## Inputs
- **PT source** (for reference): look at `spconv.pytorch.SubMConv3d` usage in `sam3d_objects/model/backbone/tdfy_dit/modules/sparse/conv/`
- **Plan**: `mlx_port/docs/PORT_PLAN.md`
- **Recon**: `mlx_port/docs/CUDA_DEPS.md`
- **Reference for `mx.fast.metal_kernel` API**:
  - https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html
  - `/Users/akaihuangm1/Desktop/github/sam-3d-body/mlx_port/kernels/add_layernorm_kernel.py` — example of `mx.fast.metal_kernel` usage

## Required deliverables

### 1. `mlx_port/kernels/sparse_subm_conv3d.metal` (or inline in Python)
```metal
// Submanifold 3D convolution kernel
// Inputs:
//   features (N, C_in)              — input features per voxel
//   coords   (N, 3)                 — int32 voxel coordinates
//   weights  (27, C_in, C_out)      — 3x3x3 kernel weights
//   neighbor_table (N, 27) int32    — pre-computed neighbor row indices, -1 if absent
// Output:
//   out (N, C_out)
//
// Each thread computes one output voxel × C_out chunk.
```

### 2. `mlx_port/kernels/sparse_subm_conv3d.py`
```python
import mlx.core as mx

class SparseSubmConv3d:
    """SubMConv3d with cached neighbor table (submanifold optimization)."""
    
    def __init__(self, in_ch, out_ch, kernel_size=3):
        self.weight = ...  # (k³, in_ch, out_ch) MLX array
        self.kernel = mx.fast.metal_kernel(...)
        self._neighbor_cache = {}  # coords_id -> neighbor_table

    def build_neighbor_table(self, coords: mx.array) -> mx.array:
        """Build (N, k³) int32 neighbor table from coords (N, 3).
        Returns -1 for offsets that don't exist in input set.
        Cache by hash of coords."""
        ...

    def __call__(self, features: mx.array, coords: mx.array) -> mx.array:
        nt = self.build_neighbor_table(coords)  # cached
        return self.kernel(inputs=[features, nt, self.weight], ...)
```

### 3. `mlx_port/tests/test_sparse_subm_conv3d.py`
- Random voxel coords (N=1000, in 0..32 grid), random features
- Naive numpy reference (correct but slow)
- Compare MLX kernel output vs numpy: max_abs_diff < 1e-3 fp32

## Performance budget
- Target: < 5ms per call (so 800 calls = 4 sec, not the dominant cost anymore)
- Compare to MLX's gather+matmul fallback (probably 10-30ms per call) — must be at least 2× faster

## Constraints
- Pure MLX + Metal
- Use `mx.fast.metal_kernel`
- DO NOT modify `sam3d_objects/` (read-only) — only ADD files in `mlx_port/`
- Working dir: `/Users/akaihuangm1/Desktop/github/sam-3d-objects/`
- Python: `/Users/akaihuangm1/Desktop/github/sam-3d-body/.venv/bin/python`

## Honest fallback
If `mx.fast.metal_kernel` is too cumbersome OR you can't beat MLX's gather+matmul:
- Implement pure-MLX version using `mx.gather` to assemble (N, k³, C_in) tensor, then `mx.einsum` with weights
- Document why custom Metal didn't pay off
- This is OK — accuracy matters more than speed for first pass

## Definition of done
1. `sparse_subm_conv3d.py` provides `SparseSubmConv3d` class
2. Test passes with max_abs_diff < 1e-3 vs reference
3. Per-call timing reported (compared to MLX fallback if available)
4. Report (≤ 200 words):
   - Implementation approach (custom Metal vs MLX fallback)
   - Hash table caching effectiveness
   - Speed comparison
   - Any spconv-specific quirks worth noting
