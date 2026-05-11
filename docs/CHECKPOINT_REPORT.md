# SAM 3D Objects Checkpoint Report

Agent OBJ-WEIGHTS, 2026-05-08.

## 1. Source

HuggingFace repo: [`facebook/sam-3d-objects`](https://huggingface.co/facebook/sam-3d-objects).

Status: **gated, manual approval**. Authentication test:
```
USER:               akaiii
canReadGatedRepos:  true   (token level)
facebook/sam-3d-objects:
  gated:    "manual"
  HEAD ckpt -> 403 GatedRepo, "you are not in the authorized list"
```

Meaning: the personal HF token already has `canReadGatedRepos`, but Meta
must additionally approve **this specific repo** via the access form on
<https://huggingface.co/facebook/sam-3d-objects>. Once approved the
existing token will work without reissue.

**Action required (user)**: visit the repo page in a browser, click
"Request access", fill the form, wait for approval. Then re-run
`hf auth login` is not needed — same token works.

## 2. Repo file inventory (from HF API, no download required)

| size (GB) | file                                       | role |
|----------:|--------------------------------------------|------|
|     6.690 | `checkpoints/ss_generator.ckpt`            | Stage-1 flow DiT (image -> sparse occupancy) |
|     4.907 | `checkpoints/slat_generator.ckpt`          | Stage-2 flow DiT (occupancy -> structured latent) |
|     0.364 | `checkpoints/slat_decoder_mesh.pt`         | Mesh decoder (skipped per scope) |
|     0.364 | `checkpoints/slat_decoder_mesh.ckpt`       | Mesh decoder (skipped per scope) |
|     0.171 | `checkpoints/slat_decoder_gs.ckpt`         | Latent -> Gaussian splat params |
|     0.170 | `checkpoints/slat_decoder_gs_4.ckpt`       | Latent -> Gaussian splat (high-res) |
|     0.148 | `checkpoints/ss_decoder.ckpt`              | Sparse-structure decoder |
|     0.000 | `checkpoints/ss_encoder.safetensors`       | Sparse-structure encoder (small; used for re-encoding) |
|     ~0    | `checkpoints/*.yaml`                       | Hydra configs |
|    ~0.6   | `doc/*.png`, `doc/*.gif`                   | Docs (not needed) |
|     ~12.9 | **TOTAL repo**                             |  |

For the MLX inference path (Gaussian splats only, no mesh) we need:
- `ss_generator.ckpt`            (6.69 GB)
- `slat_generator.ckpt`          (4.91 GB)
- `ss_decoder.ckpt`              (0.15 GB)
- `slat_decoder_gs.ckpt`         (0.17 GB)
- `slat_decoder_gs_4.ckpt`       (0.17 GB)
- All `*.yaml`                   (kB)
- (optional) `ss_encoder.safetensors` for round-trip testing

**Effective download: ~12.1 GB.**
We intentionally skip `slat_decoder_mesh.{pt,ckpt}` (mesh path is out of
scope per `PORT_PLAN.md`).

## 3. Source-code-derived state-dict layout

Verified by reading
`sam3d_objects/pipeline/inference_pipeline.py:306-417` and
`sam3d_objects/model/io.py`.

### 3.1 ss_generator.ckpt and slat_generator.ckpt

Both are PyTorch-Lightning checkpoints. Top-level dict has key
`"state_dict"`. The state_dict itself uses two prefixes:

| prefix                                  | what it is                          | strip rule applied at convert time |
|-----------------------------------------|-------------------------------------|------------------------------------|
| `_base_models.generator.`               | CFG-wrapped TdfyWrapper (the DiT)   | strip prefix; keys land in `*_flow.npz` |
| `_base_models.condition_embedder.`      | DINO + EmbedderFuser (image cond)   | strip prefix; keys land in `*_embedder.npz` |

The DiT side, after stripping, looks like (from
`sparse_structure_flow.py:72-211` and
`structured_latent_flow.py:77-303`):

```
backbone.t_embedder.mlp.0.{weight,bias}
backbone.t_embedder.mlp.2.{weight,bias}
backbone.adaLN_modulation.1.{weight,bias}     # only if share_mod=True
backbone.input_layer.{weight,bias}
backbone.pos_emb                               # buffer, AbsolutePositionEmbedder
backbone.blocks.{0..N-1}.{...}                 # ModulatedTransformerCrossBlock
backbone.out_layer.{weight,bias}
backbone.condition_embedder.{...}              # nested DINO+fuser (sometimes)
```

Plus the outer CFG wrapper adds nothing learnable - `ClassifierFreeGuidance`
just holds `backbone` so the prefix really is `backbone.` not
`reverse_fn.backbone.`. (The `reverse_fn` you see in
`override_ss_generator_cfg_config` is set by the surrounding generator
(`FlowMatching`/`PointmapCFG`) which itself owns `backbone` -- need to
double-check the actual prefix once we have the ckpt. The converter is
prefix-agnostic: it just strips `_base_models.generator.` and writes
whatever remains.)

### 3.2 ss_decoder.ckpt, slat_decoder_gs*.ckpt

Loaded via `instantiate_and_load_from_pretrained(state_dict_key=None)`
- **no `state_dict` wrapper, no prefix**. Keys map directly onto:
- `SparseStructureDecoder`     (`sparse_structure_vae.py:226`)
- `SLatGaussianDecoder`        (`structured_latent_vae/decoder_gs.py:15`)

Expected key shapes:
- `input_layer.weight` `(channels[0], in_C, 3, 3, 3)` Conv3d
- Tower of `ResBlock3d` -> `UpsampleBlock3d` -> `ResBlock3d` ...
- `out_layer.{weight,bias}` (Conv3d)
- For SLatGaussianDecoder: SparseTransformerBase blocks (transformer
  layers operating on sparse latents) + linear head producing GS params
  (xyz, scale, rotation, opacity, sh).

### 3.3 ss_encoder.safetensors

Tiny (<10 MB pointer). Loaded via `safetensors.torch.load_file` with
`state_dict_key=None`. Keys map directly onto `SparseStructureEncoder`
(Conv3d tower symmetric to the decoder).

## 4. Dtype expectations

PyTorch checkpoints will be a mix of:
- `float32` for everything by default
- `float16` for the DiT torso if `use_fp16=True` (see
  `sparse_structure_flow.py:163-164`). Most public DiT releases ship the
  ckpt in float32 even when training was fp16, so we expect float32 in
  the file.
- `int64` for the few index/coord buffers (e.g. `pos_emb` is float32, no
  ints expected outside coord buffers in encoder/decoder).

The converter casts everything float32 anyway (matches MLX default), so
we are robust either way.

## 5. Conversion plan (mirrors SAM 3D Body pattern)

| Output npz             | Source ckpt(s)         | Source prefix to strip                | Target MLX module |
|------------------------|------------------------|---------------------------------------|-------------------|
| `ss_flow.npz`          | ss_generator.ckpt      | `_base_models.generator.`             | `models/dit_mlx.py` (SS variant) |
| `ss_embedder.npz`      | ss_generator.ckpt      | `_base_models.condition_embedder.`    | `models/embedders_mlx.py` |
| `slat_flow.npz`        | slat_generator.ckpt    | `_base_models.generator.`             | `models/dit_mlx.py` (SLat variant) |
| `slat_embedder.npz`    | slat_generator.ckpt    | `_base_models.condition_embedder.`    | `models/embedders_mlx.py` |
| `ss_decoder.npz`       | ss_decoder.ckpt        | (none; bare state_dict)               | `models/decoder_mlx.py` (SS) |
| `slat_decoder_gs.npz`  | slat_decoder_gs.ckpt   | (none)                                | `models/decoder_mlx.py` (GS) |
| `slat_decoder_gs_4.npz`| slat_decoder_gs_4.ckpt | (none)                                | `models/decoder_mlx.py` (GS hi-res) |
| `ss_encoder.npz`       | ss_encoder.safetensors | (none)                                | optional |
| `slat_decoder_mesh.*`  | NOT CONVERTED          | -                                     | out of scope     |

Conv2d weights (DINO `patch_embed.proj.weight`) and Conv3d weights
(everything in the SS encoder/decoder tower) are transposed to MLX
channels-last layout at convert time - mirrors the SAM 3D Body
convention.

## 6. RAM budget during conversion

`ss_generator.ckpt` is 6.7 GB on disk. `torch.load(..., map_location="cpu")`
will hold the entire state_dict in RAM. With float32 arrays the
unzipped state is roughly the same size. Each subsequent np.savez
also holds a full copy briefly. We estimate **peak ~14 GB RAM** while
processing this single ckpt.

The converter releases each ckpt's state_dict before loading the next
(`del sd`), so non-overlapping checkpoints don't compound the peak.

If running on a 16 GB Mac, set MLX swap to allow this; on a 32 GB+ Mac
no special handling needed.

## 7. GO / NO-GO

**Status: GO (pending HF access approval).**

All inputs are well-typed (float32 / int64), no proprietary formats, no
mystery shape transforms. The converter is written + smoke-tested for
empty-dir behaviour. Once Meta approves access:

```bash
# 1. download
hf download facebook/sam-3d-objects \
    --local-dir checkpoints/hf-download \
    --include 'checkpoints/ss_generator.ckpt' \
    --include 'checkpoints/slat_generator.ckpt' \
    --include 'checkpoints/ss_decoder.ckpt' \
    --include 'checkpoints/slat_decoder_gs.ckpt' \
    --include 'checkpoints/slat_decoder_gs_4.ckpt' \
    --include 'checkpoints/ss_encoder.safetensors' \
    --include 'checkpoints/*.yaml'
mv checkpoints/hf-download/checkpoints checkpoints/hf

# 2. inspect (verify our prefix assumptions match reality)
.venv/bin/python mlx_port/weights/convert.py \
    --ckpt-dir checkpoints/hf --inspect

# 3. convert
.venv/bin/python mlx_port/weights/convert.py \
    --ckpt-dir checkpoints/hf \
    --out mlx_port/weights/sam3d_objects/
```

Expected output:
- ~12 GB compressed npz across 7-8 files
- KEY_MAP.md with full per-tensor mapping
