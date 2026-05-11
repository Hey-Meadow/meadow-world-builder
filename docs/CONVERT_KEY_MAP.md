# SAM 3D Objects weight key map (PyTorch -> MLX)

**Status**: scaffolded only. The actual generated table will be produced
when `convert.py` is run on the gated HF checkpoint (see
`mlx_port/docs/CHECKPOINT_REPORT.md` for access notes).

This file documents the **planned** key mapping (derived from source
code in `sam3d_objects/model/backbone/...`). The convert.py will
overwrite this file with the real per-tensor table once it is run on
real checkpoints.

## File summary (planned)

| file                     | source ckpt              | est. tensors | est. size |
|--------------------------|--------------------------|-------------:|----------:|
| `ss_flow.npz`            | ss_generator.ckpt        | ~700-1000    | ~6.5 GB   |
| `ss_embedder.npz`        | ss_generator.ckpt        | ~150-300     | ~0.4 GB   |
| `slat_flow.npz`          | slat_generator.ckpt      | ~700-1000    | ~4.7 GB   |
| `slat_embedder.npz`      | slat_generator.ckpt      | ~150-300     | ~0.4 GB   |
| `ss_decoder.npz`         | ss_decoder.ckpt          | ~50-100      | ~150 MB   |
| `slat_decoder_gs.npz`    | slat_decoder_gs.ckpt     | ~150-300     | ~170 MB   |
| `slat_decoder_gs_4.npz`  | slat_decoder_gs_4.ckpt   | ~150-300     | ~170 MB   |
| `ss_encoder.npz`         | ss_encoder.safetensors   | ~50          | ~10 MB    |

## Tensor layout conventions

- **Conv2d weights** `(out, in, kH, kW)` are transposed to MLX channels-last
  `(out, kH, kW, in)`. Affects DINO `patch_embed.proj.weight` and any
  2D convs in the embedder fuser.
- **Conv3d weights** `(out, in, kD, kH, kW)` are transposed to MLX
  channels-last `(out, kD, kH, kW, in)`. Affects every Conv3d in
  `SparseStructureEncoder` / `SparseStructureDecoder`.
- **Linear / LayerNorm / RMSNorm / position-embedding** tensors kept verbatim.
- **Integer buffers** (`*.faces`, `*.indices`, `*.coords`) stay int64.
- Everything else cast to float32.

## Sample mapping (representative keys per group)

These are the planned mappings, based on source code reading:

| group        | pt_key (post strip)                                  | mx_key (same)                                     | notes |
|--------------|------------------------------------------------------|---------------------------------------------------|-------|
| ss_flow      | `backbone.t_embedder.mlp.0.weight`                   | `backbone.t_embedder.mlp.0.weight`                | timestep MLP fc1 |
| ss_flow      | `backbone.t_embedder.mlp.0.bias`                     | `backbone.t_embedder.mlp.0.bias`                  |  |
| ss_flow      | `backbone.t_embedder.mlp.2.weight`                   | `backbone.t_embedder.mlp.2.weight`                | timestep MLP fc2 |
| ss_flow      | `backbone.input_layer.weight`                        | `backbone.input_layer.weight`                     | patch embed Linear |
| ss_flow      | `backbone.pos_emb`                                   | `backbone.pos_emb`                                | absolute 3D pos buffer |
| ss_flow      | `backbone.blocks.0.norm1.weight`                     | `backbone.blocks.0.norm1.weight`                  | LayerNorm |
| ss_flow      | `backbone.blocks.0.attn.to_qkv.weight`               | `backbone.blocks.0.attn.to_qkv.weight`            | self-attn fused QKV |
| ss_flow      | `backbone.blocks.0.attn.to_out.weight`               | `backbone.blocks.0.attn.to_out.weight`            |  |
| ss_flow      | `backbone.blocks.0.cross_attn.to_q.weight`           | `backbone.blocks.0.cross_attn.to_q.weight`        | cross-attn Q |
| ss_flow      | `backbone.blocks.0.cross_attn.to_kv.weight`          | `backbone.blocks.0.cross_attn.to_kv.weight`       | cross-attn fused KV |
| ss_flow      | `backbone.blocks.0.mlp.fc1.weight`                   | `backbone.blocks.0.mlp.fc1.weight`                | feedforward |
| ss_flow      | `backbone.blocks.0.adaLN_modulation.1.weight`        | `backbone.blocks.0.adaLN_modulation.1.weight`     | 6-param modulation Linear |
| ss_flow      | `backbone.out_layer.weight`                          | `backbone.out_layer.weight`                       |  |
| ss_embedder  | `backbone.dino.patch_embed.proj.weight`              | `backbone.dino.patch_embed.proj.weight`           | Conv2d, transposed (out,kH,kW,in) |
| ss_embedder  | `backbone.dino.blocks.0.norm1.weight`                | `backbone.dino.blocks.0.norm1.weight`             | DINO ViT block |
| ss_embedder  | `backbone.fuser.image_proj.weight`                   | `backbone.fuser.image_proj.weight`                | EmbedderFuser projection |
| slat_flow    | `backbone.input_layer.weight`                        | `backbone.input_layer.weight`                     | (similar tower as ss_flow) |
| ss_decoder   | `input_layer.weight`                                 | `input_layer.weight`                              | Conv3d, transposed (out,kD,kH,kW,in) |
| ss_decoder   | `blocks.0.0.conv1.weight`                            | `blocks.0.0.conv1.weight`                         | ResBlock3d Conv3d |
| ss_decoder   | `blocks.0.0.norm1.weight`                            | `blocks.0.0.norm1.weight`                         | GroupNorm32 |
| ss_decoder   | `out_layer.5.weight`                                 | `out_layer.5.weight`                              | final Conv3d |
| slat_decoder_gs | `input_layer.weight`                              | `input_layer.weight`                              | Linear (sparse-to-dense) |
| slat_decoder_gs | `blocks.0.norm1.weight`                           | `blocks.0.norm1.weight`                           | SparseTransformer LayerNorm |
| slat_decoder_gs | `blocks.0.attn.to_qkv.weight`                     | `blocks.0.attn.to_qkv.weight`                     |  |
| slat_decoder_gs | `out_layer_xyz.weight`                            | `out_layer_xyz.weight`                            | GS xyz head |
| slat_decoder_gs | `out_layer_scale.weight`                          | `out_layer_scale.weight`                          | GS scale head |
| slat_decoder_gs | `out_layer_rotation.weight`                       | `out_layer_rotation.weight`                       | GS rotation head |
| slat_decoder_gs | `out_layer_opacity.weight`                        | `out_layer_opacity.weight`                        | GS opacity head |
| slat_decoder_gs | `out_layer_features_dc.weight`                    | `out_layer_features_dc.weight`                    | GS color head |

The MLX-side keys are **the same as PT-side keys after prefix stripping**.
This 1:1 mapping is intentional - it lets us reuse SAM 3D Body's loader
verbatim with one helper that walks the npz and assigns tensors by exact
name match.
