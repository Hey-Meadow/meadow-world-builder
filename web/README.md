# Meadow iridescent WebGL viewer

A fork of [antimatter15/splat](https://github.com/antimatter15/splat) (MIT) with an iridescent / chrome / liquid-metal fragment-shader layer baked into the vertex stage. Loads any 3DGS `.ply`, applies a per-Gaussian thin-film hue band over a silver chrome base, drives the phase from `uniform float u_time` for live animation, and exposes 4 sliders for `strength / chrome↔candy / hue freq / animate`.

## Local test

From the **repo root**:

```bash
python3 -m http.server 8000
open http://localhost:8000/web/index.html
```

The page loads `assets/demos/oatchi.ply` by default. Drop in any other PLY via `?url=`:

```
http://localhost:8000/web/index.html?url=/assets/demos/pikmin_r.ply
```

## Deploy

The `web/` folder is a self-contained static site — `index.html` + `main.js`. Drop it on any static host (GitHub Pages, Cloudflare Pages, Netlify) along with whichever `.ply` files you want to expose.

For a public GitHub Pages deploy, run from this repo:

```bash
gh api -X POST repos/Hey-Meadow/meadow-world-builder/pages \
    -f source[branch]=feature/iridescent-shader -f source[path]=/
```

Then open `https://hey-meadow.github.io/meadow-world-builder/web/`.

## Files

- `index.html` — viewer page, slider UI, defaults
- `main.js` — antimatter15 WebGL splat renderer + Meadow iridescent shader patch (see `vertexShaderSource` near line 655)

## What the shader does

Per Gaussian, in the vertex stage:

```glsl
vec3 n = normalize(worldPos - u_scene_center);
vec3 viewDir = normalize(u_camera_pos - worldPos);
float ndotv = abs(dot(n, viewDir));
float phase = u_animate * u_time * 0.25;
float hue = mod(ndotv * u_freq + phase, 1.0);
vec3 rainbow = 0.5 + 0.5 * sin(hue * 6.28318531 + vec3(0.0, 2.0944, 4.18879));
// chrome blend with edge-bands + spec, then mix with original SH-DC colour
```

Identical maths to `meadow_wb/scripts/apply_iridescent.py` so the offline GIF and the live viewer produce the same look (modulo viewer rasteriser differences).
