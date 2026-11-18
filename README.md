# ComfyUI-MossBooster

A ComfyUI node that makes MOSS Image generation faster on consumer NVIDIA GPUs, without lowering image quality.

| Workflow (832×1264, 25 steps, CFG 3) | Before | After |
|---|---|---|
| No LoRA | 61 s | **50.5 s** |
| 4 LoRAs | 67–70 s | **50.1 s** |

Measured on an RTX 4050 Laptop GPU (6 GB) with the INT8/INT6 MOSS Image checkpoint, ComfyUI 0.45.

## Install

Clone into `ComfyUI/custom_nodes` and restart ComfyUI:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/KidiXDev/ComfyUI-MossBooster
```

On Windows, run the same `git clone` in `ComfyUI\custom_nodes\`, or download the ZIP and extract it there.

There are no extra dependencies. The INT8 attention kernel comes from `comfy_kitchen`, which ComfyUI already ships for Linux and Windows. It needs an NVIDIA RTX 20-series GPU or newer; on other GPUs the node keeps the default attention.

## Usage

Add **MOSS Booster** (category `advanced/model`) after the LoRA loader and before `ModelSamplingAuraFlow`:

```
Load Diffusion Model → Load LoRA (Model) → MOSS Booster → ModelSamplingAuraFlow → SkipLayerGuidanceDiTSimple → KSampler
```

Without LoRAs, connect it right after `Load Diffusion Model`. [`workflows/`](workflows) has both setups in API format.

| Option | Default | Effect |
|---|---|---|
| `bake_lora` | on | Merges the LoRAs into the weights once. The image is bit-identical to normal LoRA loading. |
| `int8_attention` | on | Runs the diffusion model's attention in INT8. The text encoder and VAE stay exact. |
| `int8_start_percent` | 0.0 | Keeps exact attention for this share of the first steps. 0.2 stays closer to the exact image for about 2 s more. |

## Use with Spectrum

[ComfyUI-Spectrum-MOSS](https://github.com/Rkkss/ComfyUI-Spectrum-MOSS) forecasts the model's hidden state to skip whole passes. MOSS Booster makes the passes that do run cheaper, so the two stack. Spectrum skips the same steps with or without MOSS Booster.

```
Load LoRA (Model) → MOSS Booster → ModelSamplingAuraFlow → SkipLayerGuidanceDiTSimple → Spectrum MOSS-Image (Adaptive) → KSampler
```

4 LoRAs, er_sde, CFG 2.5, 25 steps, Spectrum Adaptive (warmup 7, detail guard 1, tail 1, horizon cap 2):

| Setup | SLG 0.35–0.7 | SLG 0.0–1.0 |
|---|---|---|
| Plain | 67 s | 62 s |
| MOSS Booster | 50 s | – |
| Spectrum | 62 s | 46 s |
| Spectrum + MOSS Booster | 43 s | 33 s |

Spectrum relearns its forecast whenever SkipLayerGuidance turns on or off, so a partial SLG range limits how many steps it can skip (16 of 75 passes here, against 27 of 75 with the full range). A full-range SLG changes the image somewhat, so compare a few seeds before switching.

## How it works

A profile of one MOSS Image forward pass on the RTX 4050 (1.24 s, 4,196 tokens) showed:

| Part | Share | Notes |
|---|---|---|
| Attention (BF16 flash attention) | 37% | Already about 85% of the GPU's BF16 peak |
| INT8 matrix multiplies | 47% | About 71 TOPS, near the GPU's INT8 peak |
| Activation quantization, norms, RoPE, residuals | 13% | Memory-bound |

CFG runs the model twice per step (the prompt and negative prompt have different lengths), so each step costs about 2.5 s.

### INT8 attention

`comfy_kitchen` ships a SageAttention-style kernel. It quantizes Q, K and V to INT8 after a Hadamard rotation that preserves the dot products, and keeps softmax in FP32. At MOSS's attention shape it takes 4.3 ms instead of 10.5 ms, with a cosine similarity of 0.99986 to an FP32 reference.

The node installs it through ComfyUI's `optimized_attention_override`, so it applies only to this model. Short sequences, such as the text blocks, keep the default kernel.

The resulting images match the exact ones in composition, color and sharpness. Small details can move slightly, as they would on a different GPU.

### LoRA baking

The model is about 5.1 GB, and the GPU has about 5.8 GB, so ComfyUI's dynamic VRAM keeps some layers in RAM and streams them in each step. For those layers ComfyUI dequantizes, applies the LoRA and requantizes again on every step. The LoRA tensors also take VRAM, so even more layers get streamed. Together this added 6–9 s per image.

The node runs that same cast once per patched layer, with the same dtype and the same rounding seed. It then rebuilds the model from the merged state dict, so sampling sees a plain INT8 model with no patches. The merged weights are identical to the on-the-fly ones, so the image is identical too. Baking takes about 5 s and runs again only when the LoRAs or strengths change.

The merged copy of the model uses about 5 GB of system RAM.

## Benchmarking

[`bench/`](bench) has the scripts behind the numbers above (Linux, bash). They start a headless ComfyUI on port 8190, time workflows, and compare images. `gen.py` inserts the node into any API-format workflow; replace the `your_lora_N.safetensors` placeholders in `wf_lora.json` with LoRAs you have, or pass your own workflow.

```bash
cd bench
COMFYUI_DIR=/path/to/ComfyUI ./start.sh srv --extra-model-paths-config /path/to/extra_model_paths.yaml
python3 run.py "$(python3 gen.py wf_lora.json 1 1 0.0)" --runs 2 --prefix boost
python3 run.py wf_lora.json --runs 2 --prefix exact
/path/to/ComfyUI/.venv/bin/python cmp.py .run/out/exact_0_00001_.png .run/out/boost_0_00001_.png
./stop.sh
```

## Development

```bash
uvx ruff check .
uvx ruff format .
```

## License

[MIT](LICENSE)
