# Inference Examples

This folder contains **plug-and-play inference scripts** for **four** Diffusers video models with the **SpargeAttn** API wired into their attention processors.

Supported models:

* **WAN:** [Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers), [Wan2.1-T2V-14B](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers) and [Wan2.2-T2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers)
* **LTX-Video:** [0.9.7-dev](https://huggingface.co/Lightricks/LTX-Video-0.9.7-dev) and [spatial upscaler](https://huggingface.co/Lightricks/ltxv-spatial-upscaler-0.9.7)
* [**HunyuanVideo**](https://huggingface.co/hunyuanvideo-community/HunyuanVideo)
* [**CogVideoX-2B**](https://huggingface.co/zai-org/CogVideoX-2b)

Output videos are saved under `inference_examples/videos/<model>/<run_dir>/`. `<run_dir>` is `original/` for `--mode full` or `cdfthreshd-<val>` / `topk-<val>` for SpargeAttn modes.

## Run Inference

Run scripts **as modules** from repo root:

```bash
python -m inference_examples.<script_name> [--flags]
```

Flags:

* `--model {wan2_1-1_3b, wan2_1-14b, wan2_2-14b}`: choose different Wan models, only used in [`wan_infer.py`](./wan_infer.py)
* `--mode {full,cdfthreshd,topk}`
    - `full`: baseline sdpa
    - `cdfthreshd`: SpargeAttn API `spas_sage2_attn_meansim_cuda`
    - `topk`: SpargeAttn API `spas_sage2_attn_meansim_topk_cuda`
* `--value <float>`: set value for `cdfthreshd` and `topk`
* `--start <int> --end <int>`: slice of prompt list $[start, end)$ from `evaluate/datasets/video/prompts.txt`

Each script seeds a `torch.Generator` for reproducibility.

Examples:
```bash
# Wan
python -m inference_examples.wan_infer \
  --model wan2_2-14b \
  --mode topk --value 0.4 \
  --start 0 --end 1

# LTX-Video and others similar
python -m inference_examples.ltx_infer --mode topk --value 0.5
```

## Notes
1. Memory helpers enabled: model offload / sequential offload, VAE tiling / slicing, `decoder_chunk_size=1`.
2. Change `model_id` to your local model path.
