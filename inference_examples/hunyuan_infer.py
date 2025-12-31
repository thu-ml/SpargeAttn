import torch, argparse, gc, os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from tqdm import tqdm
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.utils import export_to_video
from inference_examples.modify_model.modify_hunyuan import set_sparge_hunyuan
from contextlib import nullcontext

prompt_path = "evaluate/datasets/video/prompts.txt"

def parse_args():
    parser = argparse.ArgumentParser(description="HunyuanVideo Inference")
    parser.add_argument("--mode", choices=["full", "cdfthreshd", "topk"], default="full", help="Inference mode with different APIs")
    parser.add_argument("--value", type=float, default=None, help="Value of API argument")
    parser.add_argument("--start", type=int, default=0, help="Starting prompt id of this run.")
    parser.add_argument("--end", type=int, default=12, help="Ending prompt id of this run.")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    if args.mode != "full":
        assert args.value is not None, "Must set the value for cdfthreshold or topk."
        dir_name = f"{args.mode}-{str(args.value).replace('.', '_')}"
    else: dir_name = "original"
    video_dir = f"inference_examples/videos/hunyuan/{dir_name}"
    os.makedirs(video_dir, exist_ok=True)

    with open(prompt_path, "r", encoding="utf-8") as file:
        prompts = file.readlines()
    selected_prompts = [p.strip() for p in prompts[args.start:args.end]]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_id = "hunyuanvideo-community/HunyuanVideo"
    pipe = HunyuanVideoPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    set_sparge_hunyuan(pipe.transformer, mode=args.mode, value=args.value)

    # Enable memory savings
    pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    pipe.vae.decoder_chunk_size = 1

    for local_i, prompt in tqdm(enumerate(selected_prompts), total=len(selected_prompts)):
        global_i = args.start + local_i
        amp = torch.autocast("cuda", torch.bfloat16, cache_enabled=False) if device == "cuda" else nullcontext()
        with amp:
            output = pipe(
                prompt=prompt,
                height=320,
                width=512,
                num_frames=61,
                num_inference_steps=30,
                generator=torch.Generator(device=device).manual_seed(42),
            ).frames[0]
            export_to_video(output, f"{video_dir}/{global_i}.mp4", fps=8)
            del output
            gc.collect()
            torch.cuda.empty_cache()
