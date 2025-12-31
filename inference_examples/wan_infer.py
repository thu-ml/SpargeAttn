import torch, argparse, gc, os
from tqdm import tqdm
from diffusers.utils import export_to_video
from diffusers import WanPipeline
from inference_examples.modify_model.modify_wan import set_sparge_wan
from contextlib import nullcontext


os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:64")
os.environ["TOKENIZERS_PARALLELISM"]="false"
negative_prompt_1 = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
negative_prompt_2 = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
prompt_path = "evaluate/datasets/video/prompts.txt"


def float2str(x: float) -> str:
    s = str(x)
    return s.replace('.', '_')


def parse_args():
    parser = argparse.ArgumentParser(description="Wan Inference")
    parser.add_argument(
        "--model",
        choices=["wan2_1-1_3b", "wan2_1-14b", "wan2_2-14b"],
        default="wan2_1-1_3b",
        help="Wan model",
    )
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
        dir_name = f"{args.mode}-{float2str(args.value)}"
    else: dir_name = "original"

    with open(prompt_path, "r", encoding="utf-8") as file:
        prompts = file.readlines()
    selected_prompts = [p.strip() for p in prompts[args.start:args.end]]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = torch.Generator(device=device).manual_seed(42)

    if args.model == "wan2_1-1_3b": model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    elif args.model == "wan2_1-14b": model_id = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
    else: model_id = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"

    pipe = WanPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
    )

    set_sparge_wan(pipe.transformer, mode=args.mode, value=args.value)
    if getattr(pipe, "transformer_2", None) is not None: # Wan2.2
        set_sparge_wan(pipe.transformer_2, mode=args.mode, value=args.value)

    pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    pipe.vae.decoder_chunk_size = 1

    video_fps = 16 if args.model == "wan2_2-14b" else 15
    video_dir = f"inference_examples/videos/{args.model}/{dir_name}"
    os.makedirs(video_dir, exist_ok=True)

    for local_i, prompt in tqdm(enumerate(selected_prompts), total=len(selected_prompts)):
        global_i = args.start + local_i
        amp_ctx = torch.autocast("cuda", torch.bfloat16, cache_enabled=False) if device == "cuda" else nullcontext()
        with amp_ctx:
            if args.model != "wan2_2-14b": # Wan2.1
                output = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt_1,
                    height=480,
                    width=832,
                    num_frames=81,
                    guidance_scale=5.0,
                    generator=gen,
                ).frames[0]
            else: # Wan2.2
                output = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt_2,
                    height=720,
                    width=1280,
                    num_frames=81,
                    guidance_scale=4.0,
                    guidance_scale_2=3.0,
                    num_inference_steps=40,
                    generator=gen,
                ).frames[0]

            export_to_video(output, f"{video_dir}/{global_i}.mp4", fps=video_fps)
            del output
            gc.collect()
            torch.cuda.empty_cache()
