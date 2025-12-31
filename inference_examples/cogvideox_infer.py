import torch, os, gc
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
import argparse
from tqdm import tqdm
from inference_examples.modify_model.modify_cogvideox import set_sparge_cogvideox


prompt_path = "evaluate/datasets/video/prompts.txt"


def parse_args():
    parser = argparse.ArgumentParser(description="CogVideoX Inference")
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
    video_dir = f"inference_examples/videos/cogvideox/{dir_name}"
    os.makedirs(video_dir, exist_ok=True)

    with open(prompt_path, "r", encoding="utf-8") as file:
        prompts = file.readlines()
    selected_prompts = [p.strip() for p in prompts[args.start:args.end]]

    pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-2b", torch_dtype=torch.bfloat16)
    set_sparge_cogvideox(pipe.transformer, mode=args.mode, value=args.value)

    pipe.enable_model_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    for local_i, prompt in tqdm(enumerate(selected_prompts), total=len(selected_prompts)):
        global_i = args.start + local_i
        video = pipe(
            prompt=prompt,
            num_videos_per_prompt=1,
            num_inference_steps=50,
            num_frames=49,
            guidance_scale=6,
            generator=torch.Generator(device="cuda").manual_seed(42),
        ).frames[0]

        export_to_video(video, f"{video_dir}/{global_i}.mp4", fps=8)
        del video
        gc.collect()
        torch.cuda.empty_cache()
