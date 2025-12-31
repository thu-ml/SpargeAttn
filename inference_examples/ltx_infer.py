import torch, os
import argparse
from diffusers import LTXConditionPipeline, LTXLatentUpsamplePipeline
from inference_examples.modify_model.modify_ltx import set_sparge_ltx
from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
from diffusers.utils import export_to_video, load_image, load_video
from tqdm import tqdm

prompt_path = "evaluate/datasets/video/prompts.txt"


def float2str(x: float) -> str:
    s = str(x)
    return s.replace('.', '_')


def round_to_nearest_resolution_acceptable_by_vae(height, width, pipe):
    height = height - (height % pipe.vae_spatial_compression_ratio)
    width = width - (width % pipe.vae_spatial_compression_ratio)
    return height, width


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "cdfthreshd", "topk"], default="full", help="Inference mode with different APIs")
    parser.add_argument("--value", type=float, default=None, help="Value of API argument")
    parser.add_argument("--start", type=int, default=0, help="Starting prompt id of this run.")
    parser.add_argument("--end", type=int, default=12, help="Ending prompt id of this run.")
    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    if args.mode != "full":
        assert args.value is not None, "Must set the value for cdfthreshold or topk."
        dir_name = f"{args.mode}-{float2str(args.value)}"
    else: dir_name = "original"
    video_dir = f"inference_examples/videos/ltx/{dir_name}"
    os.makedirs(video_dir, exist_ok=True)

    with open(prompt_path, "r", encoding="utf-8") as file:
        prompts = file.readlines()
    selected_prompts = [p.strip() for p in prompts[args.start:args.end]]

    pipe_id = "Lightricks/LTX-Video-0.9.7-dev"
    upscaler_id = "Lightricks/ltxv-spatial-upscaler-0.9.7"

    pipe = LTXConditionPipeline.from_pretrained(
        pipe_id,
        torch_dtype=torch.bfloat16,
    )
    # Inject the modified Processor
    set_sparge_ltx(pipe.transformer, mode=args.mode, value=args.value)

    pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    pipe_upsample = LTXLatentUpsamplePipeline.from_pretrained(
        upscaler_id,
        vae=pipe.vae,
        torch_dtype=torch.bfloat16,
    )
    pipe_upsample.enable_model_cpu_offload()

    for local_i, prompt in tqdm(enumerate(selected_prompts), total=len(selected_prompts)):
        global_i = args.start + local_i
        # image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/penguin.png")
        image = load_image(f"inference_examples/videos/ltx_first_frames/{global_i}.png")
        video = load_video(export_to_video([image]))  # compress the image using video compression as the model was trained on videos
        condition1 = LTXVideoCondition(video=video, frame_index=0)

        negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"
        expected_height, expected_width = 480, 832
        downscale_factor = 2 / 3
        num_frames = 96

        # Part 1. Generate video at smaller resolution
        downscaled_height, downscaled_width = int(expected_height * downscale_factor), int(expected_width * downscale_factor)
        downscaled_height, downscaled_width = round_to_nearest_resolution_acceptable_by_vae(downscaled_height, downscaled_width, pipe)
        latents = pipe(
            conditions=[condition1],
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=downscaled_width,
            height=downscaled_height,
            num_frames=num_frames,
            num_inference_steps=30,
            generator=torch.Generator().manual_seed(0),
            output_type="latent",  # Crucial: ask the pipeline to return latents, not decoded frames
        ).frames

        # Part 2. Upscale generated video using latent upsampler with fewer inference steps
        # The available latent upsampler upscales the height/width by 2x
        upscaled_height, upscaled_width = downscaled_height * 2, downscaled_width * 2
        upscaled_latents = pipe_upsample(
            latents=latents,
            output_type="latent"
        ).frames

        # Part 3. Denoise the upscaled video with few steps to improve texture (optional, but recommended)
        video = pipe(
            conditions=[condition1],
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=upscaled_width,
            height=upscaled_height,
            num_frames=num_frames,
            denoise_strength=0.4,  # Effectively, 4 inference steps out of 10
            num_inference_steps=10,
            latents=upscaled_latents,
            decode_timestep=0.05,
            image_cond_noise_scale=0.025,
            generator=torch.Generator().manual_seed(0),
            output_type="pil",
        ).frames[0]

        # Part 4. Downscale the video to the expected resolution
        video = [frame.resize((expected_width, expected_height)) for frame in video]
        export_to_video(video, f"{video_dir}/{global_i}.mp4", fps=24)
        del video

if __name__ == "__main__":
    main()
