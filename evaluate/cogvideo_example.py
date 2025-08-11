import torch, os, gc
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
from diffusers.models import CogVideoXTransformer3DModel
import argparse
from tqdm import tqdm
from modify_model.modify_cogvideo import set_spas_sage_attn_cogvideox
from spas_sage_attn.autotune import (
    extract_sparse_attention_state_dict,
    load_sparse_attention_state_dict,
)

prompt_path = "evaluate/datasets/video/prompts.txt"

def parse_args():
    parser = argparse.ArgumentParser(description="Flux Evaluation")
    parser.add_argument("--tune", action="store_true", help="tuning hyperpamameters")
    parser.add_argument('--parallel_tune', action='store_true', help='enable prallel tuning')
    parser.add_argument('--l1', type=float, default=0.06, help='l1 bound for qk sparse')
    parser.add_argument('--pv_l1', type=float, default=0.065, help='l1 bound for pv sparse')
    parser.add_argument('--compile', action='store_true', help='Compile the model')
    parser.add_argument(
        "--use_spas_sage_attn", action="store_true", help="Use Sage Attention"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose")
    parser.add_argument(
        "--out_path",
        type=str,
        default="evaluate/datasets/video/cogvideo_sparge",
        help="out_path",
    )
    parser.add_argument(
        "--model_out_path",
        type=str,
        default="evaluate/models_dict/CogVideoX-2b_0.05_0.06.pt",
        help="model_out_path",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.out_path, exist_ok=True)

    dtype_ = torch.bfloat16
    num_frames_ = 49

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(prompt_path, "r", encoding="utf-8") as file:
        prompts = file.readlines()
    
    if args.parallel_tune:
        os.environ['PARALLEL_TUNE'] = '1'
    if args.tune == True:
        os.environ["TUNE_MODE"] = "1"  # enable tune mode

        transformer = CogVideoXTransformer3DModel.from_pretrained(
            "THUDM/CogVideoX-2b",
            subfolder="transformer",
            torch_dtype=dtype_,
        )

        if args.use_spas_sage_attn:
            set_spas_sage_attn_cogvideox(transformer, verbose=args.verbose, l1=args.l1, pv_l1=args.pv_l1)

        pipe = CogVideoXPipeline.from_pretrained(
            "THUDM/CogVideoX-2b",
            transformer=transformer,
            torch_dtype=dtype_,
        ).to(device)

        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()

        pipe.enable_model_cpu_offload()

        for i, prompt in tqdm(enumerate(prompts[:5])):
            video = pipe(
                prompt.strip(),
                num_videos_per_prompt=1,
                num_inference_steps=50,
                num_frames=num_frames_,
                guidance_scale=6,
                generator=torch.Generator(device="cuda").manual_seed(42),
            ).frames[0]

            del video
            gc.collect()
            torch.cuda.empty_cache()

        saved_state_dict = extract_sparse_attention_state_dict(transformer)
        torch.save(saved_state_dict, args.model_out_path) 

    else:
        os.environ["TUNE_MODE"] = ""  # disable tune mode
        transformer = CogVideoXTransformer3DModel.from_pretrained(
            "THUDM/CogVideoX-2b",
            local_files_only=False,
            subfolder="transformer",
            torch_dtype=dtype_,
        )

        if args.use_spas_sage_attn:
            set_spas_sage_attn_cogvideox(transformer, verbose=args.verbose, l1=args.l1, pv_l1=args.pv_l1)
            # load saved state_dict
            saved_state_dict = torch.load(args.model_out_path)  
            load_sparse_attention_state_dict(transformer, saved_state_dict)

        pipe = CogVideoXPipeline.from_pretrained(
            "THUDM/CogVideoX-2b",
            transformer=transformer,
            torch_dtype=dtype_,
        ).to(device)

        if args.compile:
            pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune-no-cudagraphs")

        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()

        for i, prompt in tqdm(enumerate(prompts)):
            video = pipe(
                prompt.strip(),
                num_videos_per_prompt=1,
                num_inference_steps=50,
                num_frames=num_frames_,
                guidance_scale=6,
                generator=torch.Generator(device="cuda").manual_seed(42),
            ).frames[0]

            export_to_video(video, f"{args.out_path}/{i}.mp4", fps=8)
            del video
            gc.collect()
            torch.cuda.empty_cache()
