import torch, os, gc
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
from diffusers.models import CogVideoXTransformer3DModel
import argparse
from tqdm import tqdm

from modify_model.modify_cogvideo import set_rls_attn_cogvideox # Corrected import


prompt_path = "evaluate/datasets/video/prompts.txt"

def parse_args():
    parser = argparse.ArgumentParser(description="CogVideoX Evaluation with RLSAttn")
    # Remove SAGE tuning arguments
    # parser.add_argument("--tune", action="store_true", help="tuning hyperpamameters")
    # parser.add_argument('--parallel_tune', action='store_true', help='enable prallel tuning')
    # parser.add_argument('--l1', type=float, default=0.06, help='l1 bound for qk sparse')
    # parser.add_argument('--pv_l1', type=float, default=0.065, help='l1 bound for pv sparse')
    parser.add_argument(
        "--model_out_path",
        type=str,
        default="evaluate/models_dict/CogVideoX-RLS.pt", # No longer needed
        help="model_out_path",
    )
    # parser.add_argument(
    #     "--use_spas_sage_attn", action="store_true", help="Use Sage Attention" # Replace this
    # )

    # Add RLSAttn arguments
    parser.add_argument(
        "--use_rls_attn", action="store_true", help="Use RLS Attention"
    )
    parser.add_argument(
        "--split_size", type=int, default=1111, help="Split size for RLSAttn blockification" # Default, adjust as needed
    )
    parser.add_argument(
        "--attn_dropout", type=float, default=0.0, help="Dropout probability for RLSAttn"
    )


    parser.add_argument('--compile', action='store_true', help='Compile the model')
    parser.add_argument("--verbose", action="store_true", help="Verbose output during setup")
    parser.add_argument(
        "--out_path",
        type=str,
        default="evaluate/datasets/video/cogvideo_rlsattn", # Changed default output path
        help="Output path for generated videos",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.out_path, exist_ok=True)

    dtype_ = torch.bfloat16
    num_frames_ = 49 # Make sure sequence length (related to num_frames) is divisible by split_size if needed

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(prompt_path, "r", encoding="utf-8") as file:
        prompts = file.readlines()

    # --- Remove Tuning Logic ---
    # The structure simplifies greatly as RLSAttn doesn't have the SAGE tuning process

    # Load the base transformer model
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        "THUDM/CogVideoX-2b",
        # local_files_only=False, # Allow downloading if needed
        subfolder="transformer",
        torch_dtype=dtype_,
    )

    # Conditionally replace attention with RLSAttn
    if args.use_rls_attn:
        if args.split_size <= 0:
             raise ValueError("--split_size must be a positive integer")
        print(f"Using RLSAttn with split_size={args.split_size}")
        set_rls_attn_cogvideox(
            transformer,
            split_size=args.split_size,
            attention_dropout=args.attn_dropout,
            verbose=args.verbose
        )
        # No state dict loading needed for RLSAttn unless it has trainable/loadable params

    # Load the pipeline with the potentially modified transformer
    pipe = CogVideoXPipeline.from_pretrained(
        "THUDM/CogVideoX-2b",
        transformer=transformer,
        torch_dtype=dtype_,
    ).to(device)

    # Optional: Compile the model (transformer part)
    if args.compile:
        print("Compiling the transformer...")
        # Make sure compilation works with the custom attention module
        try:
            pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune-no-cudagraphs")
            print("Compilation successful.")
        except Exception as e:
            print(f"Compilation failed: {e}. Proceeding without compilation.")


    # Enable memory optimizations
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    # pipe.enable_model_cpu_offload() # Can still use this if memory is tight

    # Generation loop
    print(f"Generating videos to: {args.out_path}")
    for i, prompt in tqdm(enumerate(prompts), total=len(prompts)):
        print(f"\nProcessing prompt {i+1}/{len(prompts)}: {prompt.strip()}")
        try:
            video_frames = pipe(
                prompt.strip(),
                num_videos_per_prompt=1,
                num_inference_steps=50,
                num_frames=num_frames_,
                guidance_scale=6,
                generator=torch.Generator(device="cuda").manual_seed(42),
            ).frames[0]

            # Export video
            output_filename = f"{args.out_path}/{i}.mp4"
            export_to_video(video_frames, output_filename, fps=8)
            print(f"Saved video to {output_filename}")

            # Clean up memory
            del video_frames
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error processing prompt {i}: {prompt.strip()}")
            print(f"Error details: {e}")
            # Optionally: clean up memory even on error
            gc.collect()
            torch.cuda.empty_cache()
            # Continue to the next prompt
            continue

    print("Finished generating all videos.")