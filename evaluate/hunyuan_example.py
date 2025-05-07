import torch, os, gc
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.utils import export_to_video
import argparse
from tqdm import tqdm
from spas_sage_attn.autotune import (
    extract_sparse_attention_state_dict,
    load_sparse_attention_state_dict,
)
from modify_model.modify_hunyuan import set_spas_sage_attn_hunyuan

file_path = 'evaluate/datasets/video/prompts.txt'
model_id = "hunyuanvideo-community/HunyuanVideo"


def parse_args():
    parser = argparse.ArgumentParser(description="hunyuan Evaluation")
    
    ## sparge part
    parser.add_argument("--use_spas_sage_attn", action="store_true", help="Use Sage Attention")
    parser.add_argument("--tune", action="store_true", help="tuning hyperpamameters")
    parser.add_argument('--parallel_tune', action='store_true', help='enable prallel tuning')
    parser.add_argument('--l1', type=float, default=0.06, help='l1 bound for qk sparse')
    parser.add_argument('--pv_l1', type=float, default=0.065, help='l1 bound for pv sparse')
    parser.add_argument("--verbose", action="store_true", help="Verbose of sparge")
    parser.add_argument("--tune_pv", action="store_true", help="Verbose of sparge")
    parser.add_argument("--base_seed", type=int, default=0, help="base seed")
    parser.add_argument(
        "--out_path",
        type=str,
        default="evaluate/datasets/video/hunyuan_sparge",
        help="out_path",
    )
    parser.add_argument(
        "--model_out_path",
        type=str,
        default="evaluate/models_dict/hunyuan_saved_state_dict.pt",
        help="model_out_path",
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.out_path, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(file_path, 'r', encoding='utf-8') as file:
        prompts = file.readlines()

    if args.parallel_tune:
        os.environ['PARALLEL_TUNE'] = '1'
    if args.tune:
        os.environ["TUNE_MODE"] = "1"  # enable tune mode
    

    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        model_id, subfolder="transformer", torch_dtype=torch.bfloat16
    )
    from modify_model.modify_hunyuan import forward
    transformer.forward = forward.__get__(transformer, HunyuanVideoTransformer3DModel)
    
    
    if args.use_spas_sage_attn:
        set_spas_sage_attn_hunyuan(transformer, verbose=args.verbose)
        if not args.tune:
            saved_state_dict = torch.load(args.model_out_path)
            load_sparse_attention_state_dict(transformer, saved_state_dict)
    
    pipe = HunyuanVideoPipeline.from_pretrained(model_id, transformer=transformer, torch_dtype=torch.float16)
    pipe.enable_model_cpu_offload()
    pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    generator = torch.Generator(device).manual_seed(args.base_seed)

    for i, prompt in tqdm(enumerate(prompts)):
        video = pipe(prompt=prompt,
            height=544,
            width=960,
            num_frames=61,
            num_inference_steps=30,
            generator=generator,
        ).frames[0]
        export_to_video(video, f"{args.out_path}/{i}.mp4", fps=15)
        del video
        gc.collect()
        torch.cuda.empty_cache()

    if args.use_spas_sage_attn and args.tune:
        saved_state_dict = extract_sparse_attention_state_dict(transformer)
        torch.save(saved_state_dict, args.model_out_path)  # +args.model_name.split("/")[-1]+".pt"


