import argparse
import logging
import torch
import os
from omegaconf import OmegaConf
from tqdm import tqdm
import json
from torchvision.io import write_video
from einops import rearrange
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from utils.distributed import launch_distributed_job
from pipeline import MemoryModelPipeline
from datasets import video_bench_dataset
from utils.misc import set_seed
from evaluate.vae_metrics.lpips import calculate_lpips, calculate_psnr, calculate_ssim

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, help="Path to the config file")
parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint folder")
parser.add_argument("--video_path", type=str, default='VPData/mf_test/000000000_16fps',help="Path to the video dataset")
parser.add_argument("--csv_path", type=str, default='VPData/mf_test/mf_test.csv',help="Path to the csv")
parser.add_argument("--output_folder", type=str, help="Output folder")
parser.add_argument("--num_output_frames", type=int, default=21,
                    help="Number of overlap frames between sliding windows")
parser.add_argument("--low_memory", action="store_true", help="Whether to use low_memory")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate per prompt")
args = parser.parse_args()

local_rank = int(os.environ["LOCAL_RANK"])
rank = int(os.environ["RANK"])
# torch.cuda.set_device(local_rank)
launch_distributed_job()

device = torch.cuda.current_device() 
device = torch.device(f"cuda:{device}")
print(f'rank:{local_rank} use cuda:{torch.cuda.current_device()}')
world_size = dist.get_world_size()
set_seed(args.seed + local_rank)
print(f"Using {world_size} GPUs for inference")


dist.barrier()

torch.set_grad_enabled(False)

config = OmegaConf.load(args.config_path)
default_config = OmegaConf.load("configs/default_config.yaml")
config = OmegaConf.merge(default_config, config)

if args.checkpoint_path:
    config.checkpoint_path = args.checkpoint_path
# Initialize pipeline
# if hasattr(config, 'denoising_step_list'):
    # Few-step inference
pipeline = MemoryModelPipeline(config, device=device)

pipeline = pipeline.to(dtype=torch.bfloat16)
pipeline.generator.to(device=device)
pipeline.vae.to(device=device)


# Create dataset
dataset = video_bench_dataset(csv_path=args.csv_path,video_folder=args.video_path,num_frame_per_block=config.num_frame_per_block)
num_videos= len(dataset)
print(f"Number of videos: {num_videos}")

if dist.is_initialized():
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False)
else:
    sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=1, drop_last=True)

# Create output directory (only on main process to avoid race conditions)
result_dir = os.path.join(args.output_folder, f"result")
if local_rank == 0:
    os.makedirs(result_dir, exist_ok=True)


if dist.is_initialized():
    dist.barrier()

all_results=[]

for i, batch_data in tqdm(enumerate(dataloader),total=len(dataloader), disable=(local_rank != 0)):
    # return {"name": video_name, "video": video, "caption": caption}
    video_name = batch_data['name'][0]
    video = batch_data['video'].to(device=device, dtype=torch.bfloat16)
    print(f'video_shape:{video.shape}') # [1, 105, 3, 1080, 1920])
    prompts = batch_data['caption']
    print(f'prompts:{prompts}')

    num_generated_frames = 0  # Number of generated (latent) frames
    prompt = prompts[0]
    initial_latent = pipeline.vae.encode_to_latent(video).to(device=device, dtype=torch.bfloat16)

    print(prompt)
    if prompt == "none":
        output_name = os.path.splitext(video_name)[0] + "-none"
        prompts = [""]
    else:
        output_name = prompt[:200]
    output_path = os.path.join(args.output_folder, f'{output_name}-{args.num_samples-1}.mp4')
    inference_func = pipeline.inference_block_compress_multi_step
    if args.low_memory and args.num_samples != 1: #  多个sample依次进行
        pass
    else:
        # Generate 81 frames
        output_video,latent,vae_loss = inference_func(
            # noise=sampled_noises,
                text_prompts=prompts,
                initial_latent=initial_latent,
                block_mask_type='history_compress',
        )
        psnr_score = calculate_psnr((video+1)/2.0,output_video)
        ssim_score = calculate_ssim((video+1)/2.0,output_video)
        lpips_score = calculate_lpips(video,(output_video*2.0)-1,device=device)
        all_results.append({
            "video_name": video_name,
            "ssim": ssim_score,
            "lpips": lpips_score,
            "psnr": psnr_score,
            "vae_loss": vae_loss
        })
        current_video = rearrange(output_video, 'b t c h w -> b t h w c').cpu()
        video = 255.0 * current_video
        pipeline.vae.model.clear_cache()

    # Save the video if the current prompt is not a dummy prompt
    for seed_idx in range(args.num_samples):
        # All processes save their videos
        if prompt == "":
            prompt = "none"
        output_path = os.path.join(args.output_folder, f'{output_name}-{seed_idx}.mp4')
        write_video(output_path, video[seed_idx], fps=16)
    torch.cuda.empty_cache()

dist.barrier()
# 聚合result
gathered_results_list = None
gathered_results_list = [None] * world_size
dist.all_gather_object(gathered_results_list, all_results)
if rank == 0:
    all_results = [item for sublist in gathered_results_list for item in sublist]
    output_file = os.path.join(result_dir, 'result_all.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4)
    avg_results={}
    avg_results['avg_psnr'] = sum([res['psnr'] for res in all_results]) / len(all_results)
    avg_results['avg_ssim'] = sum([res['ssim'] for res in all_results]) / len(all_results)
    avg_results['avg_lpips'] = sum([res['lpips'] for res in all_results]) / len(all_results)
    avg_results['avg_vae_loss'] = sum([res['vae_loss'] for res in all_results]) / len(all_results)

    print(f"Average Results: {avg_results}")
    output_file = os.path.join(result_dir, 'avg_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(avg_results, f, indent=4)

dist.barrier()
dist.destroy_process_group()
