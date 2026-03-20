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
from pipeline.memory_generation_pipeline import memory_generation_pipeline
from datasets import mag_bench_dataset
from utils.misc import set_seed
from datetime import timedelta
from evaluate.vae_metrics.lpips import calculate_lpips, calculate_psnr, calculate_ssim, calculate_lpips_find_match


parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, help="Path to the config file")
parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint folder")
parser.add_argument("--memory_checkpoint_path", type=str, help="Path to the checkpoint folder")
parser.add_argument("--video_path", type=str, help="Path to the video dataset")
parser.add_argument("--csv_path", type=str, help="Path to the csv")
parser.add_argument("--output_folder", type=str, help="Output folder")
parser.add_argument(
    "--num_output_frames",
    type=int,
    default=21,
    help="Number of overlap frames between sliding windows",
)
parser.add_argument(
    "--low_memory", action="store_true", help="Whether to use low_memory"
)
parser.add_argument("--use_prompt", action="store_true", help="Whether to use prompt")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument(
    "--num_samples",
    type=int,
    default=1,
    help="Number of samples to generate per prompt",
)
parser.add_argument("--use_memory_model", action="store_true", help="Whether to use_memory_model")
parser.add_argument("--use_gt", action="store_true", help="Whether to gt as memory")
args = parser.parse_args()

local_rank = int(os.environ["LOCAL_RANK"])
rank = int(os.environ["RANK"])
# torch.cuda.set_device(local_rank)
launch_distributed_job()

device = torch.cuda.current_device()
device = torch.device(f"cuda:{device}")
print(f"rank:{local_rank} use cuda:{torch.cuda.current_device()}")
world_size = dist.get_world_size()
set_seed(args.seed + local_rank)
print(f"Using {world_size} GPUs for inference")


dist.barrier()

torch.set_grad_enabled(False)

config = OmegaConf.load(args.config_path)
default_config = OmegaConf.load("configs/default_config.yaml")
config = OmegaConf.merge(default_config, config)

if args.checkpoint_path:
    config.generator_ckpt = args.checkpoint_path
    config.memory_ckpt = args.memory_checkpoint_path

pipeline = memory_generation_pipeline(config, device=device)

pipeline = pipeline.to(dtype=torch.bfloat16, device=device)


# Create dataset
dataset = mag_bench_dataset(csv_path=args.csv_path, video_folder=args.video_path)
num_videos = len(dataset)
print(f"Number of videos: {num_videos}")

if dist.is_initialized():
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False)
else:
    sampler = SequentialSampler(dataset)
dataloader = DataLoader(
    dataset, batch_size=1, sampler=sampler, num_workers=4, drop_last=False
)

# Create output directory (only on main process to avoid race conditions)
result_dir = os.path.join(args.output_folder, f"result")
if local_rank == 0:
    os.makedirs(args.output_folder, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

if dist.is_initialized():
    dist.barrier()

all_results=[]
for i, batch_data in tqdm(
    enumerate(dataloader), total=len(dataloader), disable=(local_rank != 0)
):
    # return {"name": video_name, "video": video, "caption": caption}
    video_name = batch_data["name"][0]
    video = batch_data["video"].to(device=device, dtype=torch.bfloat16)
    print(f"video_shape:{video.shape}")  # [1, 105, 3, 1080, 1920])
    prompts = batch_data["caption"]
    switch_frame = batch_data["switch_frame"].item()
    print(f"switch_frame:{switch_frame}")

    video_before, video_after = (
        video[:, :switch_frame], video[:, switch_frame:],
    )
    args.num_output_frames = video_after.shape[1] // 4
    
    all_video = []
    num_generated_frames = 0  # Number of generated (latent) frames
    prompt = prompts[0]
    initial_latent = pipeline.vae.encode_to_latent(video).to(
        device=device, dtype=torch.bfloat16
    )
    initial_latent, gt_latent = (
        initial_latent[:, : -args.num_output_frames],
        initial_latent[:, -args.num_output_frames :],
    )  # [1, num_frames, 16, 60, 104]
    initial_latent = initial_latent.expand(
        args.num_samples, -1, -1, -1, -1
    )  # [num_samples, num_frames, 16, 60, 104]
    gt_latent = gt_latent.expand(
        args.num_samples, -1, -1, -1, -1
    )  # [num_samples, num_frames, 16, 60, 104]
    # print(prompt)
    if not args.use_prompt:
        output_name = os.path.splitext(video_name)[0] + "-none"
        prompts = [""] # * args.num_samples
    else:
        output_name = os.path.splitext(video_name)[0] + "-caption"
    output_path = os.path.join(
        args.output_folder, f"{output_name}-{args.num_samples-1}.mp4"
    )

    sampled_noises = torch.randn(
        [args.num_samples, args.num_output_frames, 16, 60, 104],
        device=device,
        dtype=torch.bfloat16,
    )
    pipeline.init_state(sampled_noises)
    if args.use_memory_model:
        inference_func = pipeline.inference_w_initial_video
    else:
        inference_func = pipeline.inference_w_initial_video_only_generator
    # Generate 81 frames
    print(prompts)
    output_video, output = inference_func(
        noise=sampled_noises,
        text_prompts=prompts,
        initial_latent=initial_latent,
        gt_latent=gt_latent if args.use_gt else None,
    )
    lpips_score,sim_indices = calculate_lpips_find_match(video_after,(output_video[:, switch_frame:]*2.0)-1,device=device)
    video_after_matched = video_after[:, sim_indices]
    psnr_score = calculate_psnr((video_after_matched+1)/2.0,output_video[:, switch_frame:])
    ssim_score = calculate_ssim((video_after_matched+1)/2.0,output_video[:, switch_frame:])
    # lpips_score = calculate_lpips(video_after,(output_video[:, switch_frame:]*2.0)-1,device=device)
    all_results.append({
        "video_name": video_name,
        "ssim": ssim_score,
        "lpips": lpips_score,
        "psnr": psnr_score,
        "sim_indices": sim_indices.cpu().tolist()
    })
    print(f"psnr:{psnr_score},ssim:{ssim_score},lpips:{lpips_score}")
    current_video = rearrange(output_video, "b t c h w -> b t h w c").cpu()
    output_video = 255.0 * current_video
    pipeline.vae.model.clear_cache()

    # Save the video if the current prompt is not a dummy prompt
    for seed_idx in range(args.num_samples):
        # All processes save their videos
        # if prompt == "":
        #     prompt = "none"
        output_path = os.path.join(args.output_folder, f"{output_name}-{seed_idx}.mp4")
        write_video(output_path, output_video[seed_idx], fps=16)
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

    print(f"Average Results: {avg_results}")
    output_file = os.path.join(result_dir, 'avg_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(avg_results, f, indent=4)

dist.barrier()
dist.destroy_process_group()
