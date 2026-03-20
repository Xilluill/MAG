import argparse
import logging
import torch
import os
import time
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision import transforms
from torchvision.io import write_video
from einops import rearrange
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from utils.distributed import launch_distributed_job
from pipeline.memory_generation_pipeline import memory_generation_pipeline

from datasets import TextDataset
from utils.misc import set_seed

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, help="Path to the config file")
parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint folder")
parser.add_argument("--memory_checkpoint_path", type=str, help="Path to the checkpoint folder")
parser.add_argument("--data_path", type=str, help="Path to the dataset")
parser.add_argument("--extended_prompt_path", default=None, type=str, help="Path to the extended prompt")
parser.add_argument("--output_folder", type=str, help="Output folder")
parser.add_argument("--num_output_frames", type=int, default=21,
                    help="Number of overlap frames between sliding windows")
parser.add_argument("--low_memory", action="store_true", help="Whether to use low_memory")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate per prompt")

args = parser.parse_args()

local_rank = int(os.environ["LOCAL_RANK"])
rank = int(os.environ["RANK"])
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
    config.generator_ckpt = args.checkpoint_path
    config.memory_ckpt = args.memory_checkpoint_path
# Initialize pipeline
pipeline = memory_generation_pipeline(config, device=device)

pipeline = pipeline.to(dtype=torch.bfloat16,device=device)


# Create dataset

dataset = TextDataset(prompt_path=args.data_path)
num_prompts = len(dataset)
print(f"Number of prompts: {num_prompts}")

if dist.is_initialized():
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False)
else:
    sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=4, drop_last=False)

# Create output directory (only on main process to avoid race conditions)
if local_rank == 0:
    os.makedirs(args.output_folder, exist_ok=True)

if dist.is_initialized():
    dist.barrier()

for i, batch_data in tqdm(enumerate(dataloader),total=len(dataloader), disable=(local_rank != 0)):
    idx = batch_data['idx'].item()
    # For DataLoader batch_size=1, the batch_data is already a single item, but in a batch container
    # Unpack the batch data for convenience
    if isinstance(batch_data, dict):
        batch = batch_data
    elif isinstance(batch_data, list):
        batch = batch_data[0]  # First (and only) item in the batch

    all_video = []
    num_generated_frames = 0  # Number of generated (latent) frames

    # For text-to-video, batch is just the text prompt
    if 'input_pt' in batch:
        prompts_embedd = batch['input_pt'].to(device=device, dtype=torch.bfloat16)# torch.Size([1, 512, 4096])
        prompts_embedd = prompts_embedd.repeat(args.num_samples, 1, 1)
        prompt = batch['prompts'][0]
    else:
        prompt = batch['prompts'][0]
        extended_prompt = batch['extended_prompts'][0] if 'extended_prompts' in batch else None
        if extended_prompt is not None:
            prompts = [extended_prompt] * args.num_samples
        else:
            prompts = [prompt] * args.num_samples

    output_path = os.path.join(args.output_folder, f'{prompt[:200]}-{args.num_samples-1}.mp4')
    # logger.info(f"Rank {rank}: idx:{idx}, prompt:{prompt:200}")
    # continue
    if os.path.exists(output_path):
        # print(f"Rank {local_rank}: Video for prompt idx {idx} already exists at {output_path}, skipping...")
        continue
    sampled_noises = torch.randn(
        [args.num_samples, args.num_output_frames, 16, 60, 104], device=device, dtype=torch.bfloat16
    )
    pipeline.init_state(sampled_noises)
    inference_func = pipeline.inference_w_text_encoder
    if args.low_memory and args.num_samples != 1: #  多个sample依次进行
        pass
    else:
        # Generate 81 frames
        video, latents = inference_func(
            noise=sampled_noises,
            text_prompts=prompts
        )
        current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
        all_video.append(current_video)
        video = 255.0 * torch.cat(all_video, dim=0)
        pipeline.vae.model.clear_cache()

    # Save the video if the current prompt is not a dummy prompt
    if idx < num_prompts:
        for seed_idx in range(args.num_samples):
            # All processes save their videos
            output_path = os.path.join(args.output_folder, f'{prompt[:200]}-{seed_idx}.mp4')
            write_video(output_path, video[seed_idx], fps=16)
    torch.cuda.empty_cache()

print(f"Rank {local_rank} finished all prompts.")
dist.barrier()
print(f"Rank {local_rank} passed the barrier, destroying process group.")
dist.destroy_process_group()
