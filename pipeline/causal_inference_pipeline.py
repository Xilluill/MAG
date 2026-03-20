from typing import List, Optional
from numpy import block
import torch

from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper

from demo_utils.memory import (
    gpu,
    get_cuda_free_memory_gb,
    DynamicSwapInstaller,
    move_model_to_device_with_memory_preservation,
)


class BasePipeline(torch.nn.Module):
    def __init__(self, args, device, generator=None, vae=None):
        super().__init__()
        # Step 1: Initialize all models
        model_kwargs = {}
        self.local_attn_size = getattr(args, "local_attn_size", -1)
        self.sink_size = getattr(args, "sink_size", 0)
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.timestep_shift = getattr(args, "timestep_shift", 5)
        model_kwargs["local_attn_size"] = self.local_attn_size
        model_kwargs["sink_size"] = self.sink_size
        model_kwargs["num_frame_per_block"] = self.num_frame_per_block
        model_kwargs["timestep_shift"] = self.timestep_shift
        if args.checkpoint_path:
            model_kwargs["model_path"] = args.checkpoint_path
        self.generator = WanDiffusionWrapper(**model_kwargs, is_causal=True)
        self.vae = WanVAEWrapper(device) if vae is None else vae

        # Step 2: Initialize all causal hyperparmeters
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long
        )
        if args.warp_denoising_step:
            timesteps = torch.cat(
                (self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32))
            )
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560

        self.kv_cache1 = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.independent_first_frame = args.independent_first_frame
        self.local_attn_size = self.generator.model.local_attn_size

        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []
        if self.local_attn_size != -1:
            # Use the local attention size to compute the KV cache size
            # kv_cache_size = 21 * self.frame_seq_length
            kv_cache_size = (
                self.local_attn_size + self.sink_size + self.num_frame_per_block
            ) * self.frame_seq_length
        else:
            # Use the default KV cache size
            kv_cache_size = 43 * self.frame_seq_length

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append(
                {
                    "k": torch.zeros(
                        [batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device
                    ),
                    "v": torch.zeros(
                        [batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device
                    ),
                    "global_end_index": torch.tensor(
                        [0], dtype=torch.long, device=device
                    ),
                    "local_end_index": torch.tensor(
                        [0], dtype=torch.long, device=device
                    ),
                }
            )

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append(
                {
                    "k": torch.zeros(
                        [batch_size, 512, 12, 128], dtype=dtype, device=device
                    ),
                    "v": torch.zeros(
                        [batch_size, 512, 12, 128], dtype=dtype, device=device
                    ),
                    "is_init": False,
                }
            )
        self.crossattn_cache = crossattn_cache


class MemoryModelPipeline(BasePipeline):
    def __init__(self, args, device, generator=None, text_encoder=None, vae=None):
        super().__init__(args, device, generator, vae)
        self.text_encoder = (
            WanTextEncoder(device) if text_encoder is None else text_encoder
        )
        self.is_lora_enabled=False
        self.empty_condition_dict=None

    def output_vae(
        self,
        # noise: torch.Tensor,
        text_prompts: List[str],
        initial_latent: Optional[torch.Tensor],
    ) -> torch.Tensor:

        video = self.vae.decode_to_pixel(initial_latent, use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)
        return video

    def inference_block_compress_multi_step(
        self,
        # noise: torch.Tensor,
        text_prompts: List[str],
        initial_latent: Optional[torch.Tensor],
        block_mask_type='compress',
    ) -> torch.Tensor:

        local_attn_size = self.local_attn_size
        num_clean_frames = initial_latent.shape[1]
        num_noisy_frames = num_clean_frames
        batch_size = initial_latent.shape[0]
        # 做memory forcing 最后几帧是不参与的
        # noisy_input = estimated_clean_image_or_video[:, :num_noisy_frames]
        num_denoising_steps = len(self.denoising_step_list)

        # current_timestep = self.denoising_step_list[indices].to(self.device)
        noisy_input = torch.randn_like(initial_latent[:, :num_noisy_frames])
        timestep_shape_one_tensor = torch.ones(
            [batch_size, num_noisy_frames],
            device=initial_latent.device,
            dtype=torch.int64,
        )

        if self.empty_condition_dict is None:
            conditional_dict = self.text_encoder(text_prompts=text_prompts)
            # conditional_dict['prompt_embeds'] = conditional_dict['prompt_embeds']
            self.empty_condition_dict = conditional_dict
        else:
            conditional_dict = self.empty_condition_dict

        for index, current_timestep in enumerate(self.denoising_step_list):
            timestep = timestep_shape_one_tensor * current_timestep.to(initial_latent.device)

            exit_flag = index == 3
            if not exit_flag:
                with torch.no_grad():
                    flow_pred = self.generator.model(
                        x=noisy_input.permute(
                            0, 2, 1, 3, 4
                        ),  # List of input video tensors, each with shape [C_in, F, H, W]
                        context=conditional_dict["prompt_embeds"],
                        t=timestep,
                        seq_len=327600,  # [1, 21, 16, 60, 104]
                        clean_x=initial_latent.permute(0, 2, 1, 3, 4),
                        block_mask_type=block_mask_type,
                    ).permute(0, 2, 1, 3, 4)
                    denoised_pred = self.scheduler._convert_flow_pred_to_x0(
                        flow_pred=flow_pred.flatten(0, 1),
                        xt=noisy_input.flatten(0, 1),
                        timestep=timestep.flatten(0, 1),
                    ).unflatten(0, flow_pred.shape[:2])
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep * timestep_shape_one_tensor,
                    ).unflatten(0, denoised_pred.shape[:2])
            else:
                flow_pred = self.generator.model(
                    x=noisy_input.permute(
                        0, 2, 1, 3, 4
                    ),  # List of input video tensors, each with shape [C_in, F, H, W]
                    context=conditional_dict["prompt_embeds"],
                    t=timestep,
                    seq_len=327600,  # [1, 21, 16, 60, 104]
                    clean_x=initial_latent.permute(0, 2, 1, 3, 4),
                    block_mask_type=block_mask_type,
                ).permute(0, 2, 1, 3, 4)
                output = self.scheduler._convert_flow_pred_to_x0(
                    flow_pred=flow_pred.flatten(0, 1),
                    xt=noisy_input.flatten(0, 1),
                    timestep=timestep.flatten(0, 1),
                ).unflatten(0, flow_pred.shape[:2])
                print(f"exit_flag reached at step {index}")
                break

        # Step 4: Decode the output
        loss = torch.nn.functional.mse_loss(
            output.double(),
            initial_latent[:, :num_noisy_frames].double(),
            reduction="mean",
        )
        print(f"MSE loss for MF multi-step: {loss.item()}")
        output = output.to(self.vae.device)
        video = self.vae.decode_to_pixel(output, use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        return video, output,loss.item()