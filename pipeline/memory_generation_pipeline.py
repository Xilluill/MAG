import time
from einops import rearrange
from torch import nn
import torch.distributed as dist
import torch
from wan.modules.causal_model import CausalWanModel
from wan.modules.model import WanModel
from utils.wan_wrapper import WanTextEncoder, WanVAEWrapper
from utils.scheduler import FlowMatchScheduler
import torch.nn.functional as F
from model.memory_generation_model import memory_generation_model
from model.long_dmd_model import long_dmd_model


class memory_generation_pipeline(memory_generation_model):
    def __init__(self, args, device):
        super(long_dmd_model, self).__init__()
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.num_training_frames = getattr(
            args, "num_training_frames", 21
        )  # 可能更大了
        self.teacher_model_frames = 21
        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560
        self.local_attn_size = getattr(
            args, "local_attn_size", -1
        )  # 这里是多大 kv cache就是多大
        self.sink_size = getattr(
            args, "sink_size", 0
        )  # kv cache保留数量，local_attn_size包含sink token和最新的一些token
        self.cache_mode = args.cache_mode
        self._initialize_models(args, device)

        self.device = device
        self.args = args
        self.dtype = torch.bfloat16 if args.mixed_precision else torch.float32
        if hasattr(args, "denoising_step_list"):
            self.denoising_step_list = torch.tensor(
                args.denoising_step_list, dtype=torch.long
            )
            if args.warp_denoising_step:
                timesteps = torch.cat(
                    (
                        self.scheduler.timesteps.cpu(),
                        torch.tensor([0], dtype=torch.float32),
                    )
                )
                self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        if getattr(self.scheduler, "alphas_cumprod", None) is not None:
            self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        else:
            self.scheduler.alphas_cumprod = None
        self.eval()

    def _initialize_cache(
        self, batch_size, kv_cache_size, sink_size, dtype, device
    ):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append(
                {
                    "k": torch.zeros(
                        [batch_size, kv_cache_size, 12, 128],
                        dtype=dtype,
                        device=device,
                    ),
                    "v": torch.zeros(
                        [batch_size, kv_cache_size, 12, 128],
                        dtype=dtype,
                        device=device,
                    ),
                    "global_end_index": torch.tensor(
                        [0], dtype=torch.long, device=device
                    ),
                    "local_end_index": torch.tensor(
                        [0], dtype=torch.long, device=device
                    ),
                    "sink_k": torch.zeros(
                        [batch_size, sink_size, 12, 128],
                        dtype=dtype,
                        device=device,
                    ),
                    "sink_v": torch.zeros(
                        [batch_size, sink_size, 12, 128],
                        dtype=dtype,
                        device=device,
                    ),
                }
            )
        self.kv_cache1 = kv_cache1  # always store the clean cache

    def init_state(self, noise):
        batch_size = noise.shape[0]
        if self.local_attn_size != -1 and self.cache_mode == "original":
            kv_cache_size = (
                self.local_attn_size + self.sink_size + self.num_frame_per_block
            ) * self.frame_seq_length
        elif self.local_attn_size == -1:
            kv_cache_size = 40 * self.frame_seq_length
        elif self.local_attn_size != -1 and self.cache_mode != "original":
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        if getattr(self, "kv_cache1", None) is None:
            self._initialize_cache(
                batch_size=batch_size,
                kv_cache_size=kv_cache_size,  # 因为这里进行了修改所以重写函数
                sink_size=self.num_frame_per_block * self.frame_seq_length,
                dtype=noise.dtype,
                device=noise.device,
            )
        else:
            self.clean_kv_cache(clean_global=True)

        if getattr(self, "crossattn_cache", None) is None:
            self.crossattn_cache = self._initialize_crossattn_cache(
                batch_size=batch_size, dtype=noise.dtype, device=noise.device
            )
        else:
            self.clean_crossattn_cache(self.crossattn_cache)

        if getattr(self, "crossattn_cache_empty", None) is None:  # 这个不需要clean
            self.crossattn_cache_empty = self._initialize_crossattn_cache(
                batch_size=batch_size, dtype=noise.dtype, device=noise.device
            )

        self.empty_conditional_dict = self.text_encoder(text_prompts=[""] * batch_size)

    def _initialize_models(self, args, device="cpu"):

        self.generator = CausalWanModel.from_pretrained(
            args.generator_ckpt,
            local_attn_size=self.local_attn_size,
            sink_size=self.sink_size,
            num_frame_per_block=self.num_frame_per_block,
            is_inference_mode=True,
            cache_mode=self.cache_mode,
        )
        self.generator.requires_grad_(False)

        self.memory_model = CausalWanModel.from_pretrained(
            args.memory_ckpt,
            local_attn_size=self.local_attn_size,
            sink_size=self.sink_size,
            num_frame_per_block=self.num_frame_per_block,
            is_inference_mode=True,
            cache_mode=self.cache_mode,
        )
        self.memory_model.requires_grad_(False)

        self.text_encoder = WanTextEncoder(device)
        self.text_encoder.requires_grad_(False)

        self.vae = WanVAEWrapper(device).to(torch.bfloat16)
        self.vae.requires_grad_(False)

        scheduler = FlowMatchScheduler(
            shift=args.timestep_shift, sigma_min=0.0, extra_one_step=True
        )
        scheduler.set_timesteps(1000, training=True)
        self.scheduler = scheduler

    def inference_w_text_encoder(self, noise, text_prompts):
        batch_size, num_frames, num_channels, height, width = noise.shape
        num_full_blocks = num_frames // self.num_frame_per_block

        num_denoising_steps = len(self.denoising_step_list)

        conditional_dict = self.text_encoder(text_prompts=text_prompts)
        conditional_dict["prompt_embeds"] = conditional_dict["prompt_embeds"].to(
            noise.device
        )
        output = torch.zeros(
            [
                batch_size,
                num_frames,
                num_channels,
                height,
                width,
            ],
            device=noise.device,
            dtype=noise.dtype,
        )

        all_num_frames = [self.num_frame_per_block] * num_full_blocks

        current_start_frame = 0
        for block_index, current_num_frames in enumerate(all_num_frames):
            noisy_input = noise[
                :,
                current_start_frame : current_start_frame + current_num_frames,
            ]
            timestep_shape_one_tensor = torch.ones(
                [batch_size, current_num_frames],
                device=noise.device,
                dtype=torch.int64,
            )
            # Step 3.1: Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):

                timestep = timestep_shape_one_tensor * current_timestep
                if index != num_denoising_steps - 1:
                    denoised_pred = self.generator_wrapper(
                        noisy_input, timestep, conditional_dict, current_start_frame
                    )
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep * timestep_shape_one_tensor,
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    denoised_pred = self.generator_wrapper(
                        noisy_input, timestep, conditional_dict, current_start_frame
                    )
                    break

            # Step 3.2: rerun with timestep zero to update the cache
            context_timestep = timestep_shape_one_tensor * 0  # self.context_noise
            with torch.no_grad():
                self.memory_wrapper(
                    denoised_pred,
                    context_timestep,
                    self.empty_conditional_dict,
                    current_start_frame,
                    update_cache=True,
                )
            # Step 3.3: record the model's output
            output[
                :, current_start_frame : current_start_frame + current_num_frames
            ] = denoised_pred
            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        # Step 4: Decode the output
        output = output.to(self.vae.device)
        video = self.vae.decode_to_pixel(output, use_cache=False, offload_cpu=True)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        return video, output

    def inference_wo_text_encoder(self, noise, text_prompts_embedds):

        batch_size, num_frames, num_channels, height, width = noise.shape
        num_full_blocks = num_frames // self.num_frame_per_block

        num_denoising_steps = len(self.denoising_step_list)

        conditional_dict = {}
        conditional_dict["prompt_embeds"] = text_prompts_embedds
        output = torch.zeros(
            [
                batch_size,
                num_frames,
                num_channels,
                height,
                width,
            ],
            device=noise.device,
            dtype=noise.dtype,
        )

        all_num_frames = [self.num_frame_per_block] * num_full_blocks

        current_start_frame = 0
        for block_index, current_num_frames in enumerate(all_num_frames):
            noisy_input = noise[
                :,
                current_start_frame : current_start_frame + current_num_frames,
            ]
            timestep_shape_one_tensor = torch.ones(
                [batch_size, current_num_frames],
                device=noise.device,
                dtype=torch.int64,
            )
            # Step 3.1: Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):

                timestep = timestep_shape_one_tensor * current_timestep
                if index != num_denoising_steps - 1:
                    denoised_pred = self.generator_wrapper(
                        noisy_input, timestep, conditional_dict, current_start_frame
                    )
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep * timestep_shape_one_tensor,
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    denoised_pred = self.generator_wrapper(
                        noisy_input, timestep, conditional_dict, current_start_frame
                    )
                    break

            # Step 3.2: rerun with timestep zero to update the cache
            context_timestep = timestep_shape_one_tensor * 0  # self.context_noise
            with torch.no_grad():
                self.memory_wrapper(
                    denoised_pred,
                    context_timestep,
                    self.empty_conditional_dict,
                    current_start_frame,
                    update_cache=True,
                )
            # Step 3.3: record the model's output
            output[
                :, current_start_frame : current_start_frame + current_num_frames
            ] = denoised_pred
            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        # Step 4: Decode the output
        output = output.to(self.vae.device)
        video = self.vae.decode_to_pixel(output, use_cache=False, offload_cpu=True)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        return video, output

    def inference_wo_text_encoder_only_generator(self, noise, text_prompts_embedds):

        start_time = time.time()
        batch_size, num_frames, num_channels, height, width = noise.shape
        num_full_blocks = num_frames // self.num_frame_per_block

        num_denoising_steps = len(self.denoising_step_list)

        conditional_dict = {}
        conditional_dict["prompt_embeds"] = text_prompts_embedds
        output = torch.zeros(
            [
                batch_size,
                num_frames,
                num_channels,
                height,
                width,
            ],
            device=noise.device,
            dtype=noise.dtype,
        )

        all_num_frames = [self.num_frame_per_block] * num_full_blocks

        # exit_flags = self.generate_and_sync_list(
        #     len(all_num_frames), num_denoising_steps, device=noise.device
        # )
        current_start_frame = 0
        for block_index, current_num_frames in enumerate(all_num_frames):
            block_start_time = time.time()
            noisy_input = noise[
                :,
                current_start_frame : current_start_frame + current_num_frames,
            ]
            timestep_shape_one_tensor = torch.ones(
                [batch_size, current_num_frames],
                device=noise.device,
                dtype=torch.int64,
            )
            # Step 3.1: Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):

                timestep = timestep_shape_one_tensor * current_timestep
                if index != num_denoising_steps - 1:
                    denoised_pred = self.generator_wrapper(
                        noisy_input, timestep, conditional_dict, current_start_frame
                    )
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep * timestep_shape_one_tensor,
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    denoised_pred = self.generator_wrapper(
                        noisy_input, timestep, conditional_dict, current_start_frame
                    )
                    break

            # Step 3.2: rerun with timestep zero to update the cache
            context_timestep = timestep_shape_one_tensor * 0  # self.context_noise
            with torch.no_grad():
                self.generator_wrapper(
                    denoised_pred,
                    context_timestep,
                    conditional_dict,
                    current_start_frame,
                    update_cache=True,
                )
            # Step 3.3: record the model's output
            output[
                :, current_start_frame : current_start_frame + current_num_frames
            ] = denoised_pred
            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames
            block_end_time = time.time()
            block_elapsed_time = block_end_time - block_start_time
            pixel_frame_num = 12
            fps = pixel_frame_num / block_elapsed_time
            print(f"Block {block_index} generation FPS: {fps:.2f} frames/second.")

        end_time = time.time()
        elapsed_time = end_time - start_time
        pixel_frame_num = (num_frames - 1)*4 + 1
        fps = pixel_frame_num / elapsed_time
        print(f"generation FPS: {fps:.2f} frames/second.")
        # Step 4: Decode the output
        output = output.to(self.vae.device)
        video = self.vae.decode_to_pixel(output, use_cache=False, offload_cpu=True)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        return video, output

    def inference_w_initial_video(
        self,
        noise: torch.Tensor,
        text_prompts,
        initial_latent: torch.Tensor,
        gt_latent: torch.Tensor = None,
    ) -> torch.Tensor:
        batch_size, num_frames, num_channels, height, width = noise.shape
        num_initial_frames = initial_latent.shape[1]

        num_blocks = num_frames // self.num_frame_per_block

        num_output_frames = num_frames + num_initial_frames
        conditional_dict = self.text_encoder(text_prompts=text_prompts)
        conditional_dict["prompt_embeds"] = conditional_dict["prompt_embeds"].to(
            noise.device
        )

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype,
        )
        output[:, :num_initial_frames] = initial_latent
        # Step 1: Initialize KV cache to all zeros

        # Step 2: Cache context feature
        current_start_frame = 0
        if num_initial_frames == 1:
            self.memory_wrapper(
                initial_latent,
                timestep_shape_one_tensor * 0,
                self.empty_conditional_dict,
                current_start_frame,
                update_cache=True,
            )
        else:
            num_initial_blocks = num_initial_frames // self.num_frame_per_block
            initial_num_frames = [self.num_frame_per_block] * num_initial_blocks
            for current_num_frames in initial_num_frames:
                timestep_shape_one_tensor = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64,
                )
                current_ref_latents = initial_latent[
                    :, current_start_frame : current_start_frame + current_num_frames
                ]
                self.memory_wrapper(
                    current_ref_latents,
                    timestep_shape_one_tensor * 0,
                    self.empty_conditional_dict,
                    current_start_frame,
                    update_cache=True,
                )
                current_start_frame += current_num_frames

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks

        for current_num_frames in all_num_frames:

            noisy_input = noise[
                :,
                current_start_frame
                - num_initial_frames : current_start_frame
                - num_initial_frames
                + current_num_frames,
            ]
            timestep_shape_one_tensor = torch.ones(
                [batch_size, current_num_frames],
                device=noise.device,
                dtype=torch.int64,
            )
            # Step 3.1: Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):

                timestep = timestep_shape_one_tensor * current_timestep

                if index < len(self.denoising_step_list) - 1:
                    denoised_pred = self.generator_wrapper(
                        noisy_input, timestep, conditional_dict, current_start_frame
                    )
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep * timestep_shape_one_tensor,
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    denoised_pred = self.generator_wrapper(
                        noisy_input, timestep, conditional_dict, current_start_frame
                    )
                    break

            # Step 3.2: record the model's output
            output[
                :, current_start_frame : current_start_frame + current_num_frames
            ] = denoised_pred

            # Step 3.3: rerun with timestep zero to update KV cache using clean context
            context_timestep = torch.ones_like(timestep) * 0
            if gt_latent is not None:
                memory_input = gt_latent[
                    :,
                    current_start_frame
                    - num_initial_frames : current_start_frame
                    - num_initial_frames
                    + current_num_frames,
                ]
            else:
                memory_input = denoised_pred
            with torch.no_grad():
                self.memory_wrapper(
                    memory_input,
                    context_timestep,
                    self.empty_conditional_dict,
                    current_start_frame,
                    update_cache=True,
                )

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        # Step 4: Decode the output
        output = output.to(self.vae.device)
        video = self.vae.decode_to_pixel(output, use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        return video, output

    def inference_w_initial_video_only_generator(
        self,
        noise: torch.Tensor,
        text_prompts,
        initial_latent: torch.Tensor,
        gt_latent: torch.Tensor = None,
    ) -> torch.Tensor:
        batch_size, num_frames, num_channels, height, width = noise.shape
        num_initial_frames = initial_latent.shape[1]

        num_blocks = num_frames // self.num_frame_per_block

        num_output_frames = num_frames + num_initial_frames
        conditional_dict = self.text_encoder(text_prompts=text_prompts)
        conditional_dict["prompt_embeds"] = conditional_dict["prompt_embeds"].to(
            noise.device
        )

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype,
        )
        output[:, :num_initial_frames] = initial_latent
        # Step 1: Initialize KV cache to all zeros

        # Step 2: Cache context feature
        current_start_frame = 0
        if num_initial_frames == 1:
            self.generator_wrapper(
                initial_latent,
                timestep_shape_one_tensor * 0,
                conditional_dict,
                current_start_frame,
                update_cache=True,
            )
        else:
            num_initial_blocks = num_initial_frames // self.num_frame_per_block
            initial_num_frames = [self.num_frame_per_block] * num_initial_blocks
            for current_num_frames in initial_num_frames:
                current_ref_latents = initial_latent[
                    :, current_start_frame : current_start_frame + current_num_frames
                ]
                timestep_shape_one_tensor = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64,
                )
                self.generator_wrapper(
                    current_ref_latents,
                    timestep_shape_one_tensor * 0,
                    conditional_dict,
                    current_start_frame,
                    update_cache=True,
                )
                current_start_frame += current_num_frames

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks

        for current_num_frames in all_num_frames:

            noisy_input = noise[
                :,
                current_start_frame
                - num_initial_frames : current_start_frame
                - num_initial_frames
                + current_num_frames,
            ]
            timestep_shape_one_tensor = torch.ones(
                [batch_size, current_num_frames],
                device=noise.device,
                dtype=torch.int64,
            )
            # Step 3.1: Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):

                timestep = timestep_shape_one_tensor * current_timestep

                if index < len(self.denoising_step_list) - 1:
                    denoised_pred = self.generator_wrapper(
                        noisy_input, timestep, conditional_dict, current_start_frame
                    )
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep * timestep_shape_one_tensor,
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    denoised_pred = self.generator_wrapper(
                        noisy_input, timestep, conditional_dict, current_start_frame
                    )
                    break

            # Step 3.2: record the model's output
            output[
                :, current_start_frame : current_start_frame + current_num_frames
            ] = denoised_pred

            # Step 3.3: rerun with timestep zero to update KV cache using clean context
            context_timestep = torch.ones_like(timestep) * 0
            if gt_latent is not None:
                memory_input = gt_latent[
                    :,
                    current_start_frame
                    - num_initial_frames : current_start_frame
                    - num_initial_frames
                    + current_num_frames,
                ]
            else:
                memory_input = denoised_pred
            with torch.no_grad():
                self.generator_wrapper(
                    memory_input,
                    context_timestep,
                    conditional_dict,
                    current_start_frame,
                    update_cache=True,
                )

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        # Step 4: Decode the output
        output = output.to(self.vae.device)
        video = self.vae.decode_to_pixel(output, use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        return video, output
