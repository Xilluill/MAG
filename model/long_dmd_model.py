from typing import Tuple
from einops import rearrange
from torch import nn
import torch.distributed as dist
import torch
from wan.modules.causal_model import CausalWanModel
from wan.modules.model import WanModel
from utils.wan_wrapper import WanTextEncoder, WanVAEWrapper
from utils.scheduler import FlowMatchScheduler
import torch.nn.functional as F


class StreamingState:
    def __init__(
        self,
        noise,
        conditional_dict,
        unconditional_dict,
        empty_conditional_dict,
        num_chunks,
    ):
        self.noise = noise
        self.conditional_dict = conditional_dict
        self.unconditional_dict = unconditional_dict
        self.empty_conditional_dict = empty_conditional_dict
        self.global_start_frame = 0
        self.chunk_index = 0
        self.num_chunks = num_chunks
        self.store_latent: torch.Tensor = None
        self.pixel_video: torch.Tensor = None
        self.latent_video: torch.Tensor = None


class long_dmd_model(nn.Module):
    def __init__(self, args, device):
        super().__init__()
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
        self.empty_condition_p = getattr(args, "empty_condition_p", 0.2)
        self.compress_empty_condition_p = getattr(args, "compress_empty_condition_p", 0.2)
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

        self.same_step_across_blocks = getattr(
            args, "same_step_across_blocks", True
        )  # 就是Tru
        # Step 2: Initialize all dmd hyperparameters
        self.num_train_timestep = args.num_train_timestep
        self.min_step = int(0.02 * self.num_train_timestep)
        self.max_step = int(0.98 * self.num_train_timestep)
        if hasattr(args, "real_guidance_scale"):
            self.real_guidance_scale = args.real_guidance_scale
            self.fake_guidance_scale = args.fake_guidance_scale
        else:
            self.real_guidance_scale = args.guidance_scale
            self.fake_guidance_scale = 0.0
        self.timestep_shift = getattr(args, "timestep_shift", 1.0)  # 5
        self.ts_schedule = getattr(args, "ts_schedule", True)  # False
        self.ts_schedule_max = getattr(args, "ts_schedule_max", False)  # False
        self.min_score_timestep = getattr(args, "min_score_timestep", 0)  # 0

        if getattr(self.scheduler, "alphas_cumprod", None) is not None:
            self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        else:
            self.scheduler.alphas_cumprod = None
        # self.num_frame_kv_cache = self.num_training_frames

    def _initialize_models(self, args, device="cpu"):
        self.real_model_name = getattr(args, "real_name", "Wan2.1-T2V-1.3B")
        self.fake_model_name = getattr(args, "fake_name", "Wan2.1-T2V-1.3B")

        self.generator = CausalWanModel.from_pretrained(
            args.generator_ckpt,
            local_attn_size=self.local_attn_size,
            sink_size=self.sink_size,
            num_frame_per_block=self.num_frame_per_block,
            cache_mode = self.cache_mode,
            is_inference_mode=False,
        )
        self.generator.requires_grad_(True)

        # self.real_score = WanDiffusionWrapper(model_name=self.real_model_name, is_causal=False)
        self.real_score = WanModel.from_pretrained(f"wan_models/{self.real_model_name}")
        self.real_score.requires_grad_(False)

        self.fake_score = WanModel.from_pretrained(f"wan_models/{self.fake_model_name}")
        self.fake_score.requires_grad_(True)

        self.text_encoder = WanTextEncoder(device)
        self.text_encoder.requires_grad_(False)

        self.vae = WanVAEWrapper(device).to(torch.bfloat16)
        self.vae.requires_grad_(False)

        scheduler = FlowMatchScheduler(
            shift=args.timestep_shift, sigma_min=0.0, extra_one_step=True
        )
        scheduler.set_timesteps(1000, training=True)
        self.scheduler = scheduler

        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()
            self.fake_score.enable_gradient_checkpointing()

    def _get_timestep(
        self,
        min_timestep: int,
        max_timestep: int,
        batch_size: int,
        num_frame: int,
        num_frame_per_block: int,
        uniform_timestep: bool = False,
    ) -> torch.Tensor:
        """
        Randomly generate a timestep tensor based on the generator's task type. It uniformly samples a timestep
        from the range [min_timestep, max_timestep], and returns a tensor of shape [batch_size, num_frame].
        - If uniform_timestep, it will use the same timestep for all frames.
        - If not uniform_timestep, it will use a different timestep for each block.
        """
        if uniform_timestep:
            timestep = torch.randint(
                min_timestep,
                max_timestep,
                [batch_size, 1],
                device=self.device,
                dtype=torch.long,
            ).repeat(1, num_frame)
            return timestep
        else:
            timestep = torch.randint(
                min_timestep,
                max_timestep,
                [batch_size, num_frame],
                device=self.device,
                dtype=torch.long,
            )
            # make the noise level the same within every block
            if self.independent_first_frame:
                # the first frame is always kept the same
                timestep_from_second = timestep[:, 1:]
                timestep_from_second = timestep_from_second.reshape(
                    timestep_from_second.shape[0], -1, num_frame_per_block
                )
                timestep_from_second[:, :, 1:] = timestep_from_second[:, :, 0:1]
                timestep_from_second = timestep_from_second.reshape(
                    timestep_from_second.shape[0], -1
                )
                timestep = torch.cat([timestep[:, 0:1], timestep_from_second], dim=1)
            else:
                timestep = timestep.reshape(timestep.shape[0], -1, num_frame_per_block)
                timestep[:, :, 1:] = timestep[:, :, 0:1]
                timestep = timestep.reshape(timestep.shape[0], -1)
            return timestep

    def _compute_kl_grad(
        self,
        noisy_image_or_video: torch.Tensor,
        estimated_clean_image_or_video: torch.Tensor,
        timestep: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        normalization: bool = True,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the KL grad (eq 7 in https://arxiv.org/abs/2311.18828).
        Input:
            - noisy_image_or_video: a tensor with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - estimated_clean_image_or_video: a tensor with shape [B, F, C, H, W] representing the estimated clean image or video.
            - timestep: a tensor with shape [B, F] containing the randomly generated timestep.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - normalization: a boolean indicating whether to normalize the gradient.
        Output:
            - kl_grad: a tensor representing the KL grad.
            - kl_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        # Step 1: Compute the fake score
        flow_pred = self.fake_score(  # 走这里
            noisy_image_or_video.permute(0, 2, 1, 3, 4),
            t=timestep[:, 0],
            context=conditional_dict["prompt_embeds"],
            seq_len=32760,
        ).permute(0, 2, 1, 3, 4)
        pred_fake_image_cond = self.scheduler._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1),
        ).unflatten(0, flow_pred.shape[:2])

        if self.fake_guidance_scale != 0.0:
            flow_pred = self.fake_score(  # 走这里
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=timestep[:, 0],
                context=unconditional_dict["prompt_embeds"],
                seq_len=32760,
            ).permute(0, 2, 1, 3, 4)
            pred_fake_image_uncond = self.scheduler._convert_flow_pred_to_x0(
                flow_pred=flow_pred.flatten(0, 1),
                xt=noisy_image_or_video.flatten(0, 1),
                timestep=timestep.flatten(0, 1),
            ).unflatten(0, flow_pred.shape[:2])

            pred_fake_image = (
                pred_fake_image_cond
                + (pred_fake_image_cond - pred_fake_image_uncond)
                * self.fake_guidance_scale
            )
        else:
            pred_fake_image = pred_fake_image_cond

        # Step 2: Compute the real score
        # We compute the conditional and unconditional prediction
        # and add them together to achieve cfg (https://arxiv.org/abs/2207.12598)
        flow_pred = self.real_score(
            noisy_image_or_video.permute(0, 2, 1, 3, 4),
            t=timestep[:, 0],
            context=conditional_dict["prompt_embeds"],
            seq_len=32760,
        ).permute(0, 2, 1, 3, 4)

        pred_real_image_cond = self.scheduler._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1),
        ).unflatten(0, flow_pred.shape[:2])

        flow_pred = self.real_score(
            noisy_image_or_video.permute(0, 2, 1, 3, 4),
            t=timestep[:, 0],
            context=unconditional_dict["prompt_embeds"],
            seq_len=32760,
        ).permute(0, 2, 1, 3, 4)

        pred_real_image_uncond = self.scheduler._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1),
        ).unflatten(0, flow_pred.shape[:2])

        pred_real_image = (
            pred_real_image_cond
            + (pred_real_image_cond - pred_real_image_uncond) * self.real_guidance_scale
        )

        # Step 3: Compute the DMD gradient (DMD paper eq. 7).
        grad = pred_fake_image - pred_real_image

        # TODO: Change the normalizer for causal teacher
        if normalization:
            # Step 4: Gradient normalization (DMD paper eq. 8).
            p_real = estimated_clean_image_or_video - pred_real_image
            normalizer = torch.abs(p_real).mean(dim=[1, 2, 3, 4], keepdim=True)
            grad = grad / normalizer
        grad = torch.nan_to_num(grad)

        return grad, {
            "dmdtrain_gradient_norm": torch.mean(torch.abs(grad)).detach(),
            "timestep": timestep.detach(),
        }

    def compute_distribution_matching_loss(
        self,
        image_or_video: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        gradient_mask: torch.Tensor = None,
        denoised_timestep_from: int = 0,
        denoised_timestep_to: int = 0,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the DMD loss (eq 7 in https://arxiv.org/abs/2311.18828).
        Input:
            - image_or_video: a tensor with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - gradient_mask: a boolean tensor with the same shape as image_or_video indicating which pixels to compute loss .
        Output:
            - dmd_loss: a scalar tensor representing the DMD loss.
            - dmd_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        original_latent = image_or_video

        batch_size, num_frame = image_or_video.shape[:2]

        with torch.no_grad():
            # Step 1: Randomly sample timestep based on the given schedule and corresponding noise
            min_timestep = (
                denoised_timestep_to
                if self.ts_schedule and denoised_timestep_to is not None
                else self.min_score_timestep
            )
            max_timestep = (
                denoised_timestep_from
                if self.ts_schedule_max and denoised_timestep_from is not None
                else self.num_train_timestep
            )
            timestep = self._get_timestep(
                min_timestep,
                max_timestep,
                batch_size,
                num_frame,
                self.num_frame_per_block,
                uniform_timestep=True,
            )

            # TODO:should we change it to `timestep = self.scheduler.timesteps[timestep]`?
            if self.timestep_shift > 1:
                timestep = (
                    self.timestep_shift
                    * (timestep / 1000)
                    / (1 + (self.timestep_shift - 1) * (timestep / 1000))
                    * 1000
                )
            timestep = timestep.clamp(self.min_step, self.max_step)

            noise = torch.randn_like(image_or_video)
            noisy_latent = (
                self.scheduler.add_noise(
                    image_or_video.flatten(0, 1),
                    noise.flatten(0, 1),
                    timestep.flatten(0, 1),
                )
                .detach()
                .unflatten(0, (batch_size, num_frame))
            )

            # Step 2: Compute the KL grad
            grad, dmd_log_dict = self._compute_kl_grad(
                noisy_image_or_video=noisy_latent,
                estimated_clean_image_or_video=original_latent,
                timestep=timestep,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
            )

        if gradient_mask is not None:
            dmd_loss = 0.5 * F.mse_loss(
                original_latent.double()[gradient_mask],
                (original_latent.double() - grad.double()).detach()[gradient_mask],
                reduction="mean",
            )
        else:
            dmd_loss = 0.5 * F.mse_loss(
                original_latent.double(),
                (original_latent.double() - grad.double()).detach(),
                reduction="mean",
            )
        return dmd_loss, dmd_log_dict

    def dmd_loss(
        self,
    ) -> Tuple[torch.Tensor, dict]:
        """ """
        if self.local_attn_size != -1:
            self.update_window_cache(self.local_attn_size * self.frame_seq_length)
            self.clean_kv_cache()
        # with torch.no_grad():

        if self.state.chunk_index > 0:
            # empty_condition_p 根据这个进行随机
            use_empty_condition = torch.rand(1).item() < self.empty_condition_p
        else:
            use_empty_condition = False
        pred_video = self.generate_next_chunk(
            use_empty_condition=use_empty_condition
        )  # 这里可能返回的是18帧

        if  pred_video.shape[1] != self.teacher_model_frames:
            pred_video_21 = torch.cat([self.state.store_latent, pred_video], dim=1)
        else:
            pred_video_21 = pred_video

        _, store_latent = self._process_first_frame_encoding(pred_video)
        self.state.store_latent = store_latent  # 每次都会保留3潜帧 不带梯度的

        # Step 2: Compute the DMD loss
        dmd_loss, dmd_log_dict = self.compute_distribution_matching_loss(
            image_or_video=pred_video_21,
            conditional_dict=self.state.conditional_dict,
            unconditional_dict=self.state.unconditional_dict,
            gradient_mask=None,
        )
        zero_tensor = torch.zeros_like(dmd_loss)
        return dmd_loss, zero_tensor, dmd_log_dict

    def critic_loss(self, image_or_video_shape) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noise and train the critic with generated samples.
        The noisy input to the generator is backward simulated.
        This removes the need of any datasets during distillation.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
        Output:
            - loss: a scalar tensor representing the generator loss.
            - critic_log_dict: a dictionary containing the intermediate tensors for logging.
        """

        # Step 1: Run generator on backward simulated noisy input
        if self.local_attn_size != -1:
            self.update_window_cache(self.local_attn_size * self.frame_seq_length)
            self.clean_kv_cache()

        if self.state.chunk_index > 0:
            # empty_condition_p 根据这个进行随机
            use_empty_condition = torch.rand(1).item() < self.empty_condition_p
        else:
            use_empty_condition = False
        with torch.no_grad():
            pred_video = self.generate_next_chunk(
                use_empty_condition=use_empty_condition
            )  # 这里可能返回的是18帧

        if  pred_video.shape[1] != self.teacher_model_frames:
            pred_video_21 = torch.cat([self.state.store_latent, pred_video], dim=1)
        else:
            pred_video_21 = pred_video

        _, store_latent = self._process_first_frame_encoding(pred_video)
        self.state.store_latent = store_latent  # 每次都会保留3潜帧 不带梯度的

        conditional_dict = self.state.conditional_dict
        # Step 2: Compute the fake prediction
        min_timestep = self.min_score_timestep
        max_timestep = self.num_train_timestep

        critic_timestep = self._get_timestep(
            min_timestep,
            max_timestep,
            image_or_video_shape[0],
            image_or_video_shape[1],
            self.num_frame_per_block,
            uniform_timestep=True,
        )

        if self.timestep_shift > 1:
            critic_timestep = (
                self.timestep_shift
                * (critic_timestep / 1000)
                / (1 + (self.timestep_shift - 1) * (critic_timestep / 1000))
                * 1000
            )

        critic_timestep = critic_timestep.clamp(self.min_step, self.max_step)

        critic_noise = torch.randn_like(pred_video_21)
        noisy_pred_video = self.scheduler.add_noise(
            pred_video_21.flatten(0, 1),
            critic_noise.flatten(0, 1),
            critic_timestep.flatten(0, 1),
        ).unflatten(0, image_or_video_shape[:2])

        flow_pred = self.fake_score(
            noisy_pred_video.permute(0, 2, 1, 3, 4),
            t=critic_timestep[:, 0],
            context=conditional_dict["prompt_embeds"],
            seq_len=32760,
        ).permute(0, 2, 1, 3, 4)
        # torch.mean((flow_pred - (critic_noise - generated_image)) ** 2)
        denoising_loss = F.mse_loss(
            flow_pred.double(),
            (critic_noise - pred_video_21).double(),
            reduction="mean",
        )

        # Step 5: Debugging Log
        critic_log_dict = {"critic_timestep": critic_timestep.detach()}

        return denoising_loss, critic_log_dict

    def generator_wrapper(
        self,
        noisy_input,
        timestep,
        conditional_dict,
        current_start_frame,
        update_cache=False,
        use_cross_cache=True
    ):
        flow_pred = self.generator(
            noisy_input.permute(0, 2, 1, 3, 4),
            t=timestep,
            context=conditional_dict["prompt_embeds"],
            seq_len=327600,
            kv_cache=self.kv_cache1,
            crossattn_cache=self.crossattn_cache if use_cross_cache else [None]*self.num_transformer_blocks, 
            current_start=current_start_frame * self.frame_seq_length,
            update_cache=update_cache,
        ).permute(0, 2, 1, 3, 4)
        if not update_cache:
            denoised_pred = self.scheduler._convert_flow_pred_to_x0(
                flow_pred=flow_pred.flatten(0, 1),
                xt=noisy_input.flatten(0, 1),
                timestep=timestep.flatten(0, 1),
            ).unflatten(0, flow_pred.shape[:2])
            return denoised_pred

    def generate_and_sync_list(self, num_blocks, num_denoising_steps, device):
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            # Generate random indices
            indices = torch.randint(
                low=0, high=num_denoising_steps, size=(num_blocks,), device=device
            )
            # if self.last_step_only:
            #     indices = torch.ones_like(indices) * (num_denoising_steps - 1)
        else:
            indices = torch.empty(num_blocks, dtype=torch.long, device=device)

        dist.broadcast(indices, src=0)  # Broadcast the random indices to all ranks
        return indices.tolist()

    def _generate_chunk(
        self, noise, global_start_frame, conditional_dict,
    ):

        batch_size, num_frames, num_channels, height, width = noise.shape
        num_full_blocks = num_frames // self.num_frame_per_block
        generate_first_frame = (num_frames % self.num_frame_per_block)==1
        # if generate_first_frame:
        #     num_blocks += 1
        num_denoising_steps = len(self.denoising_step_list)

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
        if generate_first_frame:
            all_num_frames = [1] + all_num_frames
        exit_flags = self.generate_and_sync_list(
            len(all_num_frames), num_denoising_steps, device=noise.device
        )
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
                if self.same_step_across_blocks:
                    exit_flag = index == exit_flags[0]
                else:
                    exit_flag = index == exit_flags[block_index]# Only backprop at the randomly selected timestep (consistent across all ranks)
                timestep = timestep_shape_one_tensor * current_timestep
                if not exit_flag:
                    with torch.no_grad():
                        denoised_pred = self.generator_wrapper(
                            noisy_input,
                            timestep,
                            conditional_dict,
                            global_start_frame + current_start_frame,
                            use_cross_cache = False
                        )
                        next_timestep = self.denoising_step_list[index + 1]
                        noisy_input = self.scheduler.add_noise(
                            denoised_pred.flatten(0, 1),
                            torch.randn_like(denoised_pred.flatten(0, 1)),
                            next_timestep * timestep_shape_one_tensor,
                        ).unflatten(0, denoised_pred.shape[:2])
                else:
                    denoised_pred = self.generator_wrapper(
                        noisy_input,
                        timestep,
                        conditional_dict,
                        global_start_frame + current_start_frame,
                        use_cross_cache = False
                    )
                    break

            # Step 3.2: rerun with timestep zero to update the cache
            context_timestep = timestep_shape_one_tensor * 0  # self.context_noise
            with torch.no_grad():
                self.generator_wrapper(
                    denoised_pred,
                    context_timestep,
                    conditional_dict,
                    global_start_frame + current_start_frame,
                    update_cache=True,
                    use_cross_cache = False
                )
            # Step 3.3: record the model's output
            output[
                :, current_start_frame : current_start_frame + current_num_frames
            ] = denoised_pred
            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        return output

    def init_state(self, noise_shape, text_prompts, negative_prompts, num_chunks):
        batch_size = noise_shape[0]

        # Step 2: Extract the conditional infos
        noise = torch.randn(noise_shape, device=self.device, dtype=self.dtype)
        # num_frames = noise_shape[1]
        with torch.no_grad():
            conditional_dict = self.text_encoder(text_prompts=text_prompts)

            # 如果state已经存在就不需要再重新算负prompt
            if getattr(self, "state", None) is None:
                unconditional_dict = self.text_encoder(text_prompts=negative_prompts)
                empty_conditional_dict = self.text_encoder(
                    text_prompts=[""] * batch_size
                )
            else:
                unconditional_dict = self.state.unconditional_dict
                empty_conditional_dict = self.state.empty_conditional_dict

        if getattr(self, "kv_cache1", None) is None:
            self._initialize_cache(
                batch_size=batch_size,
                kv_cache_size= self.teacher_model_frames * self.frame_seq_length,
                sink_size=self.sink_size * self.frame_seq_length,
                window_size=self.local_attn_size * self.frame_seq_length,
                dtype=noise.dtype,
                device=noise.device,
            )
        else:
            self.clean_cache()

        self.crossattn_cache = None
        # if getattr(self, "crossattn_cache", None) is None:
        #     self.crossattn_cache = self._initialize_crossattn_cache(
        #         batch_size=batch_size, dtype=noise.dtype, device=noise.device
        #     )
        # else:
        #     self.clean_crossattn_cache(self.crossattn_cache)
        self.state = StreamingState(
            noise,
            conditional_dict,
            unconditional_dict,
            empty_conditional_dict,
            num_chunks,
        )
        self.vae.model.clear_cache()

    def finish(self):
        # 判断chunk是否生成完
        return self.state.chunk_index >= self.state.num_chunks

    def generate_next_chunk(self, use_empty_condition=False) -> torch.Tensor:

        noise = self.state.noise
        if use_empty_condition:
            conditional_dict = self.state.empty_conditional_dict
        else:
            conditional_dict = self.state.conditional_dict
        batch_size, total_num_frames, num_channels, height, width = noise.shape
        global_start_frame = self.state.global_start_frame
        if self.state.chunk_index == 0:
            num_frames = self.teacher_model_frames
        else:
            num_frames = self.teacher_model_frames - self.num_frame_per_block
        global_end_frame = global_start_frame + num_frames
        # assert num_frames % self.num_frame_per_block == 0  # num_frames 一定是21+ 18 * N
        chunk_noise = noise[:, global_start_frame:global_end_frame]

        output = self._generate_chunk(chunk_noise, global_start_frame, conditional_dict)
        self.state.global_start_frame = global_end_frame
        self.state.chunk_index += 1
        # total_output[:, global_start_frame:global_end_frame] = output.detach()

        return output

    def _initialize_cache(
        self, batch_size, kv_cache_size, sink_size, window_size, dtype, device
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
                    "window_k": torch.zeros(
                        [batch_size, window_size, 12, 128],
                        dtype=dtype,
                        device=device,
                    ),
                    "window_v": torch.zeros(
                        [batch_size, window_size, 12, 128],
                        dtype=dtype,
                        device=device,
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

    def clean_cache(self):
        """
        Clean the KV cache by resetting all cached tensors to zero.
        """
        for kv_cache in self.kv_cache1:
            kv_cache["k"].zero_()
            kv_cache["v"].zero_()
            kv_cache["global_end_index"].zero_()
            kv_cache["local_end_index"].zero_()
            kv_cache["sink_k"].zero_()
            kv_cache["sink_v"].zero_()
            kv_cache["window_k"].zero_()
            kv_cache["window_v"].zero_()

    def clean_crossattn_cache(self,crossattn_cache):
        """
        Clean the cross-attention cache by resetting all cached tensors to zero.
        """
        for kv_cache in crossattn_cache:
            kv_cache["k"].zero_()
            kv_cache["v"].zero_()
            kv_cache["is_init"].fill_(False)

    def clean_kv_cache(self, clean_global=False):
        """
        Clean the KV cache by resetting all cached tensors to zero.
        """
        for kv_cache in self.kv_cache1:
            kv_cache["k"].zero_()
            kv_cache["v"].zero_()
            if clean_global:
                kv_cache["global_end_index"].zero_()
            kv_cache["local_end_index"].zero_()

    def update_window_cache(self, window_size):
        """
        将kv cache最后几帧放入window cache
        """
        for i in range(self.num_transformer_blocks):
            kv_cache_end_index = self.kv_cache1[i]["local_end_index"]
            if kv_cache_end_index - window_size < 0:
                continue
            self.kv_cache1[i]["window_k"][:, :window_size] = self.kv_cache1[i]["k"][
                :, kv_cache_end_index - window_size : kv_cache_end_index
            ]
            self.kv_cache1[i]["window_v"][:, :window_size] = self.kv_cache1[i]["v"][
                :, kv_cache_end_index - window_size : kv_cache_end_index
            ]

    # def update_sink_cache(self):
    #     """
    #     将kv cache开头帧放入sink cache
    #     """
    #     for i in range(self.num_transformer_blocks):
    #         self.kv_cache1[i]["sink_k"] = self.kv_cache1[i]["k"][:,:3].clone()
    #         self.kv_cache1[i]["sink_v"] = self.kv_cache1[i]["v"][:,:3].clone()

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
                    "is_init": torch.tensor([False],dtype=torch.bool,device=device)
                }
            )
        return crossattn_cache

    def _process_first_frame_encoding(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Apply special encoding to the first frame, following the logic in _run_generator.

        Args:
            frames: frame sequence [batch_size, num_frames, C, H, W]

        Returns:
            processed_frames: processed frame sequence where the first frame is re-encoded as an image latent
        """
        total_frames = frames.shape[1]

        if total_frames <= 1:
            # Only one or zero frames, return as is
            return frames

        with torch.no_grad():
            # Decode the frames to be processed into pixels
            pixels = self.vae.decode_to_pixel(frames, use_cache=True)

            # Take the last frame's pixel representation
            last_frame_pixel = pixels[:, -9:, ...].to(self.dtype)
            # last_frame_pixel = rearrange(last_frame_pixel, "b t c h w -> b c t h w")

            # Re-encode as image latent
            store_latent = self.vae.encode_to_latent(last_frame_pixel).to(
                self.dtype
            )  # 应该就是3潜帧了

        # remaining_frames = frames[:, -(process_frames - 1) :, ...]
        # processed_frames = torch.cat([image_latent, remaining_frames], dim=1)
        # debug时拼接到state
        # if self.state.pixel_video is None:
        #     self.state.pixel_video = pixels.cpu()
        # else:
        #     self.state.pixel_video = torch.cat([self.state.pixel_video, pixels.cpu()], dim=1)
        # if self.state.latent_video is None:
        #     self.state.latent_video = frames.detach().cpu()
        # else:
        #     self.state.latent_video = torch.cat(
        #         [self.state.latent_video, frames.detach().cpu()], dim=1
        #     )
        return pixels, store_latent
