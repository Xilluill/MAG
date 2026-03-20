from typing import Tuple
from einops import rearrange
from torch import nn
import torch.distributed as dist
import torch
from wan.modules.causal_model import CausalWanModel
from wan.modules.model import WanModel
from pipeline import SelfForcingPipeline
from utils.loss import get_denoising_loss
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from utils.scheduler import FlowMatchScheduler
import torch.nn.functional as F


class dmd_model(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.num_training_frames = getattr(args, "num_training_frames", 21)  # 就是21
        self.local_attn_size = getattr(
            args, "local_attn_size", -1
        )  # 这里是多大 kv cache就是多大
        self.sink_size = getattr(
            args, "sink_size", 0
        )  # kv cache保留数量，local_attn_size包含sink token和最新的一些token
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
        self.denoising_loss_func = get_denoising_loss(args.denoising_loss_type)()
        # self.generator.num_frame_per_block = self.num_frame_per_block
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

        # if self.local_attn_size != -1:
        #     # Use the local frame size to compute the KV cache size
        #     self.num_frame_kv_cache = self.local_attn_size
        # else:
        #     # Use the default KV cache size
        self.num_frame_kv_cache = self.num_training_frames

    def _initialize_models(self, args, device="cpu"):
        self.real_model_name = getattr(args, "real_name", "Wan2.1-T2V-1.3B")
        self.fake_model_name = getattr(args, "fake_name", "Wan2.1-T2V-1.3B")

        self.generator = CausalWanModel.from_pretrained(
            args.generator_ckpt,
            local_attn_size=self.local_attn_size,
            sink_size=self.sink_size,
            num_frame_per_block=self.num_frame_per_block,
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

        self.vae = WanVAEWrapper(device)
        self.vae.requires_grad_(False)

        scheduler = FlowMatchScheduler(
            shift=args.timestep_shift, sigma_min=0.0, extra_one_step=True
        )
        scheduler.set_timesteps(1000, training=True)
        self.scheduler = scheduler

        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()
            self.fake_score.enable_gradient_checkpointing()
        self.inference_pipeline = None

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

    def _run_generator(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        initial_latent: torch.tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Optionally simulate the generator's input from noise using backward simulation
        and then run the generator for one-step.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
            - initial_latent: a tensor containing the initial latents [B, F, C, H, W].
        Output:
            - pred_image: a tensor with shape [B, F, C, H, W].
            - denoised_timestep: an integer
        """
        # Step 1: Sample noise and backward simulate the generator's input
        assert getattr(
            self.args, "backward_simulation", True
        ), "Backward simulation needs to be enabled"
        if initial_latent is not None:
            conditional_dict["initial_latent"] = initial_latent
        if self.args.i2v:
            noise_shape = [
                image_or_video_shape[0],
                image_or_video_shape[1] - 1,
                *image_or_video_shape[2:],
            ]
        else:
            noise_shape = image_or_video_shape.copy()

        # During training, the number of generated frames should be uniformly sampled from
        # [21, self.num_training_frames], but still being a multiple of self.num_frame_per_block
        min_num_frames = 20 if self.args.independent_first_frame else 21
        max_num_frames = (
            self.num_training_frames - 1
            if self.args.independent_first_frame
            else self.num_training_frames
        )
        assert max_num_frames % self.num_frame_per_block == 0
        assert min_num_frames % self.num_frame_per_block == 0
        max_num_blocks = max_num_frames // self.num_frame_per_block
        min_num_blocks = min_num_frames // self.num_frame_per_block
        num_generated_blocks = torch.randint(
            min_num_blocks, max_num_blocks + 1, (1,), device=self.device
        )
        dist.broadcast(num_generated_blocks, src=0)  # 7
        num_generated_blocks = num_generated_blocks.item()
        num_generated_frames = num_generated_blocks * self.num_frame_per_block  # 21
        if self.args.independent_first_frame and initial_latent is None:
            num_generated_frames += 1
            min_num_frames += 1
        # Sync num_generated_frames across all processes
        noise_shape[1] = num_generated_frames
        noise = torch.randn(noise_shape, device=self.device, dtype=self.dtype)
        pred_image_or_video, noisy_video, denoised_timestep_from, denoised_timestep_to,exit_step= (
            self._consistency_backward_simulation(
                noise=noise,
                **conditional_dict,
            )
        )
        # Slice last 21 frames
        if pred_image_or_video.shape[1] > 21:
            with torch.no_grad():
                # Reencode to get image latent
                latent_to_decode = pred_image_or_video[:, :-20, ...]
                # Deccode to video
                pixels = self.vae.decode_to_pixel(latent_to_decode)
                frame = pixels[:, -1:, ...].to(self.dtype)
                frame = rearrange(frame, "b t c h w -> b c t h w")
                # Encode frame to get image latent
                image_latent = self.vae.encode_to_latent(frame).to(self.dtype)
            pred_image_or_video_last_21 = torch.cat(
                [image_latent, pred_image_or_video[:, -20:, ...]], dim=1
            )
        else:
            pred_image_or_video_last_21 = pred_image_or_video

        # if num_generated_frames != min_num_frames:
        #     # Currently, we do not use gradient for the first chunk, since it contains image latents
        #     gradient_mask = torch.ones_like(
        #         pred_image_or_video_last_21, dtype=torch.bool
        #     )
        #     if self.args.independent_first_frame:
        #         gradient_mask[:, :1] = False
        #     else:
        #         gradient_mask[:, : self.num_frame_per_block] = False
        # else:
        #     gradient_mask = None

        pred_image_or_video_last_21 = pred_image_or_video_last_21.to(self.dtype)
        return (
            pred_image_or_video_last_21,
            noisy_video,
            denoised_timestep_from,
            denoised_timestep_to,
            exit_step
        )

    def _consistency_backward_simulation(
        self, noise: torch.Tensor, **conditional_dict: dict
    ) -> torch.Tensor:
        """
        Simulate the generator's input from noise to avoid training/inference mismatch.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Here we use the consistency sampler (https://arxiv.org/abs/2303.01469)
        Input:
            - noise: a tensor sampled from N(0, 1) with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
        Output:
            - output: a tensor with shape [B, T, F, C, H, W].
            T is the total number of timesteps. output[0] is a pure noise and output[i] and i>0
            represents the x0 prediction at each timestep.
        """
        if self.inference_pipeline is None:
            self._initialize_inference_pipeline()

        return self.inference_pipeline.inference_with_trajectory(
            noise=noise, **conditional_dict
        )

    def _initialize_inference_pipeline(self):
        """
        Lazy initialize the inference pipeline during the first backward simulation run.
        Here we encapsulate the inference code with a model-dependent outside function.
        We pass our FSDP-wrapped modules into the pipeline to save memory.
        """
        self.inference_pipeline = SelfForcingPipeline(
            denoising_step_list=self.denoising_step_list,
            scheduler=self.scheduler,
            generator=self.generator,
            num_frame_per_block=self.num_frame_per_block,
            independent_first_frame=self.args.independent_first_frame,
            same_step_across_blocks=self.args.same_step_across_blocks,
            last_step_only=self.args.last_step_only,
            num_frame_kv_cache=self.num_frame_kv_cache,
            context_noise=self.args.context_noise,
        )

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

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noise and compute the DMD loss.
        The noisy input to the generator is backward simulated.
        This removes the need of any datasets during distillation.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
        Output:
            - loss: a scalar tensor representing the generator loss.
            - generator_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        # Step 1: Unroll generator to obtain fake videos
        pred_video, noisy_video, denoised_timestep_from, denoised_timestep_to,exit_step = (
            self._run_generator(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                initial_latent=initial_latent,
            )
        )
        # normal_video = self.vae.decode_to_pixel(pred_video.float(), use_cache=False)
        # normal_video = rearrange(normal_video, 'b t c h w -> b t h w c').cpu()
        # normal_video = (normal_video * 0.5 + 0.5).clamp(0, 1)
        # normal_video = 255.0 * normal_video
        # output_path = 'test_pred.mp4'
        # from torchvision.io import write_video
        # write_video(output_path, normal_video[0], fps=16)

        # Step 2: Compute the DMD loss
        dmd_loss, dmd_log_dict = self.compute_distribution_matching_loss(
            image_or_video=pred_video,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            gradient_mask=None,
            denoised_timestep_from=denoised_timestep_from,
            denoised_timestep_to=denoised_timestep_to,
        )
        # zero_tensor = torch.zeros_like(dmd_loss,requires_grad=False)
        return dmd_loss, dmd_log_dict

    def critic_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noise and train the critic with generated samples.
        The noisy input to the generator is backward simulated.
        This removes the need of any datasets during distillation.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
        Output:
            - loss: a scalar tensor representing the generator loss.
            - critic_log_dict: a dictionary containing the intermediate tensors for logging.
        """

        # Step 1: Run generator on backward simulated noisy input
        with torch.no_grad():
            generated_image, _, denoised_timestep_from, denoised_timestep_to,_ = (
                self._run_generator(
                    image_or_video_shape=image_or_video_shape,
                    conditional_dict=conditional_dict,
                    initial_latent=initial_latent,
                )
            )

        # Step 2: Compute the fake prediction
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

        critic_noise = torch.randn_like(generated_image)
        noisy_generated_image = self.scheduler.add_noise(
            generated_image.flatten(0, 1),
            critic_noise.flatten(0, 1),
            critic_timestep.flatten(0, 1),
        ).unflatten(0, image_or_video_shape[:2])

        flow_pred = self.fake_score(
            noisy_generated_image.permute(0, 2, 1, 3, 4),
            t=critic_timestep[:, 0],
            context=conditional_dict["prompt_embeds"],
            seq_len=32760,
        ).permute(0, 2, 1, 3, 4)
        # torch.mean((flow_pred - (critic_noise - generated_image)) ** 2)
        denoising_loss = F.mse_loss(
            flow_pred.double(),
            (critic_noise - generated_image).double(),
            reduction="mean",
        )

        # Step 5: Debugging Log
        critic_log_dict = {"critic_timestep": critic_timestep.detach()}

        return denoising_loss, critic_log_dict