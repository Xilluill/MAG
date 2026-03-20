
import torch
from model.long_dmd_model import long_dmd_model,StreamingState
import torch.nn.functional as F

class compress_model(long_dmd_model):
    def __init__(self, args, device):
        super().__init__(args, device)
    
    def init_state(self,batch_size, text_prompts, negative_prompts):
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

        self.state = StreamingState(
            None,
            conditional_dict,
            unconditional_dict,
            empty_conditional_dict,
            None,
        )
        self.vae.model.clear_cache()

    def block_compress_loss(
        self, estimated_clean_image_or_video=None, conditional_dict=None
    ) -> torch.Tensor:

        use_empty_condition = torch.rand(1).item() < self.compress_empty_condition_p
        if estimated_clean_image_or_video is None:
            num_train_chunks = 4
            num_frames = 21 + (num_train_chunks - 1) * 18
            estimated_clean_image_or_video = self.state.latent_video[:, :num_frames].to(
                self.device
            )
            if use_empty_condition:
                conditional_dict = self.state.empty_conditional_dict
            else:
                conditional_dict = self.state.conditional_dict
        else:
            num_frames = estimated_clean_image_or_video.shape[1]
            estimated_clean_image_or_video = estimated_clean_image_or_video.to(
                self.device
            )

            if use_empty_condition:  # 随机替换成 空，这个分支condition 是有输入的
                conditional_dict = self.state.empty_conditional_dict

        num_clean_frames = estimated_clean_image_or_video.shape[1]
        num_noisy_frames = num_clean_frames
        batch_size = estimated_clean_image_or_video.shape[0]
        # 做memory forcing 最后几帧是不参与的
        noisy_input = estimated_clean_image_or_video[:, :num_noisy_frames]
        num_denoising_steps = len(self.denoising_step_list)
        indices = torch.randint(
            low=0, high=num_denoising_steps, size=(1,), device="cpu"
        )
        current_timestep = self.denoising_step_list[indices].to(self.device)

        timestep_shape_one_tensor = torch.ones(
            [batch_size, num_noisy_frames],
            device=estimated_clean_image_or_video.device,
            dtype=torch.int64,
        )
        timestep = timestep_shape_one_tensor * current_timestep.to(self.device)
        noisy_input = self.scheduler.add_noise(
            noisy_input.flatten(0, 1),
            torch.randn_like(noisy_input.flatten(0, 1)),
            timestep,
        ).unflatten(0, noisy_input.shape[:2])
        
        # 随机位置编码的起始点，最多120帧，减去当前帧数
        start_index = torch.randint(0, 120 - num_noisy_frames + 1, (1,)).item()
        start_index = start_index // self.num_frame_per_block * self.num_frame_per_block
        flow_pred = self.generator(
            x=noisy_input.permute(
                0, 2, 1, 3, 4
            ),  # List of input video tensors, each with shape [C_in, F, H, W]
            context=conditional_dict["prompt_embeds"],
            t=timestep,
            seq_len=327600,  # [1, 21, 16, 60, 104]
            clean_x=estimated_clean_image_or_video.permute(0, 2, 1, 3, 4),
            block_mask_type='history_compress',
            start_index=start_index
        ).permute(0, 2, 1, 3, 4)

        denoised_pred = self.scheduler._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_input.flatten(0, 1),
            timestep=timestep.flatten(0, 1),
        ).unflatten(0, flow_pred.shape[:2])

        loss = F.mse_loss(
            denoised_pred.double(),
            estimated_clean_image_or_video[:, :num_noisy_frames].double(),
            reduction="mean",
        )
        return loss
    
    def multi_step_block_compress_loss(
        self, estimated_clean_image_or_video=None, conditional_dict=None
    ) -> torch.Tensor:

        use_empty_condition = torch.rand(1).item() < self.compress_empty_condition_p
        if estimated_clean_image_or_video is None:
            num_train_chunks = 4
            num_frames = 21 + (num_train_chunks - 1) * 18
            estimated_clean_image_or_video = self.state.latent_video[:, :num_frames].to(
                self.device
            )
            if use_empty_condition:
                conditional_dict = self.state.empty_conditional_dict
            else:
                conditional_dict = self.state.conditional_dict
        else:
            num_frames = estimated_clean_image_or_video.shape[1]
            estimated_clean_image_or_video = estimated_clean_image_or_video.to(
                self.device
            )

            if use_empty_condition:  # 随机替换成 空，这个分支condition 是有输入的
                conditional_dict = self.state.empty_conditional_dict

        num_frame_per_block = self.num_frame_per_block
        num_training_frames = self.num_training_frames
        local_attn_size = self.local_attn_size
        num_clean_frames = estimated_clean_image_or_video.shape[1]
        num_noisy_frames = num_clean_frames
        batch_size = estimated_clean_image_or_video.shape[0]
        # 做memory forcing 最后几帧是不参与的
        # noisy_input = estimated_clean_image_or_video[:, :num_noisy_frames]
        num_denoising_steps = len(self.denoising_step_list)
        exit_flags = self.generate_and_sync_list(
            1, num_denoising_steps, device=estimated_clean_image_or_video.device
        )
        # print("exit_flags:", exit_flags)
        # current_timestep = self.denoising_step_list[indices].to(self.device)
        noisy_input = torch.randn_like(
            estimated_clean_image_or_video[:, :num_noisy_frames]
        )
        timestep_shape_one_tensor = torch.ones(
                [batch_size, num_noisy_frames],
                device=estimated_clean_image_or_video.device,
                dtype=torch.int64,
            )
        for index, current_timestep in enumerate(self.denoising_step_list):
            timestep = timestep_shape_one_tensor * current_timestep.to(self.device)

            exit_flag = index == exit_flags[0]
            if not exit_flag:
                with torch.no_grad():
                    flow_pred = self.generator(
                        x=noisy_input.permute(
                            0, 2, 1, 3, 4
                        ),  # List of input video tensors, each with shape [C_in, F, H, W]
                        context=conditional_dict["prompt_embeds"],
                        t=timestep,
                        seq_len=327600,  # [1, 21, 16, 60, 104]
                        clean_x=estimated_clean_image_or_video.permute(0, 2, 1, 3, 4),
                        block_mask_type='compress'
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
                flow_pred = self.generator(
                    x=noisy_input.permute(
                        0, 2, 1, 3, 4
                    ),  # List of input video tensors, each with shape [C_in, F, H, W]
                    context=conditional_dict["prompt_embeds"],
                    t=timestep,
                    seq_len=327600,  # [1, 21, 16, 60, 104]
                    clean_x=estimated_clean_image_or_video.permute(0, 2, 1, 3, 4),
                    block_mask_type='compress'
                ).permute(0, 2, 1, 3, 4)
                denoised_pred = self.scheduler._convert_flow_pred_to_x0(
                    flow_pred=flow_pred.flatten(0, 1),
                    xt=noisy_input.flatten(0, 1),
                    timestep=timestep.flatten(0, 1),
                ).unflatten(0, flow_pred.shape[:2])
                print(f"exit_flag reached at step {index}")
                break

        loss = F.mse_loss(
            denoised_pred.double(),
            estimated_clean_image_or_video[:, :num_noisy_frames].double(),
            reduction="mean",
        )
        # output = denoised_pred.detach()
        # with torch.autocast("cuda",dtype=torch.bfloat16):
        #     video = self.vae.decode_to_pixel(output, use_cache=False)
        # video = (video * 0.5 + 0.5).clamp(0, 1)
        return loss #,video