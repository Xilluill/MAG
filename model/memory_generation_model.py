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
from model.long_dmd_model import long_dmd_model,StreamingState

class memory_generation_model(long_dmd_model):
    def __init__(self, args, device):
        super().__init__(args, device)

    def _initialize_models(self, args, device="cpu"):
        self.real_model_name = getattr(args, "real_name", "Wan2.1-T2V-1.3B")
        self.fake_model_name = getattr(args, "fake_name", "Wan2.1-T2V-1.3B")

        self.generator = CausalWanModel.from_pretrained(
            args.generator_ckpt,
            local_attn_size=self.local_attn_size,
            sink_size=self.sink_size,
            num_frame_per_block=self.num_frame_per_block,
            is_inference_mode=False,
            cache_mode=self.cache_mode
        )
        self.generator.requires_grad_(True)

        self.memory_model = CausalWanModel.from_pretrained(
            args.memory_ckpt,
            local_attn_size=self.local_attn_size,
            sink_size=self.sink_size,
            num_frame_per_block=self.num_frame_per_block,
            is_inference_mode=False,
            cache_mode=self.cache_mode
        )
        self.memory_model.requires_grad_(False)
        # self.generator.base_model.requires_grad_(True)
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
                kv_cache_size= 43 * self.frame_seq_length, # 因为这里进行了修改所以重写函数
                sink_size=self.sink_size * self.frame_seq_length,
                dtype=noise.dtype,
                device=noise.device,
            )
        else:
            self.clean_kv_cache(clean_global=True)

        self.crossattn_cache = None
        self.crossattn_cache_empty = None
        # if getattr(self, "crossattn_cache", None) is None:
        #     self.crossattn_cache = self._initialize_crossattn_cache(
        #         batch_size=batch_size, dtype=noise.dtype, device=noise.device
        #     )
        # else:
        #     self.clean_crossattn_cache(self.crossattn_cache)

        # if getattr(self, "crossattn_cache_empty", None) is None: #这个不需要clean
        #     self.crossattn_cache_empty = self._initialize_crossattn_cache(
        #         batch_size=batch_size, dtype=noise.dtype, device=noise.device
        #     )
        self.state = StreamingState(
            noise,
            conditional_dict,
            unconditional_dict,
            empty_conditional_dict,
            num_chunks,
        )
        self.vae.model.clear_cache()

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

    def generator_wrapper(
        self,
        noisy_input,
        timestep,
        conditional_dict,
        current_start_frame,
        update_cache=False,
        use_cross_cache=True,
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

    def memory_wrapper(
        self,
        noisy_input,
        timestep,
        conditional_dict,
        current_start_frame,
        update_cache=False,
        use_cross_cache=True,
    ):
        assert update_cache is True, 'only update cache use memory model'
        flow_pred = self.memory_model(
            noisy_input.permute(0, 2, 1, 3, 4),
            t=timestep,
            context=conditional_dict["prompt_embeds"],
            seq_len=327600,
            kv_cache=self.kv_cache1,
            crossattn_cache=self.crossattn_cache_empty if use_cross_cache else [None]*self.num_transformer_blocks,
            current_start=current_start_frame * self.frame_seq_length,
            update_cache=update_cache,
        ).permute(0, 2, 1, 3, 4)

    def _generate_chunk(
        self, noise, global_start_frame, conditional_dict, use_cross_cache=True
    ):

        batch_size, num_frames, num_channels, height, width = noise.shape
        num_full_blocks = num_frames // self.num_frame_per_block
        generate_first_frame = (num_frames % self.num_frame_per_block) == 1
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
                    exit_flag = (
                        index == exit_flags[block_index]
                    )  # Only backprop at the randomly selected timestep (consistent across all ranks)
                timestep = timestep_shape_one_tensor * current_timestep
                if not exit_flag:
                    with torch.no_grad():
                        denoised_pred = self.generator_wrapper(
                            noisy_input,
                            timestep,
                            conditional_dict,
                            global_start_frame + current_start_frame,
                            use_cross_cache=False
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
                        use_cross_cache=False
                    )
                    break

            # Step 3.2: rerun with timestep zero to update the cache
            context_timestep = timestep_shape_one_tensor * 0  # self.context_noise
            with torch.no_grad():
                self.memory_wrapper(
                    denoised_pred,
                    context_timestep,
                    self.state.empty_conditional_dict,
                    global_start_frame + current_start_frame,
                    update_cache=True,
                    use_cross_cache=False
                )
            # Step 3.3: record the model's output
            output[
                :, current_start_frame : current_start_frame + current_num_frames
            ] = denoised_pred
            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        return output

class block_cache_long_model(memory_generation_model):
    def __init__(self, args, device):
        super().__init__(args, device)
        self.cache_mode = args.cache_mode

    def _initialize_models(self, args, device="cpu"):
        self.real_model_name = getattr(args, "real_name", "Wan2.1-T2V-1.3B")
        self.fake_model_name = getattr(args, "fake_name", "Wan2.1-T2V-1.3B")

        self.generator = CausalWanModel.from_pretrained(
            args.generator_ckpt,
            local_attn_size=self.local_attn_size,
            sink_size=self.sink_size,
            num_frame_per_block=self.num_frame_per_block,
            is_inference_mode=False,
            cache_mode=self.cache_mode
        )
        self.generator.requires_grad_(True)
        # self.generator.base_model.requires_grad_(True)
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

    def _generate_chunk(
        self, noise, global_start_frame, conditional_dict,
    ):

        batch_size, num_frames, num_channels, height, width = noise.shape
        num_full_blocks = num_frames // self.num_frame_per_block
        generate_first_frame = (num_frames % self.num_frame_per_block) == 1
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
                    exit_flag = (
                        index == exit_flags[block_index]
                    )  # Only backprop at the randomly selected timestep (consistent across all ranks)
                timestep = timestep_shape_one_tensor * current_timestep
                if not exit_flag:
                    with torch.no_grad():
                        denoised_pred = self.generator_wrapper(
                            noisy_input,
                            timestep,
                            conditional_dict,
                            global_start_frame + current_start_frame,
                            use_cross_cache=False
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
                        use_cross_cache=False
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
                    use_cross_cache=False
                )
            # Step 3.3: record the model's output
            output[
                :, current_start_frame : current_start_frame + current_num_frames
            ] = denoised_pred
            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        return output