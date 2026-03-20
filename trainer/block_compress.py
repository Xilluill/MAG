from datasets import VPData_Dataset, TextDataset
from omegaconf import OmegaConf
import torch
import wandb
from datetime import timedelta
import os
from tqdm import tqdm
import argparse
from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.utils import set_seed
from utils.distributed import EMA_FSDP, fsdp_wrap
# torch.cuda.memory._record_memory_history()
from accelerate.utils import InitProcessGroupKwargs
from utils.scheduler import FlowMatchScheduler
from model.compress_model import compress_model

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
torch.set_num_threads(8)


class Trainer:
    def __init__(
        self,
        model: compress_model,
        generator_optimizer,
        train_dataloader,
        video_dataloader,
        accelerator: Accelerator,
        config,
    ):

        self.model = model
        self.model.eval()
        # self.optimizer = optimizer
        self.generator_optimizer = generator_optimizer
        self.train_dataloader = train_dataloader
        self.video_dataloader = video_dataloader
        self.accelerator = accelerator
        self.config = config
        set_seed(config.seed, device_specific=True)
        self.device = accelerator.device
        self.global_step = 0
        self.generator_step = 0
        self.critic_step = 0
        self.start_step = 0
        checkpoint_path = config.get("ckpt_path", None)
        if checkpoint_path is not None:
            self.accelerator.print(f"Loading model from {checkpoint_path}")
            self.accelerator.load_state(checkpoint_path)
            self.global_step = config.ckpt_step
            self.start_step = config.ckpt_step

        scheduler = FlowMatchScheduler(
            shift=config.timestep_shift, sigma_min=0.0, extra_one_step=True
        )
        scheduler.set_timesteps(1000, training=True)
        self.scheduler = scheduler
        self.scheduler.timesteps = self.scheduler.timesteps.to(accelerator.device)

        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)

        self.gradient_accumulation_steps = 2  # config.gradient_accumulation_steps
        accelerator.print("mixed precision:", accelerator.state.mixed_precision)

        self.generator_ema = None

    def _get_timestep(
        self,
        min_timestep: int,
        max_timestep: int,
        batch_size: int,
        num_frame: int,
        num_frame_per_block: int,
        uniform_timestep: bool = False,
        independent_first_frame: bool = False,
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
            if independent_first_frame:
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

    def save_checkpoint(self, save_directory):

        torch.cuda.empty_cache()
        self.accelerator.save_state(save_directory)
        self.accelerator.print(f"Saved final checkpoint to {save_directory}")
        # self.accelerator.get_state_dict

    def cycle(self, dataloader):
        while True:
            for data in dataloader:
                yield data

    def cycle_with_check(self, dataloader):
        while True:
            for data in dataloader:
                if data[0] != []:
                    yield data

    def init_state(self):
        batch = next(self.train_dataloader)
        text_prompts = batch["prompts"]
        negative_prompts = [self.config.negative_prompt]

        image_or_video_shape = list(self.config.image_or_video_shape)
        batch_size = image_or_video_shape[0]

        self.model.init_state(
            batch_size, text_prompts, negative_prompts
        )

    def block_compress_step(self):
        video_batch = next(self.video_dataloader)
        video_name, video_features, text_features = video_batch
        video_features = torch.stack(video_features)
        text_features = torch.stack(text_features)
        # debug print出三个的详细信息
        self.accelerator.print(
            f"rank:{self.accelerator.process_index} Video Name: {video_name}"
        )
        self.accelerator.print(
            f"rank:{self.accelerator.process_index} Video Features Shape: {video_features.shape}"
        )
        
        conditional_dict = {"prompt_embeds": text_features}
        block_compress_loss = self.model.block_compress_loss(video_features, conditional_dict)
        # current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
        # video = 255.0 * current_video
        # output_path = os.path.join('videos/train_debug', f'{block_compress_loss.item():.3f}.mp4')
        # write_video(output_path, video[0], fps=16)
        # self.model.vae.model.clear_cache()
        return block_compress_loss

    def train(self, max_train_steps):
        pbar = tqdm(
            total=max_train_steps,
            desc="Training",
            disable=not self.accelerator.is_main_process,
        )
        self.train_dataloader = self.cycle(self.train_dataloader)
        self.video_dataloader = self.cycle_with_check(self.video_dataloader)
        self.accelerator.print("Starting training...")

        self.init_state()
        while self.global_step < max_train_steps:
            # self.accelerator.print(f"开始循环 threads = {torch.get_num_threads()}")
            if self.global_step >= max_train_steps:
                break
            
            if (self.global_step >= self.config.ema_start_step) and (self.generator_ema is None) and (self.config.ema_weight > 0):
                self.generator_ema = EMA_FSDP(self.model.generator, decay=self.config.ema_weight)
                self.accelerator.register_for_checkpointing(self.generator_ema)
            # for chunk_index in range(num_chunks):
            with self.accelerator.accumulate(self.model):

                block_compress_loss = self.block_compress_step()
                self.accelerator.backward(block_compress_loss)
                if self.accelerator.sync_gradients:
                    generator_grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.generator.parameters(),
                        self.max_grad_norm_generator,
                    )
                    self.generator_optimizer.step()
                    self.generator_optimizer.zero_grad(set_to_none=True)
                    if self.generator_ema is not None:
                            self.generator_ema.update(self.model.generator)
                    total_bc_grad_norm = self.accelerator.reduce(
                        generator_grad_norm, reduction="mean"
                    )
                    total_block_compress_loss = self.accelerator.reduce(
                        block_compress_loss.detach(), reduction="mean"
                    )

            # 当梯度累计没有更新时，以下都不会执行，会导致全局步骤不增加
            if self.accelerator.sync_gradients and self.accelerator.is_main_process:
                wandb_loss_dict = {}

                self.accelerator.print(
                    f"Step {self.global_step}, bc_loss: {total_block_compress_loss.item()}"
                )
                wandb_loss_dict.update(
                    {
                        "block_compress_loss": total_block_compress_loss.item(),
                        "total_bc_grad_norm": total_bc_grad_norm.item(),
                        "step": self.global_step,
                    }
                )
                self.accelerator.log(wandb_loss_dict, step=self.global_step)
                pbar.update(1)

            if self.accelerator.sync_gradients:
                if self.global_step % 200 == 0 and self.global_step > self.start_step:
                    save_directory = os.path.join(
                        self.config.logdir, f"checkpoint-step-{self.global_step}"
                    )
                    self.save_checkpoint(save_directory)
                    self.accelerator.wait_for_everyone()  # 等待所有进程完成
                self.global_step += 1  # 所有事情完成后再更新步骤

        save_directory = os.path.join(self.config.logdir, "final_checkpoint")
        self.save_checkpoint(save_directory)

        self.accelerator.print("Finished training.")
        self.accelerator.end_training()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/block_compress.yaml",
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default="logs/block_compress_debug",
        help="Path to the directory to save logs",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="",
        help="continue training from this checkpoint",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="VPData/compress_train_filter_suffix.csv",
    )
    parser.add_argument(
        "--video_folder",
        type=str,
        default="VPData/train_filter_video_16fps_suffix_pt_bf16",
    )
    # parser.add_argument("--wandb-save-dir", type=str, default="logs/wandb", help="Path to the directory to save wandb logs")

    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)

    # get the filename of config_path
    # config_name = os.path.basename(args.config_path).split(".")[0]
    config.logdir = os.path.join(args.logdir, config.wandb_name)  # config_name)
    if args.ckpt_path:
        config.ckpt_step = int(args.ckpt_path[args.ckpt_path.rfind("-") + 1 :])
        config.ckpt_path = args.ckpt_path
        print(f"Using ckpt_path: {config.ckpt_path},step: {config.ckpt_step}")

    kwargs = InitProcessGroupKwargs(backend="nccl", timeout=timedelta(seconds=5000))

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="wandb",
        kwargs_handlers=[kwargs],
    )
    # if accelerator.is_main_process:
    #         torch.cuda.memory._record_memory_history()
    # model: CausalWanModel = CausalWanModel.from_pretrained(f"wan_models/Wan2.1-T2V-1.3B/")
    model = compress_model(
        config, device=torch.device(f"cuda:{torch.cuda.current_device()}")
    )
    del model.fake_score
    del model.real_score
    # model.enable_gradient_checkpointing()
    generator_optimizer = torch.optim.AdamW(
        [param for param in model.generator.parameters() if param.requires_grad],
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
    )
    torch.manual_seed(config.seed)
    # dataset = VPData_Dataset(csv_path=args.csv_path, video_folder=args.video_folder, num_frame_per_block= config.num_frame_per_block)
    if args.ckpt_path:
        dataset = TextDataset(
            config.data_path, start_index=config.ckpt_step * config.total_batch_size
        )
    else:
        dataset = TextDataset(
            config.data_path, start_index=300 * config.total_batch_size
        )
    accelerator.print(
        f"[rank:{accelerator.process_index}] Dataset length: {len(dataset)}"
    )
    video_dataset = VPData_Dataset(
        csv_path=args.csv_path,
        video_folder=args.video_folder,
        num_frame_per_block=config.num_frame_per_block,
        start_index=0,
        min_len= 21 // config.num_frame_per_block * config.num_frame_per_block
    )
    accelerator.print(
        f"[rank:{accelerator.process_index}] video Dataset length: {len(video_dataset)}"
    )
    # train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, num_workers=2, pin_memory=True,collate_fn=dataset.collate_fn, shuffle=False, drop_last=True)
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=2,
        pin_memory=True,
        shuffle=False,
        drop_last=True,
    )

    video_dataloader = torch.utils.data.DataLoader(
        video_dataset,
        batch_size=config.batch_size,
        num_workers=2,
        pin_memory=True,
        shuffle=False,
        drop_last=True,
        collate_fn=video_dataset.collate_fn,
    )
    accelerator.init_trackers(
        project_name=config.wandb_project,
        config=OmegaConf.to_container(config, resolve=True),
        init_kwargs={
            "wandb": {
                "name": config.wandb_name,  # config_name,
                "dir": config.logdir,
            },
        },
    )
    (
        model.generator,
        generator_optimizer,
        train_dataloader,
        video_dataloader,
    ) = accelerator.prepare(
        model.generator,
        generator_optimizer,
        train_dataloader,
        video_dataloader,
    )
    model.text_encoder = fsdp_wrap(
        model.text_encoder,
        sharding_strategy=config.sharding_strategy,
        mixed_precision=config.mixed_precision,
        wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
    )
    torch.cuda.empty_cache()
    trainer = Trainer(
        model=model,
        generator_optimizer=generator_optimizer,
        train_dataloader=train_dataloader,
        video_dataloader=video_dataloader,
        accelerator=accelerator,
        config=config,
    )
    trainer.train(max_train_steps=config.max_train_steps)


if __name__ == "__main__":
    main()
