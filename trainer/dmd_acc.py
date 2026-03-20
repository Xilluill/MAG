from datasets import TextDataset
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
from model.dmd_model import dmd_model

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
torch.set_num_threads(8)

# class Scheduler:  # 用于给其他类提供单一的scheduler功能
#     def __init__(self, timestep_shift, device="cuda"):
#         scheduler = FlowMatchScheduler(shift=timestep_shift, sigma_min=0.0, extra_one_step=True)
#         scheduler.set_timesteps(1000, training=True)
#         self.scheduler = self.get_scheduler(scheduler)
#         self.scheduler.timesteps = self.scheduler.timesteps.to(device)


class Trainer:
    def __init__(
        self,
        model: dmd_model,
        generator_optimizer,
        critic_optimizer,
        train_dataloader,
        accelerator: Accelerator,
        config,
    ):

        self.model = model
        self.model.eval()
        # self.optimizer = optimizer
        self.generator_optimizer = generator_optimizer
        self.critic_optimizer = critic_optimizer
        self.train_dataloader = train_dataloader
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

        # if (self.generator_ema is None) and (self.config.ema_weight > 0):
        #     self.generator_ema = EMA_FSDP(self.model.generator, decay=self.config.ema_weight)
        #     self.accelerator.register_for_checkpointing(self.generator_ema)
        
        # self.generator_ema.update(self.model.generator)
        # self.generator_ema.update(self.model.generator)
        # save_directory = os.path.join(
        #         self.config.logdir, f"checkpoint-step-debug_ema"
        #     )
        # self.save_checkpoint(save_directory)

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

    def fwdbwd_one_step(self, batch, train_generator):
        self.model.eval()  # prevent any randomness (e.g. dropout)

        # if self.step % 20 == 0:
        #     torch.cuda.empty_cache()

        # Step 1: Get the next batch of text prompts
        text_prompts = batch["prompts"]
        if self.config.i2v:
            clean_latent = None
            image_latent = batch["ode_latent"][:, -1][
                :,
                0:1,
            ].to(device=self.device, dtype=self.dtype)
        else:
            clean_latent = None
            image_latent = None

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

        # Step 2: Extract the conditional infos
        with torch.no_grad():
            conditional_dict = self.model.text_encoder(text_prompts=text_prompts)

            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size
                )
                unconditional_dict = {
                    k: v.detach() for k, v in unconditional_dict.items()
                }
                self.unconditional_dict = (
                    unconditional_dict  # cache the unconditional_dict
                )
            else:
                unconditional_dict = self.unconditional_dict

        # Step 3: Store gradients for the generator (if training the generator)
        if train_generator:
            dmd_loss, mf_loss, generator_log_dict = self.model.generator_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent,
                initial_latent=image_latent if self.config.i2v else None,
            )
            dmd_grad = generator_log_dict["dmdtrain_gradient_norm"]

            return dmd_loss, mf_loss, dmd_grad
        else:
            # Step 4: Store gradients for the critic (if training the critic) fake模型会用fake图像在diffusion loss上训练，提供fake分布的分数
            critic_loss, critic_log_dict = self.model.critic_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent,
                initial_latent=image_latent if self.config.i2v else None,
            )

            return critic_loss

    def save_checkpoint(self, save_directory):

        self.accelerator.save_state(save_directory)
        self.accelerator.print(f"Saved final checkpoint to {save_directory}")
        # self.accelerator.get_state_dict

    def cycle(self, dataloader):
        while True:
            for data in dataloader:
                yield data

    def train(self, max_train_steps):
        pbar = tqdm(
            total=max_train_steps,
            desc="Training",
            disable=not self.accelerator.is_main_process,
        )
        self.train_dataloader = self.cycle(self.train_dataloader)
        self.accelerator.print("Starting training...")

        while self.global_step < max_train_steps:
            # self.accelerator.print(f"开始循环 threads = {torch.get_num_threads()}")
            if self.global_step >= max_train_steps:
                break
            
            if (self.global_step >= self.config.ema_start_step) and (self.generator_ema is None) and (self.config.ema_weight > 0):
                self.generator_ema = EMA_FSDP(self.model.generator, decay=self.config.ema_weight)
                self.accelerator.register_for_checkpointing(self.generator_ema)

            TRAIN_GENERATOR = self.global_step % (1+self.config.dfake_gen_update_ratio) == 0
            batch = next(self.train_dataloader)
            with self.accelerator.accumulate(self.model):
                if TRAIN_GENERATOR:
                    dmd_loss, mf_loss, dmd_grad = self.fwdbwd_one_step(batch, train_generator=True)
                    generator_loss = dmd_loss
                    self.accelerator.backward(generator_loss)
                    if self.accelerator.sync_gradients:
                        generator_grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.generator.parameters(), self.max_grad_norm_generator
                        )
                        self.generator_optimizer.step()
                        self.generator_optimizer.zero_grad(set_to_none=True)
                        if self.generator_ema is not None:
                            self.generator_ema.update(self.model.generator)
                else:
                    critic_loss = self.fwdbwd_one_step(batch,train_generator=False)
                    self.accelerator.backward(critic_loss)
                    if self.accelerator.sync_gradients:
                        fake_score_grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.fake_score.parameters(), self.max_grad_norm_critic
                        )
                        self.critic_optimizer.step()
                        self.critic_optimizer.zero_grad(set_to_none=True)

            # 当梯度累计没有更新时，以下都不会执行，会导致全局步骤不增加
            if self.accelerator.sync_gradients and self.accelerator.is_main_process:
                wandb_loss_dict = {}
                if TRAIN_GENERATOR:
                    self.accelerator.print(
                        f"Step {self.global_step}, dmd_loss: {dmd_loss.item()}, mf_loss: {mf_loss.item()},Grad Norm: {generator_grad_norm.item()}"
                    )
                    wandb_loss_dict.update(
                        {
                            "generator_loss": dmd_loss.item(),
                            "mf_loss": mf_loss.item(),
                            "generator_grad_norm": generator_grad_norm.item(),
                            "dmdtrain_gradient_norm": dmd_grad.item(),
                            "step": self.global_step,
                        }
                    )
                else:
                    self.accelerator.print(
                    f"Step {self.global_step}, critic_loss: {critic_loss.item()}, Grad Norm: {fake_score_grad_norm.item()}"
                    )
                    wandb_loss_dict.update(
                        {
                            "critic_loss": critic_loss.item(),
                            "critic_grad_norm": fake_score_grad_norm.item(),
                            "step": self.global_step,
                        }
                    )
                self.accelerator.log(wandb_loss_dict, step=self.global_step)
                pbar.update(1)
                # file_name = f"dmd_memory_step_{self.global_step}.pickle"
                # torch.cuda.memory._dump_snapshot(file_name)
                # self.accelerator.print(f"dump memory snapshot to {file_name}")
                # return # debug only run one step

            if self.accelerator.sync_gradients:
                if self.global_step % 200 == 0 and self.global_step > self.start_step:
                    save_directory = os.path.join(
                        self.config.logdir, f"checkpoint-step-{self.global_step}"
                    )
                    self.save_checkpoint(save_directory)
                    self.accelerator.wait_for_everyone()  # 等待所有进程完成
                self.global_step += 1 # 所有事情完成后再更新步骤

        save_directory = os.path.join(self.config.logdir, "final_checkpoint")
        self.save_checkpoint(save_directory)

        self.accelerator.print("Finished training.")
        self.accelerator.end_training()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/self_forcing_dmd.yaml",
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default="logs/dmd_debug",
        help="Path to the directory to save logs",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="",
        help="continue training from this checkpoint",
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
    model = dmd_model(
        config, device=torch.device(f"cuda:{torch.cuda.current_device()}")
    )
    # model.enable_gradient_checkpointing()
    generator_optimizer = torch.optim.AdamW(
        [param for param in model.generator.parameters() if param.requires_grad],
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
    )
    critic_optimizer = torch.optim.AdamW(
        [param for param in model.fake_score.parameters() if param.requires_grad],
        lr=config.lr_critic if hasattr(config, "lr_critic") else config.lr,
        betas=(config.beta1_critic, config.beta2_critic),
        weight_decay=config.weight_decay,
    )

    torch.manual_seed(config.seed)
    # dataset = VPData_Dataset(csv_path=args.csv_path, video_folder=args.video_folder, num_frame_per_block= config.num_frame_per_block)
    if args.ckpt_path:
        dataset = TextDataset(config.data_path,start_index=config.ckpt_step * config.total_batch_size)
    else:
        dataset = TextDataset(config.data_path)
    accelerator.print(
        f"[rank:{accelerator.process_index}] Dataset length: {len(dataset)}"
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
        model.fake_score,
        generator_optimizer,
        critic_optimizer,
        train_dataloader,
    ) = accelerator.prepare(
        model.generator,
        model.fake_score,
        generator_optimizer,
        critic_optimizer,
        train_dataloader,
    )
    model.text_encoder = fsdp_wrap(
        model.text_encoder,
        sharding_strategy=config.sharding_strategy,
        mixed_precision=config.mixed_precision,
        wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
    )
    model.real_score = fsdp_wrap(
        model.real_score,
        sharding_strategy=config.sharding_strategy,
        mixed_precision=config.mixed_precision,
        wrap_strategy=config.real_score_fsdp_wrap_strategy,
    )
    torch.cuda.empty_cache()
    trainer = Trainer(
        model=model,
        generator_optimizer=generator_optimizer,
        critic_optimizer=critic_optimizer,
        train_dataloader=train_dataloader,
        accelerator=accelerator,
        config=config,
    )
    trainer.train(max_train_steps=config.max_train_steps)


if __name__ == "__main__":
    main()
