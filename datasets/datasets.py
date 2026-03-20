import os
import math
import pandas as pd
import torch
import torchvision
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, prompt_path, extended_prompt_path=None, pt_path=None, start_index=0):
        with open(prompt_path, encoding="utf-8") as f:
            self.prompt_list = [line.rstrip() for line in f]

        if extended_prompt_path is not None:
            with open(extended_prompt_path, encoding="utf-8") as f:
                self.extended_prompt_list = [line.rstrip() for line in f]
            assert len(self.extended_prompt_list) == len(self.prompt_list)
        else:
            self.extended_prompt_list = None
        self.pt_path = pt_path
        self.start_index = start_index

    def __len__(self):
        return len(self.prompt_list) - self.start_index

    def __getitem__(self, idx):
        idx = idx + self.start_index
        batch = {
            "prompts": self.prompt_list[idx],
            "idx": idx,
        }
        if self.extended_prompt_list is not None:
            batch["extended_prompts"] = self.extended_prompt_list[idx]
        if self.pt_path is not None:
            input_pt = torch.load(os.path.join(self.pt_path, self.prompt_list[idx][:200]+'.pt'), map_location='cpu')
            batch["input_pt"] = input_pt.squeeze(0)
        return batch


class VPData_Dataset(Dataset):
    """Load pt file where pt file is preprocessed by vae & t5."""

    def __init__(self, csv_path, video_folder, num_frame_per_block, start_index=0, min_len=6):
        self.df = pd.read_csv(csv_path)
        print(f"Initial dataset size from {csv_path}: {len(self.df)}")
        self.video_folder = video_folder
        self.num_frame_per_block = num_frame_per_block
        self.exist_file = set(os.listdir(video_folder))
        print(f"Total existing pt files in {video_folder}: {len(self.exist_file)}")

        self.df = self.df[self.df["path"].apply(lambda x: os.path.splitext(x)[0] + ".pt" in self.exist_file)]
        self.df = self.df.reset_index(drop=True)
        print(f"Filtered dataset size after checking existing pt files: {len(self.df)}")
        self.start_index = start_index
        self.min_len = min_len

    def __getitem__(self, index):
        index = index + self.start_index
        try:
            return self.getitem(index)
        except Exception as e:
            row = self.df.loc[index]
            video_name = row["path"]
            pt_name = os.path.splitext(video_name)[0] + ".pt"
            print(f"Error processing index {index}, file {pt_name}: {e}")
            return None, None, None

    def getitem(self, index):
        try:
            row = self.df.loc[index]
            video_name = row["path"]
            pt_name = os.path.splitext(video_name)[0] + ".pt"
            pt_file_path = os.path.join(self.video_folder, pt_name)
            data = torch.load(pt_file_path, map_location="cpu")
            assert data != None, f"Error loading {pt_file_path}"
            video_features = data["video_features"].squeeze(0).to(torch.bfloat16)
            T, C, H, W = video_features.shape
            if T % self.num_frame_per_block != 0:
                new_T = (T // self.num_frame_per_block) * self.num_frame_per_block
                video_features = video_features[:new_T, ...]
            max_frame_len = 76 // self.num_frame_per_block * self.num_frame_per_block
            if T > max_frame_len:
                video_features = video_features[:max_frame_len, ...]
            T, C, H, W = video_features.shape
            assert T > self.min_len and C != 0 and H != 0 and W != 0, f"Invalid video features shape {video_features.shape} in {pt_file_path}"
            prompt_embeds = data["prompt_embeds"].squeeze(0).to(torch.bfloat16)
            return video_name, video_features, prompt_embeds
        except Exception as e:
            row = self.df.loc[index]
            video_name = row["path"]
            pt_name = os.path.splitext(video_name)[0] + ".pt"
            print(f"Error processing index {index}, file {pt_name}: {e}")
            return None, None, None

    def collate_fn(self, batch):
        video_names = [x[0] for x in batch if x[0] is not None]
        video_features = [x[1] for x in batch if x[1] is not None]
        prompt_embeds = [x[2] for x in batch if x[2] is not None]
        return video_names, video_features, prompt_embeds

    def __len__(self):
        return len(self.df)


class video_bench_dataset(Dataset):
    """Load video according to the csv file and process."""

    def __init__(self, csv_path, video_folder="videos", target_height=480, target_width=832, num_frame_per_block=3):
        self.df = pd.read_csv(csv_path)
        self.video_folder = video_folder

        videos_path_set = self._collect_video_paths(video_folder)
        self.df = self.df[self.df["path"].apply(lambda x: x in videos_path_set)]
        self.df = self.df.reset_index(drop=True)

        self.target_height = target_height
        self.target_width = target_width
        self.num_frame_per_block = num_frame_per_block

    def _collect_video_paths(self, directory):
        video_files = []
        for root, _, files in os.walk(directory):
            for filename in files:
                if filename.lower().endswith('.mp4'):
                    file_path = os.path.join(root, filename)
                    relative_path = os.path.relpath(file_path, directory)
                    video_files.append(relative_path)
        return set(video_files)

    def __getitem__(self, index):
        row = self.df.loc[index]
        video_name = row["path"]
        caption = row["caption"]
        input_path = os.path.join(self.video_folder, video_name)
        vframes, aframes, info = torchvision.io.read_video(
            filename=input_path, pts_unit="sec", output_format="TCHW"
        )
        total_frames = len(vframes)
        output_frames = (total_frames + 3) // (4 * self.num_frame_per_block) * (4 * self.num_frame_per_block) - 3
        vframes = vframes[:output_frames, ...]
        video = vframes.float().div_(127.5).sub_(1.0)
        return {"name": video_name, "video": video, "caption": caption}

    def __len__(self):
        return len(self.df)


class mag_bench_dataset(Dataset):
    """Load video according to the csv file for MAG-Bench evaluation."""

    def __init__(self, csv_path, video_folder="videos", target_height=256, target_width=256):
        self.df = pd.read_csv(csv_path)
        self.video_folder = video_folder

        videos_path_set = self._collect_video_paths(video_folder)
        self.df = self.df[self.df["video_path"].apply(lambda x: x in videos_path_set)]
        self.df = self.df.reset_index(drop=True)

        self.target_height = target_height
        self.target_width = target_width

    def _collect_video_paths(self, directory):
        video_files = []
        for root, _, files in os.walk(directory):
            for filename in files:
                if filename.lower().endswith('.mp4'):
                    file_path = os.path.join(root, filename)
                    relative_path = os.path.relpath(file_path, directory)
                    video_files.append(relative_path)
        return set(video_files)

    def __getitem__(self, index):
        row = self.df.loc[index]
        video_name = row["video_path"]
        caption = row["raw_caption"]
        druation = float(row["duration(sec)"])
        switch_frame = math.floor(druation * 0.75 * 16.0)

        input_path = os.path.join(self.video_folder, video_name)
        vframes, aframes, info = torchvision.io.read_video(
            filename=input_path, pts_unit="sec", output_format="TCHW"
        )
        total_frames = len(vframes)
        output_frames = (total_frames + 3) // 12 * 12 - 3
        vframes = vframes[:output_frames, ...]
        video = vframes.float().div_(127.5).sub_(1.0)
        assert video.shape[0] > 9, f"video {video_name} has not enough frames {video.shape[0]}"

        switch_frame = math.ceil((switch_frame + 3) / 12) * 12 - 3
        assert switch_frame < video.shape[0], f"video {video_name} switch_frame {switch_frame} exceeds total frames {video.shape[0]}"
        return {"name": video_name, "video": video, "caption": caption, "switch_frame": switch_frame}

    def __len__(self):
        return len(self.df)
