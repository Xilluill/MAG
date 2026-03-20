import os
import numpy as np
import argparse
import math
from glob import glob
from skimage.metrics import structural_similarity as compare_ssim
import imageio
import lpips
import torch
from tqdm import tqdm
import logging

# Define metric functions


def compute_psnr(img1, img2):
    mse = np.mean((img1 / 255.0 - img2 / 255.0) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def compute_ssim(img1, img2):
    if np.all(img1 == img1[0, 0, 0]) or np.all(img2 == img2[0, 0, 0]):
        return 1.0
    return compare_ssim(img1, img2, data_range=img1.max() - img1.min(), channel_axis=-1)


def compute_lpips(img1, img2, loss_fn):
    img1_tensor = (
        torch.from_numpy(img1 / 255.0)
        .float()
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to("cuda:0")
    )
    img2_tensor = (
        torch.from_numpy(img2 / 255.0)
        .float()
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to("cuda:0")
    )

    img1_tensor = img1_tensor * 2 - 1  # Normalize to [-1, 1]
    img2_tensor = img2_tensor * 2 - 1

    return loss_fn(img1_tensor, img2_tensor).item()

def compute_tensor_video_metric(video1:torch.tensor, video2:torch.tensor, loss_fn):
    # Compute metrics for two videos
    assert len(video1) == len(video2), "Videos must have the same number of frames."

    psnr_values = []
    ssim_values = []
    lpips_values = []

    for frame1, frame2 in zip(video1, video2):
        psnr = compute_psnr(frame1, frame2)
        ssim = compute_ssim(frame1, frame2)
        lpips = compute_lpips(frame1, frame2, loss_fn)

        psnr_values.append(psnr)
        ssim_values.append(ssim)
        lpips_values.append(lpips)

    return {
        "PSNR": sum(psnr_values) / len(psnr_values) if psnr_values else 0,
        "SSIM": sum(ssim_values) / len(ssim_values) if ssim_values else 0,
        "LPIPS": sum(lpips_values) / len(lpips_values) if lpips_values else 0,
    }

def read_video(file_path):
    try:
        video = imageio.get_reader(file_path)
        frames = [frame for frame in video]
        video.close()
        return frames
    except Exception as e:
        logging.error(f"Error reading video {file_path}: {e}")
        return []


def save_results(results, root1, root2, output_file="metrics.txt"):
    with open(output_file, "a") as f:
        f.write("\n")
        f.write(f"Root1: {root1}\n")
        f.write(f"Root2: {root2}\n")
        for metric, value in results.items():
            f.write(f"{metric}: {value}\n")
        f.write("\n")
    logging.info(f"Results saved to {output_file}")


def main():
    # Load video paths
    all_videos1 = sorted(glob(os.path.join(args.root1, "*mp4")))
    all_videos2 = sorted(glob(os.path.join(args.root2, "*mp4")))

    assert len(all_videos1) == len(
        all_videos2
    ), f"Number of files mismatch: {len(all_videos1)} in {args.root1}, {len(all_videos2)} in {args.root2}"

    # Metrics storage
    metric_psnr = []
    metric_ssim = []
    metric_lpips = []

    # Initialize LPIPS model if needed
    lpips_model = None
    if args.lpips:
        lpips_model = lpips.LPIPS(net="alex").to("cuda:0")
        logging.info("Initialized LPIPS model (AlexNet).")

    for vid1_path, vid2_path in tqdm(
        zip(all_videos1, all_videos2), total=len(all_videos1), desc="Processing videos"
    ):
        vid1_frames = read_video(vid1_path)
        vid2_frames = read_video(vid2_path)

        if not vid1_frames or not vid2_frames:
            logging.error(
                f"Skipping video pair due to read failure: {vid1_path}, {vid2_path}"
            )
            continue

        assert len(vid1_frames) == len(
            vid2_frames
        ), f"Frame count mismatch: {len(vid1_frames)} in {vid1_path}, {len(vid2_frames)} in {vid2_path}"

        # Process each pair of frames
        for f1, f2 in zip(vid1_frames, vid2_frames):
            if args.psnr:
                try:
                    psnr_value = compute_psnr(f1, f2)
                    metric_psnr.append(psnr_value)
                except Exception as e:
                    logging.error(f"Error computing PSNR for frames: {e}")

            if args.ssim:
                try:
                    ssim_value = compute_ssim(f1, f2)
                    metric_ssim.append(ssim_value)
                except Exception as e:
                    logging.error(f"Error computing SSIM for frames: {e}")

            if args.lpips:
                try:
                    lpips_value = compute_lpips(f1, f2, lpips_model)
                    metric_lpips.append(lpips_value)
                except Exception as e:
                    logging.error(f"Error computing LPIPS for frames: {e}")

    # Compute average metrics
    results = {}
    if args.psnr and metric_psnr:
        results["PSNR"] = sum(metric_psnr) / len(metric_psnr)
    if args.ssim and metric_ssim:
        results["SSIM"] = sum(metric_ssim) / len(metric_ssim)
    if args.lpips and metric_lpips:
        results["LPIPS"] = sum(metric_lpips) / len(metric_lpips)

    # Print and save results
    logging.info(f"Results: {results}")
    save_results(results, args.root1, args.root2)


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    # Argument parser
    parser = argparse.ArgumentParser(
        description="Compute PSNR, SSIM, and LPIPS for videos."
    )
    parser.add_argument(
        "--root1",
        "-r1",
        type=str,
        required=True,
        help="Directory for the first set of videos.",
    )
    parser.add_argument(
        "--root2",
        "-r2",
        type=str,
        required=True,
        help="Directory for the second set of videos.",
    )
    parser.add_argument("--ssim", action="store_true", default=False, help="Compute SSIM.")
    parser.add_argument("--psnr", action="store_true", default=False, help="Compute PSNR.")
    parser.add_argument(
        "--lpips", action="store_true", default=False, help="Compute LPIPS."
    )

    args = parser.parse_args()
    main()