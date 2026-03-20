from einops import rearrange
import torch
import lpips

from pytorch_msssim import ssim

spatial = True
loss_fn = lpips.LPIPS(net='alex', spatial=spatial)

def trans(x):
    if x.shape[-3] == 1:
        x = x.repeat(1, 1, 3, 1, 1)
    x = x * 2 - 1
    return x

def calculate_lpips(video_recon, inputs, device):
    # input B T C H W
    loss_fn.to(device)
    # video_recon = trans(video_recon) # 要求-1到1？
    # inputs = trans(inputs)
    video_recon = rearrange(video_recon, "b t c h w -> (b t) c h w")
    inputs = rearrange(inputs, "b t c h w -> (b t) c h w")
    lpips_score = loss_fn.forward(inputs, video_recon).mean().detach().cpu().item()
    return lpips_score

def calculate_lpips_find_match(gt_video, inputs, device):
    # input B T C H W
    loss_fn.to(device)
    # video_recon = trans(video_recon) # 要求-1到1？
    # inputs = trans(inputs)
    gt_video = rearrange(gt_video, "b t c h w -> (b t) c h w")
    inputs = rearrange(inputs, "b t c h w -> (b t) c h w")
    frames_num = gt_video.shape[0]
    lpips_score_list = []
    for i in range(frames_num):
        lpips_score_single_frame = loss_fn.forward(inputs[i:i+1], gt_video).mean(dim=(1,2,3))
        lpips_score_list.append(lpips_score_single_frame)
    lpips_score_list = torch.stack(lpips_score_list, dim=0)  # [frames_num, 1]
    min_values, min_indices = torch.min(lpips_score_list, dim=1)
    lpips_score = min_values.mean().detach().cpu().item()
    return lpips_score,min_indices
import numpy as np
import torch
from tqdm import tqdm
from einops import rearrange

def calculate_psnr(video_recon, inputs, device=None):
    # 要求0-1，选maxI=2就是-1到1
    video_recon = rearrange(video_recon, "b t c h w -> (b t) c h w")
    inputs = rearrange(inputs, "b t c h w -> (b t) c h w")
    mse = torch.mean(torch.square(inputs - video_recon), dim=(1,2,3))
    psnr = 20 * torch.log10(1 / torch.sqrt(mse))
    psnr = psnr.mean()
    if psnr == torch.inf:
        return 100
    return psnr.item()


import numpy as np
import torch
 
# def ssim(img1, img2):
#     C1 = 0.01 ** 2
#     C2 = 0.03 ** 2
#     img1 = img1.astype(np.float64)
#     img2 = img2.astype(np.float64)
#     kernel = cv2.getGaussianKernel(11, 1.5)
#     window = np.outer(kernel, kernel.transpose())
#     mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
#     mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
#     mu1_sq = mu1 ** 2
#     mu2_sq = mu2 ** 2
#     mu1_mu2 = mu1 * mu2
#     sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
#     sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
#     sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
#     ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
#                                                             (sigma1_sq + sigma2_sq + C2))
#     return ssim_map.mean()
 
 
def calculate_ssim_function(img1, img2):
    # [0,1]
    # ssim is the only metric extremely sensitive to gray being compared to b/w 
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[0] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[i], img2[i]))
            return np.array(ssims).mean()                   
        elif img1.shape[0] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

# def trans(x):
#     return x.permute(0, 2, 1, 3, 4)

def calculate_ssim(videos1, videos2):
    # input B T C H W
    assert videos1.shape == videos2.shape

    videos1 = rearrange(videos1, "b t c h w -> (b t) c h w")
    videos2 = rearrange(videos2, "b t c h w -> (b t) c h w")
    ssim_val = ssim(videos1.float(), videos2.float(), data_range=1, size_average=True)
    return ssim_val.item()

# test code / using example

def main():
    NUMBER_OF_VIDEOS = 8
    VIDEO_LENGTH = 50
    CHANNEL = 3
    SIZE = 64
    videos1 = torch.zeros(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
    videos2 = torch.zeros(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
    device = torch.device("cuda")

    import json
    result = calculate_ssim(videos1, videos2)
    print(json.dumps(result, indent=4))

if __name__ == "__main__":
    main()