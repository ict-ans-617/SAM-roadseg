import cv2  # type: ignore

from segment_anything import sam_model_registry

import numpy as np
import torch
import glob
import time
import os
import matplotlib
import matplotlib.pyplot as plt
from typing import Any, Dict, List

from SAM_decoder import SegHead, SegHeadUpConv
from segment_anything.utils.transforms import ResizeLongestSide
from torch.nn import functional as F


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        # if ann['pred_class'] == 1:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        # img[m] = [0.25526778, 0.19120787, 0.67079563, 0.35]
        img[m] = color_mask
    ax.imshow(img)


def show_anns_2(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        if ann['pred_class'] >= 0.9:
            m = ann['segmentation']
            img[m] = [0.25526778, 0.19120787, 0.67079563, 0.35]

    ax.imshow(img)


def preprocess(x):
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    pixel_mean = [123.675, 116.28, 103.53]
    pixel_std = [58.395, 57.12, 57.375]
    pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
    pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1)
    x = (x - pixel_mean) / pixel_std

    # Pad
    h, w = x.shape[-2:]
    padh = 1024 - h
    padw = 1024 - w
    x = F.pad(x, (0, padw, 0, padh))
    return x

sam_model_checkpoints = {
    'h': "sam_vit_h_4b8939.pth",
    'l': "sam_vit_l_0b3195.pth",
    'b': "sam_vit_b_01ec64.pth",
}

from offroad_roadseg.common import visualize

def main():
    model_size = 'b'

    print("Loading model...")
    sam_model_path = f'./SAM-checkpoint/{sam_model_checkpoints[model_size]}'
    sam_model = sam_model_registry[f'vit_{model_size}'](checkpoint=sam_model_path)
    print(f"Loaded {sam_model_path = }")

    # device = torch.device("cuda:2")
    device = torch.device("cuda")

    sam_model.to(device)

    seg_decoder = SegHead(model_size=model_size)
    ckpt_path = f'./ckpts/sam_{model_size}_best_epoch.pth'
    seg_decoder.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    print(f"Loaded {ckpt_path = }")
    seg_decoder.eval()
    seg_decoder.to(device)

    # image_file = '/raid/yehongliang_data/ORFD_Dataset_ICRA2022/testing/x2021_0223_1756/image_data/'
    image_file = './camera/'
    # image_file = '/raid/yehongliang_data/huanbaoyuan_data_cam3_1/image_data/'

    # img_list = sorted(glob.glob(os.path.join(image_file, '*.jpg')))
    img_list = sorted(glob.glob(os.path.join(image_file, '*.png')))
    # save_path = '/raid/yehongliang_data/project_ckpt/SAM_roadseg/results/huanbaoyuan_seg_decoder_sam_l/'
    save_path = f'./camera_out_SAM_{model_size}/'
    os.makedirs(save_path, exist_ok=True)

    transform = ResizeLongestSide(1024)


    for t in img_list:
        print("Processing image:", t)
        img_name = os.path.split(t)[1]
        image = cv2.imread(t)
        # print("Image shape:", image.shape)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ori_size = rgb_image.shape[:2]

        start_time = time.time()
        input_image = transform.apply_image(rgb_image)
        input_size = input_image.shape[:2]

        input_image_torch = torch.as_tensor(input_image)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        input_image_torch = preprocess(input_image_torch)
        input_image_torch = input_image_torch.to(device)


        with torch.no_grad():
            start_encode_time = time.time()
            image_embedding = sam_model.image_encoder(input_image_torch)
            middle_time = time.time()
            encode_time = middle_time - start_encode_time
            pred_mask = seg_decoder(image_embedding)
            decode_time = time.time() - middle_time
            pred_mask = sam_model.postprocess_masks(pred_mask, input_size, ori_size)

        print(f"{encode_time = }")
        print(f"{decode_time = }")
        infer_time = time.time() - start_time
        print(f"{infer_time = }")
        # fps = 1 / infer_time
        # print(f"{fps = }")

        image = visualize(pred_mask=pred_mask, image=image)

        save_image_path = os.path.join(save_path, img_name)
        cv2.imwrite(save_image_path, image)
        print(f"Save to {save_image_path = }")



if __name__ == '__main__':
    main()