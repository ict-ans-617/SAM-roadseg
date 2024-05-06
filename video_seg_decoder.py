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

from test_seg_decoder import show_anns, show_anns_2, preprocess, sam_model_checkpoints

from offroad_roadseg.common import visualize

from offroad_roadseg.video_utils import open_video_capture, read_video_frame

def main():
    model_size = 'l'

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

    video_capture = open_video_capture()


    transform = ResizeLongestSide(1024)


    while True:
        ret, frame = read_video_frame(video_capture)
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        image = frame[210:730, 280:1800]
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

        cv2.imshow("frame", image)
        if cv2.waitKey(1) == ord('q'):
            break



if __name__ == '__main__':
    main()
