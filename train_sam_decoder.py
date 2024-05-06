import argparse
import datetime, time
import os
gpu_ids = '0'
# os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
print("gpu_ids: ", gpu_ids)
import cv2
import numpy as np
import random
import shutil

import torch

import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils import data

from datasets.ORFD_dataset_decoder import orfddataset
from segment_anything import sam_model_registry
from torch.nn.functional import threshold, normalize

from tool import Adjust_learning_rate, confusion_matrix, getScores

from SAM_decoder import SegHead, SegHeadUpConv


start = datetime.datetime.now()

EXP_NAME = 'st_exp_0002'
# EXP_NAME = 'exp_0909'
START_EPOCH = 0
GPU_IDS = None
BATCH_SIZE = 2
# DATA_ROOT = '/home/zjsys/data/ORFD_Dataset_ICRA2022/'
DATA_ROOT = '/home/zj/code/ORFD_Dataset_ICRA2022/'

IGNORE_LABEL = 255
INPUT_SIZE = '1280,720'
LEARNING_RATE = 1e-3
LR_POLICY = 'poly'
POWER = 0.9
RANDOM_SEED = 1234
CKPT_DIR = '/home/zj/code/kyxz_sam_roadseg/ckpts/'
# CKPT_DIR = '/home/zjsys/yehongliang/SAM_roadseg/ckpts/'


def get_arguments():
    parser = argparse.ArgumentParser(description="ORFD Network")
    parser.add_argument("--exp_name", type=str, default=EXP_NAME,
                        help="experiment name")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data-root", type=str, default=DATA_ROOT,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--num-classes", type=int, default=2,
                        help="Number of classes to predict (including background).")

    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--lr-policy", type=str, default=LR_POLICY,
                        help="which lr policy to choose, poly|lambda|cosine")
    parser.add_argument("--power", type=float, default=0.9,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument('--lr-decay-epochs', type=int, default=25,
                        help='multiply by a gamma every lr_decay_epoch epochs')
    parser.add_argument('--lr-gamma', type=float, default=0.9,
                        help='gamma factor for lr_scheduler')
    parser.add_argument('--warm-steps', type=float, default=1000,
                        help='warm steps for cosine policy')

    parser.add_argument("--ckpt-dir", type=str, default=CKPT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--gpu_ids", type=str, default=GPU_IDS,
                        help="choose gpu device.")
    parser.add_argument("--start-epoch", type=int, default=START_EPOCH,
                        help="choose the number of recurrence.")
    # parser.add_argument("--epochs", type=int, default=20,
    parser.add_argument("--epochs", type=int, default=2,
                        help="choose the number of recurrence.")

    return parser.parse_args()


args = get_arguments()


def main():
    save_ckpt_path = os.path.join(args.ckpt_dir, args.exp_name)
    if not os.path.exists(save_ckpt_path):
        os.makedirs(save_ckpt_path)

    w, h = map(int, args.input_size.split(','))
    original_image_size = (h, w)
    input_size = (576, 1024)

    cudnn.enabled = True
    # cudnn related setting
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True  # False
    torch.backends.cudnn.enabled = True
    torch.cuda.empty_cache()

    sam_model = sam_model_registry['default'](checkpoint='/home/zj/code/kyxz_sam_roadseg/SAM-checkpoint/sam_vit_h_4b8939.pth')
    # sam_model = sam_model_registry['default'](checkpoint='/home/zjsys/weights/sam/sam_vit_h_4b8939.pth')
    # sam_model = sam_model_registry['vit_l'](checkpoint='/raid/yehongliang_data/SAM_ckpts/sam_vit_l_0b3195.pth')

    device = torch.device("cuda:" + gpu_ids)
    # device = torch.device("cpu")

    sam_model.to(device)

    seg_decoder = SegHead()
    seg_decoder.train()
    seg_decoder.to(device)

    for name, parameter in sam_model.image_encoder.named_parameters():
        parameter.requires_grad = False

    for name, parameter in sam_model.prompt_encoder.named_parameters():
        parameter.requires_grad = False

    for name, parameter in sam_model.mask_decoder.named_parameters():
        parameter.requires_grad = False


    dataset_train = orfddataset(args.data_root, mode='train', target_size=sam_model.image_encoder.img_size)
    train_num_samples = len(dataset_train)
    print("train data num:", train_num_samples)
    train_dataloader = data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

    dataset_test = orfddataset(args.data_root, mode='test', target_size=sam_model.image_encoder.img_size)
    test_num_samples = len(dataset_test)
    print("test data num:", test_num_samples)
    test_dataloader = data.DataLoader(dataset_test, batch_size=1, shuffle=True, num_workers=4)

    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(device)

    lr_adjust = Adjust_learning_rate(args)

    optimizer = optim.AdamW(
        seg_decoder.parameters(),
        lr = args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2
    )

    total_iters = args.epochs * len(train_dataloader)
    total_iter_per_batch = len(train_dataloader)
    print("total iters:", total_iters)

    best_f_score = 0
    best_epoch = 0
    temp = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        seg_decoder.train()

        for i_iter, batch in enumerate(train_dataloader):
            iter_lr = i_iter + epoch * len(train_dataloader)
            lr = lr_adjust.adjust_lr(optimizer=optimizer, iter_lr=iter_lr, total_iters=total_iters, epoch=epoch)
            optimizer.zero_grad()

            image_batch = batch['rgb_image'].to(device)

            labels = batch['label'].to(device)

            with torch.no_grad():
                image_embedding = sam_model.image_encoder(image_batch)

            image_embedding1 = image_embedding[0]
            print("Shape of image_embedding:", image_embedding1.shape)
            pred_mask = seg_decoder(image_embedding)
            loss = criterion(pred_mask, labels)
            loss.requires_grad_(True)
            loss.backward()
            optimizer.step()

            if i_iter % 10 == 0:
                tim = time.time()
                print('epoch:{},iter:{}/{},loss:{:.3f},lr:{:.3e},time:{:.1f}'.
                      format(epoch, i_iter, total_iter_per_batch, loss.data.cpu().numpy(), lr, tim - temp))
                temp = tim

        if epoch % 1 == 0:
            print("----->Epoch:", epoch)
            seg_decoder.eval()

            print('valid processing')
            ckpt_epoch = os.path.join(save_ckpt_path, 'net_sam_decoder.pth')
            torch.save(seg_decoder.state_dict(), ckpt_epoch)

            conf_mat = np.zeros((args.num_classes, args.num_classes), dtype=np.float64)

            with torch.no_grad():
                for i_iter, batch in enumerate(test_dataloader):
                    image_batch = batch['rgb_image'].to(device)
                    labels = batch['label'].to(device)

                    image_embedding = sam_model.image_encoder(image_batch)
                    pred_mask = seg_decoder(image_embedding)
                    _, pred = torch.max(pred_mask.cpu(), 1)
                    pred = pred.float().detach().int().numpy()

                    gt = labels.cpu().numpy()

                    conf_mat += confusion_matrix(gt, pred, 2)

            globalacc, pre, recall, F_score, iou = getScores(conf_mat)

            print('glob acc:{0:.3f}, pre:{1:.3f}, recall:{2:.3f}, F_score:{3:.3f}, IoU:{4:.3f}'.format(globalacc, pre,
                                                                                                       recall, F_score,
                                                                                                       iou))
            
                # 将结果写入txt文件
            result_str = 'Epoch: {}, glob acc: {:.3f}, pre: {:.3f}, recall: {:.3f}, F_score: {:.3f}, IoU: {:.3f}\n'.format(
                epoch, globalacc, pre, recall, F_score, iou)
            with open('result_log.txt', 'a') as f:
                f.write(result_str)
            is_best_f_score = F_score > best_f_score
            best_f_score = max(F_score, best_f_score)

            if is_best_f_score:
                best_epoch = epoch
                print("Best F_score epoch: ", epoch)
                best_ckpt_epoch = os.path.join(save_ckpt_path, 'best_epoch.pth')
                shutil.copyfile(ckpt_epoch, best_ckpt_epoch)

            print("best epoch:", best_epoch)


    end = datetime.datetime.now()
    print(end - start, 'seconds')
    print(end)

if __name__ == '__main__':
    main()
