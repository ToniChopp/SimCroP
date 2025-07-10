# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
import time
import numpy as np
import torch
#
#

cpu_num = 1
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)
torch.multiprocessing.set_sharing_strategy('file_system')


import nibabel as nib
import SimpleITK as sitk
from torch.cuda.amp import GradScaler, autocast
from utils.cache_utils import get_loader
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import Activations, AsDiscrete, Compose
from monai.utils.enums import MetricReduction
from functools import partial
from monai.data import decollate_batch
import scipy.ndimage as ndimage
import time
from medpy import metric
from monai.networks.nets.unetr import UNETR as UNETR_p16

def get_args(args):

    if args.task == "LUNA16":
        args.data_dir = "/data2/fenghetang/data/Dataset009_AMOS/"
        args.out_channels = 16
        args.pretrained_dir = args.model + "_" + args.name + "_Dataset009_AMOS" 
    elif args.task == "BTCV":
        args.data_dir = "/data2/fenghetang/data/Dataset001_BTCV/"
        args.out_channels = 14
        args.pretrained_dir = "{}_{}_{}".format(args.model, args.name, args.data_dir.split("/")[-2])    
    
    args.json_list = "dataset.json"
    
    print(args.pretrained_dir)

    return args




parser = argparse.ArgumentParser(description="inference pipeline")
parser.add_argument(
    "--pretrained_dir",
    default="/data/fenghetang/mae/downsteam/runs/+_amos/",
    type=str,
)
parser.add_argument(
    "--name",
    default="full",
    choices=["r1", "r10", "full", "os", "os1", "os2", "os3", "full_fix"],
)
parser.add_argument(
    "--model",
    default="",
    choices=["UNETR"],
)
parser.add_argument(
    "--save",
    default=True,
    type=bool,
)
parser.add_argument("--data", default="amos", choices=["btcv"], help="dataset")

parser.add_argument("--data_dir", default="/data2/fenghetang/data/Dataset009_AMOS/", type=str, help="dataset directory")
parser.add_argument("--json_list", default="dataset.json", type=str, help="dataset json file")
parser.add_argument(
    "--pretrained_model_name", default="model.pt", type=str, help="pretrained model name"
)
parser.add_argument("--mlp_dim", default=1536 * 4, type=int, help="mlp dimention in ViT encoder")
parser.add_argument("--hidden_size", default=1536, type=int, help="hidden size dimention in ViT encoder")
parser.add_argument("--feature_size", default=32, type=int, help="feature size dimention")
parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=16, type=int, help="number of output channels")
parser.add_argument("--num_heads", default=16, type=int, help="number of attention heads in ViT encoder")
parser.add_argument("--res_block", action="store_true", help="use residual blocks")
parser.add_argument("--conv_block", action="store_true", help="use conv blocks")
parser.add_argument("--a_min", default=-1000.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=1000.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=-1.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.0, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.0, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=1.0, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=224, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=224, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=112, type=int, help="roi size in z direction")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--pos_embed", default="conv", type=str, help="type of position embedding")
parser.add_argument("--norm_name", default="instance", type=str, help="normalization layer type in decoder")
parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")
parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")





def dice(x, y):
    tmp = time.time()
    intersect = torch.sum(torch.sum(torch.sum(x * y)))
    y_sum = torch.sum(torch.sum(torch.sum(y)))
    #print("dice cal time:", time.time()-tmp)
    if y_sum == 0:
        return 0.0
    x_sum = torch.sum(torch.sum(torch.sum(x)))
    return 2 * intersect / (x_sum + y_sum)


def resample_3d(img, target_size):
    imx, imy, imz = img.shape
    tx, ty, tz = target_size
    zoom_ratio = (float(tx) / float(imx), float(ty) / float(imy), float(tz) / float(imz))
    img_resampled = ndimage.zoom(img, zoom_ratio, order=0, prefilter=False)
    return img_resampled


def calculate_metric_percase(_pred, _gt):
    pred = np.zeros_like(_pred)
    gt = np.zeros_like(_gt)
    pred[_pred] = 1
    gt[_gt] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        asd = metric.binary.asd(pred, gt)
        return dice, hd95, asd
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1, 0, 0
    else:
        return 0, 0, 0


def val_epoch(model, loader, epoch, acc_func, args, model_inferer=None, post_label=None, post_pred=None):
    model.eval()
    start_time = time.time()
    val_dice = []
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data["image"], batch_data["label"]
            data, target = data.cuda(args.rank), target.cuda(args.rank)
            with autocast(enabled=args.amp):
                if model_inferer is not None:
                    logits = model_inferer(data)
                else:
                    logits = model(data)
            if not logits.is_cuda:
                target = target.cpu()
            val_labels_list = decollate_batch(target)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            acc = acc_func(y_pred=val_output_convert, y=val_labels_convert)
            acc = acc.cuda(args.rank)
            acc_list = acc.detach().cpu().numpy()
            avg_acc = np.mean([np.nanmean(l) for l in acc_list])

            print(
                "Val {}/{} {}/{}".format(epoch, 0, idx, len(loader)),
                "acc",
                avg_acc,
                "time {:.2f}s".format(time.time() - start_time),
            )
            val_dice.append(avg_acc)
            start_time = time.time()
    return np.mean(val_dice)


def cal_dice(val_outputs, val_labels):
    val_outputs = torch.softmax(val_outputs, 1).cpu().numpy()
    val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)
    val_labels = val_labels.cpu().numpy()[:, 0, :, :, :]
    dice_list_sub = []
    for i in range(1, args.out_channels):
        organ_Dice = dice(val_outputs[0] == i, val_labels[0] == i)
        dice_list_sub.append(organ_Dice)
    mean_dice = np.mean(dice_list_sub)
    return mean_dice


def save_pred(val_outputs, original_affine, target_shape, path):
    #val_outputs = torch.softmax(val_outputs, 1).cpu().numpy()
    val_outputs = val_outputs[0].cpu().numpy()
    val_outputs = resample_3d(val_outputs, target_shape)
    nib.save(nib.Nifti1Image(val_outputs.astype(np.uint8), original_affine), path)


def save_img(val_outputs, original_affine, target_shape, path):
    val_outputs = resample_3d(val_outputs[0][0].cpu().numpy(), target_shape)
    nib.save(nib.Nifti1Image(val_outputs, original_affine), path)


def bulid_model(args, name, pretrained_pth):


    if args.model == "MAE":
        model = UNETR_p16(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            feature_size=args.feature_size,
            hidden_size=args.hidden_size,
            mlp_dim=args.mlp_dim,
            num_heads=args.num_heads,
            pos_embed="perceptron",
            norm_name=args.norm_name,
            conv_block=True,
            res_block=True,
            dropout_rate=args.dropout_rate,
            qkv_bias=True
        ).cuda()

    print("Use pretrained weights {}".format(os.path.join(pretrained_pth)))
    model_dict = torch.load(pretrained_pth)['state_dict']
    model.load_state_dict(model_dict, strict=True)
    model.eval()
    return model


def main():
    args = get_args(parser.parse_args())
    #args.pretrained_dir = args.pretrained_dir.replace("+", args.name).replace("amos", args.data_dir.split("/")[-2])
    args.test_mode = True
    args.testing = False
    dice_list_case = []
    dice_list_case_organ = []
    cal_dice_list_case_organ = []
    cal_hd95_list_case_organ = []
    cal_asd_list_case_organ = []
    args.amp = not args.noamp

    post_label = AsDiscrete(to_onehot=args.out_channels, n_classes=args.out_channels)
    post_pred = AsDiscrete(argmax=True, to_onehot=args.out_channels, n_classes=args.out_channels)
    dice_acc = DiceMetric(include_background=True, reduction=MetricReduction.MEAN, get_not_nans=True)
    inf_size = [args.roi_x, args.roi_y, args.roi_z]


    val_loader = get_loader(args)
    name = args.name
    pretrained_dir = args.pretrained_dir
    
    model_name = args.pretrained_model_name
    pretrained_pth = os.path.join(os.path.join("./runs", args.pretrained_dir), model_name)
    model = bulid_model(args=args, name=name, pretrained_pth=pretrained_pth)

    model_inferer = partial(
        sliding_window_inference,
        roi_size=inf_size,
        sw_batch_size=1,
        predictor=model,
        overlap=args.infer_overlap,
    )
    print("save {}".format(args.save))
    with torch.no_grad():
        total = len(val_loader)
        count = 0
        for i, batch in enumerate(val_loader):
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            h, w, d = batch["image_meta_dict"]["spatial_shape"][0]
            
            target_shape = (h, w, d)
            
            img_name = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-1]
            
            print(img_name, target_shape)
            
            original_affine = batch["image_meta_dict"]["affine"][0].numpy()
            
            tmp = time.time()
            # modify (96, 96, 96)
            val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model, overlap=args.infer_overlap)
            val_outputs = torch.softmax(val_outputs, 1)
            val_outputs = torch.argmax(val_outputs, dim=1).type(torch.uint8)
            
            #print("Inference on case {} time: {}".format(img_name, time.time()-tmp))
            if args.save:
                if not os.path.exists("./result/{}/{}/{}/".format(args.model, name, args.data_dir.split("/")[-2])):
                    os.makedirs("./result/{}/{}/{}/".format(args.model, name, args.data_dir.split("/")[-2]))
                if not os.path.exists("./result/{}/{}/{}/".format(args.model, name, args.data_dir.split("/")[-2]) + img_name):    
                    save_pred(val_outputs, original_affine, target_shape, "./result/{}/{}/{}/".format(args.model, name, args.data_dir.split("/")[-2]) + img_name)
            
            #tmp = time.time()
            print(val_labels.shape)
            val_labels = val_labels[:, 0, :, :, :]
            dice_list_sub = []
            dice_cal = []
            hd95_cal = []
            asd_cal = []
            #print("deal time: {}".format(time.time()-tmp))
            for i in range(1, args.out_channels):
                
                #tmp = time.time()

                organ_Dice = dice(val_outputs[0] == i, val_labels[0] == i)

                if isinstance(organ_Dice, torch.Tensor):
                    organ_Dice = organ_Dice.cpu().numpy()
                else:
                    organ_Dice = float(organ_Dice)

                hd95_cal.append(0)
                asd_cal.append(0)
                dice_cal.append(0)
                dice_list_sub.append(organ_Dice)
            mean_dice = np.mean(dice_list_sub)
            cal_hd95 = np.mean(hd95_cal)
            cal_asd = np.mean(asd_cal)
            cal_dice = np.mean(dice_cal)
            dice_list_case.append(mean_dice)
            cal_hd95_list_case_organ.append(cal_hd95)
            cal_asd_list_case_organ.append(cal_asd)
            cal_dice_list_case_organ.append(cal_dice)
            dice_list_case_organ.append(dice_list_sub)
            print("{}/{} time: {}".format(count, total, time.time()-tmp))
            count += 1
    print("Overall Mean Dice: {}, Organ Dice: {}".format(np.mean(dice_list_case), np.mean(dice_list_case_organ, axis=0).tolist()))
    print("Overall Mean Dice: {}, HD95: {}, ASD: {}".format(np.mean(cal_dice_list_case_organ),
                                                            np.mean(cal_hd95_list_case_organ, axis=0),
                                                            np.mean(cal_asd_list_case_organ, axis=0)))
    #with open('./log/inference/{}_{}_{}.txt'.format(args.model, args.name, args.data_dir.split("/")[-2]), 'w') as f:  # 设置文件对象
    #    f.write("Overall Mean Dice: {}, Organ Dice: {}".format(np.mean(dice_list_case), np.mean(dice_list_case_organ, axis=0).tolist()))


if __name__ == "__main__":
    main()
