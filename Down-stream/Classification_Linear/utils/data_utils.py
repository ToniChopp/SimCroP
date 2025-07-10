import logging
import numpy as np
import torch
import torch.distributed

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler

from utils.dist_util import get_world_size
from .dataset import CT

from monai.transforms.compose import Compose
from monai.transforms.utility.array import AddChannel, ToTensor
from monai.transforms.spatial.array import Orientation, Spacing, RandRotate90, RandFlip, RandRotate
from monai.transforms.croppad.array import RandSpatialCrop, SpatialPad, CenterSpatialCrop, RandScaleCrop
from monai.transforms.intensity.array import RandShiftIntensity, ScaleIntensityRange, RandScaleIntensity, RandGaussianNoise

logger = logging.getLogger(__name__)


def get_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    transform_train = Compose([
        AddChannel(),
        Orientation(axcodes="RAS",image_only=True),
        Spacing(
            pixdim=(args.spacing_x, args.spacing_y, args.spacing_z),
            mode="bilinear",
            image_only=True,
        ),
        ScaleIntensityRange(
            a_min=args.a_min,
            a_max=args.a_max,
            b_min=args.b_min,
            b_max=args.b_max,
            clip=True,
        ),
        RandScaleCrop(
            roi_scale=(args.roi_scale, args.roi_scale, args.roi_scale),
            max_roi_scale=(1.0, 1.0, 1.0),
            random_center=True,
            random_size=True,
        ),
        RandSpatialCrop(
            roi_size=(args.roi_x, args.roi_y, args.roi_z),
            random_size=False,
            random_center=True,
        ),
        SpatialPad(
            spatial_size=(args.roi_x, args.roi_y, args.roi_z),
            mode="reflect"
        ),
        RandFlip(
            prob=args.RandFlip_prob,
            spatial_axis=0,
        ),
        RandFlip(
            prob=args.RandFlip_prob,
            spatial_axis=1,
        ),
        RandFlip(
            prob=args.RandFlip_prob,
            spatial_axis=2,
        ),
        RandShiftIntensity(
            offsets=0.10,
            prob=args.RandShiftIntensity_prob,
        ),
        RandGaussianNoise(
            prob=args.RandGaussianNoise_prob,
        ),
        ToTensor()
    ])

    transform_test = Compose([
        AddChannel(),
        Orientation(axcodes="RAS",image_only=True),
        Spacing(
            pixdim=(args.spacing_x, args.spacing_y, args.spacing_z),
            mode="bilinear",
            image_only=True,
        ),
        ScaleIntensityRange(
            a_min=args.a_min,
            a_max=args.a_max,
            b_min=args.b_min,
            b_max=args.b_max,
            clip=True,
        ),
        CenterSpatialCrop(
            roi_size=(args.roi_x, args.roi_y, args.roi_z),
        ),
        SpatialPad(
            spatial_size=(args.roi_x, args.roi_y, args.roi_z),
            # mode="constant",
            # constant_values=-1,
            mode="reflect"
        ),
        ToTensor()
    ])


    if args.stage == "test":
        test_set = CT(root = args.dataset_path, data_volume = args.data_volume, task=args.task, split="test", transform= transform_test)
        print("test dataset:",len(test_set))
        if args.local_rank == 0:
            torch.distributed.barrier()
        test_sampler = SequentialSampler(test_set)
        test_loader = DataLoader(test_set,
                            sampler=test_sampler,
                            batch_size=args.eval_batch_size//get_world_size(),
                            num_workers=4,
                            pin_memory=True) if test_set is not None else None

        return test_loader

    train_set = CT(root = args.dataset_path, data_volume = args.data_volume, task=args.task, split="train", transform= transform_train)
    val_set = CT(root = args.dataset_path, data_volume = args.data_volume, task=args.task, split="val", transform= transform_test)
    print("train_loader",len(train_set ))
    print("test_loader",len(val_set))
    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler(train_set) if args.local_rank == -1 else DistributedSampler(train_set)
    val_sampler = SequentialSampler(val_set)
    train_loader = DataLoader(train_set,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size//get_world_size(),
                              num_workers=8,
                              pin_memory=True)
    val_loader = DataLoader(val_set,
                             sampler=val_sampler,
                             batch_size=args.eval_batch_size//get_world_size(),
                             num_workers=8,
                             pin_memory=True) if val_set is not None else None

    return train_loader, val_loader
