
import os
import torch
import torch.utils.data as data
import nibabel as nib
import numpy as np


class CT(data.Dataset):
    
    def __init__(self, root, data_volume, task, split="train", transform=None):
        super(CT, self)
        if data_volume == '1':
            train_label_data = "train_list_1.csv"
        if data_volume == '10':
            train_label_data = "train_list_10.csv"
        if data_volume == '100':
            train_label_data = "train_list.csv"
        test_label_data = "test_list.csv"
        val_label_data = "val_list.csv"
        
        self.split = split
        self.root = root
        self.transform = transform
        self.task = task
        self.listImagePaths = []
        self.listImageLabels = []
        
        if self.split == "train":
            data_label = train_label_data
        
        elif self.split == "val":
            data_label = val_label_data
                 
        elif self.split == "test":
            data_label = test_label_data
           
        #---- Open file, get image paths and labels
        
        fileDescriptor = open(os.path.join("./" + self.task, data_label), "r")
        
        #---- get into the loop
        line = True
        while line:
                
            line = fileDescriptor.readline()
            
            #--- if not empty
            if line and not str(line).startswith("VolumeName") and not str(line).startswith("NoteAcc_DEID") \
                and not str(line).startswith("im"):
          
                lineItems = line.split(",")
                imagePath = lineItems[0]
                imageLabel = lineItems[1:]
                imageLabel = [int(i) for i in imageLabel]
                
                self.listImagePaths.append(imagePath)
                self.listImageLabels.append(imageLabel)   
            
        # ipdb.set_trace()
        fileDescriptor.close()

    def __getitem__(self, index):
        
        imagePath = self.listImagePaths[index]
        try:
            if self.task == "CT-Rate":
                paths = imagePath.split("_")
                image_path = paths[0] + "_" + paths[1] + "/"  + paths[0] + "_" + paths[1] + paths[2] + "/" + imagePath
                imagePath = os.path.join(self.root + paths[0] + "_preprocessed", image_path)
                imageData = nib.load(imagePath)
                imageData = np.asanyarray(imageData.dataobj)
            elif self.task == "CC-CCII":
                imagePath = os.path.join(self.root + "images_resized", imagePath)
                imageData = nib.load(imagePath)
                imageData = np.asanyarray(imageData.dataobj)
                imageData = imageData.transpose(1, 2, 0)
            elif self.task == "RadChestCT":
                imagePath = os.path.join(self.root + "images", imagePath + ".nii.gz")
                imageData = nib.load(imagePath)
                imageData = np.asanyarray(imageData.dataobj)
            elif self.task == "LUNA16":
                imagePath = os.path.join(self.root + "images", imagePath)
                imageData = nib.load(imagePath)
                imageData = np.asanyarray(imageData.dataobj)
        except:
            print(imagePath)
            raise ValueError("Image not found")

        imageLabel= torch.FloatTensor(self.listImageLabels[index])
        
        if self.transform != None: imageData = self.transform(imageData)

        # img_data = np.array(imageData[0])
        # affine = np.array([[1.5, 0. , 0. , 0. ],
        #     [0. , 1.5, 0. , 0. ],
        #     [0. , 0. , 3., 0. ],
        #     [0. , 0. , 0. , 1. ]])
        # nib.save(nib.Nifti1Image(img_data, affine), "test.nii.gz")
         
        return imageData, imageLabel

    def __len__(self):
        
        return len(self.listImagePaths)

        
if __name__ == "__main__":
    from monai.transforms.compose import Compose
    from monai.transforms.utility.array import AddChannel, ToTensor
    from monai.transforms.spatial.array import Orientation, Spacing, RandRotate90, RandFlip, RandRotate
    from monai.transforms.croppad.array import RandSpatialCrop, SpatialPad, CenterSpatialCrop, RandScaleCrop
    from monai.transforms.intensity.array import RandShiftIntensity, ScaleIntensityRange, RandScaleIntensity
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['vit_tiny_patch16', 'vit_base_patch16', 'vit_large_patch16', 'vit_large_patch32'],
                        default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    # Required parameters

    parser.add_argument("--stage", type=str, default="train", help="train or test?")
    
    parser.add_argument("--task", choices=["CT-Rate", "RadChestCT", "CC-CCII", "LUNA16"],
                        default="CT-Rate",
                        help="Which finetune task to take.")
    parser.add_argument("--num_classes",default = 14, type=int, help="the number of class")
    parser.add_argument("--pretrained_path", type=str, default="checkpoint/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")

    parser.add_argument("--spacing_x", type=float, default=1.0, help="spacing in x direction")
    parser.add_argument("--spacing_y", type=float, default=1.0, help="spacing in y direction")
    parser.add_argument("--spacing_z", type=float, default=1.0, help="spacing in z direction")
    parser.add_argument("--a_min", type=float, default=-1000.0, help="minimum value of intensity")
    parser.add_argument("--a_max", type=float, default=1000.0, help="maximum value of intensity")
    parser.add_argument("--b_min", type=float, default=-1.0, help="minimum value of intensity after ScaleIntensityRange")
    parser.add_argument("--b_max", type=float, default=1.0, help="maximum value of intensity after ScaleIntensityRange")
    parser.add_argument("--roi_x", type=int, default=224, help="roi size in x direction")
    parser.add_argument("--roi_y", type=int, default=224, help="roi size in y direction")
    parser.add_argument("--roi_z", type=int, default=112, help="roi size in z direction")
    parser.add_argument("--RandFlip_prob", type=float, default=0.2, help="probability of RandFlip")
    parser.add_argument("--RandRotate_prob", type=float, default=0.8, help="probability of RandRotate 180°")
    parser.add_argument("--RandScaleIntensity_prob", type=float, default=0.1, help="probability of RandScaleIntensity")
    parser.add_argument("--RandShiftIntensity_prob", type=float, default=0.1, help="probability of RandShiftIntensity")
    args = parser.parse_args()

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
            roi_scale=(0.8, 0.8, 0.8),
            max_roi_scale=(1.2, 1.2, 1.2),
            random_center=True,
            random_size=True,
        ),
        RandSpatialCrop(
            roi_size=(args.roi_x, args.roi_y, args.roi_z),
            random_size=False,
            random_center=False,
        ),
        SpatialPad(
            spatial_size=(args.roi_x, args.roi_y, args.roi_z),
            # mode="constant",
            # constant_values=-1,
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
        RandRotate(
            range_x=(np.pi, np.pi),
            range_y=(np.pi, np.pi),
            range_z=(np.pi, np.pi),
            prob=args.RandRotate_prob,
        ),
        # RandScaleIntensity(
        #     factors=0.10,
        #     prob=0.10,
        # ),
        RandShiftIntensity(
            offsets=0.10,
            prob=args.RandShiftIntensity_prob,
        ),
        ToTensor()
    ])
    
    ds = CT(
        root = "../../../../Data/CT-Rate/",
        data_volume="1",
        task="CT-Rate",
        split="train",
        transform=transform_train
    )

    print(len(ds))
    print(ds[0])