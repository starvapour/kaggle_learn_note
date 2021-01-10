from albumentations import (Blur,Flip,ShiftScaleRotate,GridDistortion,ElasticTransform,HorizontalFlip,CenterCrop,RandomResizedCrop,
                            HueSaturationValue,Transpose,RandomBrightnessContrast,CLAHE,RandomCrop,Cutout,CoarseDropout,
                            CoarseDropout,Normalize,ToFloat,OneOf,Compose,Resize,RandomRain,RandomFog,Lambda
                            ,ChannelDropout,ISONoise,VerticalFlip,RandomGamma,RandomRotate90,RandomSizedCrop,ToGray,BboxParams,MotionBlur,MedianBlur)
from albumentations.pytorch import ToTensorV2
import cv2

input_length = 224

def get_train_transforms():
    return Compose(
        [RandomResizedCrop(input_length, input_length),
         #RandomCrop(224, 224),
         OneOf([
             RandomGamma(gamma_limit=(60, 120), p=0.9),
             RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.9),
             CLAHE(clip_limit=4.0, tile_grid_size=(4, 4), p=0.9),
         ]),
         OneOf([
             Blur(blur_limit=3, p=1),
             MotionBlur(blur_limit=3, p=1),
             MedianBlur(blur_limit=3, p=1)
         ], p=0.5),
         HorizontalFlip(p=0.5),
         VerticalFlip(p=0.5),
         HueSaturationValue(hue_shift_limit=0.2,sat_shift_limit=0.2,val_shift_limit=0.2,p=0.5),
         ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20,
                                         interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, p=1),
         Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
         CoarseDropout(p=0.5),
         ToTensorV2(p=1.0),]
    )


def get_test_transforms():
    return Compose(
        [Resize(input_length, input_length),
         Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
         ToTensorV2(p=1.0),]
    )