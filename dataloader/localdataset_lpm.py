import os
import glob
import torch
import random
import numpy as np
from PIL import Image
from functools import partial

from torch import nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode, transforms
from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS
from torch.utils import data as data
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union
from .realesrgan import RealESRGAN_degradation
from myutils.img_util import convert_image_to_fn
import pickle
from transformers import AutoProcessor
import torch.nn.functional as F
from test_var import pt_to_numpy, numpy_to_pil
import pandas as pd
import scipy.io
def exists(x):
    return x is not None


def has_file_allowed_extension(filename: str, extensions: Union[str, Tuple[str, ...]]) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions if isinstance(extensions, str) else tuple(extensions))


class LocalImageDataset_LPM(data.Dataset):
    def __init__(self, 
                pngtxt_dir="/datasets/6393/3658/datasets/sr_datasets/train_pasd_datasets/pngtxt_dir/", 
                image_size=512,
                tokenizer=None,
                accelerator=None,
                control_type=None,
                null_text_ratio=0.0,
                center_crop=False,
                random_flip=True,
                resize_bak=True,
                convert_image_to="RGB",
        ):
        super(LocalImageDataset_LPM, self).__init__()
        self.tokenizer = tokenizer
        self.control_type = control_type
        self.resize_bak = resize_bak
        self.null_text_ratio = null_text_ratio

        self.degradation = RealESRGAN_degradation('/home/quyunpeng/hart/dataloader/params_realesrgan.yml', device='cpu')
        self.resize_scale = 1.25
        center_crop = True

        self.crop_preproc = transforms.Compose([
            transforms.Resize(round(self.resize_scale*image_size), interpolation=InterpolationMode.LANCZOS),
            transforms.CenterCrop(image_size) if center_crop else transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
        ])
        self.crop_preproc_1024 = transforms.Compose([
            transforms.Resize(round(2.0*image_size), interpolation=InterpolationMode.LANCZOS),
            transforms.CenterCrop(image_size) if center_crop else transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
        ])
        self.neg_resize_preproc = transforms.Compose([
            transforms.Resize(image_size, interpolation=InterpolationMode.LANCZOS),
            transforms.CenterCrop(image_size) if center_crop else transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
        ])
        self.neg_crop_preproc = transforms.Compose([
            transforms.CenterCrop(image_size) if center_crop else transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
        ])
        ram_mean = [0.485, 0.456, 0.406]
        ram_std = [0.229, 0.224, 0.225]
        self.ram_normalize = transforms.Normalize(mean=ram_mean, std=ram_std)

        self.img_preproc = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.toPIL = transforms.ToPILImage()
        self.processor = AutoProcessor.from_pretrained("/home/quyunpeng/VAR/checkpoints/clip")

        self.img_paths = []
        self.neg_paths = []
        hq_scale = 0.8
        lq_scale = 0.6
        lq_patch_scale = 0.5
        lq_spaq_scale = 0.4
        lq_ava_scale = 0.32
        
        file_path = '/datasets/6393/3568/datasets/QA/IQAs/spaq/annotations.xlsx'
        df = pd.read_excel(file_path, skiprows=1)
        score = df.values[:,1].min() + df.values[:,1].max()*lq_spaq_scale
        filtered_df = df.values[df.values[:, 1] < score][:, 0]  # 筛选第二列大于 0.4 的行
        filtered_df = file_path.replace("annotations.xlsx", 'imgs/') + filtered_df
        self.neg_paths.extend(filtered_df.tolist())
        print(len(self.neg_paths))

        file_path  = '/datasets/6393/3568/datasets/QA/IQAs/koniq10k/koniq10k_scores_and_distributions.csv'
        df = pd.read_csv(file_path, skiprows=1)
        score = df.values[:,-1].min() + df.values[:,-1].max()*lq_scale
        filtered_df = df.values[df.values[:, -1] < score][:, 0]  # 筛选第二列大于 0.4 的行
        filtered_df = file_path.replace("koniq10k_scores_and_distributions.csv", 'imgs/') + filtered_df
        self.neg_paths.extend(filtered_df.tolist())
        print(len(self.neg_paths))

        file_path  = '/datasets/6393/3568/datasets/QA/IQAs/clive/AllMOS_release.mat'
        mos_mat = scipy.io.loadmat(file_path)['AllMOS_release']
        file_path  = '/datasets/6393/3568/datasets/QA/IQAs/clive/AllImages_release.mat'
        image_mat = scipy.io.loadmat(file_path)['AllImages_release']
        df = np.concatenate((image_mat, mos_mat.transpose()), axis = 1)
        score = mos_mat.min() + mos_mat.max()*lq_scale
        filtered_df = df[df[:,-1] < score][:, 0]  # 筛选第二列大于 0.4 的行
        filtered_df =  [file_path.replace("AllImages_release.mat", 'imgs/') + item.tolist()[0] for item in filtered_df]
        self.neg_paths.extend(filtered_df)
        print(len(self.neg_paths))

        file_path  = '/datasets/6393/3568/datasets/QA/IQAs/kadid-10k/dmos.csv'
        df = pd.read_csv(file_path, skiprows=1)
        score = df.values[:,2].min() + df.values[:, 2].max()*lq_scale
        filtered_df = df.values[df.values[:, 2] < score][:, 0]  # 筛选第二列大于 0.4 的行
        filtered_df = file_path.replace("dmos.csv", 'images/') + filtered_df
        self.neg_paths.extend(filtered_df.tolist())
        print(len(self.neg_paths))

        file_path  = '/datasets_share/xieqizhi/FLIVE_Database/database/labels_image.csv'
        df = pd.read_csv(file_path, skiprows=1)
        score = df.values[:,-1].min() + df.values[:, -1].max()*lq_scale
        filtered_df = df.values[df.values[:, -1] < score][:, 0]  # 筛选第二列大于 0.4 的行
        filtered_df = file_path.replace("labels_image.csv", '/') + filtered_df
        self.neg_paths.extend(filtered_df.tolist())
        print(len(self.neg_paths))

        file_path  = '/datasets_share/xieqizhi/FLIVE_Database/database/labels_patch.csv'
        df = pd.read_csv(file_path, skiprows=1)
        score = df.values[:,1].min() + df.values[:, 1].max()*lq_patch_scale
        filtered_df = df.values[df.values[:, -1] < score][:, 0]  # 筛选第二列大于 0.4 的行
        filtered_df = file_path.replace("labels_patch.csv", '/') + filtered_df
        self.neg_paths.extend(filtered_df.tolist())
        print(len(self.neg_paths))

        file_path  = '/home/quyunpeng/VAR/dataloader/meta_info_AVADataset.csv'
        df = pd.read_csv(file_path, skiprows=1)
        score = df.values[:,1].min() + df.values[:, 1].max()*lq_ava_scale
        filtered_df = df.values[df.values[:, 1] < score][:, 0]  # 筛选第二列大于 0.4 的行
        filtered_df = '/share/xupengcheng/aesthetics_datasets/AVA/AVA_dataset/image/' + filtered_df
        self.neg_paths.extend(filtered_df.tolist())
        delete_path = []
        for path in self.neg_paths:
            if os.path.exists(path)==False:
                delete_path.append(path)
        for path in delete_path:
            self.neg_paths.remove(path)
            print(path)
        print(len(self.neg_paths))
    
        file_path = '/home/lpm/image_attrs/balanced/1000w-prior100w-k4000-per1200.pkl'
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        num_classes = len(data)
        idx = 0
        extensions = IMG_EXTENSIONS 
        if extensions is not None:
            def is_valid_file(x: str) -> bool:
                return has_file_allowed_extension(x, extensions)  # type: ignore[arg-type]
        is_valid_file = cast(Callable[[str], bool], is_valid_file)
        for target_class in range(num_classes):
            while idx not in data:
                idx = idx + 1
            if idx in data:                    
                data_class = data[idx][:]
                for path in data_class:
                    if is_valid_file(path):
                        item = path #, target_class
                        self.img_paths.append(item)
                idx = idx + 1
        print(len(self.img_paths))


        pngtxt_dir = "/home/zhaokai05/datasets/sr_datasets/train_pasd_datasets/pngtxt_dir/"
        data_folders = os.listdir(pngtxt_dir)
        for data_folder in data_folders:
            self.img_paths.extend(sorted(glob.glob(f'{pngtxt_dir}/{data_folder}/*.png'))[:])
        print(len(self.img_paths))


        self.labels = torch.zeros(len(self.img_paths))
        self.neg_labels = torch.ones(len(self.neg_paths))
        self.img_labels = torch.cat((self.labels, self.neg_labels), dim=0).tolist()
        self.img_paths.extend(self.neg_paths)
        print(len(self.img_paths))


    def tokenize_caption(self, caption):
        if random.random() < self.null_text_ratio:
            caption = ""
            
        inputs = self.tokenizer(
            caption, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )

        return inputs.input_ids

    def __getitem__(self, index):
        example = dict()

        # load image
        img_path = self.img_paths[index]
        label_B = self.img_labels[index]
        txt_path = img_path.replace(".png", ".txt")
        image = Image.open(img_path).convert('RGB')

        if label_B == 0:
            image = self.crop_preproc(image)
        elif label_B==1:
            image = self.neg_resize_preproc(image)
        GT_image_t, LR_image_t = self.degradation.degrade_process(np.asarray(image)/255., resize_bak=self.resize_bak)
        example["conditioning_pixel_values"] = LR_image_t.squeeze(0) * 2.0 - 1.0
        example["pixel_values"] = GT_image_t.squeeze(0) * 2.0 - 1.0

        ram_values = F.interpolate(LR_image_t, size=(384, 384), mode='bicubic')
        ram_values = ram_values.clamp(0.0, 1.0)
        example["ram_values"] = self.ram_normalize(ram_values.squeeze(0))

        example["label_B"] = int(label_B)
        example['img_path'] = img_path
        # if label_B == 0:
        #     if torch.rand(1) < 0.02:
        #         example["label_B"] = int(1)
        #         example["pixel_values"] = example["conditioning_pixel_values"]

        return example

    def __len__(self):
        return len(self.img_paths)