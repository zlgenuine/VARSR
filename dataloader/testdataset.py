import os
import glob
import torch
import random
import numpy as np
from PIL import Image
from functools import partial
from torch import nn
from torchvision import transforms
from torch.utils import data as data
import torch.nn.functional as F
from PIL import ImageFile
from transformers import AutoProcessor
ImageFile.LOAD_TRUNCATED_IMAGES = True
def exists(x):
    return x is not None

class TestDataset(data.Dataset):
    def __init__(self, 
                pngtxt_dir="/datasets_share_1/quyunpeng/trainset", 
                image_size=512,
                tokenizer=None,
                null_text_ratio=0.0,
                original_image_ratio = 0.0,
                center_crop=False,
                random_flip=True,
                resize_bak=True,
                convert_image_to="RGB",
        ):
        super(TestDataset, self).__init__()
        self.tokenizer = tokenizer
        self.resize_bak = resize_bak
        self.null_text_ratio = null_text_ratio
        self.original_image_ratio = original_image_ratio
        self.processor = AutoProcessor.from_pretrained("/home/quyunpeng/VAR/checkpoints/clip")

        self.img_preproc = transforms.Compose([
            transforms.ToTensor(),
        ])
        ram_mean = [0.485, 0.456, 0.406]
        ram_std = [0.229, 0.224, 0.225]
        self.ram_normalize = transforms.Normalize(mean=ram_mean, std=ram_std)
        self.img_paths = []
        self.img_paths.extend(sorted(glob.glob(f'{pngtxt_dir}/HR/*'))[:])

        print(len(self.img_paths))
        #self.img_paths = self.img_paths[0:4]

    def tokenize_caption(self, caption):            
        inputs = self.tokenizer(
            caption, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )

        return inputs.input_ids

    def __getitem__(self, index):
        example = dict()

        # load image
        img_path = self.img_paths[index]
        GT_image = Image.open(img_path).convert('RGB')
        scale = 1.0
        resolution = round(512 * scale)
        GT_image_t = self.img_preproc(GT_image.resize((resolution, resolution)))
        example["pixel_values"] = GT_image_t.squeeze(0) * 2.0 - 1.0
        example['path'] = img_path

        img_path = self.img_paths[index].replace("/HR/", "/LR/")
        LR_image_t = Image.open(img_path).convert('RGB')
        if LR_image_t.size[-1] != resolution:
            example["conditioning_pixel_values"] = self.img_preproc(LR_image_t.resize((resolution, resolution))).squeeze(0) * 2.0 - 1.0
        else:
            example["conditioning_pixel_values"] = self.img_preproc(LR_image_t).squeeze(0) * 2.0 - 1.0

        ram_values = F.interpolate(self.img_preproc(LR_image_t).unsqueeze(0), size=(round(384*scale),round(384*scale)), mode='bicubic')
        ram_values = ram_values.clamp(0.0, 1.0)
        example["ram_values"] = self.ram_normalize(ram_values.squeeze(0))


        txt_path = self.img_paths[index].replace("/HR/", "/highlevel_prompt_GT/").replace(".png", ".txt")
        try:
            fp = open(txt_path, "r")
            high_caption = fp.readlines()[0].lstrip()
            fp.close()
        except:
            try: 
                txt_path = self.img_paths[index].replace("/HR/", "/highlevel_prompt/").replace(".png", ".txt")
                fp.close()
                fp = open(txt_path, "r")
                high_caption = fp.readlines()[0].lstrip()
                fp.close()
            except:
                high_caption = ""
        if self.tokenizer is not None:
            example["highlevel_prompt"] = self.tokenize_caption(high_caption).squeeze(0)


        txt_path = self.img_paths[index].replace("/HR/", "/lowlevel_prompt_q/").replace(".png", ".txt")
        try:
            fp = open(txt_path, "r")
            caption = fp.readlines()[0].lstrip()
            fp.close()
        except:
            caption = ""
        if self.tokenizer is not None:
            example["lowlevel_prompt"] = self.tokenize_caption(caption).squeeze(0)
        
        
        txt_path = self.img_paths[index].replace("/HR/", "/label/").replace(".png", ".txt")
        fp = open(txt_path, "r")
        label = fp.readlines()[0].lstrip()
        example["label_B"] = 0#torch.tensor(int(label))
        fp.close()
    
        return example

    def __len__(self):
        return len(self.img_paths)