import tqdm
import h5py
import clip
import torch
from torch.utils.data import DataLoader
from extract.dataloader_video import VideoCLIPDataset
import os
from tqdm import trange

import math
import urllib.request
import clip
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

def main():
    frame_num = 32
    dataset = VideoCLIPDataset(None, frame_num, "/home/thong/ay2324_projects/vidl_projects/youtube_data/videos_2/*.mp4")
    print(len(dataset))

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=16,
        shuffle=False
    )
    data_iter = iter(dataloader)

    clip_model = "ViT-B/32"
    model, preprocess = clip.load(clip_model, device="cuda", jit=False)

    dataset_feats = h5py.File("/home/thong/ay2324_projects/vidl_projects/mist/mist_data/feats/ytdata_2/frame_feat/clip_patch_feat_all.h5", 'w')
    dataset_feats.create_dataset("features", (len(dataset), 32, 17, 512))
    dataset_feats.create_dataset("ids", (len(dataset), ), 'S50')

    global_index = 0
    video_ids = {}
    data_iter = iter(dataloader)    

    for i in trange(len(dataset)):
        try:
            for j in trange(min(frame_num, dataset[i]['video'].shape[0])):
                with torch.no_grad():
                    image_features = model.encode_image(dataset[i]['video'][j].cuda())
                dataset_feats['features'][global_index, j] = image_features.detach().cpu().numpy()
            
            dataset_feats['ids'][global_index] = dataset[i]['vid'].split('/')[-1].encode('ascii', "ignore")
            global_index += 1
        except BaseException as error:
            continue
            
    dataset_feats.close()


if __name__ == '__main__':
    main()