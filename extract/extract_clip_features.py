import tqdm
import h5py
import clip
import torch
from torch.utils.data import DataLoader
from dataloader_video import VideoCLIPDataset
import os

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
    dataset = VideoCLIPDataset(None, frame_num, "../nextqa/videos/*.mp4")
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

    dataset_feats = h5py.File("./mist_data/feats/nextqa/frame_feat/clip_patch_feat_all.h5", 'w')
    dataset_feats.create_dataset("features", (len(dataset), 32, 17, 512))
    dataset_feats.create_dataset("ids", (len(dataset), ), 'S20')

    global_index = 0
    video_ids = {}
    data_iter = iter(dataloader)
    for batch in tqdm.tqdm(data_iter):
        batch_size = batch['video'].shape[0]
        for i in range(batch_size):
            for j in range(frame_num):
                with torch.no_grad():
                    image_features = model.encode_image(batch['video'][i][j].cuda())
                dataset_feats['features'][global_index, j] = image_features.detach().cpu().numpy()
            dataset_feats['ids'][global_index] = batch['vid'][i].encode('ascii', "ignore")
            global_index += 1
    dataset_feats.close()


if __name__ == '__main__':
    main()