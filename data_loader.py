import pandas as pd
import os
import torch
import numpy as np
from PIL import Image
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.io import imread
from torch.autograd import Variable
from torch.utils.data import Dataset
from torchvision import transforms

# Each batch will have 3 things : true image, its captions(5), and false image(real image but image
# corresponding to an incorrect caption).
# Discriminator is trained in such a way tt true_img + caption corresponds to a real example and
# false_img + caption corresponds to a fake example.


class Text2ImageDataset(Dataset):

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.img_files = pd.read_pickle('file_names.pkl')
        self.embeddings = pd.read_pickle('embed_pca.pkl')
        self.load_flower_dataset()

    def load_flower_dataset(self):
        # It will return two things : a list of image file names, a dictionary of 5 captions per image
        # with image file name as the key of the dictionary and 5 values(captions) for each key.

        self.encoded_captions = {}

        for (i, img) in enumerate(self.img_files):
            self.encoded_captions[img] = self.embeddings[i][0:5, :]

    def read_image(self, image_file_name):
        
        image_file_name += '.jpg'
        file_name = os.path.join(self.data_dir, image_file_name)
        image = imread(file_name)
        
        if image.shape != (64, 64, 3):
            image = resize(image, (64, 64, 3), anti_aliasing = True)

        return np.divide(np.array(image, dtype='float64'), 255.)

    def get_false_img(self, index):
        false_img_id = np.random.randint(len(self.img_files))
        if false_img_id != index:
            return self.img_files[false_img_id]

        return self.get_false_img(index)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):

        sample = {}
        sample['true_imgs'] = torch.tensor(self.read_image(self.img_files[index]), requires_grad=False, dtype=torch.float64)
        sample['false_imgs'] = torch.tensor(self.read_image(self.get_false_img(index)), requires_grad=False, dtype=torch.float64)
        
        embeddings = self.encoded_captions[self.img_files[index]]
        embeddings = embeddings[np.random.choice(embeddings.shape[0], 2, replace=False), :]
        embeddings = np.mean(embeddings, axis=0).reshape(1, -1)
        sample['true_embed'] = torch.tensor(embeddings, requires_grad=False, dtype=torch.float64)

        return sample

