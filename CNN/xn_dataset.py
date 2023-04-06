import pandas as pd
import numpy as np
import os
import torch
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset
from skimage import io, transform
import random
import matplotlib.pyplot as plt
from PIL import Image
import time

class XNDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, image_dir, image_type='max_contrast_window', transform=None, reduced=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.h_frame = pd.read_csv(csv_file)
        print(len(self.h_frame))
        self.root_dir = root_dir
        self.transform = transform
        self.image_dir = image_dir
        self.image_type = image_type
        self.hem_dirs = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']
        self.reduced = reduced

    def __len__(self):
        return len(self.h_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            print(idx.shape)
            idx = idx.tolist()

        while True:
            try:
                # Get the hemorrhage types from the dataframe
                h_types = self.h_frame.iloc[idx, 4:]
                h_types = np.array(h_types).astype(np.int32)
                # print(self.h_frame.iloc[idx])


                # Figure out which image directory the image is in
                dirs = [self.hem_dirs[i] for i in range(len(h_types)) if h_types[i] != 0]
                # print(h_types)
                # print(dirs)
                if len(dirs) == 0:
                    hem_dir = 'normal'
                elif len(dirs) > 1:
                    hem_dir = 'multi'
                else:
                    hem_dir = dirs[0]

                # Create the image directory path
                img_name = os.path.join(self.root_dir, self.image_dir, hem_dir, self.image_type,
                                        self.h_frame.iloc[idx, 2] + '.jpg')
                

                # Convert numpy arrays to torhc Tensors
                # Read a PIL image
                image = Image.open(img_name)
                
                # Define a transform to convert PIL 
                # image to a Torch tensor
                transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.RandomHorizontalFlip(),

                    transforms.PILToTensor(),
                ])

                # No horizontal flip for validation dataset
                if self.reduced:
                    transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.PILToTensor(),
                    ])
                  
                # transform = transforms.PILToTensor()
                # Convert the PIL image to Torch tensor
                image = transform(image)
                # image = torch.Tensor(io.imread(img_name))

                # image.show()
                # h_types = torch.Tensor(h_types)
                # Full Classification
                # h_types = int("".join(str(x) for x in h_types), 2)

                # Half Classification
                # print(dirs)
                if len(dirs) == 0:
                    h_types = 0
                elif len(dirs) > 1:
                    h_types = 6
                else:
                    h_types = self.hem_dirs.index(dirs[0]) + 1
                # print(h_types)

                # h_types = torch.Tensor(h_types)

                # Apply transform if necessary
                if self.transform:
                    image = self.transform(image)
                return image.float(), h_types
            except FileNotFoundError:
                idx = random.randrange(1, self.__len__())
                pass

# xn = XNDataset(csv_file='hemorrhage-labels.csv', root_dir='./', image_dir='xn_image_data')

# dataloader = torch.utils.data.DataLoader(xn, batch_size=4,
#                                                  shuffle=True, num_workers=0)

# # Get a batch of training data
# inputs, classes = next(iter(dataloader))
# print(inputs.shape)





