import csv
import pandas as pd
import numpy as np
from PIL import Image
import os

frame = pd.read_csv("hemorrhage-labels.csv")
new_frame = pd.read_csv("hemorrhage-labels.csv")

hem_dirs = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']
print(len(new_frame))
drop_indices = []

for idx in range(len(frame)):
    # Get the hemorrhage types from the dataframe
    try:
        h_types = frame.iloc[idx, 2:]
        h_types = np.array(h_types).astype(np.int32)
        # print(self.h_frame.iloc[idx])


        # Figure out which image directory the image is in
        dirs = [hem_dirs[i] for i in range(len(h_types)) if h_types[i] != 0]
        if len(dirs) == 0:
            hem_dir = 'normal'
        elif len(dirs) > 1:
            hem_dir = 'multi'
        else:
            hem_dir = dirs[0]

        # Create the image directory path
        img_name = os.path.join('./xn_image_dataset', hem_dir, 'max_contrast_window',
                                frame.iloc[idx, 0] + '.jpg')
        

        # Convert numpy arrays to torhc Tensors
        # Read a PIL image
        image = Image.open(img_name)

    except FileNotFoundError:
    	drop_indices.append(idx)
        
new_frame = new_frame.drop(drop_indices)
print(len(new_frame))
new_frame.to_csv("hemorrhage-labels-edited.csv")  

