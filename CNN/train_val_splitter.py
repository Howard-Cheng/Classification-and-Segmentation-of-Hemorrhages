import pandas as pd
import numpy as np
import os
import time
from sklearn.model_selection import train_test_split


o_frame = pd.read_csv('hemorrhage-labels-edited.csv')
# print(len(o_frame))
# print(o_frame.iloc[2, 2])


train, test = train_test_split(o_frame, test_size=0.2, shuffle=True)

print(len(train))
print(len(test))
train.to_csv("hemorrhage-labels-edited-train.csv") 
test.to_csv("hemorrhage-labels-edited-val.csv") 
