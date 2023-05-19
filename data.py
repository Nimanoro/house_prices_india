import torch 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import math
import torch.optim as optim
#loading the data with panda library
data = pd.read_csv('Hyderabad.csv')
data=data.drop('Location', axis=1)
data_array = data.to_numpy()
data_tensor = torch.tensor(data_array, dtype=torch.float32)