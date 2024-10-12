# %%
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# %%
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
print(x_data)
# %%
np_arr = np.array(data)
x_np = torch.from_numpy(np_arr)
print(x_np)
# %%
x_ones = torch.ones_like(x_data)  # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float)  # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")
# %%
print(f"Zeroes Tensor: \n {torch.zeros_like(x_data)} \n")
# %%
print(f"Device tensor: {x_np.device}")
# %%
# Matrix Mult betwenn two tensors
tensor = torch.ones(4, 4)
y1 = tensor @ tensor.T
print(y1)
# %%
