
# %%
import os
import functools
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import einops
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from IPython.display import display
from jaxtyping import Float, Int
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from tqdm.notebook import tqdm

# Make sure exercises are in the path
chapter = r"chapter0_fundamentals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part2_cnns"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

import part2_cnns.tests as tests
from part2_cnns.utils import print_param_count
from plotly_utils import line

MAIN = __name__ == "__main__"

device = t.device('mps' if t.backends.mps.is_available() else 'cuda' if t.cuda.is_available() else 'cpu')

# %%
class MyModule(nn.Module):
    def __init__(self):
        super.__init__()
        self.weights = nn.Parameter(weights)
        self.biases = nn.Parameter(biases)
    
    def forward(self,x: t.Tensor) -> t.Tensor:
        pass 

    def extra_repr(self) -> str:
        return f"{self}"
# %%
class ReLU(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        return t.maximum(x, t.tensor(0.0))

tests.test_relu(ReLU)
# %%
class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias = True) :
        '''
        A simple linear transformation.
        '''
        super().__init__()

        weights = (2*t.rand(out_features,in_features)-1)/np.sqrt(in_features)
        biases = (2*t.rand(out_features,)-1)/np.sqrt(in_features)

        self.weight = nn.Parameter(weights)
        if bias:
            self.bias = nn.Parameter(biases)
        else:
            self.bias = None

    def forward(self, x: t.Tensor) -> t.Tensor:
       '''
       x: shape (*, in_features)
       Return: shape (*, out_features)
       '''
       x = einops.einsum(x, self.weight, "... in_feats, ... out_feats in_feats -> ... out_feats")
       if self.bias is not None:
         x += self.bias
       return x

    def extra_repr(self)->str:
        return f"weight: {self.weight}, \n bias: {self.bias is not None}"

tests.test_linear_parameters(Linear, bias=False)
tests.test_linear_parameters(Linear, bias=True)
tests.test_linear_forward(Linear, bias=False)
tests.test_linear_forward(Linear, bias=True)
        
# %%
class Flatten(nn.Module):
    
    def __init__(self, start_dim: int =1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim
            
    def forward(self, input: t.Tensor) -> t.Tensor:    
        '''
        Flatten out dimensions from start_dim to end_dim inclusive of both.
        input: shape (n0 ... nK)
        output: shape (n0 ... n(start_dim)...n(end_dim) ... nK)
        '''
        shape = input.shape
        start_dim = self.start_dim
        end_dim = self.end_dim if self.end_dim >= 0 else len(shape) + self.end_dim
    
        left_dims = shape[:start_dim]
       
        flatten_dims = 1
        for dim in shape[start_dim:end_dim+1]:
            flatten_dims *= dim
       
        right_dims = shape[end_dim+1:]
        
        new_shape = left_dims + (flatten_dims,) + right_dims
        
        return t.reshape(input,new_shape)
                

    def extra_repr(self) -> str:
        return ", ".join([f"{key}={getattr(self, key)}" for key in ["start_dim", "end_dim"]])

 
x = t.arange(24).reshape(2,3,4)
y = Flatten(start_dim=0)(x)
print(y.shape,y)  
tests.test_flatten(Flatten)
# %%
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.flatten = Flatten()
        self.linear1 = Linear(28*28,100)
        self.relu = ReLU()
        self.linear2 = Linear(100,10)
    
    
    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.linear2(self.relu(self.linear1(self.flatten(x))))

    
    
tests.test_mlp_module(SimpleMLP)
tests.test_mlp_forward(SimpleMLP)
# %%
