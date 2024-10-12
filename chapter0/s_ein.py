# %%
import math
import os
import sys
from pathlib import Path

import einops
import numpy as np
import torch as t

# Make sure exercises are in the path
chapter = r"chapter0_fundamentals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part0_prereqs"
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part0_prereqs.tests as tests
from part0_prereqs.utils import display_array_as_img
from plotly_utils import bar, imshow, line

MAIN = __name__ == "__main__"
# %%
arr = np.load(section_dir / "numbers.npy")

hlep = einops.rearrange(arr, "b c h w -> c h (b w)")

# %%
hlep.shape
display_array_as_img(hlep)
arr[0].shape

# %%
zer = arr[0]
zer = einops.repeat(zer, "c h w -> c (2 h) w")
display_array_as_img(zer)

# %%
zerone = arr[0:2]
zerone = einops.repeat(zerone, "b c h w -> c (b h) (2 w)")
display_array_as_img(zerone)

# %%
# Looooong Booooy

zeeero = arr[0]
zeeero = einops.repeat(zeeero, "c h w -> c (h repeat) w", repeat=2)
display_array_as_img(zeeero)

# %%

# Greyscale
display_array_as_img(einops.repeat(arr[0], "c h w -> h (c w)"))
