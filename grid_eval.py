import numpy as np
import os
import time
cwd = os.getcwd()

from Starfish.grid_tools import IsoPHOENIXGridInterface
from Starfish.grid_tools.instruments import CRIRES_K
from Starfish.grid_tools import HDF5Creator
from Starfish.emulator import Emulator
# ranges = [[5700, 8600], [4.0, 6.0], [-0.5, 0.5]]  # T, logg, Z
ranges = [[3500, 4900], [3.5, 4.5], [-0.5, 0.5], [31, 220]]  # T, logg, Z, C12/C13

k2166_wl_range = [20000, 24800]
# k2166_wl_range = [21660-100, 21660+100]


if 'data2' in cwd:
    path = '/data2/dario/Isotopes/'
else:
    path = '/home/dario/phd/Starfish/data/'


grid = IsoPHOENIXGridInterface(path, wl_range=k2166_wl_range)
grid.load_flux([4300, 4.0, -0.5, 61])

