import numpy as np
import os
cwd = os.getcwd()

from Starfish.grid_tools import IsoPHOENIXGridInterface
from Starfish.grid_tools.instruments import CRIRES_K
from Starfish.grid_tools import HDF5Creator
from Starfish.emulator import Emulator
# ranges = [[5700, 8600], [4.0, 6.0], [-0.5, 0.5]]  # T, logg, Z
ranges = [[3900, 4800], [3.0, 4.5], [-0.5, 0.5], [31, 301]]  # T, logg, Z, C12/C13

# k2166_wl_range = [19000, 25000]
k2166_wl_range = [21660-100, 21660+100]


if 'picos' in cwd:
    path = '/data2/dario/Isotopes/'
elif 'dario' in cwd:
    
    path = '/home/dario/phd/Starfish/data/'


grid = IsoPHOENIXGridInterface(path, wl_range=k2166_wl_range)



grid_name = "IsoPHOENIX_CRIRES_test.hdf5"
creator = HDF5Creator(
    grid, grid_name, instrument=CRIRES_K(), ranges=ranges)
creator.process_grid()


emu = Emulator.from_grid(grid_name)

kwargs = {"method": "Nelder-Mead", "options": {"maxiter": 10}}
emu.train(**kwargs)

