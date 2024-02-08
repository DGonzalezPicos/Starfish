import numpy as np
import os
import time
cwd = os.getcwd()

from Starfish.grid_tools import CustomPHOENIXGridInterface, IsoPHOENIXGridInterface
from Starfish.grid_tools.instruments import CRIRES_K
from Starfish.grid_tools import HDF5Creator
from Starfish.emulator import Emulator
# ranges = [[5700, 8600], [4.0, 6.0], [-0.5, 0.5]]  # T, logg, Z
ranges = [[3500, 4900], [3.5, 4.5], [-0.5, 0.5]]  # T, logg, Z, C12/C13

k2166_wl_range = [21800, 23800]
# k2166_wl_range = [21660-100, 21660+100]


if 'data2' in cwd:
    path = '/data2/dario/Isotopes/'
else:
    path = '/home/dario/phd/Starfish/data/'


grid = CustomPHOENIXGridInterface(path, wl_range=k2166_wl_range)
# iso_grid = IsoPHOENIXGridInterface(path, wl_range=k2166_wl_range)

process_grid = True
cache = False
grid_name = "PHOENIX_CRIRES.hdf5"

if process_grid:
    # check if grid exists
    if os.path.exists(grid_name) and cache:
        print(f'Found {grid_name}')
    else:
        print(f'Creating {grid_name}')
        creator = HDF5Creator(
            grid, grid_name, instrument=CRIRES_K(), ranges=ranges)
        creator.process_grid()


emu = Emulator.from_grid(grid_name)
train_emu = True

if train_emu:
    kwargs = {"method": "Nelder-Mead", "options": {"maxiter": 1e4}}
    print('Training emulator...')
    start = time.time()
    emu.train(**kwargs)
    print(f'--> Training done in {time.time()-start:.2e} s')
    emu_name = grid_name.replace('.hdf5', '_emu.hdf5')
    emu.save(emu_name)
    print(f'--> Saved emulator to {emu_name}')
    
    # from Starfish.emulator.plotting import plot_emulator

    # plot_emulator(emu)
    # plt.savefig(emu_name.replace('.hdf5', '.png'), dpi=200, bbox_inches='tight')