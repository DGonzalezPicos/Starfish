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

process_grid = True

if process_grid:
    grid_name = "IsoPHOENIX_CRIRES_extended.hdf5"
    # check if grid exists
    if os.path.exists(grid_name):
        print(f'Found {grid_name}')
    else:
        print(f'Creating {grid_name}')
        creator = HDF5Creator(
            grid, grid_name, instrument=CRIRES_K(), ranges=ranges)
        creator.process_grid()


emu = Emulator.from_grid(grid_name)
train_emu = True

if train_emu:
    kwargs = {"method": "Nelder-Mead", "options": {"maxiter": 1e5}}
    print('Training emulator...')
    start = time.time()
    emu.train(**kwargs)
    print(f'--> Training done in {time.time()-start:.2e} s')
    emu.save("CRIRES_emu_extended.hdf5")
    print('--> Saved emulator to CRIRES_emu_extended.hdf5')
    
    from Starfish.emulator.plotting import plot_emulator

    plot_emulator(emu)
    plt.savefig("CRIRES_emu_extended.png", dpi=200, bbox_inches='tight')