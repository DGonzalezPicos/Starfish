import numpy as np
import os
cwd = os.getcwd()

import pathlib
import h5py

import matplotlib.pyplot as plt


path = pathlib.Path('/data2/dario/Isotopes/')
Z = -0.5
Z_dict = {-1.0 : 'Z-1.0', -0.5 : 'Z-0.5', 0.0 : 'Z-0.0', 0.5 : 'Z+0.5'}

fig, ax = plt.subplots(3, 1, figsize=(10, 5), sharex=True)

for i, Z in enumerate([-1.0, 0.0, 0.5]):


    files = sorted((path/Z_dict[Z]).glob('*.h5'))
    print(f' - Found {len(files)} files for Z={Z}')

    file = files[0]
    fh5 = h5py.File(file,'r')
    flux = 10.**fh5['PHOENIX_SPECTRUM/flux'][()]
    # inspect the keys
    # print(list(fh5.keys()))
    print(fh5.keys())
    print(fh5['PHOENIX_SPECTRUM'].keys())
    print(f'Mean flux: {np.mean(flux)}')
    print(f'Min flux: {np.min(flux)}')
    print(f'Max flux: {np.max(flux)}')
    
    ax[i].plot(flux, label=f'Z={Z}')
    ax[i].legend()
    
plt.show()
