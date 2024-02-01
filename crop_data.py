import pathlib
import numpy as np
import matplotlib.pyplot as plt
import h5py
import tqdm

from astropy.io import fits

# read a spectrum, this is generic

path = pathlib.Path('/data2/dario/Isotopes/')
out_path = path / 'Z-0.0'
out_path.mkdir(exist_ok=True)

files_full = sorted((path / 'full_Z-0.0/').glob('*.h5'))
files = [f for f in files_full if 'O_ratio' not in f.name]



print(f'Found {len(files)} files')
print(f'First file: {files[0]}')

def read_file(file):
    
    with h5py.File(file, 'r') as fh5:
        bb = fh5['PHOENIX_SPECTRUM/bb'][()]
        chunky = fh5['PHOENIX_SPECTRUM/chunky'][()]
        flux = fh5['PHOENIX_SPECTRUM/flux'][()]
        nwl = fh5['PHOENIX_SPECTRUM/nwl'][()]
        wl = fh5['PHOENIX_SPECTRUM/wl'][()]
    return bb, chunky, flux, nwl, wl

def write_file(file, bb, chunky, flux, nwl, wl):
    with h5py.File(file, 'w') as fh5:
        # fh5.create_dataset('PHOENIX_SPECTRUM/wl', data=wl)
        fh5.create_dataset('PHOENIX_SPECTRUM/bb', data=bb)
        fh5.create_dataset('PHOENIX_SPECTRUM/chunky', data=chunky)
        fh5.create_dataset('PHOENIX_SPECTRUM/flux', data=flux)
        fh5.create_dataset('PHOENIX_SPECTRUM/nwl', data=nwl)
        fh5.create_dataset('PHOENIX_SPECTRUM/wl', data=wl)        


file_wave = path / "WAVE_PHOENIX-NewEra-ACES-COND-2023.h5"

if file_wave.exists():
    print(f'Found {file_wave}')
    ref_wl = h5py.File(file_wave,'r')['WAVE'][()] # in Angstroms

else:
    file_ref = sorted((path / 'Z-0.5').glob('*.h5'))[0]
    ref_wl = read_file(file_ref)[-1]
    
    fh5 = h5py.File(path / file_wave, 'w')
    fh5.create_dataset('WAVE', data=ref_wl)
    fh5.close()
    print(f'- wrote {path / file_wave}')

print(f'Refence cenwave is {np.median(ref_wl):.2f} A')

# save cropped spectrum
for i, f in enumerate(tqdm.tqdm(files)):
    if 'O_ratio' in f.name:
        continue
    # print(f' Cenwave for file {f.name}: {np.median(wl):.2f} A')
    # wl, flux = read_file(f)
    bb, chunky, flux, nwl, wl = read_file(f)
    # print(f' Shapes: {bb.shape}, {chunky.shape}, {flux.shape}, {nwl.shape}, {wl.shape}')
    
    # wl /= 10. # convert to Angstroms
    # print(f'Cenwave for file {f.name}: {np.median(wl):.2f} A', end='\r')
    
    mask = (wl >= ref_wl[0]) & (wl <= ref_wl[-1])
    assert np.allclose(wl[mask], ref_wl), f'Wavelengths do not match for {f.name}'
    assert np.sum(np.isnan(flux[mask])) == 0, f'NaNs in flux for {f.name}'
    
    if i % 1000 == 0:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.plot(wl, flux, lw=0.9)
        ax.plot(wl[mask], flux[mask], lw=0.9, alpha=0.7)
        ax.set_xlabel('Wavelength [A]')
        ax.set_ylabel('Flux')
        ax.set_title(f.name)
        
        xlim = [wl[mask][0], wl[mask][-1]]
        pad = 0.15 * np.diff(xlim)
        xlim[0] -= pad
        xlim[1] += pad
        
        ylim = [np.nanmin(flux[mask]), np.nanmax(flux[mask])]
        
        ax.set(xlim=xlim, ylim=ylim)
        
        fig.savefig(out_path / f'{f.name}.png', dpi=200, bbox_inches='tight')
        plt.close(fig)
        
    
    write_file(out_path / f.name, bb[mask], chunky, flux[mask], nwl, wl[mask])
    # print(f' - wrote {out_path / f.name}', end='\r')


