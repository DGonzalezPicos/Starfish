import pathlib
import numpy as np
import h5py

from astropy.io import fits

# read a spectrum, this is generic

path = pathlib.Path('/data2/dario/Isotopes/')
out_path = path / 'Z-0.0'
out_path.mkdir(exist_ok=True)

files = sorted((path / 'full_Z-0.0/').glob('*.h5'))



print(f'Found {len(files)} files')
print(f'First file: {files[0]}')

def read_file(file):
    with h5py.File(file, 'r') as fh5:
        wl = fh5['PHOENIX_SPECTRUM/wl'][()]
        fl = 10.**fh5['PHOENIX_SPECTRUM/flux'][()]
    return wl, fl

def write_file(file, fl):
    with h5py.File(file, 'w') as fh5:
        # fh5.create_dataset('PHOENIX_SPECTRUM/wl', data=wl)
        fh5.create_dataset('PHOENIX_SPECTRUM/flux', data=fl)
        


file_wave = path / "WAVE_PHOENIX-NewEra-ACES-COND-2023.h5"

if file_wave.exists():
    print(f'Found {file_wave}')
    ref_wl = h5py.File(file_wave,'r')['WAVE'][()] / 10. # in Angstroms

else:
    file_ref = sorted((path / 'Z-0.5').glob('*.h5'))[0]
    ref_wl, _ = read_file(file_ref)
    ref_wl *= 10. # convert to Angstroms
    
    fh5 = h5py.File(path / file_wave, 'w')
    fh5.create_dataset('WAVE', data=ref_wl)
    fh5.close()
    print(f'- wrote {path / file_wave}')

print(f'Refence cenwave is {np.median(ref_wl):.2f} A')

# save cropped spectrum
for f in files:
    if 'O_ratio' in f.name:
        continue
    # print(f' Cenwave for file {f.name}: {np.median(wl):.2f} A')
    wl, flux = read_file(f)
    wl /= 10. # convert to Angstroms
    # print(f'Cenwave for file {f.name}: {np.median(wl):.2f} A', end='\r')
    
    mask = (wl >= ref_wl[0]) & (wl <= ref_wl[-1])
    assert np.allclose(wl[mask], ref_wl), f'Wavelengths do not match for {f.name}'
    assert np.sum(np.isnan(flux[mask])) == 0, f'NaNs in flux for {f.name}'
    
    
    write_file(out_path / f.name, flux[mask])
    print(f' - wrote {out_path / f.name}', end='\r')


