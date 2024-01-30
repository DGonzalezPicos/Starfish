import pathlib
import subprocess

path = pathlib.Path('/data2/dario/Isotopes/')
Z_dirs = ['Z-1.0', 'Z-0.5', 'Z-0.0', 'Z+0.5']

out_path = path / 'O_ratio'
out_path.mkdir(exist_ok=True)

for Z in Z_dirs:
    files = sorted((path / Z).glob('*.h5'))
    print(f'Found {len(files)} files')
    files_O = sorted((path / Z).glob('*O_ratio*.h5'))
    print(f'Found {len(files_O)} O_ratio files')
    pattern = str(path / Z / '*O_ratio*.h5')
    
    out_path_Z = out_path / Z
    out_path_Z.mkdir(exist_ok=True)
    subprocess.run(['mv'] + [pattern] + [str(out_path_Z)])
    
    