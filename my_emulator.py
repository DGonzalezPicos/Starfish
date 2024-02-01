import matplotlib.pyplot as plt
import numpy as np
from Starfish.models import SpectrumModel


from Starfish.spectrum import Spectrum
from Starfish.transforms import doppler_shift, instrumental_broaden, rotational_broaden
from Starfish.grid_tools.utils import blackbody

old_settings = np.seterr(all='ignore')  #seterr to known value

wave, flux, err = np.loadtxt("data/GQLupA/SCIENCE_GQLup_PRIMARY.dat", unpack=True)
_, transm, cont = np.loadtxt("data/GQLupA/SCIENCE_GQLup_PRIMARY_molecfit_transm.dat", unpack=True)

wave_cm = wave * 1e-6
bb = blackbody(12700, wave_cm)
cont/= (bb / bb.mean())

telluric_model = np.copy(transm)
# transm = np.ones_like(wave)

n_edge = 60

mask = np.isnan(flux) # True for NaNs


zeros = transm < 0.25
zeros = np.convolve(zeros, np.ones(15) / 15, mode="same") > 0

flux_corr = np.divide(flux, transm * cont, where=~zeros)
err_corr  = np.divide(err, transm * cont, where=~zeros)
flux_corr[zeros] = np.nan
mask |= zeros

# apply edge mask
flux_corr[:n_edge] = np.nan
flux_corr[-n_edge:] = np.nan
mask |= np.isnan(flux_corr)

data = Spectrum(wave[~mask], flux_corr[~mask], err_corr[~mask])
print('Loaded data')

from Starfish.emulator import Emulator
from Starfish.transforms import instrumental_broaden, rotational_broaden, doppler_shift
emu = Emulator.load("CRIRES_emu.hdf5")
print(f'Loaded emulator from CRIRES_emu.hdf5')
fluxes = emu.bulk_fluxes
emu.wl = doppler_shift(emu.wl, -32.0)
fluxes = rotational_broaden(emu.wl, fluxes, 10.)
fluxes = instrumental_broaden(emu.wl, fluxes, 3.0)
eigs = fluxes[:-2]
flux_mean, flux_std = fluxes[-2:]


fig, ax = plt.subplots(1, 1, figsize=(10, 5))

for T in np.arange(4200, 4400, 27):
    # flux = emu.load_flux([T, 4.0, -0.5, 31])
    weights, cov = emu([T, 4.0324, -0.1, 36])
    X = emu.eigenspectra * flux_std
    flux = weights @ X + flux_mean
    # emu_cov = X.T @ weights @ X
    ax.plot(emu.wl, flux, label=f'T={T}')

ax.legend()

# plt.show()
fig.savefig("emulator_test.png", dpi=200, bbox_inches='tight')
print(f' Saved plot to emulator_test.png')

# model = SpectrumModel(
#     "CRIRES_emu.hdf5",
#     data,
#     grid_params=[4300, 4.0, -0.5, 31],
#     Av=0.0,
#     global_cov=dict(log_amp=22.0, log_ls=2.0,),
    
# )
# print(model)
# model.plot()
# plt.savefig("emulator_test.png", dpi=200, bbox_inches='tight')
# print(f'Saved plot to emulator_test.png')
# plt.close()