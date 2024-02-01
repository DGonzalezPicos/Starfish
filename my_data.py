import numpy as np
import os
import matplotlib.pyplot as plt

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

# Load grid
from Starfish.grid_tools import IsoPHOENIXGridInterface
k2166_wl_range = [23000, 24000]

grid = IsoPHOENIXGridInterface('/data2/dario/Isotopes/', wl_range=k2166_wl_range)
model = grid.load_flux([4500, 4.0, -0.5, 91])
grid.wl = doppler_shift(grid.wl, -32.0)
model = rotational_broaden(grid.wl, model, 10.)
model = instrumental_broaden(grid.wl, model, 3.0)



fig, ax = plt.subplots(1, 1, figsize=(10, 5))
# ax.plot(wave, flux, label="Flux")
ax.plot(wave, flux_corr)
ax.fill_between(wave, flux_corr - err_corr, flux_corr + err_corr, alpha=0.5)

# ax.plot(wave, telluric_model, label="Transm", color='r', alpha=0.7)
ax.plot(grid.wl * 1e-1, model / np.nanmedian(model), label="Model", alpha=0.6)

ax.set(xlim=(2320, 2360))
# plt.show()
# plt.show()
# data = Spectrum(waves=)

# data.plot()
plt.savefig("data/my_data.png", dpi=200, bbox_inches='tight')