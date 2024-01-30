import numpy as np
import os
import matplotlib.pyplot as plt

from Starfish.spectrum import Spectrum

old_settings = np.seterr(all='ignore')  #seterr to known value

wave, flux, err = np.loadtxt("data/GQLupA/SCIENCE_GQLup_PRIMARY.dat", unpack=True)
_, transm, cont = np.loadtxt("data/GQLupA/SCIENCE_GQLup_PRIMARY_molecfit_transm.dat", unpack=True)

n_edge = 60

mask = np.isnan(flux) # True for NaNs


zeros = transm < 0.45
zeros = np.convolve(zeros, np.ones(15) / 15, mode="same") > 0

flux_corr = np.divide(flux, transm * cont, where=~zeros)
err_corr  = np.divide(err, transm * cont, where=~zeros)
flux_corr[zeros] = np.nan
mask |= zeros

# apply edge mask
flux_corr[:n_edge] = np.nan
flux_corr[-n_edge:] = np.nan
mask |= np.isnan(flux_corr)

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
# ax.plot(wave, flux, label="Flux")
ax.plot(wave, flux_corr, label="Transmission")
ax.fill_between(wave, flux_corr - err_corr, flux_corr + err_corr, alpha=0.5)


plt.show()
# data = Spectrum(waves=)

# data.plot()
# plt.savefig("data/example_spec.png", dpi=200, bbox_inches='tight')