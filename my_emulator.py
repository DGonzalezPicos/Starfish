import matplotlib.pyplot as plt
import numpy as np
from Starfish.models import SpectrumModel


from Starfish.spectrum import Spectrum
from Starfish.transforms import doppler_shift, instrumental_broaden, rotational_broaden
from Starfish.grid_tools.utils import blackbody
from Starfish.utils import solve_linear
from Starfish.spline_model import SplineModel

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

assert np.sum(np.isnan(wave)) == 0, f'Found NaNs in wavelength array'
# data = Spectrum(wave[~mask], flux_corr[~mask], err_corr[~mask])



print('Loaded data')

from Starfish.emulator import Emulator
from Starfish.transforms import instrumental_broaden, rotational_broaden, doppler_shift, rebin
emu = Emulator.load("CRIRES_emu.hdf5")
print(f'Loaded emulator from CRIRES_emu.hdf5')
fluxes = emu.bulk_fluxes

# crop data to match the limits of the emulator (23000, 24000) A
# convert wave to Angstroms
wave *= 10.

wave_mask = (wave > emu.wl.min()) & (wave < emu.wl.max())
wave = wave[wave_mask]
flux_corr = flux_corr[wave_mask]
err_corr = err_corr[wave_mask]


# emu.wl = doppler_shift(emu.wl, -32.0)
# fluxes = rotational_broaden(emu.wl, fluxes, 10.)
# fluxes = instrumental_broaden(emu.wl, fluxes, 3.0)
eigs = fluxes[:-2]
flux_mean, flux_std = fluxes[-2:]
# flux_mean_rb = rebin(wave, emu.wl, flux_mean).reshape(7,3,2048)


T = 4322.
weights, cov = emu([T, 3.51, 0.0, 36])
X = emu.eigenspectra * flux_std

flux = weights @ X + flux_mean
emu.wl = doppler_shift(emu.wl, -32.0)
flux = rotational_broaden(emu.wl, flux, 5.)
flux = instrumental_broaden(emu.wl, flux, 3.0)
flux = rebin(wave, emu.wl, flux)

plot_eigenspectra = False

if plot_eigenspectra:
    fig, ax = plt.subplots(len(X)+2, 1, figsize=(14, 10), sharex=True)

    colors = plt.get_cmap('viridis')(np.linspace(0, 1, len(X)))
    for i, X_i in enumerate(X):
        ax[i+2].plot(emu.wl, X_i, label=f'X_{i}', color=colors[i])

    ax[0].plot(wave, flux, label=f'PCA reconstructed', color='k')
    ax[1].plot(emu.wl, flux_mean, label=f'Mean Flux', color='darkgreen')
    [axi.legend() for axi in ax]

    ax[-1].set(xlim=(emu.wl.min(), emu.wl.max()), xlabel='Wavelength [A]')
    plt.show()
    
    
# Reshape data to (n_orders, n_detectors, n_pixels) 
n_orders, n_detectors, n_pixels = 1, 3, 2048
new_shape = (n_orders, n_detectors, n_pixels)
wave = wave.reshape(new_shape)
flux_corr = flux_corr.reshape(new_shape)
err_corr = err_corr.reshape(new_shape)

flux = flux.reshape(new_shape) # model flux

fig, ax = plt.subplots(2,1, figsize=(14, 10), 
                       sharex=True, 
                       gridspec_kw={'height_ratios': [3, 1]})

logL = 0.0
for order in range(n_orders):
    for det in range(n_detectors):
        ax[0].plot(wave[order,det], flux_corr[order,det], label='', color='k')

        # assert np.sum(np.isnan(err_corr[order,det])) == 0, f'Found NaNs in error array'
        
        nans = np.isnan(flux_corr[order,det]) | np.isnan(err_corr[order,det])
        cov = np.diag(err_corr[order,det,~nans]**2)
        cov_inv = np.diag(1/err_corr[order,det,~nans]**2)
        
        smodel = SplineModel(N_knots=11, spline_degree=3)(flux[order,det,~nans])

        
        phi = solve_linear(flux_corr[order,det,~nans], smodel, cov_inv)
        smodel_flux = phi @ smodel
        
        N_pix = (~nans).sum()
        logL += -0.5 * N_pix * np.log(2*np.pi) #- 0.5 * np.log(np.linalg.det(cov))
        
        ax[0].plot(wave[order,det,~nans], smodel_flux, label=f'Spline model', color='darkorange')

        residuals = flux_corr[order,det,~nans] - smodel_flux
        logL += -0.5 * residuals.T @ cov_inv @ residuals
        ax[1].plot(wave[order,det,~nans], residuals, color='k')
        
print(f'logL = {logL:.2e}')
   
ax[1].axhline(0, ls='--', color='k')
ax[1].set(ylabel='Residuals', xlabel='Wavelength [A]', xlim=(emu.wl.min(), emu.wl.max()))
ax[0].legend()
plt.show()