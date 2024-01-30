import numpy as np
import os
import matplotlib.pyplot as plt

from Starfish.spectrum import Spectrum


data = Spectrum.load("data/example_spec.hdf5")

data.plot()
plt.savefig("data/example_spec.png", dpi=200, bbox_inches='tight')