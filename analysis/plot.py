import h5py
import numpy
import matplotlib.pyplot as plt

h5file = h5py.File("./ANALYSIS/DEFAULT/direct.h5", "r")
fluxes = h5file['target_flux_evolution']['expected']
red_fluxes = h5file['red_flux_evolution'][:]
xs = numpy.arange(0,fluxes.shape[0])
plt.semilogy(xs, fluxes, color="cornflowerblue")
plt.semilogy(xs, red_fluxes, color="tomato")
plt.xlabel("iteration")
plt.ylabel("flux evolution (tau$^{-1}$)")
plt.savefig("flux_evo.pdf")
