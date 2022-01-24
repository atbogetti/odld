import h5py
import sys

last = int(sys.argv[1])

f = h5py.File('west.h5', 'r')['summary']

walltime = f['walltime'][0:last].sum()
cputime = f['cputime'][0:last].sum()

print("walltime: ", walltime, "s")
print("cputime: ", cputime, "s")
