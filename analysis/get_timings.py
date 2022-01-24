import h5py

f = h5py.File('west.h5', 'r')['summary']

walltime = f['walltime'].sum()
cputime = f['cputime'].sum()

print("walltime: ", walltime, "s")
print("cputime: ", cputime, "s")
