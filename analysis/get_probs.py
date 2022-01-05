import h5py

f = h5py.File("west.h5", "r")

mins = f['summary']['min_seg_prob'][:500]
maxs = f['summary']['max_seg_prob'][:500]
nsegs = f['summary']['n_particles'][:500]

weights = []

for i in range(1,500):
    path = "iterations/iter_" + str(i).zfill(8)
    weight = f[path]['seg_index']['weight'].sum()
    weights.append(round(weight,10))

print("seg count:", nsegs)
print("weight sum:", weights)
print("max:", maxs.max())
print("min:", mins.min())
#print("all max:", maxs)
#print("all min:", mins)
