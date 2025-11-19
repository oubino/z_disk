import numpy as np
from numpy import random
from perpl.relative_positions import getdistances
import matplotlib.pyplot as plt

# generate random localisations in 1d 
x = random.rand(1000)

# scale up data to a line of length...
length = 20.
x = x * length
x = np.expand_dims(x, axis=1)

# calculate the distances in 1d
d = getdistances(
    x, 500.0, verbose=False
)[1]

 # Make all axial separations positive.
d = abs(d)
d = np.sort(d)
d = d[::2]

# calculate the histogram of distances
a1, a2 = np.histogram(d, bins=100)
bin_centres = (a2[:-1] + a2[1:]) / 2

# calculate the expected distribution of distances
pdf = 2 * (1/length) * (1 - bin_centres/length)
bin_width = a2[1] - a2[0]
y =  pdf * bin_width * len(d)

# plot
plt.stairs(a1, a2)
plt.plot(bin_centres,y)
plt.savefig("sandpit/hist1d.svg")