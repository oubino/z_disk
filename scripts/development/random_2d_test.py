import numpy as np
from numpy import random
from perpl.relative_positions import getdistances
import matplotlib.pyplot as plt

# generate random localisations in 2d
x = random.rand(1000)
y = random.rand(1000)

# scale up data to a square of size
square_size = 100.
x *= square_size
y *= square_size
xy = np.stack([x,y], axis=1)

# calculate the distances in 2d 
d = getdistances(
    xy, 1000.0, verbose=False
)[1]
d = (d[:,0] ** 2 + d[:,1] ** 2) ** 0.5

# calculate the histogram of distances
a1, a2 = np.histogram(d, bins=100)
bin_centres = (a2[:-1] + a2[1:]) / 2

# calculate the expected distribution of distances
pdf = (4 * bin_centres/(square_size ** 4)) * ((np.pi / 2) * square_size ** 2 - 2 * square_size * bin_centres + 0.5 * bin_centres ** 2)
bin_width = a2[1] - a2[0]
y =  pdf * bin_width * len(d)

# plot
plt.stairs(a1, a2)
plt.plot(bin_centres,y)
plt.savefig("sandpit/hist2d.svg")