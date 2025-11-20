import numpy as np
from numpy import random
from perpl.relative_positions import getdistances
import matplotlib.pyplot as plt

# generate random localisations in 2d
x = random.rand(1000)
y = random.rand(1000)

# scale up data to a square of size
square_size = 200.
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
bg_x = np.linspace(0,square_size*np.sqrt(2),100)
bin_width = bg_x[1] - bg_x[0]
pdf = (4 * bg_x/(square_size ** 4)) * ((np.pi / 2) * square_size ** 2 - 2 * square_size * bg_x + 0.5 * bg_x ** 2)
bg =  pdf * bin_width * len(d)

# model the linear bit
pdf_2 = (2 * np.pi * bg_x/(square_size ** 2) - 8 * bg_x ** 2 * square_size ** -3) * bin_width * len(d)
pdf_1 = (2 * np.pi * bg_x/(square_size ** 2)) * bin_width * len(d)

# plot
plt.stairs(a1, a2, label="bg")
plt.plot(bg_x,bg, label ="model_bg")
plt.plot(bg_x, pdf_1,c = 'r', label="model_bg_linear_term")
plt.plot(bg_x, pdf_2,c = 'y', label="model_bg_linear_and_quad_term")
plt.xlim(0,square_size)
plt.ylim(0,10000)
plt.legend()
plt.savefig("sandpit/hist2d.svg")