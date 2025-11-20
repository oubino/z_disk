import numpy as np
from numpy import random
from perpl.relative_positions import getdistances
from perpl.io import plotting
import matplotlib.pyplot as plt
from scipy.special import i0

def churchman_term2d(d, mu, sigma):
    if ((mu * d / sigma**2) < 700).all():
        p = (d / sigma**2) * (
            np.exp(-(mu**2 + d**2) / (2 * sigma**2)) * i0(d * mu / sigma**2)
        )
    else:  # Approximate overly large i0()
        p = (
            1
            / (np.sqrt(2 * np.pi) * sigma)
            * np.sqrt(d / mu)
            * np.exp(-((d - mu) ** 2) / (2 * sigma**2))
        )
    return p

# generate random localisations in 2d
x = random.rand(1000)
y = random.rand(1000)

# scale up data to a square of size
square_size = 100.
x *= square_size
y *= square_size
xy = np.stack([x,y], axis=1)

# kde params
loc_precision = 3.
fitlength = 80.
normalise = True

# calculate the distances in 2d 
d = getdistances(
    xy, 1000.0, verbose=False
)[1]
d = (d[:,0] ** 2 + d[:,1] ** 2) ** 0.5

# calculate kde for distances

increment = np.round(fitlength / len(d))
if increment == 0:
    increment = 1
calculation_points = np.arange(
    0, fitlength + 1.0, increment
)

churchman = plotting.estimate_rpd_churchman_2d

rpd = churchman(
    input_distances=d,
    calculation_points=calculation_points,
    combined_precision=(np.sqrt(2) * loc_precision),
)

# normalise 
if normalise:
    y_expt = rpd[calculation_points > 0]
    x_expt = calculation_points[calculation_points > 0]
else:
    y_expt = rpd
    x_expt = calculation_points

# calculate the bg

# generate locations based on distribution of distances
#_, a2 = np.histogram(d, bins=2000)
#bin_centres = (a2[:-1] + a2[1:]) / 2
#bin_width = a2[1] - a2[0]

# generate the bg
bg = 0 * x_expt
sigma = (np.sqrt(2) * loc_precision)

bg_x = np.linspace(0,square_size*np.sqrt(2), 500)
bin_width = bg_x[1] - bg_x[0]
for i, mu in enumerate(bg_x):
#for i, mu in enumerate(np.arange(0, fitlength + 1.0, 0.07)):
#for i, mu in enumerate(bin_centres):

    churchman_term = churchman_term2d(x_expt, mu, sigma)
    frequency_term = (4 * mu/(square_size ** 4)) * ((np.pi / 2) * square_size ** 2 - 2 * square_size * mu + 0.5 * mu ** 2)
    frequency_term =  frequency_term * bin_width * len(d)

    bg += churchman_term * frequency_term

# normalise the data
if normalise:
    y_expt /= x_expt
    bg /= x_expt

# plot the data
plt.scatter(x_expt, y_expt, label="distances")
plt.plot(x_expt, bg, label="bg", c="r")
plt.xlim(0,square_size)
plt.legend()
plt.savefig("sandpit/2d_test_kde.svg")
plt.close()

#pdf = (4 * bin_centres/(square_size ** 4)) * ((np.pi / 2) * square_size ** 2 - 2 * square_size * bin_centres + 0.5 * bin_centres ** 2)