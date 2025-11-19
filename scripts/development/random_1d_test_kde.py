import numpy as np
from numpy import random
from perpl.relative_positions import getdistances
from perpl.io import plotting
import matplotlib.pyplot as plt

# generate random localisations in 1d 
x = random.rand(500)

# scale up data to a line of length...
length = 50.
x = x * length
x = np.expand_dims(x, axis=1)

# params for generating kde
loc_precision = 3.
fitlength = 10.
normalise = False

# calculate the distances in 1d 
d = getdistances(
    x, 500.0, verbose=False
)[1]

 # Make all axial separations positive.
d = abs(d)
d = np.sort(d)
d = d[::2]

# calculate the kde for distances
increment = np.round(fitlength / len(d))
if increment == 0:
    increment = 1
calculation_points = np.arange(
    0, fitlength + 1.0, increment
)

churchman = plotting.estimate_rpd_churchman_1d

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
_, a2 = np.histogram(d, bins=2000)
bin_centres = (a2[:-1] + a2[1:]) / 2
bin_width = a2[1] - a2[0]

# generate the bg
bg = 0 * x_expt
sigma = (np.sqrt(2) * loc_precision)
for i, mu in enumerate(bin_centres):

    churchman_term = (
                np.sqrt(2 / np.pi)
                * 1.0
                / sigma
                * np.exp(-(mu**2 + x_expt**2) / (2 * sigma**2))
                * np.cosh(mu * x_expt / sigma**2)
            )
    frequency_term = 2 * (1/length) * (1 - mu/length)
    frequency_term =  frequency_term * bin_width * len(d)

    bg += churchman_term * frequency_term

# normalise the data
if normalise:
    y_expt /= x_expt
    bg /= x_expt

# plot the data
plt.scatter(x_expt, y_expt, label="distances")
plt.plot(x_expt, bg, label="bg")
plt.legend()
plt.savefig("sandpit/1d_test_kde.svg")
plt.close()

# ----- foo
#pdf = 2 * (1/length) * (1 - bin_centres/length)
#bin_width = a2[1] - a2[0]
#y =  pdf * bin_width * len(d)
#plt.hist(d.flatten(), histtype="step", bins=100)
#plt.hist(np.repeat(bin_centres, y.astype(int)), histtype="step", bins=100)
#plt.show()