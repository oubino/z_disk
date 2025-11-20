import numpy as np
from numpy import random
from perpl.relative_positions import getdistances
from perpl.io import plotting
import matplotlib.pyplot as plt

def churchman_term1d(d, mu, sigma):
    if ((mu * d / sigma**2) < 500).all():
        p = (
            np.sqrt(2 / np.pi)
            * 1.0
            / sigma
            * (
                np.exp(-(mu**2 + d**2) / (2 * sigma**2))
                * np.cosh(mu * d / sigma**2)
            )
        )
    else:
        p = (
            (1.0 / 2.0)
            * np.sqrt(2 / np.pi)
            * 1.0
            / sigma
            * (np.exp(-((d - mu) ** 2) / (2 * sigma**2)))
        )
    return p

def foo(x, length, loc_precision, fitlength, normalise):
    # scale up data to a line of length...
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

    return d, x_expt, y_expt

# generate pattern of localisations
#x = np.repeat([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], 1)
x = np.arange(0,1,0.05)
x_no_bg = x

# generate random localisations in 1d
k = 0.5
if k == 0:
    nbg = 0
else:
    nbg = len(x)/(1/k - 1)
    nbg = int(nbg)
    k = nbg/(nbg + len(x))
    xbg = random.rand(nbg)

    # concat
    x = np.concatenate((x, xbg))

# scale up data to a line of length...
length = 200.

# params for generating kde
loc_precision = 3.
fitlength = 30.
normalise = False

d, x_expt, y_expt = foo(x, length, loc_precision, fitlength, normalise)
d_no_bg, x_expt_no_bg, y_expt_no_bg = foo(x_no_bg, length, loc_precision, fitlength, normalise)
if k != 0:
    d_bg, x_expt_bg, y_expt_bg = foo(xbg, length, loc_precision, fitlength, normalise)

# calculate the bg

# generate locations based on distribution of distances
#_, a2 = np.histogram(d, bins=2000)
#bin_centres = (a2[:-1] + a2[1:]) / 2
#bin_width = a2[1] - a2[0]

# generate the bg
if k!= 0:
    bg = 0 * x_expt
    sigma = (np.sqrt(2) * loc_precision)
    bg_x = np.linspace(0, length, 500)
    bin_width = bg_x[1] - bg_x[0]
    for i, mu in enumerate(bg_x):

        churchman_term = churchman_term1d(x_expt, mu, sigma)
        frequency_term = 2 * (1/length) * (1 - mu/length)
        frequency_term =  frequency_term * bin_width * len(d) * k

        bg += churchman_term * frequency_term

# normalise the data
if normalise:
    y_expt /= x_expt
    if k != 0:
        bg /= x_expt
        y_expt_no_bg /= x_expt
        y_expt_bg /= x_expt_bg

# plot the data
plt.scatter(x_expt, y_expt, label="distances")
if k != 0:
    plt.plot(x_expt, bg, label="bg model")
    plt.scatter(x_expt_no_bg, y_expt_no_bg, label="distances_no_bg")
    plt.scatter(x_expt_no_bg, y_expt - y_expt_no_bg, label="actual y - y with no bg")
    plt.scatter(x_expt_no_bg, y_expt_no_bg + bg, label="distances_no_bg_plus_bg")
    plt.scatter(x_expt_bg, y_expt_bg, label="actual bg")
plt.legend()
plt.savefig("sandpit/1d_test_kde_data.svg")
plt.close()

# ----- foo
#pdf = 2 * (1/length) * (1 - bin_centres/length)
#bin_width = a2[1] - a2[0]
#y =  pdf * bin_width * len(d)
#plt.hist(d.flatten(), histtype="step", bins=100)
#plt.hist(np.repeat(bin_centres, y.astype(int)), histtype="step", bins=100)
#plt.show()