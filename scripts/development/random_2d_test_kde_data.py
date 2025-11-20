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

def foo(xy, loc_precision, fitlength, normalise):
    # scale up data to a square of size
    
    # calculate the distances in 2d 
    d = getdistances(
        xy, 1000.0, verbose=False
    )[1]
    d = (d[:,0] ** 2 + d[:,1] ** 2) ** 0.5

    # calculate the kde for distances
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

    return d, x_expt, y_expt

# generate random localisations in 2d
#x = random.rand(1000)
#y = random.rand(1000)

xs = np.linspace(0, 1, 20)
ys = np.linspace(0, 1, 11)

# scale up data to a square of size
square_size = 100.

xs *= square_size
ys *= square_size
xy = np.array([(x, y) for x in xs for y in ys])
xy_no_bg = xy

# generate random localisations in 1d
k = .3
if k == 0:
    nbg = 0
else:
    nbg = len(xy)/(1/k - 1)
    nbg = int(nbg)
    k = nbg/(nbg + len(xy))

    # generate random localisations in 2d
    xbg = random.rand(nbg)
    ybg = random.rand(nbg)

    # scale up data to a square of size
    square_size = 100.
    xbg *= square_size
    ybg *= square_size
    xybg = np.stack([xbg,ybg], axis=1)
    
    # concat
    xy = np.concatenate((xy, xybg))

plt.scatter(xy[:,0], xy[:,1])
plt.savefig("2d_rand.png")
plt.close()
#plt.show()

# kde params
loc_precision = 3.
fitlength = 80.
normalise = True

d, x_expt, y_expt = foo(xy, loc_precision, fitlength, normalise)
d_no_bg, x_expt_no_bg, y_expt_no_bg = foo(xy_no_bg, loc_precision, fitlength, normalise)
if k != 0:
    d_bg, x_expt_bg, y_expt_bg = foo(xybg, loc_precision, fitlength, normalise)

assert (x_expt == x_expt_no_bg).all()
if k !=0 :
    assert (x_expt == x_expt_bg).all()
# calculate the bg

# generate locations based on distribution of distances
#_, a2 = np.histogram(d, bins=2000)
#bin_centres = (a2[:-1] + a2[1:]) / 2
#bin_width = a2[1] - a2[0]

# generate the bg
if k!= 0:
    bg = 0 * x_expt
    sigma = (np.sqrt(2) * loc_precision)

    bg_x = np.linspace(0,square_size*np.sqrt(2), 500)
    bin_width = bg_x[1] - bg_x[0]
    for i, mu in enumerate(bg_x):

        churchman_term = churchman_term2d(x_expt, mu, sigma)
        frequency_term = (4 * mu/(square_size ** 4)) * ((np.pi / 2) * square_size ** 2 - 2 * square_size * mu + 0.5 * mu ** 2)
        frequency_term =  frequency_term * bin_width * len(d) * k

        bg += churchman_term * frequency_term

# normalise the data
if normalise:
    y_expt /= x_expt
    if k != 0:
        bg /= x_expt
        y_expt_no_bg /= x_expt
        y_expt_bg /= x_expt

# plot the data
plt.scatter(x_expt, y_expt, label="distances")
if k != 0:
    plt.plot(x_expt, bg, label="bg")
    plt.scatter(x_expt, y_expt_no_bg, label="distances_no_bg")
    plt.scatter(x_expt, y_expt - y_expt_no_bg, label="actual y - y with no bg")
    plt.scatter(x_expt, y_expt_no_bg + bg, label="distances_no_bg_plus_bg")
    plt.scatter(x_expt_bg, y_expt_bg, label="actual bg")

plt.legend()
plt.savefig("sandpit/2d_test_kde_data.svg")
plt.close()

#pdf = (4 * bin_centres/(square_size ** 4)) * ((np.pi / 2) * square_size ** 2 - 2 * square_size * bin_centres + 0.5 * bin_centres ** 2)