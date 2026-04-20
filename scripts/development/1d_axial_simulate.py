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
            * (np.exp(-(mu**2 + d**2) / (2 * sigma**2)) * np.cosh(mu * d / sigma**2))
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


# generate random localisations in 1d
xy1 = (0.,0.)
xy2 = (0.,6.)
xy3 = (26.,26.)
xy4 = (26.,32.)

xy = np.stack([xy1, xy2, xy3, xy4])

print(xy.shape)

for i in range(10):
    i += 1
    xy1 = xy[0] + np.array([0.0, i*18.0])
    xy2 = xy[1] + np.array([0.0, i*18.0])
    xy3 = xy[2] + np.array([0.0, i*18.0])
    xy4 = xy[3] + np.array([0.0, i*18.0])

    xy_ = np.stack([xy1, xy2, xy3, xy4])
    xy = np.concatenate([xy, xy_], axis=0)

fig, ax = plt.subplots(figsize=(10,10))
plt.scatter(xy[:,0], xy[:,1])
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.savefig("sandpit/alpha_actinin_pattern.svg")
plt.close()


# params for generating kde
loc_precision = 3.0
fitlength = 100.0
normalise = False

# calculate the distances in 1d
d = getdistances(xy, 500.0, verbose=False)[1]

# calculate the kde for distances
increment = np.round(fitlength / len(d))
if increment == 0:
    increment = 1
calculation_points = np.arange(0, fitlength + 1.0, increment)

churchman = plotting.estimate_rpd_churchman_1d

d = d[:,1]
d = abs(d)
d = np.sort(d)
d = d[::2]

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

# normalise the data
if normalise:
    y_expt /= x_expt

# plot the data
plt.scatter(x_expt, y_expt, label="distances")
plt.legend()
plt.show()
plt.savefig("sandpit/alpha_actinin_sep.svg")
plt.close()

# ----- foo
# pdf = 2 * (1/length) * (1 - bin_centres/length)
# bin_width = a2[1] - a2[0]
# y =  pdf * bin_width * len(d)
# plt.hist(d.flatten(), histtype="step", bins=100)
# plt.hist(np.repeat(bin_centres, y.astype(int)), histtype="step", bins=100)
# plt.show()
