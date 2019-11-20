from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from fpm_analysis.phase_result import FPMResult, latex_phi

data = '/home/alejandro/git/fpm_analysis/hdf5_data/rock1'

result = FPMResult()
result.load_hdf5(data)

xc = 're/be'
yc = 'b/r'

x, y = result.get_x_y(xc, yc)
x_label, y_label = result.get_x_y_labels(xc, yc)
result.plot_result(x=result.unwrapped['r'], y=result.unwrapped['b'],
                   x_label=f'${latex_phi["r"]}$', y_label=f'${latex_phi["b"]}$')

plt.plot(result.unwrapped['r'][1100:1150, 400:650].flatten(), result.unwrapped['b'][1100:1150, 400:650].flatten(), 'go', markersize=0.5)
plt.show()
exit()

plt.hist2d(result.unwrapped['r'].flatten(), result.unwrapped['g'].flatten())
# plt.imshow(result.unwrapped['r'] / result.unwrapped['g'])
plt.show()


def fitness(phases: Tuple[float, float, float]):
    return result.fitness(*result.get_x_y(xc, yc, -phases[0], -phases[1], -phases[2]))


limit = (result._unwrapped['r'].min(), result._unwrapped['r'].max())

res = minimize(fitness, np.array([0, 0, 0]),
               bounds=[limit, limit, limit])

print(res)

x, y = result.get_x_y(xc, yc, *res.x)
result.plot_result(x, y)
plt.show()
