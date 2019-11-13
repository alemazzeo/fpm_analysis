from typing import Tuple

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from fpm_analysis.simulate_phase import SimulatedPhaseResult, glass_BK7, glass_SF10, glass_BAF10

# Configuration

SIZE = 128
MATERIALS = [glass_BK7, glass_SF10, glass_BAF10]
STEPS = 32
DH_MAX = 1.0
OFFSET = 0.0
NOISE = 0.001
NOISE_STD = 1.0
NOISE_CENTER = 0.0

xc = 'r/g'
yc = 'b/g'

r_shift = 0
g_shift = 0
b_shift = 0

step_size = SIZE // STEPS
dh = DH_MAX / STEPS

# Report

print(f'SIZE: {SIZE}')
print(f'MATERIALS: {",".join([m["name"] for m in MATERIALS])}')
print(f'STEPS: {STEPS}')
print(f'STEP_X_SIZE: {step_size}')
print(f'DH: {dh}')
print(f'OFFSET: {OFFSET}')
print(f'NOISE: Gaussian\n'
      f'       {NOISE} amplitude\n'
      f'       {NOISE_STD} std\n'
      f'       {NOISE_CENTER} center')

# Simulation

simulated_result = SimulatedPhaseResult(shape=(SIZE, SIZE))

simulated_result.reset_sim()
heights_mask, materials_mask = simulated_result.make_ladder(materials=MATERIALS,
                                                            n_steps=STEPS,
                                                            dh=dh,
                                                            offset=OFFSET)

simulated_result.global_r_shift = r_shift
simulated_result.global_b_shift = b_shift
simulated_result.global_g_shift = g_shift

simulated_result.add_random_normal_noise(NOISE)

x, y = simulated_result.get_x_y(xc, yc)
x_label, y_label = simulated_result.get_x_y_labels(xc, yc)

simulated_result.plot_simulation(x=x, y=y, x_label=x_label, y_label=y_label)
plt.hist2d(simulated_result._last_x.flatten(), simulated_result._last_y.flatten(), bins=100)
plt.show()

# N = 100
# fitness = np.zeros((N, N, N), dtype=float)
# r_interval = np.linspace(-100, 100, N)
# g_interval = np.linspace(-100, 100, N)
# b_interval = np.linspace(-100, 100, N)
# for i, rs in enumerate(r_interval):
#     for j, gs in enumerate(g_interval):
#         for k, bs in enumerate(b_interval):
#             fitness[i][j][k] = simulated_result.fitness(rs, gs, bs, xc, yc)


def fitness(phases: Tuple[float, float, float], xc, yc):
    x, y = simulated_result.get_x_y(xc, yc, r_shift=-phases[0], g_shift=-phases[1], b_shift=-phases[2])
    print(phases)
    return simulated_result.fitness(x, y)


res = minimize(fitness, np.array([0, 0, 0]), args=(xc, yc),
               bounds=[(-100, 100), (-100, 100), (-100, 100)])

print(res)
print(r_shift, g_shift, b_shift)

simulated_result.get_x_y(xc, yc, *res.x)
simulated_result.plot_simulation()
plt.hist2d(simulated_result._last_x.flatten(), simulated_result._last_y.flatten(), bins=100)
plt.show()
