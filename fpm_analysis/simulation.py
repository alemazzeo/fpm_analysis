from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from fpm_analysis.simulate_phase import SimulatedPhaseResult, glass_BK7, glass_SF10, glass_BAF10

# Configuration

SIZE = 128
MATERIALS = [glass_BK7, glass_SF10, glass_BAF10]
STEPS = 32
DH_MAX = 10.0
OFFSET = -5.0
NOISE = 0.01
NOISE_STD = 1.0
NOISE_CENTER = 0.0

xc = 're/be'
yc = 'r/b'

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

# r = 1
# N = 101
# stats = np.zeros([4, N, N, 2], dtype=float)
#
# phases_r = np.linspace(-r, r, N)
# phases_b = np.linspace(-r, r, N)
#
# for i, r in enumerate(phases_r):
#     for j, b in enumerate(phases_b):
#         x, y = simulated_result.get_x_y(xc, yc, r_shift=r, b_shift=b)
#         for p, var in enumerate([x, y]):
#             stats[0, i, j, p] = np.mean(var)
#             stats[1, i, j, p] = np.std(var)
#             stats[2, i, j, p] = np.max(var)
#             stats[3, i, j, p] = np.min(var)
#
# for i, title in enumerate(['mean', 'std', 'max', 'min']):
#     fig, ax = plt.subplots(1, 3)
#     ax[0].imshow(stats[i, :, :, 0])
#     ax[1].imshow(stats[i, :, :, 1])
#     ax[2].imshow(stats[i, :, :, 1]**2 + stats[i, :, :, 0]**2)
#     ax[2].plot(*np.unravel_index(np.argmax(stats[i, :, :, 1]), (N, N)), 'go')
#     plt.title(title)
#     plt.show()
#
# np.savez('stats.npz', mean=stats[0], std=stats[1], max=stats[2], min=stats[3])
# exit()

simulated_result.plot_simulation(x, y)

plt.plot(simulated_result.unwrapped['r'].flatten(), simulated_result.unwrapped['b'].flatten(), 'go')
plt.show()

def optimize(phases, c1='r', c2='g', fake_c1_shift = 13, fake_c2_shift = 7):
    plt.ion()
    ax: Tuple[plt.Axes, plt.Axes]
    fig, ax = plt.subplots(1, 3)
    results = []
    d1 = phases[c1] - fake_c1_shift
    d2 = phases[c2] - fake_c2_shift

    def _fitness(shift, x, y):
        r1 = x + shift
        r2 = y
        fitness = np.abs(np.mean(r1.flatten()[np.abs(r2.flatten()) < 1.0]))
        results.append(fitness)
        r = r1 / r2
        ax[0].clear()
        ax[0].imshow(r)
        ax[1].clear()
        ax[1].hist(r.flatten(), bins=20)
        ax[2].clear()
        ax[2].plot(results if len(results) < 100 else results[-100:])
        plt.pause(0.1)
        print(f'Fitness: {fitness} - {c1.upper()}: {shift[0]} - {c2.upper()}: {shift[1]}')
        return fitness

    res = minimize(_fitness, np.array([0.0, 0.0]))
    plt.close(fig)
    plt.ioff()
    fig, ax = plt.subplots(1, 3)
    _fitness(res.x)
    plt.show()
    return res


print(optimize(simulated_result.unwrapped, 'r', 'b'))
exit()
# simulated_result.plot_simulation(x=x, y=y, x_label=x_label, y_label=y_label)

# N = 100
# fitness = np.zeros((N, N, N), dtype=float)
# r_interval = np.linspace(-100, 100, N)
# g_interval = np.linspace(-100, 100, N)
# b_interval = np.linspace(-100, 100, N)
# for i, rs in enumerate(r_interval):
#     for j, gs in enumerate(g_interval):
#         for k, bs in enumerate(b_interval):
#             fitness[i][j][k] = simulated_result.fitness(rs, gs, bs, xc, yc)

#
# def fitness(phases: Tuple[float, float, float], xc, yc):
#     x, y = simulated_result.get_x_y(xc, yc, r_shift=-phases[0], g_shift=-phases[1], b_shift=-phases[2])
#     print(phases)
#     return simulated_result.fitness(x, y)
#
#
# limit = (simulated_result._unwrapped['r'].min(), simulated_result._unwrapped['r'].max())
#
# res = minimize(fitness, np.array([20, 30, 50]), args=(xc, yc))
# # bounds=[limit, limit, limit])
#
# print(res)
#
# x, y = simulated_result.get_x_y(xc, yc, *res.x)
# simulated_result.plot_simulation(x, y)
# plt.show()
