from typing import Tuple

import matplotlib

default_color_cycler = [c['color'] for c in matplotlib.rcParams['axes.prop_cycle']]

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from matplotlib.widgets import Slider, Button, RadioButtons

from fpm_analysis.simulate_phase import SimulatedPhaseResult, glass_BK7, glass_SF10, glass_BAF10

SIZE = 128
MATERIALS = [glass_BK7, glass_SF10, glass_BAF10]
STEPS = 32
DH_MAX = 1.0
OFFSET = 0.0
NOISE = 0.1
NOISE_STD = 1.0
NOISE_CENTER = 0.0

step_size = SIZE // STEPS
dh = DH_MAX / STEPS

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

simulated_result = SimulatedPhaseResult(shape=(SIZE, SIZE))

simulated_result.reset_sim()
heights_mask, materials_mask = simulated_result.make_ladder(materials=MATERIALS,
                                                            n_steps=STEPS,
                                                            dh=dh,
                                                            offset=OFFSET)
simulated_result.add_random_normal_noise(NOISE)


def exp_phase_shift(color: str = 'r', shift: float = 0.0, factor: float = 60):
    """ Adds phase and returns unwrapped the selected color """
    return np.exp((simulated_result.unwrapped[color] + shift) / factor)


def phase_shift(color: str = 'r', shift: float = 0.0):
    """ Adds phase and returns unwrapped the selected color """
    return simulated_result.unwrapped[color] + shift


r_shift = 0
g_shift = 0
b_shift = 0

re = exp_phase_shift('r', r_shift)
ge = exp_phase_shift('g', b_shift)
be = exp_phase_shift('b', g_shift)

r = phase_shift('r', r_shift)
g = phase_shift('g', b_shift)
b = phase_shift('b', g_shift)

x = re / ge
x_label = r'$e^{\phi_r} / e^{\phi_g}$'
y = b / g
y_label = r'$\phi_b / \phi_g $'

grid = plt.GridSpec(4, 3, wspace=0.4, hspace=0)
ax1: plt.Axes = plt.subplot(grid[0, 0])
plt.setp(ax1.get_xticklabels(), visible=False)
plt.setp(ax1.get_yticklabels(), visible=False)
ax2: plt.Axes = plt.subplot(grid[1, 0], sharex=ax1)
plt.setp(ax2.get_xticklabels(), visible=False)
plt.setp(ax2.get_yticklabels(), visible=False)
ax3: plt.Axes = plt.subplot(grid[2, 0], sharex=ax1)
plt.setp(ax3.get_xticklabels(), visible=False)
plt.setp(ax3.get_yticklabels(), visible=False)
ax4: plt.Axes = plt.subplot(grid[:, 1:])

r1 = ax1.imshow(re[0:SIZE // 2, :], cmap='hot')
plt.colorbar(r1, ax=ax1, shrink=0.66)
ax1.set_ylabel('Phase\n(red)', fontsize=14)
r2 = ax2.imshow(materials_mask[0:SIZE // 2, :], cmap=colors.ListedColormap(default_color_cycler[0:len(MATERIALS)]))
cbar = plt.colorbar(r2, ax=ax2, shrink=0.66, ticks=[i for i in range(len(MATERIALS))])
cbar.ax.set_yticklabels([m["name"] for m in MATERIALS])
ax2.set_ylabel('Material\n', fontsize=14)
r3 = ax3.imshow(heights_mask[0:SIZE // 2, :], cmap='Blues')
plt.colorbar(r3, ax=ax3, shrink=0.66)
ax3.set_ylabel(f'Heights\n(um)', fontsize=14)

for i, m in enumerate(MATERIALS):
    ax4.plot(x[materials_mask == i].flatten(), y[materials_mask == i].flatten(),
             ls='', marker='.', label=f'{m["name"]}')
ax4.set_xlabel(xlabel=x_label, fontsize=14)
ax4.set_ylabel(ylabel=y_label, fontsize=14)

plt.legend()
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()

# t = np.arange(0.0, 1.0, 0.001)
# a0 = 5
# f0 = 3
# delta_f = 5.0
# s = a0 * np.sin(2 * np.pi * f0 * t)
# l, = plt.plot(t, s, lw=2)
#
# ax_controls: plt.Axes = plt.subplot2grid((8, 3), (7,0), )
# ax_controls.clear()
# sfreq = Slider(ax_controls, 'Freq', 0.1, 30.0, valinit=f0, valstep=delta_f)


plt.show()
