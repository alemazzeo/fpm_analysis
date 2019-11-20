import logging
from typing import Tuple

import matplotlib

default_color_cycler = [c['color'] for c in matplotlib.rcParams['axes.prop_cycle']]

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

from fpm_analysis.phase_result import FPMResult, COLORS
from skimage.restoration import unwrap_phase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def constant_index_sphere(shape: tuple, x: float, y: float, radius: float = 1.0,
                          n_index: float = 1.0, wavelength: float = 0.630):
    base = np.zeros(shape)
    xx, yy = np.meshgrid(range(shape[1]), range(shape[0]))
    r = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
    base += (radius - r) * n_index * 2 / wavelength
    mask = r > radius
    base[mask] = 0
    return base


def constant_index_step_x(shape: Tuple[int, int], interval: Tuple[int, int], height: float = 1.0,
                          n_index: float = 1.0, wavelength: float = 0.630):
    step = np.zeros(shape)
    x0 = interval[0]
    x1 = interval[1]
    step[:, x0:x1] = 2 * np.pi * n_index * height / wavelength
    return step


glass_SF10 = {'name': 'SF10',
              'r': 1.7234,  # 630 nm
              'g': 1.7390,  # 520 nm
              'b': 1.7507}  # 470 nm

glass_BAF10 = {'name': 'BAF10',
               'r': 1.6672,  # 630 nm
               'g': 1.6760,  # 520 nm
               'b': 1.6823}  # 470 nm

glass_BK7 = {'name': 'BK7',
             'r': 1.5152,  # 630 nm
             'g': 1.5202,  # 520 nm
             'b': 1.5236}  # 470 nm


class SimulatedPhaseResult(FPMResult):
    def __init__(self, shape=(128, 128), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._shape = shape
        self._unwrapped = {color: np.zeros(shape, dtype=float) for color in COLORS}
        self._layer_sim = {color: np.zeros(shape, dtype=float) for color in COLORS}
        self._layer_noise = {color: np.zeros(shape, dtype=float) for color in COLORS}
        self._phase_shifts = {color: 0.0 for color in COLORS}

        self._materials_list = []
        self._materials_mask = np.zeros(shape, dtype=float)
        self._heights_mask = np.zeros(shape, dtype=float)

    def update(self):
        self._unwrapped = {color: (self._layer_sim[color]
                                   + self._layer_noise[color]
                                   + self._phase_shifts[color])
                           for color in COLORS}
        self._phase = {c: w - np.floor(w / (2 * np.pi)) * 2 * np.pi
                       for c, w in self._unwrapped.items()}
        # self._unwrapped = {c: unwrap_phase(w, wrap_around=True) for c, w in self._phase.items()}

    def set_phase_shift(self, **kwargs):
        for color, value in kwargs.items():
            self._phase_shifts[color] = value
        self.update()

    def reset_sim(self):
        self._layer_sim = {color: np.zeros(self._shape, dtype=float) for color in COLORS}
        self._layer_noise = {color: np.zeros(self._shape, dtype=float) for color in COLORS}

    def add_sphere(self, x: float, y: float, radius: float = 1.0, material=None):
        if material is None:
            material = glass_SF10
        for c, w in COLORS.items():
            self._layer_sim[c] += constant_index_sphere(shape=self._shape, x=x, y=y, radius=radius,
                                                        n_index=material[c], wavelength=w)
        self.update()

    def add_random_spheres(self, radius: float = 1.0, n: int = 1, material=None):
        positions = np.random.rand(n, 2)
        positions[:, 0] *= self._shape[1]
        positions[:, 1] *= self._shape[0]
        for pos in positions:
            self.add_sphere(x=pos[0], y=pos[1], radius=radius, material=material)

    def add_step_x(self, interval: Tuple[int, int], height: float = 1.0, material=None):
        if material is None:
            material = glass_SF10
        for c, w in COLORS.items():
            self._layer_sim[c] += constant_index_step_x(shape=self._shape, height=height, interval=interval,
                                                        n_index=material[c], wavelength=w)
        self.update()

    def make_ladder(self, materials=None, n_steps=4, dh=10.0, offset=0.0):
        if materials is None:
            materials = [glass_SF10, ]
        self._materials_list = materials
        n = len(materials)
        dx = self._shape[1] // n_steps
        interval = [(dx * i, dx * (i + 1)) for i in range(n_steps)]
        interval[-1] = (interval[-1][0], self._shape[1])
        materials = [materials[i % n] for i in range(n_steps)]
        params = [{'interval': interval[i],
                   'height': dh * (i+1) + offset,
                   'material': materials[i]} for i in range(n_steps)]
        heights_mask = np.zeros(self._shape, dtype=float)
        materials_mask = np.zeros(self._shape, dtype=int)
        for i, param in enumerate(params):
            x0 = param['interval'][0]
            x1 = param['interval'][1]
            heights_mask[:, x0:x1] = param['height']
            materials_mask[:, x0:x1] = materials.index(param['material'])
            self.add_step_x(**param)
        self._materials_mask = materials_mask
        self._heights_mask = heights_mask

        return heights_mask, materials_mask

    def add_random_normal_noise(self, amplitude: float = 0.0, std: float = 1.0, center: float = 0.0):
        for c, w in COLORS.items():
            self._layer_sim[c] += np.random.normal(loc=center, scale=std, size=self._shape) * amplitude
        self.update()

    def add_normal_noise(self, amplitude: float = 0.0, std: float = 1.0, center: float = 0.0):
        noise = np.random.normal(loc=center, scale=std, size=self._shape) * amplitude
        for c, w in COLORS.items():
            self._layer_sim[c] += noise
        self.update()

    def plot_scatter(self, x=None, y=None, x_label=None, y_label=None,
                     mask=None, mask_names=None, ax=None, pick_event=None):
        if mask is None:
            mask = self._materials_mask

        if mask_names is None:
            mask_names = [x['name'] for x in self._materials_list]

        super().plot_scatter(x, y, x_label, y_label, mask, mask_names, ax, pick_event)

    def plot_simulation(self, x=None, y=None, x_label=None, y_label=None):

        grid = plt.GridSpec(4, 3, wspace=0.4, hspace=0)
        ax1: plt.Axes = plt.subplot(grid[0:2, 0])
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax1.get_yticklabels(), visible=False)
        ax2: plt.Axes = plt.subplot(grid[2, 0], sharex=ax1)
        plt.setp(ax2.get_xticklabels(), visible=False)
        plt.setp(ax2.get_yticklabels(), visible=False)
        ax3: plt.Axes = plt.subplot(grid[3, 0], sharex=ax1)
        plt.setp(ax3.get_xticklabels(), visible=False)
        plt.setp(ax3.get_yticklabels(), visible=False)
        ax4: plt.Axes = plt.subplot(grid[:, 1:])

        r1 = ax1.imshow(self.unwrapped['r'], cmap='hot')
        selected, = ax1.plot([], [], ls='', marker='.', alpha=0.5)
        plt.colorbar(r1, ax=ax1, shrink=0.66)
        ax1.set_ylabel('Phase\n(red)', fontsize=14)
        r2 = ax2.imshow(self._materials_mask[0:self._shape[0] // 2, :],
                        cmap=colors.ListedColormap(default_color_cycler[0:len(self._materials_list)]))
        cbar = plt.colorbar(r2, ax=ax2, shrink=0.66, ticks=[i for i in range(len(self._materials_list))])
        cbar.ax.set_yticklabels([m["name"] for m in self._materials_list])
        ax2.set_ylabel('Material\n', fontsize=14)
        r3 = ax3.imshow(self._heights_mask[0:self._shape[0] // 2, :], cmap='Blues')
        plt.colorbar(r3, ax=ax3, shrink=0.66)
        ax3.set_ylabel(f'Heights\n(um)', fontsize=14)

        def on_pick(event):
            yp, xp = np.unravel_index(event.real_id, x.shape)
            selected.set_data(xp, yp)
            plt.draw()

        self.plot_scatter(x, y, x_label, y_label, ax=ax4, pick_event=on_pick)

        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.show()


if __name__ == '__main__':
    simulated_result = SimulatedPhaseResult(shape=(128, 128))  # random_phase=['r', 'b'])
    # simulated_sample.add_random_spheres(radius=8, n=20, material=glass_SF10)
    # simulated_result.add_random_spheres(radius=16, n=20, material=glass_BK7)
    # simulated_result.add_random_spheres(radius=32, n=5, material=glass_SF10)
    # simulated_result.add_step_x(interval=(0, 32), height=10.0, material=glass_SF10)
    # simulated_result.add_step_x(interval=(32, 64), height=20.0, material=glass_BK7)
    # simulated_result.add_step_x(interval=(64, 96), height=30.0, material=glass_SF10)
    # simulated_result.add_step_x(interval=(96, 128), height=40.0, material=glass_BK7)
    fig, ax = plt.subplots(1, 2)
    simulated_result.reset_sim()
    simulated_result.make_ladder(same_material=False, n_steps=4, dh=10.0, offset=0.0)
    simulated_result.add_random_normal_noise(1.0)
    simulated_result.plot_phase(color='r', ax=ax[0])
    simulated_result.plot_quotient('r/b', ax=ax[1])
    plt.show()
    # simulated_result.plot_quotient_2d(x='r/g', y='b/g')
