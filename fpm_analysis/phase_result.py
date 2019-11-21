import logging
import pathlib
from functools import wraps
from typing import Dict, Tuple

import click
import h5py
import matplotlib.pyplot as plt
import numpy as np
from skimage.restoration import unwrap_phase

plt.rcParams['toolbar'] = 'toolmanager'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

COLORS = {'r': 0.630, 'g': 0.520, 'b': 0.480}

latex_phi = {'re': r'e^{\phi_r}',
             'ge': r'e^{\phi_g}',
             'be': r'e^{\phi_b}',
             'r': r'\phi_r',
             'g': r'\phi_g',
             'b': r'\phi_b'}


def optional_axes(rows=1, columns=1):
    def decorator(func):
        @wraps(func)
        def wrapped(self, ax=None, force_show=False, subplot_kw=None, fig_kw=None, *args, **kwargs):
            if ax is None:
                if fig_kw is None:
                    fig_kw = {}
                fig, ax = plt.subplots(rows, columns, subplot_kw=subplot_kw, **fig_kw)
                force_show = True
            r = func(self, ax=ax, *args, **kwargs)
            if force_show is True:
                plt.show()
            return r

        return wrapped

    return decorator


def apply_roi(data, roi: Tuple[Tuple[int, int], Tuple[int, int]]):
    if roi is not None:
        if isinstance(data, np.ndarray):
            data = data[roi[1][0]:roi[1][1], roi[0][0]:roi[0][1]]
        elif isinstance(data, (tuple, list)):
            data = [d[roi[1][0]:roi[1][1], roi[0][0]:roi[0][1]] for d in data]
        elif isinstance(data, dict):
            data = {c: d[roi[1][0]:roi[1][1], roi[0][0]:roi[0][1]] for c, d in data.items()}
        else:
            raise TypeError
    return data


class FPMResult:
    def __init__(self, *args, **kwargs):
        self._sample: Dict[str, np.ndarray] = {}
        self._amplitude: Dict[str, np.ndarray] = {}
        self._phase: Dict[str, np.ndarray] = {}
        self._unwrapped: Dict[str, np.ndarray] = {}
        self._diff: Dict[str, np.ndarray] = {}
        self._colors = None
        self._unwrap = True
        self.global_r_shift = 0.0
        self.global_g_shift = 0.0
        self.global_b_shift = 0.0
        self._last_x = None
        self._last_y = None
        self._last_x_label = None
        self._last_y_label = None

    def load_hdf5(self, data):
        data = pathlib.Path(data)
        files = [str(file.resolve()) for file in sorted(data.glob('*.h5py'))]
        str_files = "\n".join(files)
        logger.info(f'Files founded {str_files}')
        for file in files:
            color = file.split('_')[-1].split('.')[0]
            logger.info(f'Open file {file} for color "{color}"')
            f = h5py.File(file, 'r')
            positions = sorted(f.keys())
            # TODO: Join multiple positions
            self._sample[color] = f[positions[0]]
            self._phase[color] = np.angle(self._sample[color])
            # self._phase[color] -= self._phase[color].min()
            self._amplitude[color] = np.absolute(self._sample[color])
            self._unwrapped[color] = unwrap_phase(self._phase[color], wrap_around=True)
            # self._unwrapped[color] -= self._unwrapped[color].min()
            logger.info('Sample added')

    @property
    def unwrapped(self):
        return self._unwrapped

    @property
    def phase(self):
        return self._phase

    def get_quotient(self, quotient: str = 'r/b', unwrap=True, roi=None, mean=False):
        phase = self._unwrapped if unwrap is True else self._phase
        phase = apply_roi(phase, roi)
        c_a = quotient.split('/')[0]
        c_b = quotient.split('/')[1]
        w = COLORS[c_a] / COLORS[c_b]
        if mean is True:
            return (phase[c_a] - np.mean(phase[c_a])) / (phase[c_b] - np.mean(phase[c_a])) * w
        else:
            return phase[c_a] / phase[c_b] * w

    @optional_axes()
    def plot_amplitude(self, ax, color, roi=None):
        ax.imshow(apply_roi(self._amplitude[color], roi))

    @optional_axes()
    def plot_phase(self, ax, color, roi=None):
        ax.imshow(apply_roi(self._phase[color], roi))

    @optional_axes()
    def plot_unwrapped(self, ax, color, roi=None):
        ax.imshow(apply_roi(self._unwrapped[color], roi))

    @optional_axes()
    def plot_quotient(self, ax, q='r/g', roi=None, unwrap=True):
        ax.imshow(self.get_quotient(quotient=q, unwrap=unwrap, roi=roi))

    @optional_axes()
    def plot_quotient_2d(self, ax, x: str = 'r/g', y: str = 'g/b', roi=None, unwrap=True):
        xc = self.get_quotient(x, roi=roi, unwrap=unwrap)
        yc = self.get_quotient(y, roi=roi, unwrap=unwrap)
        ax.plot(xc.flatten(), yc.flatten(), 'ro', markersize=0.1)

    #
    #

    def exp_phase_shift(self, color: str = 'r', shift: float = 0.0, factor: float = 60):
        """ Adds phase and returns unwrapped the selected color """
        return np.exp((self.unwrapped[color] + shift) / factor)

    def phase_shift(self, color: str = 'r', shift: float = 0.0):
        """ Adds phase and returns unwrapped the selected color """
        return self.unwrapped[color] + shift

    def _get_phase(self, a, r_shift, g_shift, b_shift):
        if 'e' in a:
            return self.exp_phase_shift(a[0], locals()[f'{a[0]}_shift'])
        else:
            return self.phase_shift(a[0], locals()[f'{a[0]}_shift'])

    def get_x_y(self, xc: str = 're/ge', yc: str = 'b/g', r_shift: float = 0, g_shift: float = 0, b_shift: float = 0):
        """ Gets x-y curves from selected quotients """
        x0, x1 = xc.replace(' ', '').split("/")
        y0, y1 = yc.replace(' ', '').split("/")

        rs = self.global_r_shift + r_shift
        gs = self.global_g_shift + g_shift
        bs = self.global_b_shift + b_shift

        a = self._get_phase(x0, rs, gs, bs)
        b = self._get_phase(x1, rs, gs, bs)
        c = self._get_phase(y0, rs, gs, bs)
        d = self._get_phase(y1, rs, gs, bs)

        self._last_x = a / b
        self._last_y = c / d

        return self._last_x, self._last_y

    def get_x_y_labels(self, x: str = 're/ge', y: str = 'b/g'):
        """ Gets labels for x and y curves"""
        x0, x1 = x.replace(' ', '').split("/")
        y0, y1 = y.replace(' ', '').split("/")

        x_label = f'${latex_phi[x0]} / {latex_phi[x1]}$'
        y_label = f'${latex_phi[y0]} / {latex_phi[y1]}$'

        self._last_x_label = x_label
        self._last_y_label = y_label

        return self._last_x_label, self._last_y_label

    def plot_scatter(self, x=None, y=None, x_label=None, y_label=None,
                     mask=None, mask_names=None, ax=None, pick_event=None, alpha=0.005):

        if x is None:
            x = self._last_x
        if y is None:
            y = self._last_y
        if x_label is None:
            x_label = self._last_x_label
        if y_label is None:
            y_label = self._last_y_label

        if ax is None:
            fig: plt.Figure
            ax: plt.Axes
            fig, ax = plt.subplots(1, 1)
        fig = ax.get_figure()

        x = x.flatten()
        y = y.flatten()
        ax.data_index = np.arange(len(x))
        ax.curves = []
        if mask is None:
            ax.mask = np.zeros_like(x, dtype=int)
            ax.mask_names = ['All data']
            ax.curves.append(ax.plot(x.flatten(), y.flatten(), ls='', marker='.',
                                     alpha=alpha, picker=50)[0])
        else:
            ax.mask = np.asarray(mask.flatten(), dtype=int)
            ax.mask_names = mask_names
            if mask_names is not None:
                for i, m in enumerate(ax.mask_names):
                    ax.curves.append(ax.plot(x[ax.mask == i], y[ax.mask  == i], ls='', marker='.',
                                             alpha=alpha, label=f'{m}', picker=50)[0])

            else:
                raise ValueError

        ax.selected = ax.plot([], [], ls='', marker='.', alpha=0.5, label='Selected')[0]
        ax.selected_map = None
        ax.set_xlabel(xlabel=x_label, fontsize=14)
        ax.set_ylabel(ylabel=y_label, fontsize=14)
        ax.set_title(f'Fitness: {self.fitness():.2e}')
        ax.legend()

        def on_pick(event):
            if event.artist in ax.curves:
                event.real_id = np.arange(ax.mask.size)[ax.mask == ax.curves.index(event.artist)][event.ind]
            elif event.artist is ax.selected:
                event.real_id = ax.selected_map[event.ind]
            else:
                return

            if pick_event is not None:
                pick_event(event)

        def toggle_main_curves(axes, visibility: bool):
            for c in axes.curves:
                c.set_visible(visibility)

        ax.toggle_main_curves = toggle_main_curves

        def reset(axes):
            axes.toggle_main_curves(True)
            axes.selected.set_data([], [])

        ax.reset = reset

        fig.canvas.mpl_connect('pick_event', on_pick)

    def plot_image(self, phase, color='r', cmap='gray', ax=None, click_event=None):
        if ax is None:
            fig: plt.Figure
            ax: plt.Axes
            fig, ax = plt.subplots(1, 1)
        fig = ax.get_figure()

        r = ax.imshow(phase[color], cmap=cmap)
        p, = ax.plot([], [], 'o', ms=14, markerfacecolor="None",
                     markeredgecolor='black', markeredgewidth=1)
        plt.colorbar(r, ax=ax, shrink=0.66)
        ax.point = p
        ax.selected = ax.plot([], [], ls='', marker='.', alpha=0.5, color=color, label='Selected')[0]
        ax.set_ylabel(f'Phase ({color})', fontsize=14)
        ax.color = color
        ax.phase = phase
        xx, yy = np.meshgrid(np.arange(phase[color].shape[0]), np.arange(phase[color].shape[1]))
        ax.xx = xx
        ax.yy = yy
        ax.mask = np.zeros_like(phase[color], dtype=bool)

        def on_click(event):
            if click_event is not None:
                if event.inaxes is ax:
                    click_event(event)

        def reset(axes):
            axes.selected.set_data([], [])
            axes.mask = np.zeros_like(phase[color], dtype=bool)

        ax.reset = reset
        fig.canvas.mpl_connect('button_press_event', on_click)

    def plot_result(self, x=None, y=None, x_label=None, y_label=None):

        grid = plt.GridSpec(3, 3, wspace=0.4, hspace=0)
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

        def on_pick(event):
            yp, xp = np.unravel_index(event.real_id, x.shape)
            ax2.selected.set_data(xp, yp)
            plt.draw()

        def on_click(event):
            ax = event.inaxes
            ax.mask[:] = False
            ax.mask[((ax.yy - event.ydata) ** 2 + (ax.xx - event.xdata) ** 2) < 100] = True
            ax.point.set_data(event.xdata, event.ydata)
            ax4.selected.set_data(x[ax.mask], y[ax.mask])
            ax4.selected.selected_map = ax4.data_index[ax.mask.flatten()]
            plt.draw()

        self.plot_image(self.unwrapped, 'r', cmap='Reds', ax=ax1)
        self.plot_image(self.unwrapped, 'g', cmap='Greens', ax=ax2, click_event=on_click)
        self.plot_image(self.unwrapped, 'b', cmap='Blues', ax=ax3)
        self.plot_scatter(x, y, x_label, y_label, ax=ax4, pick_event=on_pick)

        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.show()

    def _fitness(self, x=None, y=None):
        if x is None:
            x = self._last_x
        if y is None:
            y = self._last_y

        hist2d = np.histogram2d(x=x.flatten(), y=y.flatten(), bins=100)[0]
        return np.std(hist2d.flatten())

    def fitness(self, x=None, y=None):
        if x is None:
            x = self._last_x
        if y is None:
            y = self._last_y

        x_mean = np.mean(x)
        y_mean = np.mean(y)
        return np.sum((x - x_mean) ** 2 + (y - y_mean) ** 2)


@click.command()
@click.option('--unwrap', is_flag=True)
@click.argument('data', type=click.Path(dir_okay=True, file_okay=False, exists=True))
def plot_quotient(data, unwrap):
    result = FPMResult()
    result.load_hdf5(data)
    result.plot_amplitude(color='r')
    result.plot_phase(color='r')
    result.plot_unwrapped(color='r')
    result.plot_quotient_2d(x='r/g', y='b/g', unwrap=unwrap)


if __name__ == '__main__':
    plot_quotient()

    # 3684 5544

    # array([[3, 0],
    #        [6, 7],
    #        [3, 6],
    #        [2, 2]])
