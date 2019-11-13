import logging
import pathlib
from functools import wraps
from typing import Dict, Tuple

import click
import h5py
import matplotlib.pyplot as plt
import numpy as np
from skimage.restoration import unwrap_phase

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
            self._unwrapped[color] = unwrap_phase(self._phase[color])
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

    def _get_phase(self, a, r_shift, b_shift, g_shift):
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

    def fitness(self, x=None, y=None):
        if x is None:
            x = self._last_x
        if y is None:
            y = self._last_y

        hist2d = np.histogram2d(x=x.flatten(), y=y.flatten(), bins=100)[0]
        return np.std(hist2d.flatten())


    def _fitness(self, x=None, y=None):
        if x is None:
            x = self._last_x
        if y is None:
            y = self._last_y

        x_mean = np.mean(x)
        y_mean = np.mean(y)
        return np.sum((x - x_mean) ** 2) + np.sum((y - y_mean) ** 2)


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
