import logging
import pathlib
from typing import Dict

import click
import h5py
import matplotlib.pyplot as plt
import numpy as np
from skimage.restoration import unwrap_phase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

COLORS = {'r': 0.630, 'g': 0.520, 'b': 0.480}


class FPMResult:
    def __init__(self, *args, **kwargs):
        self._sample: Dict[str, np.ndarray] = {}
        self._amplitude: Dict[str, np.ndarray] = {}
        self._phase: Dict[str, np.ndarray] = {}
        self._unwrapped: Dict[str, np.ndarray] = {}
        self._colors = None

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

    def plot(self, color='r', unwrap=True, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 2, sharex='all', sharey='all')
        phase = self._unwrapped if unwrap is True else self._phase
        ax[0].imshow(self._amplitude[color])
        ax[1].imshow(phase[color])
        if ax is None:
            plt.show()

    def plot_amplitude(self, color='r', ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        ax.imshow(self._amplitude[color])
        if ax is None:
            plt.show()

    def plot_phase(self, color='r', unwrap=True, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        phase = self._unwrapped if unwrap is True else self._phase
        ax.imshow(phase[color])
        if ax is None:
            plt.show()

        # plt.hist(phase[color].flatten())
        # plt.show()

    def plot_diff(self, subtract: str = 'r-b', unwrap=True, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        phase = self._unwrapped if unwrap is True else self._phase
        c_a = subtract.split('-')[0]
        c_b = subtract.split('-')[1]
        ax.imshow(np.abs(phase[c_a] - phase[c_b]))
        if ax is None:
            plt.show()

    def get_quotient(self, quotient: str = 'r/b', unwrap=True):
        phase = self._unwrapped if unwrap is True else self._phase
        c_a = quotient.split('/')[0]
        c_b = quotient.split('/')[1]
        w = wx = COLORS[c_a] / COLORS[c_b]
        return phase[c_a] / phase[c_b] * w

    def get_quotient_2d(self, x: str = 'r/g', y: str = 'g/b', unwrap=True):
        xc = self.get_quotient(x, unwrap)
        yc = self.get_quotient(y, unwrap)
        return xc, yc

    def plot_quotient(self, quotient: str = 'r/b', unwrap=True, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        q = self.get_quotient(quotient, unwrap)
        r = ax.imshow(q)
        if ax is None:
            plt.show()
        else:
            return r

    def plot_quotient_2d(self, x: str = 'r/g', y: str = 'g/b', unwrap=True, threshold=None,
                         roi=None, ax=None):
        if roi is None:
            if ax is None:
                fig, ax = plt.subplots(1, 1)
            else:
                if not isinstance(ax, plt.Axes):
                    raise TypeError
        else:
            if ax is None:
                fig, ax = plt.subplots(1, 2, sharex='all', sharey='all')
            else:
                if not all([isinstance(x, plt.Axes) for x in ax]):
                    raise TypeError

        xc, yc = self.get_quotient_2d(x, y, unwrap)
        if threshold is None and roi is None:
            plt.plot(xc.flatten(), yc.flatten(), 'ro', markersize=0.1)
        else:
            if threshold is not None and roi is None:
                mask = self._unwrapped['r'].flatten() > threshold
                _mask = self._unwrapped['r'].flatten() <= threshold

                ax[0].imshow(self._unwrapped['r'])
                ax[1].imshow(self._unwrapped['r'] > threshold)
                plt.tight_layout()
                fig_manager = plt.get_current_fig_manager()
                fig_manager.window.showMaximized()

            elif roi is not None and threshold is None:
                mask = np.zeros_like(self._unwrapped['r'], dtype=bool)
                mask[:] = False
                mask[roi[1][0]:roi[1][1], roi[0][0]:roi[0][1]] = True

                _mask = np.ones_like(self._unwrapped['r'], dtype=bool)
                _mask[:] = True
                _mask[roi[1][0]:roi[1][1], roi[0][0]:roi[0][1]] = False

                fig, ax = plt.subplots(1, 2, sharex='all', sharey='all')
                ax[0].imshow(self._unwrapped['r'])
                ax[1].imshow(mask)
                plt.tight_layout()
                fig_manager = plt.get_current_fig_manager()
                fig_manager.window.showMaximized()

                mask = mask.flatten()
                _mask = _mask.flatten()
            else:
                raise ValueError
            fig, ax = plt.subplots(1, 2, sharex='all', sharey='all')
            ax[0].plot(xc.flatten()[mask], yc.flatten()[mask], 'ro', markersize=0.5, alpha=0.5)
            ax[1].plot(xc.flatten()[_mask], yc.flatten()[_mask], 'bo', markersize=0.5, alpha=0.5)
            ax[0].set_xlim(-30, 30)
            ax[0].set_ylim(-10, 10)
            plt.tight_layout()
            fig_manager = plt.get_current_fig_manager()
            fig_manager.window.showMaximized()
        if ax is None:
            plt.show()


@click.command()
@click.option('--unwrap', is_flag=True)
@click.argument('data', type=click.Path(dir_okay=True, file_okay=False, exists=True))
def plot_quotient(data, unwrap):
    result = FPMResult()
    result.load_hdf5(data)
    # result.plot(color='r', unwrap=unwrap)
    # result.plot(color='g', unwrap=unwrap)
    # result.plot(color='b', unwrap=unwrap)
    # result.plot_phase('r', unwrap=unwrap)
    # result.plot_phase('g', unwrap=unwrap)
    # result.plot_phase('b', unwrap=unwrap)
    # result.plot_quotient('r/g', unwrap=unwrap)
    # result.plot_quotient('b/g', unwrap=unwrap)
    # result.plot_quotient('r/b', unwrap=unwrap)
    result.plot_quotient_2d(x='r/g', y='b/g', unwrap=unwrap, threshold=0.0)
    result.plot_quotient_2d(x='r/g', y='b/g', unwrap=unwrap, roi=((0, 200), (1000, 1200)))


if __name__ == '__main__':
    plot_quotient()

    # 3684 5544

    # array([[3, 0],
    #        [6, 7],
    #        [3, 6],
    #        [2, 2]])
