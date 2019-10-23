from typing import List, Dict

import h5py
import matplotlib.pyplot as plt
import numpy as np

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Number of positions to use
N = 1
# Files h5py for R/G/B
input_files: Dict[str, str] = {'r': 'image_red.h5py', 'g': 'image_green.h5py', 'b': 'image_blue.h5py'}
# List of colors obtained from files
colors: List[str] = [color for color, _ in input_files.items()]
logger.info(f'Available colors: {" ".join(colors)}')
# Dict with colors as keys for list of phases
color_phases: Dict[str, List[np.ndarray]] = {color: None for color in colors}

# Fill color_phases from input files
for color, file in input_files.items():
    f = h5py.File(file, 'r')
    positions = sorted(f.keys())
    assert N <= len(positions)
    color_phases[color]: np.ndarray = [np.angle(f[positions[i]]) for i in range(N)]


def get_phase_quotients(position: int, x: str = 'r/g', y: str = 'g/b'):
    phase_x = color_phases[x.split('/')[0]][position] / color_phases[x.split('/')[1]][position]
    phase_y = color_phases[y.split('/')[0]][position] / color_phases[y.split('/')[1]][position]
    return phase_x, phase_y


def plot_phase(color: str, position: int, ax: plt.Axes = None):
    if ax is None:
        ax: plt.Axes = plt.subplots(1, 1, figsize=(8, 6))[1]
    ax.imshow(color_phases[color][position], cmap='gray', interpolation='nearest')
    plt.show()


def plot_phase_quotients(phase_x: np.ndarray, phase_y: np.ndarray, position: int, ax: plt.Axes = None):
    if ax is None:
        ax: plt.Axes = plt.subplots(1, 1, figsize=(8, 6))[1]
    ax.plot(phase_x.flatten, phase_y.flatten, 'ro', alpha=0.8)
    plt.show()


fig, axs = plt.subplots(1, 2)
plot_phase(color='r', position=0, ax=axs[0])
phase_x, phase_y = get_phase_quotients(position=0)
plot_phase_quotients(position=0, phase_x=phase_x, phase_y=phase_y, ax=axs[0])
