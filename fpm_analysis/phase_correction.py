import matplotlib.pyplot as plt
import numpy as np

from fpm_analysis.simulate_phase import SimulatedPhaseResult

SIZE = 128
SAME_MATERIAL = False
STEPS = 16
DH_MAX = 10.0
OFFSET = 0.0
NOISE = 0.001
step_size = SIZE // STEPS
dh = DH_MAX / STEPS

simulated_result = SimulatedPhaseResult(shape=(SIZE, SIZE))
simulated_result.reset_sim()
simulated_result.make_ladder(same_material=SAME_MATERIAL, n_steps=STEPS, dh=dh, offset=OFFSET)
simulated_result.add_random_normal_noise(NOISE)

test_points = np.asarray([(i * step_size + step_size // 2, SIZE // 2) for i in range(STEPS)]).T

plt.imshow(simulated_result.unwrapped['r'] / np.pi / 2)
plt.plot(*test_points, 'go')
plt.show()

samples = {c: np.squeeze(simulated_result.unwrapped[c][test_points[1], [test_points[0]]])
           for c in 'rgb'}

plt.plot(samples['r'] / samples['g'], samples['b'] / samples['g'], 'ro')
plt.show()
