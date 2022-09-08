import numpy as np
from jax import grad

from Generator import sim

grad_sim = grad(sim)

while True:

    initial_state = np.random.rand(2, 12, 12) * 10 - 5

    gradient = grad_sim(initial_state)

    if np.all(gradient != 0.0):
        print(gradient)
        break
