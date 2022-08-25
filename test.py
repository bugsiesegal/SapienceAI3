import jax.numpy as np
from autograd import elementwise_grad as egrad
from jax import grad, jit, vmap

from Generator import sim

grad_sim = jit(vmap(sim))

a = np.full((2, 12, 12), 0.9999)
b = np.full((2, 12, 12), 1.0001)

print((sim(a) - sim(b))/0.0002)

print(grad_sim(np.full((2, 12, 12), 1.0)))
