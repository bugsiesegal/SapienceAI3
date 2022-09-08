from Network import Brain, loss_function
import numpy as np
from jax import random


def simon_game(brain: Brain, key):
    sequence = []
    reward = 0

    done = False
    while not done:
        new_key, subkey = random.split(key)
        del key  # The old key is discarded -- we must never use it again.
        normal_sample = random.normal(subkey)
        del subkey  # The subkey is also discarded after use.
        key = new_key  # If we wanted to do this again, we would use new_key as the key.

        for n in sequence:
            brain.step([n])

        n = random.randint(key, shape=(1,), minval=1, maxval=4)[0]
        sequence.append(n)
        brain.step([n])

        for i in sequence:
            n = brain.step([0])
            reward = loss_function(n, i)
            if n != i:
                done = True
                break

    return reward, key
