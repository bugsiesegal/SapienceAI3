from jax import random

from Enviroment import simon_game
from Network import Brain
import numpy as np


def sim(x):
    x = x.reshape((2, 12, 12))
    brain = Brain(10, 1, 1)
    key = random.PRNGKey(135)
    brain.from_array(x)
    rewards = 0
    for i in range(100):
        reward, key = simon_game(brain, key=key)
        rewards += reward
    return rewards / 100
