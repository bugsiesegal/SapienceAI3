from autograd import grad

from Enviroment import simon_game
from Network import Brain


def sim(x):
    brain = Brain(10, 1, 1)

    brain.from_array(x)
    rewards = 0
    for i in range(100):
        rewards += simon_game(brain)

    return rewards / 100
