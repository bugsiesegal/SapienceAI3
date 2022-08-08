from Network import Brain
import random


def simon_game(brain: Brain):
    sequence = []
    reward = 0
    done = False
    while not done:
        for n in sequence:
            brain.step([n])

        n = random.randint(1, 4)
        sequence.append(n)
        brain.step([n])

        for i in sequence:
            if brain.step([0]) == i:
                reward += 1
            else:
                done = True
                break
    return reward
