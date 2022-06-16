from dataclasses import dataclass

from typing import Tuple

import numpy as np


class Brain:
    def __init__(self, num_neurons, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.num_neurons = num_neurons + input_size + output_size

        self.neuron_array = np.zeros(num_neurons)
        self.threshold_potential = np.empty((num_neurons, num_neurons))
        self.action_potential = np.empty((num_neurons, num_neurons))

    def step(self, input_array):
        self.neuron_array[:self.input_size] = input_array




