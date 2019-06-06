import numpy as np
from numpy import random as rd

import time


def get_random_point_in_disc(center, radius):
    radius = radius * np.sqrt(rd.random())
    angle = 2 * np.pi * rd.random()

    return np.array([
        center[0] + radius * np.cos(angle),
        center[1] + radius * np.sin(angle)
    ])


# Iterator over iterator with a timeout in seconds
def iterator_with_timeout(iterator, timeout):
    init_time = time.time()

    while time.time() - init_time < timeout:
        yield next(iterator)
