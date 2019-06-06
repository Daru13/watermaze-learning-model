import numpy as np
from numpy import random as rd

import constants as cst
import utilities as ut


class Plateform:

    center = None
    radius = cst.PLATEFORM_RADIUS

    def __init__(self):
        self.set_random_center()

    
    def set_random_center(self):
        self.center = ut.get_random_point_in_disc((cst.X_ORIGIN, cst.Y_ORIGIN),
                                                   cst.WATERMAZE_RADIUS)



class Watermaze:

    plateform = None

    center = np.array([cst.X_ORIGIN, cst.Y_ORIGIN])
    radius = cst.WATERMAZE_RADIUS

    def __init__(self):
        self.plateform = Plateform()

