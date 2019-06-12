import numpy as np
from numpy import random as rd

from constants import *
from utilities import get_random_point_in_disc




class Plateform:
    '''
    Plateform of a watermaze.
    '''

    center = None
    radius = PLATEFORM_RADIUS

    def __init__(self):
        self.set_random_center()

    
    def set_random_center(self):
        self.center = get_random_point_in_disc((X_ORIGIN, Y_ORIGIN), WATERMAZE_RADIUS)



class Watermaze:
    '''
    Watermaze in which an experiment can be ran. It contains a plateform.
    '''

    plateform = None

    center = np.array([X_ORIGIN, Y_ORIGIN])
    radius = WATERMAZE_RADIUS

    def __init__(self):
        self.set_random_plateform()

    
    def set_random_plateform(self):
        self.plateform = Plateform()

