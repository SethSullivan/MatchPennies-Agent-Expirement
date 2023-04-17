import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import data_visualization as dv
wheel = dv.ColorWheel()

class Group_Subjects():
    def __init__(self,objects):
        self.objects = objects
    def combine_all_subjects(self,metric):
        '''
        List comprehension into np array to put the subjects at index 0
        '''
        #* Loop through all attributes, and set the group attribute with all subjects combined
        for a in dir(self.objects[0]):
            if not a.startswith('__'):
                setattr(self,a,np.array([getattr(o,a) for o in self.objects]))