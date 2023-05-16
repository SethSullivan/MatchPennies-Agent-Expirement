import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import data_visualization as dv
wheel = dv.ColorWheel()

class Group_Subjects():
    def __init__(self,objects):
        self.objects = objects
    
        #* Loop through all attributes, and set the group attribute with all subjects combined
        for a in dir(self.objects[0]):
            if not a.startswith('__'):
                setattr(self,a,np.array([getattr(o,a) for o in self.objects]))