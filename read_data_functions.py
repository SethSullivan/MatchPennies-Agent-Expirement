import numpy as np
import dill
import os

def pickle_subject_objects(subject_list,load_path):
    subject_objects = []
    for i,subname in enumerate(subject_list):
        print(subname)
        subject_object = dill.load(open(load_path + f'\\{subname}\\{subname}_object.pkl', 'rb'))
        subject_objects.append(subject_object)
    return subject_objects