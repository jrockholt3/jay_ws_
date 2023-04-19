import numpy

import pickle

file = open('/home/jrockholt@ad.ufl.edu/jay_ws_/src/mink_control/files/action_mem.pkl','rb')
actions = pickle.load(file)
a = actions[0,0]

print(type(a.item()))