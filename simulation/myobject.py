import random
import numpy as np

class myobject:

    def __init__(self, x = 0.0, y = 0.0, id = random.randint(1,2000)):
#        self.xPos = x
#        self.yPos = y
        self.pos = np.array([x,y])



