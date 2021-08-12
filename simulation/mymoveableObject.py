from simulation import myobject as mo
from simulation import MyBehaivour
import numpy as np

myobject = mo.myobject

class mymoveableObject(myobject):

    def __init__(self, xPos = 0.0, yPos = 0.0, xSpeed = 0.0, ySpeed = 0.0):
        myobject.__init__(self, xPos, yPos)
        self.velocity = np.array([0,0])
        self.acceleration = np.array([0,0])
        self.speed = np.array([xSpeed,ySpeed])


    def setBehaviour(self, behaviour):
        self.behaviour = behaviour

    def update(self):
    #    try:
        self.behaviour.update()
     #   except:
     #       print("behaviour update failed")