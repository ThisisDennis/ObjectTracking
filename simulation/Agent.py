import cv2
import random
import numpy as np
from simulation import mymoveableObject as mmov
#import pygame

maxSpeed = 2000
color = (125,255,255)
r = 0
mymoveableObject = mmov.mymoveableObject

class Agent(mymoveableObject):
    objCounter = 0
    panicDist = 100
    separationDist = 0

    def __init__(self, id, xPos = 0.0, yPos = 0.0, xSpeed = maxSpeed, ySpeed = maxSpeed, radius = r, color = color):
        #self(xPos, yPos, xSpeed, ySpeed, 20.0, 1, 1, 0 ) #braucht hier eigene funktion
        b = random.randint(0, 255)
        g = random.randint(0, 255)
        r = random.randint(0, 255)
        self.color = (b, g, r)
        mymoveableObject.__init__(self, xPos, yPos, xSpeed, ySpeed)
        #self.pos = (xPos, yPos)
        self.radius = radius

        #self.objCounter += 1
        self.id = id

    def applyForce(self, force):
        self.force = np.divide(force, self.radius*self.radius)
        self.acceleration = force

    def render(self, screen):

        #rect = (self.pos[0], self.pos[0]+10, self.pos[1],self.pos[1]+ 20)
        rect = (self.pos[0],self.pos[1],20,10)
        #pygame.draw.ellipse(screen, (240,250,250), rect, 0)
        #img = cv2.ellipse(screen, (int(self.pos[0]), int(self.pos[1])), (5, 2), random.randint(0, 50), 0, 360, 255, 10)
        text =  str(self.color)

        img = cv2.circle(screen, (int(self.pos[0]), int(self.pos[1])), 5, self.color, cv2.FILLED)
        cv2.putText(screen, text, (int(self.pos[0]), int(self.pos[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
       # img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        return img