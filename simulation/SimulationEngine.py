import cv2, random, time
import numpy as np
from simulation import Agent as a
from simulation import MyObjectManager as mom
from simulation import MyBehaivour as mb
#import pygame, sys
import os

width = 728
height = 1064
agentSize = 10
fps = 25
# pygame

#screen = pygame.display.set_mode((height, width))
#img = np.zeros((height,width,3), np.uint8)

delay = 1/fps

Agent = a.Agent
MyBehaivour = mb.MyBehaivour

def nothing(x):
    pass



def update():
    global  agentSize
    rand = random.Random()
    agentSize = cv2.getTrackbarPos("number of agents", "Settings")

#    speed = cv2.getTrackbarPos("speed", "Settings")
    oldSize = simu.agents.getAgentSize()
    if agentSize > oldSize:

        for i in range(0, agentSize):
            agent = Agent(oldSize + i, rand.randint(0, width), rand.randint(0, height), (rand.uniform(0.0, 1.0)+1), rand.uniform(0.0, 1.0)-1, rand.randint(1,4))
            simu.agents.registerAgents(agent)

    elif agentSize < oldSize:
        number = oldSize - agentSize
        print(number)
        for i in range(0, number):
            simu.agents.removeAgent()


def trackbar():

    cv2.createTrackbar("number of agents", "Settings", 1, 200, nothing)

    # set Values
    cv2.setTrackbarPos("number of agents", "Settings", agentSize)

    cv2.setTrackbarMin("number of agents", "Settings", 1)


class gameEngine:
    #agents = MyObjectManager()

    #pygame.display.update()
    def getAgentsize(self):
        return self.agents.getAgentSize()

    def agentNumbers(self, agentSize):
        rand = random.Random()
        oldSize = self.agents.getAgentSize()
        if agentSize > oldSize:

            for i in range(0, agentSize):
                agent = Agent(oldSize + i, rand.randint(0, width), rand.randint(0, height),
                              (rand.uniform(0.0, 1.0) + 1), rand.uniform(0.0, 1.0) - 1, rand.randint(1, 4))
                self.agents.registerAgents(agent)

        elif agentSize < oldSize:
            number = oldSize - agentSize
            print(number)
            for i in range(0, number):
                self.agents.removeAgent()

    def __init__(self):
        #agents = myObjectManager.getExamplar()
        #trackbar()
        self.agents = mom.MyObjectManager()
        self.createAgents(agentSize)
        self.im = None

    def createAgents(self, anz):
        rand = random.Random()
        for i in range(0, anz):
            agent = Agent(i, rand.randint(0, width), rand.randint(0, height), (rand.random()+1)*5, (rand.random()-1)*5, rand.randint(1,4))
            self.agents.registerAgents(agent)

    def rederLoop(self):

    #myObjectManager = MyObjectManager()


        img = np.zeros((width, height, 3), dtype = np.uint8)
        for i in range(0, len(img)):
            img[i] = (255,238,204)  #background color
        #update()

        for i in range(0, self.agents.getAgentSize()):

            aktAgent = self.agents.getAgent(i)
            behave = MyBehaivour( self.agents, aktAgent, width, height )
            aktAgent.setBehaviour(behave)
            #pygame.display.update()
            img = aktAgent.render(img)
            aktAgent.update()

        #screen.fill((0, 0, 0))

       # aktAgent.update()
        self.im = img
        #cv2.imshow("Original", img)


simu = gameEngine()
simu.rederLoop()