import numpy as np
from  simulation import Agent

class MyBehaivour:

    def __init__(self, agents, agent, width, height, tx=-1, ty=-1):
        self.tx = tx
        self.ty = ty
        self.width = width
        self.height = height
        self.agent = agent
        self.agents = agents
        self.velocity = np.array([0,0])

    def separation(self):
        steeringForce = np.array([0, 0])

        for i in range(0, self.agents.getAgentSize()):

            if self.agent.id == i:
                continue
            bObj = self.agents.getAgent(i)

            if isinstance(bObj, object):

                if np.linalg.norm(self.agent.pos-bObj.pos) < self.agent.separationDist + bObj.separationDist:

                    help = np.subtract(self.agent.pos, bObj.pos)
                    length = np.linalg.norm(help)
                    help = np.divide(help, length)
                    steeringForce = steeringForce + help

        return steeringForce


    def update(self):
        # regel1....3 initialisieren

        #velocity = np.array([0,0])
        #self.velocity = self.velocity + self.agent.speed
        #self.agent.pos = self.agent.pos + self.velocity + self.separation()
        #self.agent.pos = self.agent.pos + self.agent.speed
        #self.agent.pos =  self.agent.pos + self.velocity + self.agent.speed
        self.agent.pos = self.agent.pos + self.agent.speed +self.agent.separationDist

        if self.agent.pos[1] > self.width or self.agent.pos[1] < 0:

            self.agent.speed[1] *= -1

        if self.agent.pos[0] > self.height or self.agent.pos[0] < 0:
            self.agent.speed[0] *= -1
