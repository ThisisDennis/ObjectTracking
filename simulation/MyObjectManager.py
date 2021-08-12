class MyObjectManager:

    def __init__(self):
        self.agents = {}

    def registerAgents(self, obj):
        self.agents[obj.id]=obj

    def removeAgent(self, obj=None):
        if obj is None:
            s= self.getAgentSize()
            self.agents.pop(s-1)
        else:
            self.agent.pop(obj.id)

    def getAgent(self, objID):
        return self.agents[objID]

    def getAgentSize(self):
        return len(self.agents)
    