'''
Created on Dec 13, 2017

@author: Marcus Ritter
'''

class State():
    
    def __init__(self, o, d, e, s):
        
        self.o = o
        self.d = d
        self.e = e
        self.s = s
        self.observation = []
        self.statestring = ""
        
        for i in range(len(self.o)):
            self.observation.append(self.o[i])
        for i in range(len(self.d)):
            self.observation.append(self.d[i])
        for i in range(len(self.e)):
            self.observation.append(self.e[i])
        for i in range(len(self.s)):
            self.observation.append(self.s[i])
            
        for i in range(len(self.observation)):
            self.statestring = self.statestring + str(self.observation[i]) + ','