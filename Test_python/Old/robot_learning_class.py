# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 09:07:15 2022

@author: Rosa Maessen
"""

from pDMP_functions import pDMP

class robot_learning():
    def __init__(self):
        self.mode = 0
        
    
    
    def learning(self, phase):
        
        if self.mode == 0:
            self.observe(phase)
            
        elif self.mode == 1:
            self.learn(phase)
            
        elif self.mode == 2:
            self.collaborative(phase)
        
        else:
            exit()
            
    
    def observe(self, phase, data):
        
    def learn(self):
        
    def collaborative(self);
        