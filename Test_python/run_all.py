# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 10:02:20 2022

@author: Rosa Maessen
"""

import learning_human_robot
import learning_robot_robot
import load_data

from params import *


runs = 14

learning_human_robot.run_file()
learning_robot_robot.run_file(runs)
load_data.run_file(runs)