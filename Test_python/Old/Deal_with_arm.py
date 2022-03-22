#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dynamics and Control Assignment 4c: dynamic control of endpoint
-------------------------------------------------------------------------------
DESCRIPTION:
2-DOF planar robot arm model with shoulder and elbow joints. The code includes
simulation environment and visualisation of the robot.

The robot is a torque-controlled robot:
- Measured joint angle vector q is provided each sample time.
- Calculate the joint torque tau in each sample time to be sent to the robot.

Important variables:
q1/tau -> shoulder joint configuration/torque
q2/tau -> elbow joint configuration/torque
p1 -> endpoint x position
p2 -> endpoint y position

TASK:
Make the robot track a given endpoint reference trajectory by using dynamic
control (i.e., impedance controller). Try impedance control without and with
the damping term to analyse the stability.
-------------------------------------------------------------------------------


INSTURCTOR: Luka Peternel
e-mail: l.peternel@tudelft.nl

"""



import numpy as np
import math
import matplotlib.pyplot as plt


plt.close('all')


'''ROBOT MODEL'''

class human_arm_2dof:
    def __init__(self, l, I, m, s):
        self.l = l # link length
        self.I = I # link moment of inertia
        self.m = m # link mass
        self.s = s # link center of mass location
        
        self.q = np.array([0.0, 0.0]) # joint position
        self.dq = np.array([0.0, 0.0]) # joint veloctiy
        self.tau = np.array([0.0, 0.0]) # joint torque

        # arm dynamics parameters
        self.k = np.array([ self.I[0] + self.I[1] + self.m[1]*(self.l[0]**2), self.m[1]*self.l[0]*self.s[1], self.I[1] ])
        
        # joint friction matrix
        self.B = np.array([[0.050, 0.025],
                           [0.025, 0.050]])
    
        # external damping matrix (e.g., when endpoint is moving inside a fluid)
        self.D = np.diag([1.0, 1.0])



    # forward kinematics
    def FK(self, q):
        '''*********** This should be added from Assignment 4a ***********'''
        q1 = self.q[0]
        q2 = self.q[1]
        
        p = np.array([self.l[0]*np.cos(q1)+self.l[1]*np.cos(q1+q2), self.l[0]*np.sin(q1)+self.l[1]*np.sin(q1+q2)])
        '''*********** This should be added from Assignment 4a ***********'''
        return p
    

    
    # arm Jacobian matrix
    def Jacobian(self):
        '''*********** This should be added from Assignment 4a ***********'''
        q1 = self.q[0]
        q2 = self.q[1]
        
        J = np.array([[-self.l[0]*np.sin(q1)-self.l[0]*np.sin(q1+q2), -self.l[1]*np.sin(q1+q2)],
                      [self.l[0]*np.cos(q1)+self.l[0]*np.cos(q1+q2), self.l[1]*np.cos(q1+q2)]])
        '''*********** This should be added from Assignment 4a ***********'''
        return J

    
    
    # forward arm dynamics
    def FD(self):
        '''*********** This should be added from previous Assignments ***********'''
        g = 0
        M = np.array([[self.m[0] * self.l[0]**2 / 4 + self.I[0] + self.m[1] * self.l[0]**2,
                       0.5 * self.m[1] *  np.cos(self.q[0] - (self.q[1]-self.q[0])) * self.l[0] * self.l[1]],
                      [0.5 * self.m[1] * np.cos(self.q[0] - (self.q[1]-self.q[0])) * self.l[0] * self.l[1],
                       self.m[1] * self.l[1]**2 / 4 + self.I[1]]])
        
        forces = np.array([tau[0]*l[0]/2-self.dq[1]**2 * 0.5 * self.m[1] * np.sin(self.q[0] - (self.q[1]-self.q[0])) 
                            - (self.m[0] * g * self.l[0] / 2+ self.m[1] * g * self.l[0]) * np.sin(self.q[0]),
                           tau[1]*l[1]/2+self.dq[0]**2 * 0.5 * self.m[1] * np.sin(self.q[0]- (self.q[1]-self.q[0])) 
                            - (self.m[1] * g * self.l[1] / 2 * np.sin((self.q[1]-self.q[0])))])
        
        # accelerations
        sol = np.linalg.solve(M, forces)
        ddq = sol.reshape(2,)
        '''*********** This should be added from previous Assignments ***********'''
        return ddq
    
    
    
    # inverse kinematics
    def IK(self, p):
        q = np.zeros([2])
        r = np.sqrt(p[0]**2+p[1]**2)
        q[1] = np.pi - math.acos((self.l[0]**2+self.l[1]**2-r**2)/(2*self.l[0]*self.l[1]))
        q[0] = math.atan2(p[1],p[0]) - math.acos((self.l[0]**2-self.l[1]**2+r**2)/(2*self.l[0]*r))
        
        return q
    
    
    
    # state change
    def state(self, q, dq, tau):
        self.q = q
        self.dq = dq
        self.tau = tau


'''SIMULATION'''

# SIMULATION PARAMETERS
dt = 0.01 # intergration step timedt = 0.01 # integration step time
dts = dt*1 # desired simulation step time (NOTE: it may not be achieved)
T = 3 # total simulation time

# ROBOT PARAMETERS
x0 = 0.0 # base x position
y0 = 0.0 # base y position
l1 = 0.3 # link 1 length
l2 = 0.3 # link 2 length (includes hand)
l = np.array([l1, l2]) # link length
I = np.array([0.025, 0.045]) # link moment of inertia
m = np.array([1.4, 1.1]) # link mass
s = np.array([0.5*l1, 0.*l2]) # link center of mass location

L_b = 0.3

x20 = x0 + 0.2*2+L_b*2
y20 = y0


# REFERENCE TRAJETORY
ts = T/dt # trajectory size
xt = np.linspace(-2,2,int(ts))
yt1 = np.sqrt(1-(abs(xt)-1)**2)
yt2 = -3*np.sqrt(1-(abs(xt)/2)**0.5)

x = np.concatenate((xt, np.flip(xt,0)), axis=0)
y = np.concatenate((yt1, np.flip(yt2,0)), axis=0)

#pr = np.array((x / 10 + 0.0, y / 10 + 0.45)) # reference endpoint trajectory

xr1a = np.linspace(0.2, 0.5, 250)
xr1b = np.linspace(0.5, 0.5, 100)
xr1 = np.concatenate((xr1a,xr1b, np.flip(xr1a)))

yr1 = np.zeros((len(xr1)))

xr2 = xr1+L_b
yr2 = yr1

pr = np.array((xr1, yr1))
pr2 = np.array((xr2, yr2))


'''*********** Student should fill in ***********'''
# IMPEDANCE CONTROLLER PARAMETERS
K = 3500 # stiffness N/m
D = 300 #2*0.7*np.sqrt(K) # damping Ns/m
'''*********** Student should fill in ***********'''

# SIMULATOR
# initialise robot model class
model = human_arm_2dof(l, I, m, s)
model2 = human_arm_2dof(l, I, m, s)

# initialise real-time plot
plt.close()
plt.figure(1)
plt.xlim(-0.1,1.1)
plt.ylim(-0.3,0.3)

# initial conditions
t = 0.0 # time
q = model.IK(pr[:,0]) # joint position
dq = np.array([0., 0.]) # joint velocity
tau = [0., 0.] # joint torque
model.state(q, dq, tau) # update initial state
p_prev = pr[:,0] # previous endpoint position
i = 0 # loop counter
state = [] # state vector

q2 = model2.IK(pr2[:,0]) # joint position
dq2 = np.array([0., 0.]) # joint velocity
tau2 = [0., 0.] # joint torque
model2.state(q2, dq2, tau2) # update initial state
p_prev2 = pr2[:,0] # previous endpoint position
state2 = [] # state vector

# robot links
x1 = l1*np.cos(q[0])
y1 = l1*np.sin(q[0])
x2 = x1+l2*np.cos(q[0]+q[1])
y2 = y1+l2*np.sin(q[0]+q[1])
link1, = plt.plot([x0,x1],[y0,y1],color='b',linewidth=3) # draw upper arm
link2, = plt.plot([x1,x2],[y1,y2],color='b',linewidth=3) # draw lower arm
shoulder, = plt.plot(x0,y0,color='k',marker='o',markersize=8) # draw shoulder / base
elbow, = plt.plot(x1,y1,color='k',marker='o',markersize=8) # draw elbow
hand, = plt.plot(x2,y2,color='k',marker='*',markersize=15) # draw hand / endpoint

x21 = x20 + l1*np.cos(q2[0])
y21 = y20 + l1*np.sin(q2[0])
x22 = x21+ l2*np.cos(q2[0]+q2[1])
y22 = y21+l2*np.sin(q2[0]+q2[1])
link21, = plt.plot([x20,x21],[y20,y21],color='r',linewidth=3) # draw upper arm
link22, = plt.plot([x21,x22],[y21,y22],color='r',linewidth=3) # draw lower arm
shoulder2, = plt.plot(x20,y20,color='k',marker='o',markersize=8) # draw shoulder / base
elbow2, = plt.plot(x21,y21,color='k',marker='o',markersize=8) # draw elbow
hand2, = plt.plot(x22,y22,color='k',marker='*',markersize=15) # draw hand / endpoint


for i in range(len(x)):
    
    # update individual link position
    x1 = l1*np.cos(q[0])
    y1 = l1*np.sin(q[0])
    x2 = x1+l2*np.cos(q[0]+q[1])
    y2 = y1+l2*np.sin(q[0]+q[1])
    
    x21 = x20 + l1*np.cos(q2[0])
    y21 = y20 + l1*np.sin(q2[0])
    x22 = x21+l2*np.cos(q2[0]+q2[1])
    y22 = y21+l2*np.sin(q2[0]+q2[1])
    
    # real-time plotting
    #ref, = plt.plot(pr[0,i],pr[1,i],color='g',marker='+') # draw reference
    link1.set_xdata([x0,x1]) # update upper arm
    link1.set_ydata([y0,y1]) # update upper arm
    link2.set_xdata([x1,x2]) # update lower arm
    link2.set_ydata([y1,y2]) # update lower arm
    shoulder.set_xdata(x0) # update shoulder / base
    shoulder.set_ydata(y0) # update shoulder / base
    elbow.set_xdata(x1) # update elbow
    elbow.set_ydata(y1) # update elbow
    hand.set_xdata(x2) # update hand / endpoint
    hand.set_ydata(y2) # update hand / endpoint
    
    link21.set_xdata([x20,x21]) # update upper arm
    link21.set_ydata([y20,y21]) # update upper arm
    link22.set_xdata([x21,x22]) # update lower arm
    link22.set_ydata([y21,y22]) # update lower arm
    shoulder2.set_xdata(x20) # update shoulder / base
    shoulder2.set_ydata(y20) # update shoulder / base
    elbow2.set_xdata(x21) # update elbow
    elbow2.set_ydata(y21) # update elbow
    hand2.set_xdata(x22) # update hand / endpoint
    hand2.set_ydata(y22) # update hand / endpoint

    plt.pause(dts) # try to keep it real time with the desired step time
        


    '''*********** Student should fill in ***********'''   
    # endpoint reference trajectory controller
    dp = model.Jacobian()@dq
    Fext = K*(pr[:,i]-model.FK(q))-D*dp
    tau = np.transpose(model.Jacobian())@Fext
    
    # Importing some variables for the sate function
    model.state(q,dq,tau)
    ddq = model.FD()
    p = model.FK(q)
    
    dp2 = model2.Jacobian()@dq2
    Fext2 = K*(pr2[:,i]-model2.FK(q2))-D*dp2
    tau2 = np.transpose(model2.Jacobian())@Fext2
    
    # Importing some variables for the sate function
    model2.state(q2,dq2,tau2)
    ddq2 = model2.FD()
    p2 = model2.FK(q2)
    '''*********** Student should fill in ***********'''

    

    # log states for analysis
    state.append([t, q[0], q[1], dq[0], dq[1], ddq[0], ddq[1], tau[0], tau[1], p[0], p[1]])   
    state2.append([t, q2[0], q2[1], dq2[0], dq2[1], ddq2[0], ddq2[1], tau2[0], tau2[1], p2[0], p2[1]])   
    
    # previous endpoint position for velocity calculation
    p_prev = p
    p_prev2 = p2
    
    # integration
    dq += ddq*dt
    q += dq*dt
    t += dt
    
    dq2 += ddq2*dt
    q2 += dq2*dt
    
    # increase loop counter
    i = i + 1



'''ANALYSIS'''

state = np.array(state)
state2 = np.array(state2)

plt.figure(3)
plt.subplot(411)
plt.title("JOINT BEHAVIOUR")
plt.plot(state[:,0],state[:,7],"b",label="shoulder")
plt.plot(state[:,0],state[:,8],"r",label="elbow")
plt.legend()
plt.ylabel("tau [Nm]")

plt.subplot(412)
plt.plot(state[:,0],state[:,5],"b")
plt.plot(state[:,0],state[:,6],"r")
plt.ylabel("ddq [rad/s2]")

plt.subplot(413)
plt.plot(state[:,0],state[:,3],"b")
plt.plot(state[:,0],state[:,4],"r")
plt.ylabel("dq [rad/s]")

plt.subplot(414)
plt.plot(state[:,0],state[:,1],"b")
plt.plot(state[:,0],state[:,2],"r")
plt.ylabel("q [rad]")
plt.xlabel("t [s]")

plt.tight_layout()
plt.show()



# Adjusted analysis plot
plt.figure(4)
plt.title("ENDPOINT BEHAVIOUR")
plt.plot(0,0,"ok",label="shoulder")
plt.plot(state[:,9],state[:,10],label="trajectory")
plt.plot(state[0,9],state[0,10],"xg",label="start point")
plt.plot(state[-1,9],state[-1,10],"+r",label="end point")
plt.plot(pr[0,:], pr[1,:], "r:", label="reference trajectory")
plt.axis('equal')
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.legend()

plt.tight_layout()
plt.show()


plt.figure(5)
plt.title("ENDPOINT BEHAVIOUR")
plt.plot(0,0,"ok",label="shoulder")
plt.plot(state2[:,9],state2[:,10],label="trajectory")
plt.plot(state2[0,9],state2[0,10],"xg",label="start point")
plt.plot(state2[-1,9],state2[-1,10],"+r",label="end point")
plt.plot(pr2[0,:], pr2[1,:], "r:", label="reference trajectory")
plt.axis('equal')
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.legend()

plt.tight_layout()
plt.show()

