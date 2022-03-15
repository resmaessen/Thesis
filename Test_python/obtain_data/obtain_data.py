# -*- coding: utf-8 -*-
"""
Control in Human-Robot Interaction Assignment 1: Haptic Rendering
-------------------------------------------------------------------------------
DESCRIPTION:
Creates a simulated haptic device (right) and VR environment (left)

The forces on the virtual haptic device are displayed using pseudo-haptics. The 
code uses the mouse as a reference point to simulate the "position" in the 
user's mind and couples with the virtual haptic device via a spring. the 
dynamics of the haptic device is a pure damper, subjected to perturbations 
from the VR environment. 

IMPORTANT VARIABLES
xc -> x and y coordinates of the center of the haptic device and of the VR
xm -> x and y coordinates of the mouse cursor 
xh -> x and y coordinates of the haptic device (shared between real and virtual panels)
fe -> x and y components of the force fedback to the haptic device from the virtual impedances
dxh -> change to the haptic handle position according to the external forces. This is what makes the simulation

TASKS:
1- Implement the impedance control of the haptic device
2- Implement an elastic element in the simulated environment
3- Implement a position dependent potential field that simulates a bump and a hole
4- Implement the collision with a 300x300 square in the bottom right corner 
5- Implement the god-object approach and compute the reaction forces from the wall

REVISIONS:
Initial release MW - 14/01/2021
Added 2 screens and Potential field -  21/01/2021
Added Collision and compressibility (LW, MW) - 25/01/2021
Added Haptic device Robot - TODO

INSTRUCTORS: Michael Wiertlewski & Luka Peternel & Laurence Willemet & Mostafa Atalla
e-mail: {m.wiertlewski,l.peternel, l.willemet,m.a.a.atalla}@tudelft.nl

Edited by: Rosa Maessen (4564200)
E-mail: R.E.S.Maessen@student.tudelft.nl
"""


import pygame
import numpy as np
import math
import matplotlib.pyplot as plt


##################### General Pygame Init #####################
##initialize pygame window
pygame.init()
window = pygame.display.set_mode((1200, 600))   ##twice 600x600 for haptic and VR
pygame.display.set_caption('Virtual Haptic Device')

screenHaptics = pygame.Surface((600,600))
screenVR = pygame.Surface((600,600))

##add nice icon from https://www.flaticon.com/authors/vectors-market
icon = pygame.image.load('robot.png')
pygame.display.set_icon(icon)

##add text on top to debugToggle the timing and forces
font = pygame.font.Font('freesansbold.ttf', 18)

pygame.mouse.set_visible(True)     ##Hide cursor by default. 'm' will toggle it
 
##set up the on-screen debugToggle
text = font.render('Virtual Haptic Device', True, (0, 0, 0),(255, 255, 255))
textRect = text.get_rect()
textRect.topleft = (10, 10)


xc,yc = screenVR.get_rect().center ##center of the screen


##initialize clock
clock = pygame.time.Clock()
FPS = 100

##define some colors
cWhite = (255,255,255)
cDarkblue = (36,90,190)
cLightblue = (0,176,240)
cRed = (255,0,0)
cOrange = (255,100,0)
cYellow = (255,255,0)
cGreen = (0, 255, 0)
cBlack = (0, 0, 0)


##################### Simulation Init #####################

'''***********  Dynamics parameters ***********'''
'''*********** Student should fill in ***********'''
##hint k/b needs to be <1
k_b = 1     # Ratio between spring and damping constant
k = 0.5      ##Stiffness between cursor and haptic display
b = k/k_b   # Damping constant
kc = 2  # s
kb = kc


'''*********** !Student should fill in ***********'''


##################### Define sprites #####################

##define sprites
hhandle = pygame.image.load('handle.png')
haptic  = pygame.Rect(320,350, 0, 0).inflate(48, 48)

cursor  = pygame.Rect(0, 0, 5, 5)
colorHaptic = cOrange ##color of the wall

xh = np.array(haptic.center)

##Set the old values to 0 to avoid jumps at init
xhold = 0
xmold = 0


'''*********** Visualisation for the VR env***********'''
'''*********** Student should fill in ***********'''
##hint use pygame.rect() to create rectangles
proxy = pygame.Rect(320,350, 0, 0).inflate(48, 48)
wall = pygame.Rect(xc, yc, 300,300)
x_compress_max = 30

xrr_max = (kb*x_compress_max)/k + (300-24 + x_compress_max)


variance = 100
mean1 = 150
mean2 = 450
bump_1A = pygame.Rect(mean1-variance,0,variance,600)
bump_1B = pygame.Rect(mean1,0,variance,600)
bump_2A = pygame.Rect(mean2-variance,0,variance,600)
bump_2B = pygame.Rect(mean2,0,variance,600)

x_bump1 =np.linspace(-variance,variance,1000) + mean1
x_bump2 =np.linspace(-variance,variance,1000) + mean2

Force_virtual = 1000

def height_map(x, mean, variance):
    variance = variance/4
    
    force = (1/(np.sqrt(2*np.pi * variance**2))) * np.exp(-0.5*(((x-mean)/variance)**2))
    if x < mean:
        force = - force
    return force*1000




def gradientRect( window, left_colour, right_colour, target_rect ):
    """ Draw a horizontal-gradient filled rectangle covering <target_rect> """
    colour_rect = pygame.Surface( ( 2, 2 ) )                                   # tiny! 2x2 bitmap
    pygame.draw.line( colour_rect, left_colour,  ( 0,0 ), ( 0,1 ) )            # left colour line
    pygame.draw.line( colour_rect, right_colour, ( 1,0 ), ( 1,1 ) )            # right colour line
    colour_rect = pygame.transform.smoothscale( colour_rect, ( target_rect.width, target_rect.height ) )  # stretch!
    window.blit( colour_rect, target_rect )                                    # paint it

'''*********** Student should fill in ***********'''



##################### Main Loop #####################

##Run the main loop
run = True
ongoingCollision = False

robotToggle = True
debugToggle = False
virtualBump = False
virtualWall = False
showInfo = False
record = False
entered = 'Empty'
size = 48

save_data = []
while run:

    #########Process events  (Mouse, Keyboard etc...)#########
    for event in pygame.event.get():
        ##If the window is close then quit 
        if event.type == pygame.QUIT:
            run = False
        elif event.type == pygame.KEYUP:
            if event.key == ord('m'):   ##Change the visibility of the mouse
                pygame.mouse.set_visible(not pygame.mouse.get_visible())  
            if event.key == ord('q'):   ##Force to quit
                run = False            
            if event.key == ord('d'):
                debugToggle = not debugToggle
                showInfo = False
            if event.key == ord('r'):
                robotToggle = not robotToggle
            '''*********** Student can add more ***********'''
            ##Toggle the wall or the height map
            if event.key == ord('b'):
                virtualBump = not virtualBump
                virtualWall = False
            if event.key == ord('w'):
                virtualWall = not virtualWall
                virtualBump = False
            if event.key == ord('i'):
                showInfo = not showInfo
                debugToggle = False
            if event.key == ord('s'):
                record = not record
                if record:
                    save_data = []
            
            '''*********** !Student can add more ***********'''


    ##Get mouse position
    cursor.center = pygame.mouse.get_pos()
    
    ######### Main simulation. Everything should  #########
    ##Compute distances and forces between blocks
    xh = np.clip(np.array(haptic.center),0,599)
    xm = np.clip(np.array(cursor.center),0,599)

    
    # ##Compute velocities
    vm = xm - xmold
    vh = xh - xhold
    
    
    '''*********** Student should fill in ***********'''
    ##here is where the forces and their effect on the handle position are calculated 
    ######### Compute forces #######
    fk = k*(xm-xh)             ##Elastic force between mouse and haptic device

    fe = np.array([0,0])
    if virtualBump:
        if haptic.colliderect(bump_1A) or haptic.colliderect(bump_1B):
            fe[0] = (height_map(xh[0], mean1, variance))
            
        elif haptic.colliderect(bump_2A) or haptic.colliderect(bump_2B): 
            fe[0] = -(height_map(xh[0], mean2, variance))
        

    if virtualWall:
        if (haptic.colliderect(wall)) == 1:
            if entered == 'Empty': 
                if haptic.center[0] >=  haptic.center[1]:
                    entered = 'Top' 
                else:
                    entered = 'Side'
            K_total = (k*kc)/(k+kc)
            x_wall = np.array([300-24, 300-24])
            fe = K_total*(x_wall-xm)

            if entered == 'Top': 
                fe[0] = 0
            elif entered == 'Side': 
                fe[1] = 0

       
    dxh = fk/b + fe/b # + fb ##replace with the valid expression that takes the force into account
    
    '''*********** !Student should fill in ***********'''

    xh = np.round(xh+dxh)  ##update new positon of the end effector
    haptic.center = xh         

    ##Update old samples for velocity computation
    xhold = xh
    xmold = xm


    ######### Update graphical elements #########
    ##Render the VR surface
    screenVR.fill(cLightblue)
    
    '''*********** Student should fill in ***********'''
    ### here goes the visualisation ofthe VR sceen. 
    ### Use pygame.draw.rect(screenVR, color, rectagle)
    ### to render rectangles. 
    
    if not (virtualBump  or virtualWall):
        pygame.draw.rect(screenVR, cOrange, haptic, border_radius=4)
        
    if virtualBump:
        gradientRect( screenVR, cLightblue, cGreen, bump_1A)
        gradientRect( screenVR, cGreen, cLightblue ,bump_1B )
        gradientRect( screenVR, cLightblue, cBlack, bump_2A)
        gradientRect( screenVR, cBlack, cLightblue ,bump_2B )
        pygame.draw.rect(screenVR, cOrange, haptic, border_radius=4) 
        
    if virtualWall:
        pygame.draw.rect(screenVR, cDarkblue, wall)

        if (haptic.colliderect(wall)) == 1:
            if entered == 'Empty': 
                if haptic.center[0] >=  haptic.center[1]:
                    entered = 'Top' 
                else:
                    entered = 'Side'

            x_wall = np.array([300-24, 300-24])
            x = (kb*x_wall - k* xh)/(kb - k)
            x_compress = x_wall - x
            
            if entered == 'Top':
                x_comp = x_compress[1]

                if x_comp < 0:
                    x_comp = 0
                elif x_comp > x_compress_max:
                    x_comp = x_compress_max

                size = 48 - x_comp
                proxy.height = size
                proxy.width = haptic.width
                proxy.center = [haptic.center[0], 300-size/2]
                
            elif entered == 'Side': 
                x_comp = x_compress[0]

                if x_comp < 0:
                    x_comp = 0
                elif x_comp > x_compress_max:
                    x_comp = x_compress_max
                
                size = 48 - x_comp
                proxy.width = size
                proxy.height = haptic.height
                proxy.center = [300-size/2, haptic.center[1]]

            pygame.draw.rect(screenVR, cOrange, proxy, border_radius=4)
                
        else: 
            entered = 'Empty'
            pygame.draw.rect(screenVR, cOrange, haptic, border_radius=4)

    if record:
        save_data.append([xh[0]])
    '''*********** !Student should fill in ***********'''



    ##Change color based on effort
    colorMaster = (255,\
             255-np.clip(np.linalg.norm(fk)*5,0,255),\
             255-np.clip(np.linalg.norm(fk)*5,0,255)) #if collide else (255, 255, 255)


    ######### Graphical output #########
    ##Render the haptic surface
    screenHaptics.fill(cWhite)
    
    pygame.draw.rect(screenHaptics, colorMaster, haptic,border_radius=4)

    ######### Robot visualization ###################
    # update individual link position
    if robotToggle:
        colorLinks = (150,150,150)
        
        #################### Define Robot #######################
        # ROBOT PARAMETERS
        l = [0.35, 0.45] # links length l1, l2
        window_scale = 400 # conversion from meters to pixels

        xrc = [300,300] ## center location of the robot in pygame
    
        pr = np.array([(xh[0]-xrc[0])/window_scale, -(xh[1]-xrc[1])/window_scale]) ##base is at (0,0) in robot coordinates
        #q = model.IK(pr)
        
        #################### Compute inverse kinematics#######################
        q = np.zeros([2])
        r = np.sqrt(pr[0]**2+pr[1]**2)
        try:
            q[1] = np.pi - math.acos((l[0]**2+l[1]**2-r**2)/(2*l[0]*l[1]))
        except:
            q[1]=0
        
        try:
            q[0] = math.atan2(pr[1],pr[0]) - math.acos((l[0]**2-l[1]**2+r**2)/(2*l[0]*r))
        except:
            q[0]=0
        
        #################### Joint positions #######################

        xr0 =       np.dot(window_scale,[0.0,                      0.0])   #Position of the base
        xr1 = xr0 + np.dot(window_scale,[l[0]*np.cos(q[0]),        l[0]*np.sin(q[0])]) #Position of the first link
        xr2 = xr1 + np.dot(window_scale,[l[1]*np.cos(q[0]+q[1]),   l[1]*np.sin(q[0]+q[1])]) #Position of the second link
        
        #################### Draw the joints and linkages #######################
        pygame.draw.lines (screenHaptics, colorLinks, False,\
                           [(xr0[0] + xrc[0], -xr0[1] + xrc[1]), \
                            (xr1[0] + xrc[0], -xr1[1] + xrc[1])], 15) # draw links
            
        pygame.draw.lines (screenHaptics, colorLinks, False,\
                           [(xr1[0] + xrc[0]      ,-xr1[1] + xrc[1]), \
                            (xr2[0] + xrc[0]      ,-xr2[1] + xrc[1])], 14)
            
        pygame.draw.circle(screenHaptics, (0, 0, 0),\
                           (int(xr0[0]) + xrc[0] ,int(-xr0[1]) + xrc[1]), 15) # draw shoulder / base
        pygame.draw.circle(screenHaptics, (200, 200, 200),\
                           (int(xr0[0]) + xrc[0] ,int(-xr0[1]) + xrc[1]), 6) # draw shoulder / base
        pygame.draw.circle(screenHaptics, (0, 0, 0),\
                           (int(xr1[0]) + xrc[0],int(-xr1[1]) + xrc[1]), 15) # draw elbow
        pygame.draw.circle(screenHaptics, (200, 200, 200),\
                           (int(xr1[0]) + xrc[0],int(-xr1[1]) + xrc[1]), 6) # draw elbow
        pygame.draw.circle(screenHaptics, (255, 0, 0),\
                           (int(xr2[0]) + xrc[0],int(-xr2[1]) + xrc[1]), 5) # draw hand / endpoint
        
    
    ### Hand visualisation
    screenHaptics.blit(hhandle,(haptic.topleft[0],haptic.topleft[1]))
    pygame.draw.line(screenHaptics, (0, 0, 0), (haptic.center),(haptic.center+2*fk))
    
    ##Fuse it back together
    window.blit(screenHaptics, (0,0))
    window.blit(screenVR, (600,0))

    ##Print status in  overlay
    if debugToggle: 
        text = font.render("FPS = " + str(round(clock.get_fps())) + \
                            "  xm = " + str(np.round(10*xr1)/10) +\
                            "  xh = " + str(np.round(10*xh)/10) +\
                            "  fk = " + str(np.round(10*fk)/10)\
                            , True, (0, 0, 0), (255, 255, 255))
        window.blit(text, textRect)
    
    if showInfo: 
        text = font.render("Keys= " + \
                            " m: mouse, " +\
                            " q: quit," +\
                            " d: debug, " +\
                            " r: robot arm," +\
                            " Simulations= b: bump, s: record" +\
                            "w: wall" \
                            ,True, (0, 0, 0), (255, 255, 255))
        window.blit(text, textRect)


    pygame.display.flip()    
    ##Slow down the loop to match FPS
    clock.tick(FPS)

pygame.display.quit()
pygame.quit()

fp = []
x = []

for i in range(600):
    if i < mean1-variance:
        fp.append(0)
    elif i < mean1+variance:
        fp.append(height_map(i, mean1, variance))
    elif i < mean2 - variance:
        fp.append(0)
    elif i < mean2 + variance:
        fp.append(-height_map(i, mean2, variance))
    else:
        fp.append(0)
    x.append(i)

plt.figure()
plt.plot(x,fp)
plt.xlabel('x-direction [pixel]')
plt.ylabel('External Force')
plt.show()