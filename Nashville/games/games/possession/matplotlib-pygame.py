import pygame
import sys
import time
import json
from pygame.locals import *

import matplotlib
matplotlib.use("Agg")
 
import matplotlib.backends.backend_agg as agg
 
import matplotlib.pyplot as plt 
#import pylab
import numpy as np
from vicon_proxy import ViconProxy
from test_vicon_buffer import get_now



def get_now_file(f,lc=[0],debug=False):
    """ Read object from json text file
    There are no static variables (for persistence) in python
    But the lc=[0] creates an anonymous list which simulates this behaviour
    http://ubuntuforums.org/showthread.php?t=403715
    """
    l = f.readline()
    lc[0]+=1

    if debug:
        if np.mod(lc[0],100)==0:
            print "line %d" % lc[0]
    try:
        obj = json.loads(l)
    except:
        print "Error parsing JSON (EOF?)"
        f.close()
        obj = None
    return obj

def update_ballpos(pos,col):
    """ Takes a [numobjects x 2 numpy] array
    representing ball positions
    updates underlying offsets on plot
    """
    if options.visualize_switch_xy:
        col.set_offsets(pos[:,::-1]) # reverse x-y direction
    else:
        col.set_offsets(pos)

class Ball:
    pos = [0.0,0.0,0.0]
    lastpos = [0.0,0.0,0.0]
    
    def __init__(self,vp,objectname):
        self.objectname = objectname
        self.simulation = isinstance(vp,file)
        if self.simulation:
            obj = get_now_file(vp)
        else:
            obj = get_now(vp)
        self.set_pos(obj)
        print "Ball created at %s" % str(self.pos)

    def set_pos(self,obj):
        pos = self.get_vicon(obj,self.objectname)
        # if object is occluded then we remember it's last known pos
        if pos is not None:
            self.lastpos = self.pos
            self.pos = pos

    def get_pos(self):
        return self.pos

    def get_lastpos(self):
        return self.lastpos


    def get_vel(self):
        return np.linalg.norm( np.asarray(self.pos) - np.asarray(self.lastpos))

    def changed(self,a):

        # print self.get_vel()
        # watch out for switches
        if self.get_vel()>1000:
            return 0
        
        if self.pos[a]>0 and self.lastpos[a]<=0:
            #print "CHANGE 1"
            return 1
        elif self.pos[a]<0 and self.lastpos[a]>=0:
            #print "CHANGE 2"
            return 2
        else:
            return 0

    def get_vicon(self,obj,objectname):
        #return 3d coordinates
        points = None

        #marker position as numpy array
        objs=obj['objs']
        if len(objs)>0:
            for o in objs:
                if o['name'] == objectname:
                    if o['oc']: # entire object is occluded
                        print "Occluded"
                        return points
                    else:
                        points = o['t']
        return points


class Options:
    xmin = xmax = ymin = ymax = 0
    visualize_switch_xy = False
    axis = 0

options = Options()

options.xmin = -5500
options.xmax = 5000
options.ymin = -6000
options.ymax = 1000
options.visualize_switch_xy = False
options.axis = 0
options.width = 1024
options.height = 768
options.vicon_file = "/Users/gwtaylor/Desktop/possession/vicon.txt"
options.line = 800

# vicon_objects = ["CBS_BB02", "CBS_BB01"]
vicon_objects = ["GouldBB01","GouldBB02"]
numobjects = len(vicon_objects)
assert(numobjects>0)


simulation_mode = False
if options.vicon_file is not None:
    simulation_mode = True

if simulation_mode:
    f = open(options.vicon_file,'r')
    # should we read ahead?
    for i in xrange(options.line):
        f.readline()
    else:
        # Initialize the object...
        vp = ViconProxy()

balls = []
for o in vicon_objects:
    balls.append(Ball(f if simulation_mode else vp,o) )

pos = np.zeros((numobjects,2))


dpi=72.0 # set dpi
# fig accepts size in inches
# so divide desired pixel width, height by dpi to get inches
w,h=(options.width/dpi,options.height/dpi)

fig = plt.figure(1,figsize=(w,h), #Inches
                 dpi=dpi,         # 100 dots per inch, so the resulting buffer is 400x400 pixels
                 )

numobjects = pos.shape[1]
#plt.ion() # turn on interactive plotting mode
#fig = plt.figure(1,figsize=(8,8))
#fig.clear()

## w = options.width/fig.get_dpi() # desired width in inches
## h = options.height/fig.get_dpi() # desired height in inches
## fig.set_size_inches(w,h,forward=True) # last arg resizes the canvas to match

ax = plt.axes()
ax.set_xlim(options.xmin,options.xmax)
ax.set_ylim(options.ymin,options.ymax)
#pyplot.axis('scaled')

# I don't know why axis('scaled') doesn't work here
# But I think the next two commands are equivalent
ax.set_aspect('equal', adjustable='box', anchor='C')
ax.set_autoscale_on(False)

# plt.draw()

#facecolors = [cm.jet(x) for x in np.random.rand(len(vicon_objects))]
facecolors = [matplotlib.cm.jet(x) for x in np.linspace(0,1,numobjects)]
if options.visualize_switch_xy:
    if options.axis==1:
        ax.axvline(linewidth=4, c='k')
    else:
        ax.axhline(linewidth=4, c='k')
    col = plt.scatter(pos[:,1],pos[:,0],c=facecolors,s=3000)
else:
    if options.axis==1:
        ax.axhline(linewidth=4, c='k')
    else:
        ax.axvline(linewidth=4, c='k')
    col = plt.scatter(pos[:,0],pos[:,1],c=facecolors,s=3000)

canvas = agg.FigureCanvasAgg(fig)
canvas.draw()
renderer = canvas.get_renderer()
raw_data = renderer.tostring_rgb()
 
pygame.init()
 
window = pygame.display.set_mode((options.width, options.height), DOUBLEBUF)
screen = pygame.display.get_surface()
 
size = canvas.get_width_height()
 
surf = pygame.image.fromstring(raw_data, size, "RGB")
screen.blit(surf, (0,0))
pygame.display.flip()

ff=0
start_time = time.time()

done = False
while not done:

    for event in pygame.event.get():
        if event.type == pygame.QUIT: sys.exit()

        if (event.type == KEYUP) or (event.type == KEYDOWN):
            print event
            if (event.key == K_ESCAPE):
                done = True
            if (event.type == KEYUP and event.key == K_g):
                print "GO!"

    if simulation_mode: # read from file
        obj = get_now_file(f,debug=True)
    else: # read from vicon_proxy
        obj = get_now(vp)

    for b in balls:
        b.set_pos(obj)

    for c,b in enumerate(balls):
        # update position for display
        pos[c,:] = b.pos[0:2] # only take x,y

    update_ballpos(pos,col)
    # print pos
    canvas.draw()
    raw_data = renderer.tostring_rgb()
    surf = pygame.image.fromstring(raw_data, size, "RGB")
    screen.blit(surf, (0,0))
    pygame.display.flip()
    ff+=1
    if ff==200:
        print "FPS %f" % (200.0/(time.time()-start_time))
        ff=0
        start_time = time.time()
    
