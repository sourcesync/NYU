import numpy as np
import time
from signal import signal,SIGTERM,SIGINT
from sys import exit

import json
from test_vicon_buffer import get_now

from matplotlib import pylab,cm
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

class Graphics:

    def __init__(self,room):
        self.room = room
        self.fig,self.ax = self.DrawRoom()
        
    def DrawRoom(self):
        # make a multicoloured plot based on room partitions
        pylab.ion()
        f = pylab.figure(1)
        # ax = f.add_subplot(111)
        ax = f.add_axes([0,0,1,1]) # also see subplots_adjust to make tight

        ax.set_xlim(self.room.xmin,self.room.xmax)
        ax.set_ylim(self.room.ymin,self.room.ymax)

        ax.set_aspect('equal') # true to room
        ax.set_axis_off() # no labeling of axis

        facecolors = [cm.jet(x) for x in np.random.rand(len(self.room.partitions))]
        for i,p in enumerate(self.room.partitions):
            #walk through Room partitions and plot a colored rectangle for each
            w = p.xmax-p.xmin
            h = p.ymax-p.ymin

            ax.add_patch(Rectangle((p.xmin,p.ymin),w,h,fc=facecolors[i],alpha=1))

        pylab.draw()
        return f,ax

    def DrawBall(self,b=None):
        c = self.room.get_center()
        self.p=self.ax.plot([c[0]],[c[1]],mfc='w',mec='k',mew=5,marker='o',ms=30,ls='None')
        #pylab.plot(c[0],c[1],mfc='w',mec='k',mew=2)
        pylab.draw()

    def UpdateBall(self,x,y):
        self.p[0].set_data(x,y)
        pylab.draw()

    def HighlightPatch(self,patchnum):
        self.ax.patches[patchnum].set_alpha(0.4)
        #self.ax.patches[patchnum].set_facecolor('k')
        pylab.draw()

    def RestorePatch(self,patchnum):
        self.ax.patches[patchnum].set_alpha(1)
        #self.ax.patches[patchnum].set_facecolor('k')
        pylab.draw()

    


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


class Room:
    """ 2D room
    """

    def __init__(self,xmin,xmax,ymin,ymax):
    
        self.xmin=xmin
        self.xmax=xmax
        self.ymin=ymin
        self.ymax=ymax

    def partition(self,N,notes=None):
        """ Partition Room into N**2 regions
        Optionally take a list of notes to assign to partitions
        """
        self.partitions = []
        
        xl = np.linspace(self.xmin,self.xmax,N+1)
        yl = np.linspace(self.ymin,self.ymax,N+1)

        # create partitions
        for i in xrange(N):
            for j in xrange(N):
                idx=i*N+j #linear index
                if notes is not None:
                    # cycle through notes in notes list
                    note = notes[np.mod(idx,len(notes))]
                else:
                    note = None
                
                self.partitions.append(Partition(idx,xl[j],xl[j+1],yl[i],yl[i+1],note))
                    
    def get_center(self):
        return self.xmin+(self.xmax-self.xmin)/2,self.ymin+(self.ymax-self.ymin)/2

    def get_partition(self,x,y):
        """ Given x,y coordinates return partition index
        If x,y are out of range, returns closest partition
        """
        N = int(np.sqrt(len(self.partitions)))

        if x>=self.xmax:
            # check upper limit
            xind = N-1
        else:
            # in case we are < xmin, don't let the index go negative
            xind = np.max([0,np.argmax(x<np.linspace(self.xmin,self.xmax,N+1))-1])

        if y>=self.ymax:
            # check upper limit
            yind = N-1
        else:
            # in case we are < xmin, don't let the index go negative
            yind = np.max([0,np.argmax(y<np.linspace(self.ymin,self.ymax,N+1))-1])

        ## print xind
        ## print yind

        #linear index
        return yind*N+ xind

    def __repr__(self):
        return "Room x: (%f,%f) y: (%f,%f)" % (self.xmin,self.xmax,\
                                              self.ymin,self.ymax)
    
        

class Partition:
    """ Represents a Partition (region) of a Room
    """
    def __init__(self,idx,xmin,xmax,ymin,ymax,note=None):
        self.idx=idx
        self.xmin=xmin
        self.xmax=xmax
        self.ymin=ymin
        self.ymax=ymax
        self.note=note

    def __repr__(self):
        return "Partition x: (%f,%f) y: (%f,%f)" % (self.xmin,self.xmax,\
                                              self.ymin,self.ymax)

    def get_center(self):
        return self.xmin+(self.xmax-self.xmin)/2,self.ymin+(self.ymax-self.ymin)/2

    def isin(self,x,y):
        """ Return True if a given x,y falls within partition
        """
        if x>=self.xmin and x<self.xmax and y>=self.ymin and y<self.ymax:
            return True
        else:
            return False


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

    def get_vel(self):
        return np.linalg.norm( np.asarray(self.pos) - np.asarray(self.lastpos))


    def get_pos(self):
        return self.pos


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

    




    
