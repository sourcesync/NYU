import numpy as np
import time
from signal import signal,SIGTERM,SIGINT
from sys import exit

import fluidsynth

from pantilt import PanTilt
from vicon_proxy import ViconProxy
import json
from test_vicon_buffer import get_now
from test_pantilt import Mapper, get_vicon

from optparse import OptionParser

from matplotlib import pylab,cm
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

parser = OptionParser()
parser.add_option("-x", "--xmin",
                  type="int", dest="xmin", default=-6500, help="x min")
parser.add_option("-X", "--xmax",
                  type="int", dest="xmax", default=6500, help="x max")
parser.add_option("-y", "--ymin",
                  type="int", dest="ymin", default=-5500, help="y min")
parser.add_option("-Y", "--ymax",
                  type="int", dest="ymax", default=0, help="y max")
parser.add_option("-t", "--max-time",
                  type="float", dest="max_time", default=20.0, help="max time (s)")
parser.add_option("-k", "--sequence-length",
                  type="int", dest="length", default=3, help="default sequence length")
parser.add_option("-p", "--partition-side-length",
                  type="int", dest="partition", default=2, help="default partition side length (for x and y)")
parser.add_option("-n", "--net-file",
                  action="store", type="string", dest="netfile", default="/Users/gwtaylor/python/SquidBall/pantilt/learning/nnlinear4.pkl", help="File containing neural net params")
parser.add_option("-f", "--file",
                  action="store", type="string", dest="vicon_file", default=None, help="Vicon file")
parser.add_option("-l", "--line",
                  type="int", dest="line", default=0, help="Read ahead this many lines in Vicon file (default-0)")
parser.add_option("-s", "--show-sleep",
                  type="float", dest="show_sleep", default=2.0, help="max time to sleep when showing pattern (s)")
parser.add_option("-o", "--vicon-object", dest="object", type="string", default="HugeRubberYellow", help="Object named in vicon (default='HugeRubberYellow')")
parser.add_option("-v", "--min-velocity",
                  type="float", dest="min_vel", default=20.0, help="The ball must drop below this velocity to register an entered region (default=20)")
parser.add_option("-c", "--crowd-mode", dest="crowd",
                  action="store_true", default=False, help="Crowd mode: length of sequence never increases (default=False)")


(options,args) = parser.parse_args()

simulation_mode=options.vicon_file is not None

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


class Sequence:
    """ Represents a sequence of partitions that the ball must pass through
    """
    def __init__(self,L,max_time,partitions,current_partition):
        self.L = L
        self.max_time = max_time # in seconds

        self.make_list(L,partitions,current_partition)


    def make_list(self,L,partitions,current_partition):
        """Make a list of partitions
        Don't start with current_partition
        """
        self.partitions = []
        complete_list = range(len(partitions))

        # current_partition always is with respect to the original,
        # complete partition list
        for l in xrange(L):
            # do not select previously selected partition
            valid_list = complete_list[0:current_partition] \
                         + complete_list[current_partition+1:]
            current_partition = valid_list[np.random.randint(len(valid_list))]
            self.partitions.append(partitions[current_partition])
        

    def start(self):
        self.start_time = time.time()

    def isover(self):
        return (time.time()-self.start_time)>self.max_time

    def timeleft(self):
        return self.max_time - (time.time()-self.start_time)

    def match(self):
        """We matched an element of the sequence; remove it
        """
        del self.partitions[0]


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

    


class Game:

    def __init__(self):
        self.r = Room(options.xmin,options.xmax,options.ymin,options.ymax)

        self.fs = fluidsynth.Synth()
        self.fs.start(driver="coreaudio")
        self.sfid = self.fs.sfload("./FluidR3_GM.sf2")
        self.notes = [69, 81, 74, 79, 57, 62, 67, 93, 86, 91]

        self.r.partition(options.partition,self.notes)

        try:            
            self.pt = PanTilt()
            # set to max speed
            self.pt.set_max_speed()

        except:
            print "Couldn't initialize pan-tilt unit"
            self.pt = None
            pass


        if simulation_mode:
            self.f = open(options.vicon_file,'r')
            # should we read ahead?
            for i in xrange(options.line):
                self.f.readline()
        else:
            try:            
                self.vp = ViconProxy()
            except:
                print "Couldn't initialize Vicon Proxy"
                self.vp = 0
                pass


        self.m = Mapper(options.netfile)

        self.graphics = Graphics(self.r)
        self.graphics.DrawBall()

    def show_sequence(self):
        self.fs.program_select(0,self.sfid,0,103)
        for p in self.s.partitions:
            pc = p.get_center()
            print pc
            ptmap = self.m.map(pc) # map to pan-tilt
            pan,tilt = ptmap[0]
            if self.pt:
                self.pt.go_block(pan,tilt,True)
            self.fs.noteon(0,p.note,127)

            self.graphics.HighlightPatch(p.idx)
            # raw_input("Press Enter to continue...")
            time.sleep(options.show_sleep)
            self.fs.noteoff(0,p.note)
            self.graphics.RestorePatch(p.idx)

    def run(self):

        # could adjust game params here (sequence length, etc.)

        # set up ball
        self.b = Ball(self.f if simulation_mode else self.vp,options.object)

        L = options.length
        T = options.max_time


        while True:
            print "Sequence length: %d, Time: %fs" % (L,T)
            WON_GAME = False            

            p = self.b.get_pos()
            print "Initializing sequence"
            self.s = Sequence(L,T,self.r.partitions,\
                              self.r.get_partition(p[0],p[1]))

            print "Displaying sequence"
            self.show_sequence()


            print "GAME ON"
            self.s.start()

            while not self.s.isover():

                if simulation_mode: # read from file
                    obj = get_now_file(self.f,debug=True)
                else: # read from vicon_proxy
                    obj = get_now(self.vp)
                self.b.set_pos(obj)
                p = self.b.get_pos()
                #print "ball: %s time left: %f" % (p,self.s.timeleft())
                #print "ball in partition: %d" % self.r.get_partition(p[0],p[1])

                v = self.b.get_vel()

                self.graphics.UpdateBall(p[0],p[1])
                
                #print "velocity: %f" % v

                #only look for matches if the ball has approximately stopped
                if self.s.partitions[0].isin(p[0],p[1]):
                    print "IN REGION"
                    if v<options.min_vel:
                        p = self.s.partitions[0]
                        self.fs.noteon(0,p.note,127)
                        self.graphics.HighlightPatch(p.idx)
                        print "MATCH"
                        # play the note for 1s then note off
                        # this is a bit stupid though since everything stops
                        time.sleep(1)
                        self.fs.noteoff(0,p.note)
                        self.graphics.RestorePatch(p.idx)
                        self.s.match()
                    

                if len(self.s.partitions)<1:
                    WON_GAME = True
                    break;

                time.sleep(1.0/10)
            if WON_GAME:
                print "SUCCESS! : time elapsed %fs" % (time.time()-self.s.start_time)

                # Applause
                self.fs.program_select(0,self.sfid,0,126)
                self.fs.noteon(0,60,127)
                time.sleep(1)
                self.fs.noteon(0,72,127)
                time.sleep(1)
                self.fs.noteon(0,84,127)
                time.sleep(3)
                self.fs.noteoff(0,60)
                self.fs.noteoff(0,72)
                self.fs.noteoff(0,84)

                # Adjust level
                if not options.crowd:
                    L = L+1
            else:
                print "Try again"


                # Buzzer sound
                self.fs.program_select(0,self.sfid,0,61)
                self.fs.noteon(0,36,127)
                time.sleep(1)
                self.fs.noteoff(0,36)

                time.sleep(3)

                # reset length of sequence to base
                L = options.length


        if simulation_mode:
            self.f.close()
        else:
            self.vp.close()
        if self.pt:
            self.pt.close()
        self.fs.delete()

        
if __name__ == "__main__":

    g = Game()

    def cleanup():
        print "CLEANUP"
        if simulation_mode:
            g.f.close()
        else:
            g.vp.close()
        g.pt.close()
        g.fs.delete()

    # atexit.register(cleanup)
    
    # Normal exit when killed
    signal(SIGTERM, lambda signum, stack_frame: cleanup())
    signal(SIGINT, lambda signum, stack_frame: cleanup())
    
    g.run()


    
