from simon_classes import *
import pygame

from pantilt import PanTiltNetwork
from test_pantilt import Mapper
from optparse import OptionParser
import sys
import numpy as np

from vicon_proxy import ViconProxy

import fluidsynth

parser = OptionParser()

parser.add_option("-r", "--room-configuration",
                  action="store", type="string", dest="roomconfig", default="room_config.txt", help="Room configuration file")
parser.add_option("-f", "--file",
                      action="store", type="string", dest="vicon_file", default=None, help="Vicon file")
parser.add_option("-o", "--object",
                      action="append", type="string", dest="objects", help="Add Vicon object")
parser.add_option("-l", "--line",
                      type="int", dest="line", default=0, help="Read ahead this many lines in Vicon file (default-0)")
parser.add_option("-v", "--min-velocity",
                  type="float", dest="min_vel", default=1000.0, help="The ball must drop below this velocity to register an entered region (default=1000)")
parser.add_option("-s", "--show-sleep",
                  type="float", dest="show_sleep", default=1.0, help="max time to sleep when showing pattern (s)")


(options,args) = parser.parse_args()


class Sound:
    """ Handles game sounds."""

    ## def instrument1(self):
    ##     self.fs.program_select(0,self.sfid,0,103)

    ## def instrument2(self):
    ##     self.fs.program_select(0,self.sfid,0,77)

    ## def instrument3(self):
    ##     self.fs.program_select(0,self.sfid,0,9)

    ## def instrument_applause(self):
    ##     self.fs.program_select(0,self.sfid,0,126)

    def set_instrument(self,i):
        self.fs.program_select(0,self.sfid,0,i)
        
    def play_note(self,note,instrument=0):
        self.set_instrument(instrument)
        self.fs.noteon(0,note,127)

    def note_off(self,note):
        self.fs.noteoff(0,note)

    def play_applause(self):
        self.set_instrument(126)
        self.fs.noteon(0,60,127)
        time.sleep(1)
        self.fs.noteon(0,72,127)
        time.sleep(1)
        self.fs.noteon(0,84,127)
        time.sleep(3)
        self.fs.noteoff(0,60)
        self.fs.noteoff(0,72)
        self.fs.noteoff(0,84)

    def __init__(self):
        self.fs = fluidsynth.Synth()
        self.fs.start(driver="coreaudio")

        self.sfid = self.fs.sfload("FluidR3_GM.sf2")

    def test(self):

        self.fs.program_select(0, self.sfid, 0, 0)

        self.fs.noteon(0, 60, 30)
        self.fs.noteon(0, 67, 30)
        self.fs.noteon(0, 76, 30)

        time.sleep(1.0)

        self.fs.noteoff(0, 60)
        self.fs.noteoff(0, 67)
        self.fs.noteoff(0, 76)

        time.sleep(1.0)


class Game:

    def __init__(self,roomconfig):

        self.TARGET_FPS = 15
        pygame.init()
        self.clock = pygame.time.Clock()
        self.roomlist = self.build_rooms(roomconfig)

        try: 
            self.sound = Sound()
        except:
            print "Error initializing sound"
            sys.exit(1)

        try:            
            self.pt = PanTiltNetwork([1,2])
            # set to max speed
            self.pt.set_max_speed()
        except:
            print "Couldn't initialize pan-tilt unit"
            self.pt = None
            pass

        try:
            assert(options.objects is not None)
        except:
            print "Make sure you define 1 or more vicon objects through -o"
            sys.exit(1)

        if options.vicon_file is not None:
            self.simulation_mode = True
        else:
            self.simulation_mode = False
            print "Running in live mode. "\
                  "Game will hang here if you are not connected to a Vicon Proxy server"

        if self.simulation_mode:
            self.f = open(options.vicon_file,'r')
            # should we read ahead?
            for i in xrange(options.line):
                self.f.readline()
        else:
            # Initialize the object...
            print "Waiting for Vicon..."
            self.vp = ViconProxy()

        # set up balls
        self.balls = []
        for o in options.objects:
            self.balls.append(Ball(self.f if self.simulation_mode else self.vp,o))

        # initial targets
        self.targets = [None]*len(self.roomlist) # target for each room
        for t in xrange(len(self.targets)):
            self.set_target(t)
        print self.targets
        for tidx in xrange(len(self.targets)):
            self.show_target(tidx)


    def run(self):
        done = False
        while not done:

            timeChange = self.clock.tick(self.TARGET_FPS)

            if self.simulation_mode: # read from file
                obj = get_now_file(self.f,debug=True)
            else: # read from vicon_proxy
                obj = get_now(self.vp)

            for j,b in enumerate(self.balls):
                b.set_pos(obj)
                print b.get_pos()

            # determine whether any of the balls have made the target
            for i,t in enumerate(self.targets):
                for j,b in enumerate(self.balls):
                    pos = b.get_pos()
                    vel = b.get_vel()

                    if t.isin(pos[0],pos[1]):
                        print "Ball %d: %d,%d in target %d" % (j,pos[0],pos[1],i)
                        print self.targets[i]

                        if vel<options.min_vel:
                            # t.note
                            print "MATCH"
                            self.sound.play_note(t.note,self.roomlist[i].instrument)
                            time.sleep(options.show_sleep)
                            self.sound.note_off(t.note)
                            # self.sound.play_applause()
                            self.set_target(i)
                            self.show_target(i)
                        

    def show_target(self,tidx):
        """ Show a given target with pan-tilt and sound. """
        t = self.targets[tidx]
        r = self.roomlist[tidx]

        pc = t.get_center()
        print "center: %d,%d" % (pc[0],pc[1])
        ptmap = r.mapper.map(pc)
        pan,tilt = ptmap[0]
        if self.pt:
            # select appropriate pan-tilt for this room
            self.pt.select(r.pantilt)
            self.pt.go_block(pan,tilt,True)
        self.sound.play_note(t.note,r.instrument)
        time.sleep(options.show_sleep)
        self.sound.note_off(t.note)
        
            
    def set_target(self,tidx):
        """ Choose targets (but do not choose partitions where balls are currently located).
        Specify target to set.

        """
        r = self.roomlist[tidx]
        #print "Room %d" % i
        bad = [] # this is a list of partitions in which any of the balls are located
        for b in self.balls:
            for n,p in enumerate(r.partitions):
                pos = b.get_pos()
                if p.isin(pos[0],pos[1]):
                    #print "Ball found in partition %d" % n
                    bad.append(n)

        # set difference between all partitions and bad partitions
        valid_list = list(set(range(len(r.partitions)))-set(bad))
        if len(valid_list)==0:
            # many balls, no valid partitions?
            current_partition = np.random.randint(len(r.partitions))
        else:
            current_partition = valid_list[np.random.randint(len(valid_list))]

        self.targets[tidx] = r.partitions[current_partition]
    #print self.targets
            

    def build_rooms(self,roomconfig):
        """Build rooms based on text file."""
        notes = [69, 81, 74, 79, 57, 62, 67, 93, 86, 91]
                
        roomlist = []
        f = open(roomconfig,'r')
        while True:
            line = f.readline()
            if not line: break
            if not line.startswith('#'): # ignore comments
                xmin,xmax,ymin,ymax,n,p,m,inst = line.split() # split on spaces
                #r = Room(*[int(n) for n in xmin,xmax,ymin,ymax]) # why is this so slow?
                r = Room(int(xmin),int(xmax),int(ymin),int(ymax))
                print "Room created"
                #r = Room(-400,400,-200,200)
                r.partition(int(n),notes)
                r.pantilt=int(p)
                r.mapper = Mapper(m) # will be a mapper for this specific pan-tilt
                r.instrument = int(inst)
                roomlist.append(r)
                
        f.close()
        return roomlist

print options.roomconfig
g = Game(options.roomconfig)
g.run()
