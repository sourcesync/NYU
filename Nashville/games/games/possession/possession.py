"""crowd2cloud python example: Possession game.

Graham Taylor, August 2010

Dependencies: pygame, numpy, matplotlib

Options are passed from command-line.
For usage instructions:
python possession.py -h 

"""
from vicon_proxy import ViconProxy,get_now

import sys
import os
import time

import json

import numpy as np

# Graphics stuff

import pygame
from pygame.locals import *

import matplotlib
matplotlib.use("Agg") # before importing pyplot
import matplotlib.backends.backend_agg as agg

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Wedge
from matplotlib.collections import PatchCollection

from optparse import OptionParser

if not pygame.mixer: print 'Warning, sound disabled'

## CLASSES ##

class Sounds:
    """Handles game sound effects."""

    def __init__(self):

        self.snd_path="./sounds"
        
        self.swoosh1_sound = self.load_sound("101432__Robinhood76__01904_air_swoosh.wav")
        self.swoosh2_sound = self.load_sound("101954__Robinhood76__01905_space_swoosh.wav")
        self.start_sound = self.load_sound("97878__Robinhood76__01817_start_beeps.wav")
        self.gameover_sound = self.load_sound("54047__guitarguy1985__buzzer.wav")
    
    def load_sound(self,wav):
        class NoneSound:
            def play(self): pass
        if not pygame.mixer:
            return NoneSound()

        #sndfile = "%s/%s" % (self.snd_path,eval(sndtag))
        fullname = os.path.join(self.snd_path,wav)

        try:
            sound = pygame.mixer.Sound(fullname)
        except pygame.error, message:
            print 'Cannot load sound:', wav
            raise SystemExit, message
        return sound


class Graphics:
    """Handles game graphics (currently through matplotlib)."""
    
    def __init__(self,options,pos):
        """Set up 2D plot given current object positions.
        
        Object positions are given in a numobjects x 2 numpy array
        
        """
        self.options = options
        numobjects = pos.shape[1]
        plt.ion() # turn on interactive plotting mode
        dpi=72.0 # set dpi (I think this is appropriate on mac)
        # fig accepts size in inches
        # so divide desired pixel width, height by dpi to get inches
        w,h=(self.options.width/dpi,self.options.height/dpi)
        fig = plt.figure(1,figsize=(w,h),dpi=dpi)
        fig.clear()

        #ISPIRO
        self.bg = pygame.image.load("img/bg.jpg")
        self.red_goal = pygame.image.load("img/red.png")
        self.green_goal = pygame.image.load("img/green.png")
        self.blue_goal = pygame.image.load("img/blue.png")
        self.black_goal = pygame.image.load("img/black.png")
        self.yellow_goal = pygame.image.load("img/yellow.png")
        self.gray_goal = pygame.image.load("img/gray.png")
        # list of all possible markers (we cycle through these)
        self.goals = [self.red_goal,self.green_goal,self.blue_goal,self.black_goal,self.yellow_goal,self.gray_goal]
        self.dialog = pygame.image.load("img/dialog.png")
        self.goal_w = self.red_goal.get_width()
        self.goal_h = self.red_goal.get_height()
        self.width = self.options.width;
        self.height = self.options.height;

        #w = self.options.width/fig.get_dpi() # desired width in inches
        #h = self.options.height/fig.get_dpi() # desired height in inches
        #fig.set_size_inches(w,h,forward=True) # last arg resizes the canvas to match

        self.ax = plt.axes([0,0,1,1]) # axes take up entire canvas
        self.ax.set_xlim(self.options.xmin,self.options.xmax)
        self.ax.set_ylim(self.options.ymin,self.options.ymax)
        #pyplot.axis('scaled')

        # I don't know why axis('scaled') doesn't work here
        # But I think the next two commands are equivalent
        self.ax.set_aspect('equal', adjustable='box', anchor='C')
        self.ax.set_autoscale_on(False)

        # No ticks on x or y axes
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        #facecolors = [cm.jet(x) for x in np.random.rand(len(vicon_objects))]
        facecolors = [cm.jet(x) for x in np.linspace(0,1,numobjects)]
        if 0: # disable matplotlib balls
            if self.options.visualize_switch_xy:
                ## if self.options.axis==1:
                ##     self.divider=self.ax.axvline(linewidth=4, c='k')
                ## else:
                ##     self.divider=self.ax.axhline(linewidth=4, c='k')
                self.col = plt.scatter(pos[:,1],pos[:,0],c=facecolors,s=3000)
            else:
                ## if self.options.axis==1:
                ##     self.divider=self.ax.axhline(linewidth=4, c='k')
                ## else:
                ##     self.divider=self.ax.axvline(linewidth=4, c='k')
                self.col = plt.scatter(pos[:,0],pos[:,1],c=facecolors,s=3000)

        # scores
        w = (self.options.xmax-self.options.xmin)
        h = (self.options.ymax-self.options.ymin)
        # based on photoshop image (note inverse y for matplotlib conversion)
        tnegx=self.options.xmin+0.075*w
        tnegy=self.options.ymin+(1-0.81)*h
        tposx=self.options.xmin+0.925*w
        tposy=self.options.ymin+(1-0.81)*h

        self.tpos = self.ax.text(tposx,tposy,str(50),
                       size=72,color='k',ha='center',va='center')
        self.tneg = self.ax.text(tnegx,tnegy,str(50),
                       size=72,color='k',ha='center',va='center')

        self.canvas = agg.FigureCanvasAgg(fig)
        self.canvas.draw()
        self.renderer = self.canvas.get_renderer()
        raw_data = self.renderer.tostring_rgb()


        pygame.mouse.set_visible(False)

        self.resolution = (options.width,options.height)

        if options.startfullscreen:
            window = pygame.display.set_mode(self.resolution,pygame.FULLSCREEN) 
            self.fullscreen = 1
        else:
            window = pygame.display.set_mode(self.resolution) # do not start full
            self.fullscreen = 0

        #self.window = pygame.display.set_mode((options.width,options.height), DOUBLEBUF)
        self.screen = pygame.display.get_surface()
        self.game_screen = pygame.Surface((self.width, self.height))


        self.set_caption("Possession: Waiting for Vicon")
 
        size = self.canvas.get_width_height()
 
        surf = pygame.image.fromstring(raw_data, size, "RGB")
        self.screen.blit(surf, (0,0))
        pygame.display.flip()

    def set_caption(self,caption):
        """Set window caption."""
        pygame.display.set_caption(caption)


    #ISPIRO
    def normalize(self,xy):
        xx = (xy[0] - self.options.xmin) / (self.options.xmax - self.options.xmin)
        yy = (xy[1] - self.options.ymin) / (self.options.ymax - self.options.ymin)
        xpix = xx*self.width - self.goal_w/2;
        ypix = (1-yy)*self.height - self.goal_h/2;
        return (xpix,ypix)

    def get_goal(self,b):
        """Return appropriate goal based on ball name.
        b is an index to the vicon balls
        If no match based on name, cycle through available colors.

        """
        name = options.objects[b]
        #print "My name is: %s" % name
        if name.find('Red')>=0:
        #    print "Red"
            return self.red_goal
        elif name.find('Green')>=0:
        #    print "Green"
            return self.green_goal
        elif name.find('Blue')>=0:
        #    print "Blue"
            return self.blue_goal
        elif name.find('Yellow')>=0:
        #    print "Yellow"
            return self.yellow_goal
        elif name.find('Black')>=0:
        #    print "Black"
            return self.black_goal
        elif name.find('Trans')>=0:
        #    print "Trans"
            return self.gray_goal
        else:
            numcolors = len(self.goals)
            color = b%numcolors
            return self.goals[color]


    def update_ballpos(self,pos):
        """ Takes a [numobjects x 2 numpy] array
        representing ball positions
        updates underlying offsets on plot
        """

        if 0:
            if self.options.visualize_switch_xy:
                self.col.set_offsets(pos[:,::-1]) # reverse x-y direction
            else:
                self.col.set_offsets(pos)
        
        #ISPIRO
        self.game_screen.blit(self.bg,(0,0))

        # print "There are %d balls" % pos.shape[0]
        numballs=pos.shape[0]
        numcolors=len(self.goals)
        for b in xrange(numballs): # loop through each object and blit a R,G or B image
            #color=b%numcolors
            goal = self.get_goal(b)
            # print "Color: %d" % color
            #self.game_screen.blit(self.goals[color],self.normalize(pos[b]))
            self.game_screen.blit(goal,self.normalize(pos[b]))

        ## self.game_screen.blit(self.red_goal,self.normalize(pos[0]))
        ## self.game_screen.blit(self.green_goal,self.normalize(pos[1]))
        ## self.game_screen.blit(self.blue_goal,self.normalize(pos[2]))

    def erase_clock(self):
        self.ax.collections.remove(self.patches) # delete patch collection (wedge)

    def draw_clock(self,theta):
        # clock

        # scores
        w = (self.options.xmax-self.options.xmin)
        h = (self.options.ymax-self.options.ymin)
        # based on photoshop image (note inverse y for matplotlib conversion)
        wedgeposx = self.options.xmin+0.9*w
        wedgeposy = self.options.ymin+(1-0.23)*h
        wedgepos = (wedgeposx,wedgeposy)
        
        ## wedgepos = (self.options.xmin+0.8*(self.options.xmax-self.options.xmin),
        ##             self.options.ymin+0.8*(self.options.ymax-self.options.ymin))
        wedgerad = 0.05*w

        w1 = Wedge(wedgepos,wedgerad,0,theta,facecolor='k',edgecolor='k',ls='solid',lw=10)
        w2 = Wedge(wedgepos,wedgerad,theta,360,facecolor='w',edgecolor='k',ls='solid',lw=10)
        patches = [w1,w2]

        # note that match_original must be set to True for colors to take effect
        # otherwise colormap will be applied
        self.patches = PatchCollection(patches, match_original=True)

        self.ax.add_collection(self.patches)

    def update_scores(self,posscore,negscore):
        self.tpos.set_text("%d" % posscore)
        self.tneg.set_text("%d" % negscore)
        if posscore>negscore:
            self.tpos.set_color('g')
            self.tneg.set_color('r')
        elif posscore<negscore:
            self.tpos.set_color('r')
            self.tneg.set_color('g')
        else:
            self.tpos.set_color('k')
            self.tneg.set_color('k')

    def redraw(self,instr=False):
        #plt.draw()

        self.canvas.draw()
        raw_data = self.renderer.tostring_rgb()
        size = self.canvas.get_width_height()
        surf = pygame.image.fromstring(raw_data, size, "RGB")
	#GWT ORIGINAL:
        #self.screen.blit(surf,(0,0))
        if instr:
            # draw dialog with lighter center region
            self.screen.blit(self.dialog,(190,212))
        else:
            self.screen.blit(self.game_screen,(0,0))
        self.screen.blit(surf,(0,0),None,pygame.BLEND_MULT)
        #ISPIRO - optionally draw the dialog mask
        #self.screen.blit(self.dialog,(190,212))

        pygame.display.flip()

    def gameover(self,postime,negtime):
        """Draw game over graphics."""
        ypos = self.options.ymin+0.5*(self.options.ymax-self.options.ymin)
        if postime<negtime:
            # positive winner
            winnerxpos = self.options.xmin+0.75*(self.options.xmax-self.options.xmin)
            gameoverxpos = self.options.xmin+0.25*(self.options.xmax-self.options.xmin)
        elif postime>negtime:
            # negative winner
            winnerxpos = self.options.xmin+0.25*(self.options.xmax-self.options.xmin)
            gameoverxpos = self.options.xmin+0.75*(self.options.xmax-self.options.xmin)
        else:
            #tie
            gameoverxpos = self.options.xmin+0.5*(self.options.xmax-self.options.xmin)

        if postime<negtime or negtime<postime:
            self.twinner = self.ax.text(winnerxpos,ypos,"WINNER!",size=100,color='g',va='center',ha='center')

        self.tgameover = self.ax.text(gameoverxpos,ypos,"GAME\nOVER",size=100,color='k',va='center',ha='center')

    def remove_gameover_text(self):
        del self.ax.texts[-1] # game over
        del self.ax.texts[-1] # winner


    def instructions(self):
        # hide balls, scores, dividing line
        # self.col.set_visible(False)
        # self.divider.set_visible(False)
        self.tpos.set_visible(False)
        self.tneg.set_visible(False)
        # instructional text
        tinfo_eng = self.ax.text(self.options.xmin+0.5*(self.options.xmax-self.options.xmin),self.options.ymin+0.66*(self.options.ymax-self.options.ymin),"Keep the balls\n on the other side!",size=60,color='k',va='center',ha='center')
        tinfo_ger = self.ax.text(self.options.xmin+0.5*(self.options.xmax-self.options.xmin),self.options.ymin+0.33*(self.options.ymax-self.options.ymin),"Werft die Balle\n auf die andere seite!",size=60,color='k',va='center',ha='center')

        self.redraw(instr=True)
        time.sleep(3)
        tinfo_eng.set_color([0.5,0.5,0.5,1])
        tinfo_ger.set_color([0.5,0.5,0.5,1])

    def countdown(self):
        
        for i in xrange(3,0,-1):
            tcountdown = self.ax.text(self.options.xmin+0.5*(self.options.xmax-self.options.xmin),self.options.ymin+0.5*(self.options.ymax-self.options.ymin),str(i),size=300,color='k',va='center',ha='center')
            self.redraw(instr=True)
            time.sleep(1)
            del self.ax.texts[-1] # delete last text
            self.redraw(instr=True)

        del self.ax.texts[-1] # delete instructional text (eng)
        del self.ax.texts[-1] # delete instructional text (ger)
        # restore balls, scores, dividing line
        # self.col.set_visible(True)
        #self.divider.set_visible(True)
        self.tpos.set_visible(True)
        self.tneg.set_visible(True)
        self.redraw()



class Ball:
    """Represents a Vicon object."""
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
        try:
            assert(obj['mode']==1) # should be in object mode
            # marker position as numpy array
            objs=obj['objs']
        except:
            print "ERROR while parsing objs. Is proxy in raw mode?"
            return points
        try:
            if len(objs)>0:
                for o in objs:
                    if o['name'] == objectname:
                        if o['oc']: # entire object is occluded
                            print "Occluded"
                            return points
                        else:
                            points = o['t']
        except:
            print "ERROR while parsing objs. Is proxy in raw mode?"
            pass
        return points


class Game:
    """Handles timing, score, game objects and gameplay."""
    def __init__(self,options):
        self.options = options

        pygame.init()

        # Related to writing game state
        self.clock = pygame.time.Clock() # currently this clock is only used for game state writes
        self.accumulator = 0 # used for game state writes
        self.write_game_state(0,0,0,0,0,0) # dummy game state means we are waiting


        # Global scores
        self.pos_global=0
        self.neg_global=0

        self.game_time = self.options.game_time
        self.last_updated = time.time()


        if self.options.vicon_file is not None:
            self.simulation_mode = True
        else:
            self.simulation_mode = False
            print "Running in live mode. "\
                  "Possession will hang here if you are not connected to a Vicon Proxy server"

        if self.simulation_mode:
            self.f = open(options.vicon_file,'r')
            # should we read ahead?
            for i in xrange(options.line):
                self.f.readline()
        else:
            # Initialize the object...
            print "Waiting for Vicon..."
            self.vp = ViconProxy()

        #Vicon objects as defined on command-line
        if options.objects is None:
            print "Automatic object detection mode"
            while options.objects is None:
                print "Looking for objects..."
                options.objects = self.lookfor_objects()
                # Handle events
                self.handle_events()
        else:
            print "Objects are defined on command line"

        self.vicon_objects = self.options.objects

        try:
            assert(self.vicon_objects is not None)
        except:
            print "Make sure you define 1 or more vicon objects through -o"
            sys.exit(1)
        
        numobjects = len(self.vicon_objects)

        # self.pos is a numobjects x 2 numpy array
        # which represents the x,y position of the objects
        self.pos = np.zeros((numobjects,2))

        self.postime = 0.0
        self.negtime = 0.0
        self.gameovertext=False #whether we have written GAME OVER on screen


        # Currently our graphics implementation (matplotlib + pygame) is a bit hacky
        # Matplotlib is set to have equal axes (in terms of units)
        # Therefore the aspect ratio of the room must equal the aspect ratio of the fixed
        # Screen resolution for matplotlib and pygame to line up
        # So here we make sure that aspect ratios are the same
        aspect = float(self.options.height)/self.options.width
        print "DEBUG: screen aspect %f" % aspect
        # keep x, but change y
        roomw = (self.options.xmax-self.options.xmin)
        roomh = (self.options.ymax-self.options.ymin)
        targetroomh = roomw*aspect
        print "DEBUG: room w: %f, target room h: %f" % (roomw,targetroomh)
        roomhcenter = self.options.ymin+roomh/2.0
        # update room y to match target aspect ratio
        self.options.ymin = roomhcenter - targetroomh/2.0
        self.options.ymax = roomhcenter + targetroomh/2.0
        print "DEBUG: new ymin: %f, new ymax: %f" % (self.options.ymin,self.options.ymax)


        self.graphics = Graphics(self.options,self.pos)
        

        self.balls = []
        for o in self.vicon_objects:
            self.balls.append(Ball(self.f if self.simulation_mode else self.vp,o) )

        self.sounds = Sounds()

        # for user keyboard input
        # cv.NamedWindow( "Status", 1)

    def lookfor_objects(self):
        objects = None
        
        if self.simulation_mode: # read from file
            obj = get_now_file(self.f,debug=True)
        else: # read from vicon_proxy
            obj = get_now(self.vp)

        try:
            assert(obj['mode']==1) # should be in object mode
            # list of objects
            objs=obj['objs']
        except:
            print "ERROR while parsing objs. Is proxy in raw mode?"
            return objects

        # add every object currently broadcasting
        objects = []    
        for o in objs:
            objects.append(o['name'])
            print "Added: %s" % o['name']

        return objects

    def start_clock(self):
        self.clock_started = time.time()
        self.last_updated = self.clock_started

    def time_elapsed(self):
        return time.time()-self.clock_started
    def time_left(self):
        return self.game_time - self.time_elapsed()

    def update(self):
        self.last_updated = time.time()

    def sincelastupdate(self):
        return time.time()-self.last_updated

    def isgameover(self):
        return self.time_elapsed() >= self.game_time


    def gameover(self,posscore,negscore):        
        self.graphics.erase_clock()

        self.graphics.gameover(self.postime,self.negtime)
     
        self.gameovertext=True

        # self.sounds.play("gameover")
        self.sounds.gameover_sound.play()

        print "Game is over, here are some stats:"
        # print "postime: %6.2f" % (postime/len(vicon_objects))
        # print "negtime: %6.2f" % (negtime/len(vicon_objects))
        if self.postime+self.negtime > 0:
            print "postime: %6.2f" % posscore # (100*postime/(postime+negtime))
            print "negtime: %6.2f" % negscore # (100*negtime/(postime+negtime))
        else:
            print "No time recorded"

        if self.postime > self.negtime:
            print "Negative side wins!"
            self.neg_global+=1
        elif self.negtime > self.postime:
            print "Positive side wins!"
            self.pos_global+=1
        else:
            print "Tie"
            self.pos_global+=1
            self.neg_global+=1

        print "Press 'g' to play again"

    def getready(self):
        # self.sounds.play("start")
        self.accumulator = 0 # used for game state writes
        self.graphics.instructions()
        self.sounds.start_sound.play()
        self.graphics.countdown()

    def handle_events(self):

        # Event handling through pygame
        # Currently only two keystrokes: ESC and 'g'
        for event in pygame.event.get():
            if event.type == pygame.QUIT: sys.exit()

            if (event.type == KEYUP) or (event.type == KEYDOWN):
                print event

                if hasattr(event, 'key') and event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_f:
                        print "f"
                        if self.graphics.fullscreen:
                            self.graphics.fullscreen=0
                            window = pygame.display.set_mode(self.graphics.resolution)
                        else:
                            self.graphics.fullscreen=1
                            window = pygame.display.set_mode(self.graphics.resolution, pygame.FULLSCREEN)


                if hasattr(event, 'key') and event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_1:
                        print "timing 10"
                        self.game_time = 10.0
                    elif event.key == pygame.K_3:
                        print "timing 30"
                        self.game_time = 30.0
                    elif event.key == pygame.K_6:
                        print "timing 60"
                        self.game_time = 60.0
                    elif event.key == pygame.K_9:
                        print "timing 90"
                        self.game_time = 90.0

                if (event.key == K_ESCAPE):
                    if self.simulation_mode:
                        self.f.close()
                    else:
                        self.vp.close()
                    done = True
                    # Delete game state file
                    print "Removing game state"
                    os.remove(options.game_state_file)

                if (self.mode==WAITING and event.type == KEYUP and event.key == K_g):
                    self.mode=GAMEON
                    print "Game ON"
                    self.graphics.set_caption("Possession")

                    if self.gameovertext:
                        self.graphics.remove_gameover_text()
                    self.getready()
                    self.postime = 0.0
                    self.negtime = 0.0

                    self.graphics.draw_clock(360)

                    self.start_clock()

                if (event.type == KEYDOWN and event.key == K_u):
                    print "Update Vicon objects"
                    options.objects = None
                    while options.objects is None:
                        print "Looking for objects..."
                        options.objects = self.lookfor_objects()
                        # Handle events
                        self.handle_events()
                    self.vicon_objects = self.options.objects


    def write_game_state(self,*state):
        """Writes the game state to a text file at a specific period.
        Name of file and period are set by command-line.

        """
        #print "Considering game state write"
        timeChange = self.clock.tick()
        self.accumulator += timeChange
        #print "Accumulator: %f" % self.accumulator
        if self.accumulator > options.game_state_period:
            self.accumulator = self.accumulator - options.game_state_period
            print "Writing game state"
            f = open(options.game_state_file, 'w')
            for s in state[:-1]:
                f.write('%f,' % s)
            f.write('%f\n' % state[-1])
            f.close()
            
    def run(self):
        """ Main game loop
        """

        self.mode = WAITING
        self.graphics.set_caption("Possession: Press 'g' to start")
        print "Press 'g' to start play"
        #mode = GAMEON # debug
        #reset_clock()

        done = False
        while not done:

            if self.simulation_mode: # read from file

                obj = get_now_file(self.f,debug=self.options.debug)

            else: # read from vicon_proxy

                obj = get_now(self.vp)
            for b in self.balls:
                b.set_pos(obj)

            # time.sleep(1.0/60)

            c=0
            for b in self.balls:
                pos = b.get_pos()
                # update position for display
                self.pos[c,:] = b.pos[0:2] # only take x,y
                c+=1

            # regardless of whether we're playing game
            # update ball position on display
            self.graphics.update_ballpos(self.pos)
            
            # accumulate time on each side
            if self.mode==GAMEON:
                s = self.sincelastupdate() # time since last update

                for b in self.balls:
                    pos=b.get_pos()
                    changed=b.changed(self.options.axis) # did it change sides?

                    if pos[self.options.axis] > 0:
                        if self.options.debug:
                            print "%s + " % b.objectname
                        self.postime += s
                    elif pos[self.options.axis] < 0:
                        if self.options.debug:
                            print "%s - " % b.objectname
                        self.negtime += s

                    # play sounds if ball changed sides
                    if changed==1:
                        # self.sounds.play("swoosh1")
                        self.sounds.swoosh1_sound.play()
                    elif changed==2:
                        # self.sounds.play("swoosh2")
                        self.sounds.swoosh2_sound.play()


                self.update() #update clock


            # update text if game is on
            if self.mode==GAMEON:
                timeleft = self.time_left()
                posscore=100-round(100*self.postime/(self.postime+self.negtime+1e-9))
                negscore=100-round(100*self.negtime/(self.postime+self.negtime+1e-9))

                # I don't see a set_radius command
                # So I just remove the patch collection, recreate wedge, collection
                # and add collection again
                self.graphics.erase_clock()
                # clock takes a single argument - hand position in degrees
                self.graphics.draw_clock(360*timeleft/self.game_time)
                self.graphics.update_scores(posscore,negscore)

            # regardless of whether we are playing or not, update plot
            self.graphics.redraw()


            if self.mode==GAMEON and g.isgameover():
                self.mode=GAMEDONE

                self.gameover(posscore,negscore)
                # do one last write
                self.write_game_state(posscore,negscore,self.time_elapsed(),self.game_time,self.pos_global,self.neg_global)

                self.mode=WAITING
                self.graphics.set_caption("Possession: Press 'g' to start")

            if self.mode==GAMEON:
                self.write_game_state(posscore,negscore,self.time_elapsed(),self.game_time,self.pos_global,self.neg_global)
            self.handle_events()


        # Outside while loop
        pygame.quit()


## GLOBAL FUNCTIONS ##

def get_now_file(f,lc=[0],debug=False):
    """ Read object from json text file; replacement for get_now().
    
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
    


# Game states
(WAITING,GAMEON,GAMEDONE)=range(0,3) #like an Enum

if __name__=="__main__":
    """Parsing of command-line options and starting game play."""

    # Parsing of command-line options
    parser = OptionParser(usage = "usage: %prog [options] -oViconObject1 [-oViconObject2 ...]")

    parser.add_option("-x", "--xmin",
                      type="int", dest="xmin", default=-4000, help="x min")
    parser.add_option("-X", "--xmax",
                      type="int", dest="xmax", default=4000, help="x max")
    parser.add_option("-y", "--ymin",
                      type="int", dest="ymin", default=-4000, help="y min")
    parser.add_option("-Y", "--ymax",
                      type="int", dest="ymax", default=4000, help="y max")
    parser.add_option("-F", "--game-state-file",
                      action="store", type="string", dest="game_state_file", default="./data.txt", help="Game state file (default './data.txt')")
    parser.add_option("-P", "--game-state-period",
                      action="store", type="float", dest="game_state_period", default=1000.0/15, help="Write game state at this period in ms (default 1/15s)")
    parser.add_option("-f", "--file",
                      action="store", type="string", dest="vicon_file", default=None, help="Vicon file")
    parser.add_option("-l", "--line",
                      type="int", dest="line", default=0, help="Read ahead this many lines in Vicon file (default-0)")

    parser.add_option("-o", "--object",
                      action="append", type="string", dest="objects", help="Add Vicon object")
    parser.add_option("-a", "--game-axis", type="int", dest="axis", default=0, help="Game axis: 0 (Vicon x) or 1 (Vicon y) (default=0)")
    parser.add_option("-t", "--game-time", type="float", dest="game_time", default=30.0, help="Game time in seconds (default=30)")

    parser.add_option("-w", "--visualize-switch-xy",
                      action="store_true", dest="visualize_switch_xy", default=False, help="Switch xy in visualization (default False)")

    parser.add_option("--figure-width", type="int", dest="width", default=1280, help="Figure width: default 1280")
    parser.add_option("--figure-height", type="int", dest="height", default=1024, help="Figure height: default 1024")

    parser.add_option("-d", "--debug",
                      action="store_true", dest="debug", default=False, help="Debug mode - extra text (default=False)")
    parser.add_option("-s", "--start-fullscreen",
                  action="store_true", dest="startfullscreen", default=False, help="Start in full-screen mode (default=False)")
    (options,args) = parser.parse_args()


    # Create a game
    g = Game(options)
    g.run()
