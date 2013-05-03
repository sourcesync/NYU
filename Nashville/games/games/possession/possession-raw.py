"""crowd2cloud python example: Possession game (raw).

Graham Taylor, August 2010

Dependencies: pygame, numpy, matplotlib

Options are passed from command-line.
For usage instructions:
python possession.py -h 

"""
print 'importing vp'
from vicon_proxy import ViconProxy
print 'done importing vp'
from test_vicon_buffer import get_now

import sys
import os
import time

import json

import numpy as np

# Graphics stuff
print 'aa'

import pygame
from pygame.locals import *

print 'before mpl'
import matplotlib
print 'yes'
matplotlib.use("Agg") # before importing pyplot
print 'yes2'
import matplotlib.backends.backend_agg as agg
print 'm1'

import matplotlib.pyplot as plt
print 'a1'
import matplotlib.cm as cm
print 'a2'
from matplotlib.patches import Wedge
print 'a3'
from matplotlib.collections import PatchCollection
print 'after mpl'

from optparse import OptionParser

if not pygame.mixer: print 'Warning, sound disabled'

## CLASSES ##

# Game states
(WAITING,GAMEON,GAMEDONE)=range(0,3) #like an Enum

side_count = 0
last_side_count = 0

print 'cc'
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

	if ( not os.path.exists( fullname ) ):
		print "sound does not exist!"
		
        try:
            sound = pygame.mixer.Sound(fullname)
        except pygame.error, message:
            print 'Cannot load sound:', wav
            raise SystemExit, message
        return sound


class Graphics:
    """Handles game graphics (currently through matplotlib)."""
 
    def __init__(self,options):
        """Set up 2D plot given current object positions.
        
        Object positions are given in a numobjects x 2 numpy array
        
        """
	global side_count, last_side_count
	side_count = 0
	last_side_count = 0

        self.options = options
        self.oldstyle = 0 # multicolour, oldstyle graphics
        self.options.scaling = 0 # scale markers based on area
        
        #numobjects = pos.shape[1]
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
        #self.green_goal = pygame.image.load("img/green_tiny.png")
        self.blue_goal = pygame.image.load("img/blue.png")
        self.black_goal = pygame.image.load("img/black.png")
        #self.yellow_goal = pygame.image.load("img/yellow.png")
        #self.gray_goal = pygame.image.load("img/gray.png")
        # list of all possible markers (we cycle through these)
        #self.goals = [self.red_goal,self.green_goal,self.blue_goal,self.black_goal,self.yellow_goal,self.gray_goal]
        #self.goals = [self.red_goal,self.green_goal,self.blue_goal]
        self.goals = [self.red_goal,self.blue_goal]
        self.dialog = pygame.image.load("img/dialog.png")
        self.goal_w = self.black_goal.get_width()
        self.goal_h = self.black_goal.get_height()
        self.width = self.options.width;
        self.height = self.options.height;


        self.col = None # this will represent circle collection

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


        ## if self.options.visualize_switch_xy:
        ##     if self.options.axis==1:
        ##         self.divider=self.ax.axvline(linewidth=4, c='k')
        ##     else:
        ##         self.divider=self.ax.axhline(linewidth=4, c='k')
        ## else:
        ##     if self.options.axis==1:
        ##         self.divider=self.ax.axhline(linewidth=4, c='k')
        ##     else:
        ##         self.divider=self.ax.axvline(linewidth=4, c='k')

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

        pygame.init()
        pygame.mouse.set_visible(False)
 
        self.resolution = (options.width,options.height)

        if options.startfullscreen:
            window = pygame.display.set_mode(self.resolution,pygame.FULLSCREEN) 
            self.fullscreen = 1
        else:
            window = pygame.display.set_mode(self.resolution) 
            self.fullscreen = 0

        #self.window = pygame.display.set_mode((options.width,options.height), DOUBLEBUF)

        self.screen = pygame.display.get_surface()
        self.game_screen = pygame.Surface((self.width, self.height))

        # If Vicon is waiting for raw data, want to have new background on screen
        self.game_screen.blit(self.bg,(0,0))
        self.screen.blit(self.game_screen,(0,0))

        self.set_caption("Possession: Waiting for Vicon")
 
        size = self.canvas.get_width_height()
 
        surf = pygame.image.fromstring(raw_data, size, "RGB")
        # We could blend in the matplotlib surf if we want to see 50:50 during wait
        # Otherwise we just see Kirill's background
        #self.screen.blit(surf, (0,0))
        pygame.display.flip()

    def draw_markers(self,pos,area):
        #ISPIRO
        self.game_screen.blit(self.bg,(0,0))
        numobjects=pos.shape[0]

	global side_count, last_side_count
	side_count = 0
 
        if self.oldstyle:

            facecolors = [cm.jet(x) for x in np.linspace(0,1,numobjects)]

            if self.col is not None:
                self.ax.collections.remove(self.col)

            # repeated scatter plots; this is not efficient
            # but number of markers is constantly changing
            if self.options.visualize_switch_xy:
                self.col = plt.scatter(pos[:,1],pos[:,0],c=facecolors,s=900)
            else:
                self.col = plt.scatter(pos[:,0],pos[:,1],c=facecolors,s=900)

        else:
            #print "There are %d balls" % numobjects
            numcolors=len(self.goals)
            for b in xrange(numobjects): # loop through each object and blit a R,G or B image

                #color used to just cycle
                #color=b%numcolors
                # now we use color 0 if ball is on pos side, else color 1
                if pos[b,self.options.axis]>0:
                    color=0
		    side_count += 1
                else:
                    color=1
                
                #print "Color: %d" % color
                if self.options.scaling:
                    self.game_screen.blit(pygame.transform.scale(self.goals[color],(area[b],area[b])),self.normalize(pos[b]))
                else:
                    self.game_screen.blit(self.goals[color],self.normalize(pos[b]))

    def set_caption(self,caption):
        """Set window caption."""
        pygame.display.set_caption(caption)


    # ISPIRO
    def normalize(self,xy):
        """
        Convert from Vicon coordinates to pixels.

        Care is taken here to do float division instead of integer division.
        """
        xx = (xy[0] - self.options.xmin)*1.0 / (self.options.xmax - self.options.xmin)
        yy = (xy[1] - self.options.ymin)*1.0 / (self.options.ymax - self.options.ymin)
        xpix = xx*self.width - self.goal_w*1.0/2;
        ypix = (1-yy)*self.height - self.goal_h*1.0/2;
        return (xpix,ypix)

    def erase_clock(self):
        #del self.ax.collections[1] # delete patch collection (wedge)
        self.ax.collections.remove(self.patches)

    def draw_clock(self,theta):
        # clock
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
            self.screen.blit(self.game_screen,(0,0))
            self.screen.blit(self.dialog,(190,212))
        else:
            self.screen.blit(self.game_screen,(0,0))
        
        self.screen.blit(surf,(0,0),None,pygame.BLEND_MULT)

        pygame.display.flip()


    def gameover(self,posscore,negscore,global_scores):
        """Draw game over graphics."""
        ypos = self.options.ymin+0.5*(self.options.ymax-self.options.ymin)
        scoreypos = self.options.ymin+0.33*(self.options.ymax-self.options.ymin)
        posxmiddle=self.options.xmin+0.75*(self.options.xmax-self.options.xmin)
        negxmiddle=self.options.xmin+0.25*(self.options.xmax-self.options.xmin)
        if posscore>negscore:
            # positive winner
            winnerxpos = posxmiddle
            gameoverxpos = negxmiddle
        elif negscore>posscore:
            # negative winner
            winnerxpos = negxmiddle
            gameoverxpos = posxmiddle
        else:
            #tie
            gameoverxpos = self.options.xmin+0.5*(self.options.xmax-self.options.xmin)

        if posscore<negscore or negscore<posscore:
            self.twinner = self.ax.text(winnerxpos,ypos,"WINNER!",size=100,color='g',va='center',ha='center')

        self.tgameover = self.ax.text(gameoverxpos,ypos,"GAME\nOVER",size=100,color='k',va='center',ha='center')

        self.tposglobal = self.ax.text(posxmiddle,scoreypos,
                                       "%d win" % global_scores[0] if global_scores[0]==1 else "%d wins" % global_scores[0],
                                       size=50,color='k',va='center',ha='center')
        self.tnegglobal = self.ax.text(negxmiddle,scoreypos,
                                       "%d win" % global_scores[1] if global_scores[1]==1 else "%d wins" % global_scores[1],
                                       size=50,color='k',va='center',ha='center')

    def remove_gameover_text(self):
        del self.ax.texts[-1:-5:-1] # remove last 4 elements: pos global, neg global, game over, winner

    def instructions(self):
        # hide balls, scores, dividing line
        if self.oldstyle:
            self.col.set_visible(False)
        #self.divider.set_visible(False)
        self.tpos.set_visible(False)
        self.tneg.set_visible(False)
        # instructional text
        tinfo_eng = self.ax.text(self.options.xmin+0.5*(self.options.xmax-self.options.xmin),self.options.ymin+0.5*(self.options.ymax-self.options.ymin),"Keep the balls\n on the other side!",size=80,color='k',va='center',ha='center')
        #tinfo_ger = self.ax.text(self.options.xmin+0.5*(self.options.xmax-self.options.xmin),self.options.ymin+0.33*(self.options.ymax-self.options.ymin),"Werft die Balle\n auf die andere seite!",size=60,color='k',va='center',ha='center')

        self.redraw(instr=True)
        time.sleep(3)
        tinfo_eng.set_color([0.5,0.5,0.5,1])
        #tinfo_ger.set_color([0.5,0.5,0.5,1])

    def countdown(self):
        
        for i in xrange(3,0,-1):
            tcountdown = self.ax.text(self.options.xmin+0.5*(self.options.xmax-self.options.xmin),self.options.ymin+0.5*(self.options.ymax-self.options.ymin),str(i),size=300,color='k',va='center',ha='center')
            self.redraw(instr=True)
            time.sleep(1)
            del self.ax.texts[-1] # delete last text
            self.redraw(instr=True)

        del self.ax.texts[-1] # delete instructional text (eng)
        #del self.ax.texts[-1] # delete instructional text (ger)
        # restore balls, scores, dividing line
        if self.oldstyle:
            self.col.set_visible(True)
        #self.divider.set_visible(True)
        self.tpos.set_visible(True)
        self.tneg.set_visible(True)
        self.redraw()

class Game:
    """Handles timing, score, game objects and gameplay."""
    def __init__(self,options):
	print "init game"
        self.options = options

        # Related to writing game state
        self.clock = pygame.time.Clock() # currently this clock is only used for game state writes
        self.accumulator = 0 # used for game state writes
        self.write_game_state(0,0,0,0,0,0) # dummy game state means we are waiting


        # Global scores
        self.pos_global=0
        self.neg_global=0

        self.game_time = self.options.game_time
        self.last_updated = time.time()
        
        # self.pos is a numobjects x 2 numpy array
        # which represents the x,y position of the objects
        # self.pos = np.zeros((numobjects,2))

        self.postime = 0.0
        self.negtime = 0.0
        self.bias = 0.0 # positive means x side has an advantage

        self.minarea=100
        self.maxarea=5000

        self.gameovertext=False #whether we have written GAME OVER on screen

        if self.options.vicon_file is not None:
            self.simulation_mode = True
        else:
            self.simulation_mode = False
            print "Running in live mode. "\
                  "Possession will hang here if you are not connected to a Vicon Proxy server"

	#self.simulation_mode = True

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

        self.graphics = Graphics(self.options)
        
        if self.simulation_mode:
            self.f = open(options.vicon_file,'r')
            # should we read ahead?
            for i in xrange(options.line):
                self.f.readline()
        else:
            # Initialize the object...
            print "Waiting for Vicon..."
            self.vp = ViconProxy()

        ## self.balls = []
        ## for o in self.vicon_objects:
        ##     self.balls.append(Ball(self.f if self.simulation_mode else self.vp,o) )

        self.sounds = Sounds()

        # for user keyboard input
        # cv.NamedWindow( "Status", 1)


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


        print "Game is over, here are some stats:"
        # print "postime: %6.2f" % (postime/len(vicon_objects))
        # print "negtime: %6.2f" % (negtime/len(vicon_objects))
        if self.postime+self.negtime > 0:
            print "postime: %6.2f" % posscore # (100*postime/(postime+negtime))
            print "negtime: %6.2f" % negscore # (100*negtime/(postime+negtime))
        else:
            print "No time recorded"

        if negscore>posscore:
            print "Negative side wins!"
            self.neg_global+=1
        elif posscore>negscore:
            print "Positive side wins!"
            self.pos_global+=1
        else:
            print "Tie"
            self.pos_global+=1
            self.neg_global+=1

        self.graphics.gameover(posscore,negscore,(self.pos_global,self.neg_global))
     
        self.gameovertext=True

        # self.sounds.play("gameover")
        self.sounds.gameover_sound.play()


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
                    # For some reason pygame.K_PLUS doesn't work
                    if event.key == 61:
                        self.bias += 0.1
                        print "bias %f" % self.bias
                    elif event.key == pygame.K_MINUS:
                        self.bias -= 0.1
                        print "bias %f" % self.bias
                    elif event.key == pygame.K_0:
                        self.bias = 0
                        print "bias reset"

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

                if (event.type == KEYDOWN and event.key == K_ESCAPE):
                    if self.simulation_mode:
                        self.f.close()
                    else:
                        self.vp.close()
                    done = True

                    # Delete game state file
                    if os.path.isfile(options.game_state_file):
                        print "Removing game state"
                        os.remove(options.game_state_file)
                    else:
                        print "No game state found"

                if (self.mode==WAITING and event.type == KEYUP and event.key == K_g):
                    self.mode=GAMEON
                    print "Game ON"
                    self.graphics.set_caption("Possession")

                    if self.gameovertext:
                        self.graphics.remove_gameover_text()
                        self.gameovertext=False
                    self.getready()
                    self.postime = 0.0
                    self.negtime = 0.0

                    self.graphics.draw_clock(360)

                    self.start_clock()


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
            #print "Writing game state"
            f = open(options.game_state_file, 'w')
            for s in state[:-1]:
                f.write('%f,' % s)
            f.write('%f\n' % state[-1])
            f.close()
            
    def run(self):
        """ Main game loop
        """

	global side_count, last_side_count

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

            ## for b in self.balls:
            ##     b.set_pos(obj)

            markers = get_vicon_raw(obj)

            # if we couldn't get a proper object (i.e. tracking mode)
            # then markers will be none
            # just wait for data but still handle events
            if markers is None:
                # maybe not raw mode, keep trying
                #print "Still waiting for data..."
                self.handle_events()
                continue


            # time.sleep(1.0/60)

            # If there are no markers, then markers will be an empty list
            # we cannot convert to array
            # So we create a single dummy neutral point far, far away

            if len(markers)>0:
                if options.swap:
                    # use x and z

		    #gw	
                    #self.pos = np.array(markers)[:,0:3:2] # extracts x,z 
                    #self.area = np.array(markers)[:,1] # extracts y (stores area)
                    self.pos = np.array(markers)[:,0:3:1] # extracts x,z 
                    self.area = np.array(markers)[:,2] # extracts y (stores area)
		    #gw

                    #self.area = np.random.uniform(low=self.minarea,high=self.maxarea,size=len(self.pos)) # DEBUG
                    # rescale area to be on 10,300
                    # note use of np.maximum (not np.max) to do element-wise maximum to threshold 
                    self.area=100+(np.maximum(0,self.area-self.minarea))*1.0/(self.maxarea-self.minarea)*(300-100)
                    # don't let final area go above self.maxarea
                    self.area=np.minimum(self.maxarea,self.area)
                    #print self.area # DEBUG
                else:
                    self.pos = np.array(markers)[:,0:2] # only take x,y
            else:
                print "NO MARKERS"
                if options.axis==0:
                    self.pos = np.array([[0,1000000]])

                else:
                    self.pos = np.array([[1000000,0]])
                self.area = np.array([0])

            ## c=0
            ## for b in self.balls:
            ##     pos = b.get_pos()
            ##     # update position for display
            ##     self.pos[c,:] = b.pos[0:2] # only take x,y
            ##     c+=1

            # regardless of whether we're playing game
            # update ball position on display
            #self.graphics.update_ballpos(self.pos)
            if options.swap:
                self.graphics.draw_markers(self.pos,self.area)
            else:
                self.graphics.draw_markers(self.pos)
                
            # accumulate time on each side
            if self.mode==GAMEON:
                s = self.sincelastupdate() # time since last update

                for m in markers:
                    if m[self.options.axis]>0:
                        self.postime+=s
                    elif m[self.options.axis]<0:
                        self.negtime+=s
                
		#gw        
                #for m in markers:
                #     pos=b.get_pos()
                #     changed=b.changed(self.options.axis) # did it change sides?
		#
                ##     if pos[self.options.axis] > 0:
                ##         if self.options.debug:
                ##             print "%s + " % b.objectname
                ##         self.postime += s
                ##     elif pos[self.options.axis] < 0:
                ##         if self.options.debug:
                ##             print "%s - " % b.objectname
                ##         self.negtime += s
		#
                # play sounds if ball changed sides
                #     if changed==1:
                #         # self.sounds.play("swoosh1")
                #         self.sounds.swoosh1_sound.play()
                #     elif changed==2:
                #         # self.sounds.play("swoosh2")
                #         self.sounds.swoosh2_sound.play()
		#
		if ( side_count != last_side_count ):	
			self.sounds.swoosh2_sound.play()
		last_side_count = side_count

                self.update() #update clock


            # update text if game is on
            if self.mode==GAMEON:
                timeleft = self.time_left()
                #posscore=100-round(100*self.postime/(self.postime+self.negtime+1e-9))
                #negscore=100-round(100*self.negtime/(self.postime+self.negtime+1e-9))

                #posscore=100-round(100*self.postime/(self.postime+self.negtime+1e-9))
                #negscore=100-round(100*self.negtime/(self.postime+self.negtime+1e-9))

                # sigmoid approach
                # p(pos wins) = sigmoid( negfrac-posfrac+bias)
                posfrac = self.postime/(self.postime+self.negtime+1e-9)
                negfrac = self.negtime/(self.postime+self.negtime+1e-9)
                posscore = 1/(1+np.exp(-(negfrac-posfrac+self.bias))) # on 0,1
                negscore = 100*(1-posscore)
                posscore = 100*posscore

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
def get_vicon_raw(obj):
        """Given vicon object, return list of marker positions."""
        objs = None

        # HACK FOR OLD FILE
        if options.vicon_file=='possession_sample-raw.txt':
            objs=obj['objs']
            return objs
        # END HACK

        try:
            assert(obj['mode']==0) # should be in raw mode
            objs=obj['objs']
        except:
            #print "ERROR while parsing objs. Is proxy in object mode?"
            pass
        return objs

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
    

if __name__=="__main__":
    """Parsing of command-line options and starting game play."""

    # Parsing of command-line options
    parser = OptionParser(usage = "usage: %prog [options]")

    parser.add_option("-p", "--swap",
                      action="store_true", dest="swap", default=False, help="Swap y and z axis on Vicon read (default=False)")
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

    parser.add_option("-a", "--game-axis", type="int", dest="axis", default=0, help="Game axis: 0 (Vicon x) or 1 (Vicon y) (default=0)")
    parser.add_option("-t", "--game-time", type="float", dest="game_time", default=30.0, help="Game time in seconds (default=30)")

    parser.add_option("-w", "--visualize-switch-xy",
                      action="store_true", dest="visualize_switch_xy", default=False, help="Switch xy in visualization (default False)")

    parser.add_option("--figure-width", type="int", dest="width", default=1280, help="Figure width: default 1024")
    parser.add_option("--figure-height", type="int", dest="height", default=1024, help="Figure height: default 768")

    parser.add_option("-d", "--debug",
                      action="store_true", dest="debug", default=False, help="Debug mode - extra text (default=False)")
    parser.add_option("-s", "--start-fullscreen",
                  action="store_true", dest="startfullscreen", default=False, help="Start in full-screen mode (default=False)")


    (options,args) = parser.parse_args()

    print'a'

    # Create a game
    g = Game(options)
    g.run()
