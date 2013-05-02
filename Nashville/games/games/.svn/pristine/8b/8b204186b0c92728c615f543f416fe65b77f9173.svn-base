#!/usr/bin/env python
#
# To kill windows:
# cv.DestroyWindow("Histogram")
# cv.DestroyWindow("CamShiftDemo")

import cv
import time
import pygame
import pygame.gfxdraw
import pygame.font
from pygame.locals import *
import numpy as np
import urllib
import math


from optparse import OptionParser

# import Image

use_grabframe = 1 #use GrabFrame/RetrieveFrame instead of QueryFrame


class Camera:
    """Represents an IP cam."""
    def __init__(self,cameraname,img_size=(640,480),fps=30,mirrored=True,avi_mode=False):
        print "Initializing %s" % cameraname
        self.name=cameraname
        self.mirrored=mirrored # display camera as mirrored (i.e. flip x)
        
        if avi_mode:
            capturestring = "save/%s.avi" % (cameraname)
        else:
            capturestring = "http://root:gr33nd0t@%s.cs.nyu.edu/axis-cgi/mjpg/video.cgi?resolution=%dx%d&fps=%d&clock=0&.mjpg" % (cameraname,img_size[0],img_size[1],fps)

        if options.fake:
            print "Faking ipcam"
            self.capture = cv.CaptureFromCAM(0)
        else:        
            self.capture = cv.CreateFileCapture(capturestring)
            
        print "%s initialized" % cameraname

class DisplayCam:

    def __init__(self):
        
        pygame.init()

        self.TARGET_FPS=30
        self.accumulator = 0
        self.clock = pygame.time.Clock() # currently this clock is only used for game state reads
        self.time = None
        self.total_time = None
        self.time = None
        self.font = pygame.font.SysFont(None,56)
        self.big_font = pygame.font.SysFont(None,220)

        
        pygame.mouse.set_visible(False)

        #Hard-coded
        self.resolution = (1024,768)
        self.cam_resolution = (640,480)

 
        if options.startfullscreen:
            window = pygame.display.set_mode(self.resolution,pygame.FULLSCREEN) 
            self.fullscreen = 1
        else:
            window = pygame.display.set_mode(self.resolution) # do not start full
            self.fullscreen = 0

        
        self.screen = pygame.display.get_surface()
        self.small_screen = pygame.Surface((self.cam_resolution))
        self.frame = pygame.image.load("img/frame2.png")

        self.cameras = []
        for c in options.cameras:
            self.cameras.append(Camera(c,img_size=self.cam_resolution,fps=self.TARGET_FPS,avi_mode=options.avi_mode))

        ## self.w = cv.GetCaptureProperty(self.capture,cv.CV_CAP_PROP_FRAME_WIDTH)
        ## self.h = cv.GetCaptureProperty(self.capture,cv.CV_CAP_PROP_FRAME_HEIGHT)
       
        #cv.NamedWindow( "DisplayCam", 1 )
     
        print( "Keys:\n"
            "    ESC - quit the program\n" )

    def advance(self):
        """Advances cameras."""
        # GrabFrame in every camera
        # Strange things happen if we only grab in the active camera
        for c in self.cameras:
            cv.GrabFrame(c.capture)

    def get_cam(self):
        p = np.random.permutation(range(len(self.cameras)))
        cam = self.cameras[p[0]] # select 1 camera
        print "camera %s" % cam.name
        pygame.display.set_caption(cam.name)
        return cam


    def run(self):

        cam = self.get_cam()        

        done=False
        while not done:
            #time.sleep(0.1)

            self.advance()

            frame = cv.RetrieveFrame(cam.capture)

            if not frame:
                print "Frame is None"
                break;

            # np.asarray only works on cvMat
            # convert IplImage to cvMat using cv.GetMat
            # Ideas from here:
            # http://code.google.com/p/pycam/source/browse/trunk/pycam/pycam/conversionUtils.py

            m = cv.GetMat(frame)
            arr = np.asarray(m)

            # Crazy! I need to reverse the RGB order in addition to taking the transpose
            # I was wondering why it looked blue and washed out
            # It was just by fluke that I discovered this
            # But this page gave me the idea that it may be reversed
            # (their comment on little-endian)
            # http://www.pygame.org/wiki/CairoPygame
            #
            # Ah, look it is also referenced here
            # Indicating that cv Image is in BGR format to begin with
            # http://code.google.com/p/pycam/source/browse/trunk/pycam/pycam/adaptors.py
            
            pygame.surfarray.blit_array(self.small_screen,arr.transpose(1,0,2)[:,:,::-1])
            scaled = pygame.transform.scale(self.small_screen, self.resolution)
            self.screen.blit(scaled,(0,0))
            self.screen.blit(self.frame,(0,0))



            timeChange = self.clock.tick()
            self.accumulator += timeChange
            
            if self.accumulator > options.game_state_period:
                self.time = None
                self.accumulator = self.accumulator - options.game_state_period
                
                if options.game_state_url is not None:
                    state_line = None
                    try:
                        f = urllib.urlopen(options.game_state_url)
                        state_line = f.read()
                    except IOError,e:
                        print "Found IO Error"
                        pass

                    if state_line is not None:
                        state_fields = state_line.split(",")
                        if len(state_fields) == 6:
                            
                            score1 = float(state_fields[0])
                            score2 = float(state_fields[1])
                            win1 = int(float(state_fields[4]))
                            win2 = int(float(state_fields[5]))
                            self.time = float(state_fields[2])
                            self.total_time = float(state_fields[3])
                            if options.team == "left":
                                self.my_score = score1
                                self.other_score = score2
                                self.wins = win1
                                self.loses = win2
                            if options.team == "right":
                                self.my_score = score2
                                self.other_score = score1
                                self.wins = win2
                                self.loses = win1

               


            if self.time is not None:
                clock_pos = (823,17)
                start_angle = 0
                rad = 90
                #self.time = self.time+1
                score_color = (0,255,0)
                if (self.my_score < 50):
                    score_color = (255,0,0)
                
                if self.time >= self.total_time:
                    stop_angle = math.pi *2
                    if self.my_score < self.other_score:
                        text_image = self.big_font.render("Game Over", True, (255,255,255))
                        self.screen.blit(text_image,(90,260))
                    else:
                        text_image = self.big_font.render("Winner!!!", True, (255,255,255))
                        self.screen.blit(text_image,(180,260))
                    
                    text_image = self.big_font.render(str(self.wins)+"-"+str(self.loses), True, (255,255,255))
                    self.screen.blit(text_image,(400,460))
                        
                else:
                    stop_angle = (1.0 - (self.time / self.total_time)) *2 * math.pi

            

                #pygame.gfxdraw.aacircle(self.screen, clock_pos[0]+rad+1, clock_pos[1]+rad, rad, (255,255,255))
                #pygame.gfxdraw.aacircle(self.screen, clock_pos[0]+rad+1, clock_pos[1]+rad+1, rad, (255,255,255))
                #pygame.gfxdraw.aacircle(self.screen, clock_pos[0]+rad, clock_pos[1]+rad+1, rad, (255,255,255))
                
                if stop_angle == math.pi * 2:
                    pygame.draw.circle(self.screen, score_color, (clock_pos[0]+rad,clock_pos[1]+rad), rad)
                else:
                    pygame.draw.arc(self.screen, score_color, (clock_pos,(rad*2,rad*2)), start_angle, stop_angle, rad)
                    pygame.draw.arc(self.screen, score_color, ((clock_pos[0]+1,clock_pos[1]),(rad*2,rad*2)), start_angle, stop_angle, rad)
                    pygame.draw.arc(self.screen, score_color, ((clock_pos[0],clock_pos[1]+1),(rad*2,rad*2)), start_angle, stop_angle, rad)
                
                pygame.gfxdraw.aacircle(self.screen, clock_pos[0]+rad, clock_pos[1]+rad, rad, (255,255,255))
                pygame.gfxdraw.aacircle(self.screen, clock_pos[0]+rad, clock_pos[1]+rad, rad+1, (255,255,255))
                pygame.gfxdraw.aacircle(self.screen, clock_pos[0]+rad, clock_pos[1]+rad, rad+2, (255,255,255))                

                total_width = 780
                total_height = 45
                
                score_width = total_width * (self.my_score / 100.0)

                pygame.draw.rect(self.screen, score_color, (18,18,score_width,total_height))
                pygame.draw.rect(self.screen, (255,255,255), (18,18,total_width,total_height), 3  )
                
                text_image = self.font.render(str(int(self.my_score))+"%", True, (0,0,0))
                self.screen.blit(text_image,(24,24))

#                text_image = self.font.render(str(int(self.wins))+"-"+str(int(self.loses)), True, (255,255,255))
#                self.screen.blit(text_image,(30,950))

                
            pygame.display.flip()

            # parr = pygame.surfarray.pixels3d(self.screen)

            # cv.ShowImage("DisplayCam", cv.fromarray(arr))

            ## pi = Image.fromstring("RGB", cv.GetSize(frame), frame.tostring())
            ## pg_img = pygame.image.frombuffer( pi.tostring(), pi.size, pi.mode )
            ## self.screen.blit( pg_img, (0,0) )
            ## pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT: sys.exit()

                if (event.type == KEYUP) or (event.type == KEYDOWN):
                    print event

                    if hasattr(event, 'key') and event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_f:
                            print "f"
                            if self.fullscreen:
                                self.fullscreen=0
                                window = pygame.display.set_mode(self.resolution)
                            else:
                                self.fullscreen=1
                                window = pygame.display.set_mode(self.resolution, pygame.FULLSCREEN)

                    if (event.type == KEYDOWN and event.key == K_ESCAPE):
                        done = True
                    if (event.type == KEYDOWN and event.key == K_SPACE):
                        # select new camera, target
                        cam = self.get_cam()
                        
        # Outside while loop
        pygame.quit()


parser = OptionParser()

parser.add_option("-o", "--object",
                      action="append", type="string", dest="objects", help="Add Vicon object")
parser.add_option("-s", "--start-fullscreen",
                  action="store_true", dest="startfullscreen", default=False, help="Start in full-screen mode (default=False)")
parser.add_option("-c", "--camera",
                      action="append", type="string", dest="cameras", help="Add ip camera")
parser.add_option("-k", "--fake-cam",
                  action="store_true", dest="fake", default=False, help="Fake ip camera with local webcam (default=False)")
parser.add_option("-a", "--avi-mode",
                  action="store_true", dest="avi_mode", default=False, help="Get camera input from avi files in save")
parser.add_option("-u", "--game-state-url",
                  action="store", type="string", dest="game_state_url", default=None, help="Load game state from URL")
parser.add_option("-t", "--team",
                  action="store", type="string", dest="team", default=None, help="Specify which team this slave represents (left/right)")
parser.add_option("-P", "--game-state-period",
                  action="store", type="float", dest="game_state_period", default=1000.0/2, help="Write game state at this period in ms (default 1/15s)")

(options,args) = parser.parse_args()


if __name__=="__main__":
    demo = DisplayCam()
    demo.run()
