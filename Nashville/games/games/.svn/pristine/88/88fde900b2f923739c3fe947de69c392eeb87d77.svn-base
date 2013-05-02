import cv
import time
import pygame
from pygame.locals import *
from matplotlib import cm
import numpy as np
import os,sys
import json

from vicon_proxy import ViconProxy
from test_vicon_buffer import get_now


from optparse import OptionParser

if not pygame.mixer: print 'Warning, sound disabled'

# Game states
(WAITING,GAMEON,GAMEDONE)=range(0,3) #like an Enum

class Sounds:
    """Handles game sound effects."""

    def __init__(self):

        self.snd_path="./sounds"
        
        #self.swoosh1_sound = self.load_sound("101432__Robinhood76__01904_air_swoosh.wav")
        self.swoosh1_sound = self.load_sound("13936__adcbicycle__8.wav")
        self.gameover_sound = self.load_sound("54047__guitarguy1985__buzzer.wav")

    
    def load_sound(self,wav):
        class NoneSound:
            def play(self): pass
        if not pygame.mixer:
            return NoneSound()

        #sndfile = "%s/%s" % (self.snd_path,eval(sndtag))
        fullname = os.path.join(self.snd_path,wav)
        print fullname
        try:
            sound = pygame.mixer.Sound(fullname)
        except pygame.error, message:
            print 'Cannot load sound:', wav
            raise SystemExit, message
        return sound


class Target:
    """Places a target on a camera."""
    
    def __init__(self,camera,radius=30):
        self.radius=radius
        self.minx=0+radius
        self.miny=0+radius
        self.maxx=cv.GetCaptureProperty(camera.capture,cv.CV_CAP_PROP_FRAME_WIDTH)-radius
        self.maxy=cv.GetCaptureProperty(camera.capture,cv.CV_CAP_PROP_FRAME_HEIGHT)-radius
        # Randomly pick a target location
        print "x %f:%f" % (self.minx,self.maxx+1)
        self.x = np.random.randint(self.minx,self.maxx+1)
        print "y %f:%f" % (self.miny,self.maxy+1)
        self.y = np.random.randint(self.miny,self.maxy+1)
        self.rot = 0
    
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

        if len(objs)>0:
            for o in objs:
                if o['name'] == objectname:
                    if o['oc']: # entire object is occluded
                        #print "Occluded"
                        return points
                    else:
                        points = o['t']
        return points

class Camera:
    """Represents an IP cam."""
    def __init__(self,cameraname,img_size=(640,400),fps=30,mirrored=True):
        print "Initializing %s" % cameraname
        self.name=cameraname
        self.mirrored=mirrored # display camera as mirrored (i.e. flip x)
        capturestring = "http://root:gr33nd0t@%s.cs.nyu.edu/axis-cgi/mjpg/video.cgi?resolution=%dx%d&fps=%d&clock=0&.mjpg" % (cameraname,img_size[0],img_size[1],fps)

        # from where to read camera parameters
        if options.vicon_file is not None:
            # canned camera parameters
            read_path = "./save/params"                    
        else:
            read_path = "../../ipcams/calibration/params"
                
        read_path = "%s/%s" % (read_path,cameraname)

        intrinsics,distortion,rvec,tvec=self.get_params(read_path)
        self.intrinsics=intrinsics
        self.distortion=distortion
        self.rvec=rvec
        self.tvec=tvec

        if options.fake:
            print "Faking ipcam"
            self.capture = cv.CaptureFromCAM(0)
        elif options.vicon_file is not None:
            capturestring = "save/%s.avi" % cameraname
            print "Loading from %s" % capturestring
            self.capture = cv.CreateFileCapture(capturestring)
        else:        
            self.capture = cv.CreateFileCapture(capturestring)
            
        print "%s initialized" % cameraname

        # ProjectPoints2 doesn't seem to work when there is a single point
        # so use a minimum of 4 (bigger if there are more than 2 objects)
        # weirdness is in here:
        # /Users/gwtaylor/src/OpenCV-2.1.0/src/cv/cvfundam.cpp (around l 870)
        # Seems to determine # points based on whether rows or columns is larger
        # so if input is 2x3 it thinks there are 3 points rather than 2 3D points
        MIN_NUM_PTS = 4
        num_pts = max(MIN_NUM_PTS,len(options.objects))
        self.object_points = cv.CreateMat(num_pts, 3, cv.CV_32FC1) #3D
        cv.Zero(self.object_points) #they should have sensible default values
        self.image_points = cv.CreateMat(num_pts, 2, cv.CV_32FC1)  #2D

    def project(self):
        cv.ProjectPoints2(self.object_points, self.rvec, self.tvec,
                          self.intrinsics, self.distortion, self.image_points)

    def match(self,target,ballidx):
        # compute distance from target center to ball pos in camera view
        b = np.array((self.image_points[ballidx,0],self.image_points[ballidx,1]))
        t = np.array( ( target.x,target.y))
        d = np.linalg.norm(b - t)
        return d< target.radius


    def get_params(self,p):

        read_file = "%s/%s.yml" % (p,'intrinsics')
        if os.path.exists(read_file):
            intrinsics=(cv.Load(read_file))
        else:
            sys.exit("Problem reading %s" % read_file)

        read_file = "%s/%s.yml" % (p,'distortion')
        if os.path.exists(read_file):
            distortion=(cv.Load(read_file))
        else:
            sys.exit("Problem reading %s" % read_file)

        read_file = "%s/%s.yml" % (p,'ext_r')
        if os.path.exists(read_file):
            ext_r=(cv.Load(read_file))
        else:
            sys.exit("Problem reading %s" % read_file)

        read_file = "%s/%s.yml" % (p,'ext_t')
        if os.path.exists(read_file):
            ext_t=(cv.Load(read_file))
        else:
            sys.exit("Problem reading %s" % read_file)


        return intrinsics,distortion,ext_r,ext_t


class Game1:
    def __init__(self):
        pygame.init()
        self.sounds = Sounds()

    def run(self):
        self.sounds.swoosh1_sound.play()
        time.sleep(10)

class Game:

    def __init__(self):

        self.TARGET_FPS = 30
        cam_w,cam_h=(640,480)
        timer_w=480
        # self.resolution = (cam_w+timer_w,cam_h)
        
        pygame.init()
        pygame.mouse.set_visible(False)

        # get resolution
        # OVERRIDE - THIS IS SUDDENLY SUPER BUGGY ON MY MACHINE (BUT NOT IAN'S)???
        # self.fullresolution = (int(pygame.display.Info().current_w), int(pygame.display.Info().current_h))
        # self.smallresolution = (int(pygame.display.Info().current_w*2/3), int(pygame.display.Info().current_h*2/3))
        self.fullresolution = (1024,768)
        self.smallresolution = (int(self.fullresolution[0]*2.0/3),int(self.fullresolution[1]*2.0/3))


        #window = pygame.display.set_mode((w,h), DOUBLEBUF)
        #window = pygame.display.set_mode((w,h), FULLSCREEN)
        if options.startfullscreen:
            self.resolution = self.fullresolution
            window = pygame.display.set_mode(self.resolution,pygame.FULLSCREEN) 
            self.fullscreen = 1

        else:
            self.resolution = self.smallresolution
            window = pygame.display.set_mode(self.resolution) # do not start full
            self.fullscreen = 0

        print self.fullresolution    
        print self.resolution

        self.sounds = Sounds()

        #ISPIRO: Load goal image. Pygame handles png transparency gracefully
        self.ball = pygame.image.load("img/gray.png")
        self.ball_w = self.ball.get_width()
        self.ball_h = self.ball.get_height()

        self.brush = pygame.image.load("img/small.png")
        self.brush_w = self.brush.get_width()
        self.brush_h = self.brush.get_height()

        self.vortex = pygame.image.load("img/vortex.png");
        self.vortex_w = self.vortex.get_width()
        self.vortex_h = self.vortex.get_height()
        
        self.sum_surface = pygame.Surface((10,10))

        self.font = pygame.font.SysFont(None,80)

        if options.vicon_file is not None:
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

        if options.objects is None:
            print "Automatic object detection mode"
            while options.objects is None:
                print "Looking for objects..."
                options.objects = self.lookfor_objects()
                # Handle events
                self.handle_events()
        else:
            print "Objects are defined on command line"

        try:
            assert(options.objects is not None)
        except:
            print "Make sure you define 1 or more vicon objects through -o"
            sys.exit(1)

        self.set_surfaces()

        # ISPIRO: Need a mask image for blitting circles
        # original mask to match camera
        self.mask = pygame.Surface((options.camerawidth,options.cameraheight))
        # resized mask
        self.bigmask = pygame.Surface((self.camsurface_w,self.camsurface_h))
        
        self.cameras = []
        for c in options.cameras:
            self.cameras.append(Camera(c,img_size = (options.camerawidth,options.cameraheight),fps=self.TARGET_FPS,mirrored=options.mirrored))

        self.balls = []
        for o in options.objects:
            self.balls.append(Ball(self.f if self.simulation_mode else self.vp,o) )

        # set up team colors
        if options.boygirl:
            # we know there are only two teams
            self.teamcolors = [cm.hsv(0.61),cm.hsv(0.86)]
        else:
            self.teamcolors = [cm.jet(x) for x in np.linspace(0.2,0.8,options.numteams)]
        # set up object colors (using a different colormap)
        # self.objectcolors = [cm.spring(x) for x in np.linspace(0,1,len(options.objects))]
        self.objectcolors = [cm.summer(x) for x in np.linspace(0,1,len(options.objects))]
        
        self.score = [0]*options.numteams
        # self.update_score()
        
        self.accumulator = 0
        self.digit = options.game_time
        self.clock = pygame.time.Clock()
        # self.update_digit()

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


    def set_surfaces(self):
        # surface representing cam view
        #self.camsurface_h = self.resolution[1] # full height


        # I had originally set the height of the cam view to be the
        # full screen height However this is a problem when we switch
        # to a non-widescreen resolution (e.g. 1280x1024 which is less
        # wide than the cam image, 640x480) So now I set a target
        # aspect ratio (based on my own mac which looks good) And
        # letterbox crop the height of the image if the screen is less
        # wide then my mac
        target_aspect = 1.6 # match my macbook pro
        self.camsurface_h = int(self.resolution[0]/target_aspect) # height of cam and score
        # offset the top of the camera view and score view (letterbox top)
        self.horiz_offset = int((self.resolution[1]-self.camsurface_h)/2)

        
        # ideally this should be based on the current camera rather than options
        # (x*1.0/y) is the recommended way to do true division:
        # http://www.python.org/dev/peps/pep-0238/
        self.camsurface_w = int((options.camerawidth*1.0/options.cameraheight)*self.camsurface_h)

        # represents the original camera image
        self.rawcamsurface = pygame.Surface((options.camerawidth,options.cameraheight))

        print "Rawcamsurface:"
        print (options.camerawidth,options.cameraheight)
        print "Camsurface:"
        print (self.camsurface_w,self.camsurface_h)
        # represents the resized camera
        self.camsurface = pygame.Surface((self.camsurface_w,self.camsurface_h))
        # surface representing entire display
        self.screen = pygame.display.get_surface()

        print "Screen surface: %d x %d" % (self.screen.get_width(),self.screen.get_height())

    def get_cam(self):
        p = np.random.permutation(range(len(self.cameras)))
        cam = self.cameras[p[0]] # select 1 camera
        print "camera %s" % cam.name
        pygame.display.set_caption(cam.name)
        return cam

    def advance(self):
        """Advances cameras."""
        # GrabFrame in every camera
        # Strange things happen if we only grab in the active camera
        for c in self.cameras:
            cv.GrabFrame(c.capture)

    def hit(self,team,mode):
        """Handles a hit of target."""
        print "HIT"
        if mode==GAMEON:
            self.score[team]+=1
            self.update_score()
        self.sounds.swoosh1_sound.play()

    def reset(self,mode):
        """Choose a new random camera and random target."""
        cam = self.get_cam()
        targets = [Target(cam,radius=options.radius) for t in xrange(options.numteams)]

        if mode==REVEAL:
            # reset mask
            self.mask = pygame.Surface((options.camerawidth,options.cameraheight)) 
            self.bigmask = pygame.Surface((self.camsurface_w,self.camsurface_h))
        return cam,targets

    def tick(self,timeChange):
        """Updates timer display."""
        self.accumulator += timeChange
        if self.accumulator > 1000:
            self.accumulator = self.accumulator - 1000
            self.digit-=1
            self.update_digit()

    def blank_score(self):
        screen_w,screen_h = self.screen.get_size()
        cam_w,cam_h = self.camsurface.get_size()
        score_rec = pygame.Rect(cam_w,(2.0/3)*screen_h,(screen_w-cam_w),(1.0/3)*screen_h)
        self.screen.fill((0,0,0),score_rec)

    def blank_digit(self):
        screen_w,screen_h = self.screen.get_size()
        cam_w,cam_h = self.camsurface.get_size()
        digit_rec = pygame.Rect(cam_w,0,(screen_w-cam_w),(2.0/3)*screen_h)
        self.screen.fill((0,0,0),digit_rec)

    def update_score(self):
        # want to blank out the score area
        # so we are not writing scores on top of scores
        screen_w,screen_h = self.screen.get_size()
        cam_w,cam_h = self.camsurface.get_size()
        score_rec = pygame.Rect(cam_w,(2.0/3)*screen_h,(screen_w-cam_w),(1.0/3)*screen_h)        

        if options.boygirl:
            #extra space for icons
            #score_rec = pygame.Rect(cam_w,(7.0/9)*screen_h,(screen_w-cam_w),(2.0/9)*screen_h)
            font = pygame.font.SysFont("Menlo",100)            
        else:        
            font = pygame.font.SysFont("Menlo",150)

        self.screen.fill((0,0,0),score_rec)
        
        #font = pygame.font.Font(None,440)


        # linspace including last elements for xpositioning
        buf=100 # pixel buffer on each side
        #xlinspace = np.linspace(cam_w,screen_w,options.numteams+2)[1:options.numteams+1]
        xlinspace = np.linspace(cam_w+buf,screen_w-buf,options.numteams)
        
        print xlinspace
        for t in xrange(options.numteams):
            font_color = (np.array(self.teamcolors[t])*255).tolist()
            image = font.render(str(self.score[t]), True, font_color)
            rect = image.get_rect()
            x = xlinspace[t]
            print x
            y = (5.0/6)*screen_h
            rect.center = (x,y)
            self.screen.blit(image,rect)

    def update_digit(self):

        # want to blank out the digit area
        # so we are not writing digits on top of digits
        screen_w,screen_h = self.screen.get_size()
        cam_w,cam_h = self.camsurface.get_size()
        digit_rec = pygame.Rect(cam_w,0,(screen_w-cam_w),(2.0/3)*screen_h)
        self.screen.fill((0,0,0),digit_rec)

        #font = pygame.font.Font(None,440)
        font = pygame.font.SysFont("Menlo",150)

        # now draw the new digit
        font_color = (255,255,255)
        image = font.render(str(self.digit), True, font_color)
        rect = image.get_rect()
        x = cam_w + (screen_w-cam_w)/2
        y = screen_h/3
        rect.center = (x,y)
        self.screen.blit(image,rect)

    def targetdraw(self,frame,cam,targets):
        m = cv.GetMat(frame)
        arr = np.asarray(m)

        
        pygame.surfarray.blit_array(self.rawcamsurface,arr.transpose(1,0,2)[:,:,::-1])
        
        # draw target
        for i,t in enumerate(targets):
            target_color = (np.array(self.teamcolors[i])*255).tolist()
            #pygame.draw.circle(self.rawcamsurface, target_color, (t.x,t.y), t.radius, 0)
            #pygame.draw.circle(self.rawcamsurface, (255,255,255,0.1), (t.x,t.y), t.radius, 4)
            t.rot = t.rot - 1
            rot_v = pygame.transform.rotate(self.vortex, t.rot)
            #rot_v = self.vortex

            self.rawcamsurface.blit(rot_v,(t.x-rot_v.get_width()/2,t.y-rot_v.get_height()/2))


        # draw markers
        cam_w,cam_h = self.rawcamsurface.get_size()
        for j,b in enumerate(self.balls):
            ball_color = (np.array(self.objectcolors[j])*255).tolist()
            x,y = (cam.image_points[j,0],cam.image_points[j,1])
            # note use of half radius (display size) to prevent circle going into clock display
            if x > 0 and x < cam_w-(t.radius/2.0) and y > 0 and y < cam_h:
                pygame.draw.circle(self.rawcamsurface, ball_color,
                               (x,y), int(t.radius/2.0), 0)
                pygame.draw.circle(self.rawcamsurface, (0,0,0),
                                   (x,y), int(t.radius/2.0), 3)
                #self.rawcamsurface.blit(self.ball,(x-self.ball_w/2,y-self.ball_h/2))
        # optionally, flip the camera view (and targets,balls) horizontally to get mirror
        # then blit to screen
        pygame.transform.scale(self.rawcamsurface, (self.camsurface_w, self.camsurface_h), self.camsurface)
        
        if cam.mirrored:
            self.screen.blit(pygame.transform.flip(self.camsurface,True,False),(0,self.horiz_offset))
        else:
            self.screen.blit(self.camsurface,(0,self.horiz_offset))

        if options.boygirl:
            self.screen.blit(maleimage,(800,(4.0/6)*480))
            self.screen.blit(femaleimage,(1075,(4.0/6)*480))

                                                              
        pygame.display.flip()

    def revealdraw(self,frame,cam):
        m = cv.GetMat(frame)
        arr = np.asarray(m)
        pygame.surfarray.blit_array(self.rawcamsurface,arr.transpose(1,0,2)[:,:,::-1])
        
        ## # draw target
        ## for i,t in enumerate(targets):
        ##     target_color = (np.array(self.teamcolors[i])*255).tolist()
        ##     pygame.draw.circle(self.camsurface, target_color, (t.x,t.y), t.radius, 0)
        ##     pygame.draw.circle(self.camsurface, (255,255,255,0.1), (t.x,t.y), t.radius, 4)
        
        # draw markers
        cam_w,cam_h = self.rawcamsurface.get_size()
        for j,b in enumerate(self.balls):
            ball_color = (np.array(self.objectcolors[j])*255).tolist()

            x,y = (int(cam.image_points[j,0]),int(cam.image_points[j,1]))

            # A couple of times the draw.circle has failed with an OverflowError like:
            ##             Traceback (most recent call last):
            ##   File "combo.py", line 763, in <module>
            ##     g.run()
            ##   File "combo.py", line 593, in run
            ##     self.revealdraw(frame,cam)
            ##   File "combo.py", line 518, in revealdraw
            ##     (x,y), rad, 0)
            ## OverflowError: signed integer is greater than maximum
            # Therefore we catch it here and recover gracefully
            # It may be related to an SDL bug
            try:
                # note use of half radius (display size) to prevent circle going into clock display
                ## if x > 0 and x < cam_w-(t.radius/2.0) and y > 0 and y < cam_h:
                ##     pygame.draw.circle(self.camsurface, ball_color,
                ##                    (x,y), int(t.radius/2.0), 0)
                ##     pygame.draw.circle(self.camsurface, (0,0,0),
                ##                        (x,y), int(t.radius/2.0), 3)
#                self.rawcamsurface.blit(self.ball, (x-self.ball_w/2,y-self.ball_h/2))

                #ISPIRO: Draw a white circle on the mask image
                rad = self.ball_w/2 - 18
                #pygame.draw.circle(self.mask, (255, 255, 255),
                #                       (x,y), int(rad), 0)
                self.mask.blit(self.brush, (x-self.brush_w/2, y-self.brush_h/2))
            except OverflowError, e:
                # Balls are not drawn
                print "OVERFLOW DETECTED"
                print "x:"+str(x)
                print "y:"+str(y)
                print "rad:"+str(rad)

        pygame.transform.scale(self.rawcamsurface, (self.camsurface_w, self.camsurface_h), self.camsurface)
        pygame.transform.scale(self.mask, (self.camsurface_w, self.camsurface_h), self.bigmask)
        
        pygame.transform.scale(self.mask, (10,10), self.sum_surface)
        value_sum = 0
        value_count = 0
        for i in range(0,10):
            for j in range(0,10):
                col = self.sum_surface.get_at((i,j))
                value_sum += col[0]
                value_count += 255
                
        
        score = int((float(value_sum) / float(value_count))*100.0)
        

        # optionally, flip the camera view (and targets,balls) horizontally to get mirror
        # then blit to screen
        self.screen.fill((0,0,0))

        if cam.mirrored:
            self.screen.blit(pygame.transform.flip(self.camsurface,True,False),(0,self.horiz_offset))
            self.screen.blit(pygame.transform.flip(self.bigmask,True,False),(0,self.horiz_offset),None,pygame.BLEND_MULT)

        else:
            self.screen.blit(self.camsurface,(0,self.horiz_offset))
            self.screen.blit(self.bigmask,(0,self.horiz_offset),None,pygame.BLEND_MULT)

        if options.boygirl:
            self.screen.blit(maleimage,(800,(4.0/6)*480))
            self.screen.blit(femaleimage,(1075,(4.0/6)*480))
        
        font_image = self.font.render(str(score)+"%", True, (255,255,255))
        self.screen.blit(font_image,(20,20))
        pygame.display.flip()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT: sys.exit()

            if (event.type == KEYUP) or (event.type == KEYDOWN):
                print event

                if hasattr(event, 'key') and event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_f:
                        print "f"
                        if self.fullscreen:
                            self.fullscreen=0
                            self.resolution = self.smallresolution
                            window = pygame.display.set_mode(self.resolution)

                            self.set_surfaces()

                            # There is currently a bug here
                            # I shouldn't have to reset self.mask when we enable fullscreen
                            # But if I don't, it looks yellow instead of white
                            if self.mode==REVEAL:
                                self.mask = pygame.Surface((options.camerawidth,options.cameraheight)) 
                                self.bigmask = pygame.Surface((self.camsurface_w,self.camsurface_h))
                             
                            if self.mode==TARGETS or self.mode==GAMEON:
                                self.update_score()
                                self.update_digit()
                        else:
                            self.fullscreen=1
                            self.resolution = self.fullresolution
                            window = pygame.display.set_mode(self.resolution, pygame.FULLSCREEN)
                            self.set_surfaces()

                            if self.mode==REVEAL:
                            # There is currently a bug here
                            # I shouldn't have to reset self.mask when we enable fullscreen
                            # But if I don't, it looks yellow instead of white
                                self.mask = pygame.Surface((options.camerawidth,options.cameraheight)) 
                                self.bigmask = pygame.Surface((self.camsurface_w,self.camsurface_h))

                            if self.mode==TARGETS or self.mode==GAMEON:
                                self.update_score()
                                self.update_digit()

                if (event.type == KEYDOWN and event.key == K_ESCAPE):
                    if self.simulation_mode:
                        self.f.close()
                    else:
                        self.vp.close()
                    done = True
                if (event.type == KEYDOWN and event.key == K_SPACE):
                    # select new camera, target
                    cam,targets = self.reset(self.mode)

                if (event.type == KEYDOWN and event.key == K_g):
                    print "GO!"
                    self.mode=GAMEON
                    self.score = [0]*options.numteams
                    self.update_score()
                    self.clock = pygame.time.Clock()
                    self.digit = options.game_time
                    self.update_digit()

                if (event.type == KEYDOWN and event.key == K_t):
                    print "TARGETS"
                    self.mode=TARGETS
                    self.update_score()
                    self.update_digit()

                if (event.type == KEYDOWN and event.key == K_r):
                    print "REVEAL"
                    self.mode=REVEAL

                    self.blank_digit()
                    self.blank_score()

                if (event.type == KEYDOWN and event.key == K_u):
                    print "Update Vicon objects"
                    options.objects = None
                    while options.objects is None:
                        print "Looking for objects..."
                        options.objects = self.lookfor_objects()
                        # Handle events
                        self.handle_events()


    def run(self):

        self.mode = REVEAL

        cam,targets = self.reset(self.mode)
        
        done = False
        while not done:

            timeChange = self.clock.tick(self.TARGET_FPS)

            self.advance()
            
            if self.simulation_mode: # read from file
                obj = get_now_file(self.f,debug=True)
            else: # read from vicon_proxy
                obj = get_now(self.vp)

            # only retrieve from chosen cam
            frame = cv.RetrieveFrame(cam.capture)

            if not frame:
                print "Frame is None"
                sys.exit(1)


            for j,b in enumerate(self.balls):
                b.set_pos(obj)
                p = b.get_pos()
                cam.object_points[j,0]=p[0]
                cam.object_points[j,1]=p[1]
                cam.object_points[j,2]=p[2]

            cam.project()


            if self.mode==GAMEON or self.mode==TARGETS:
                # determine whether any of the balls have made the target
                for i,t in enumerate(targets):
                    for j,b in enumerate(self.balls):
                        if cam.match(t,j):
                            self.hit(i,self.mode)
                            cam,targets = self.reset(self.mode)

            # update screen
            if self.mode==GAMEON:
                self.tick(timeChange)

            if self.mode==TARGETS or self.mode==GAMEON:
                self.targetdraw(frame,cam,targets)
            else:
                self.revealdraw(frame,cam)

            if self.mode==REVEAL:
                pass
                ## elements = 640*480*3*255
                ## coverage = np.sum(pygame.surfarray.array3d(self.mask))/elements
                ## print "Coverage: %f" % coverage

            if self.mode==GAMEON and self.digit==0:
                print "GAME OVER"
                self.sounds.gameover_sound.play()
                self.mode=TARGETS

            self.handle_events()

                
                        ## self.score = [0]*options.numteams
                        ## self.update_score()
                        ## self.clock = pygame.time.Clock()
                        ## self.digit = options.game_time
                        ## self.update_digit()

                        
        # Outside while loop
        pygame.quit()
 
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
        obj = None
        f.close()
        pygame.quit()
        sys.exit()
    return obj


# Game states
#(WAITING,GAMEON)=range(0,2) #like an Enum
(REVEAL,TARGETS,GAMEON)=range(0,3) #like an Enum

parser = OptionParser()

parser.add_option("-o", "--object",
                      action="append", type="string", dest="objects", help="Add Vicon object")
parser.add_option("-s", "--start-fullscreen",
                  action="store_true", dest="startfullscreen", default=False, help="Start in full-screen mode (default=False)")
parser.add_option("-c", "--camera",
                      action="append", type="string", dest="cameras", help="Add ip camera")

parser.add_option("-r", "--radius",
                      action="store", type="int", dest="radius", default=60, help="Target radius in pixels (default=60)")

parser.add_option("-n", "--numteams",
                      action="store", type="int", dest="numteams", default=1,help="Number of teams (default=1)")

parser.add_option("-f", "--file",
                      action="store", type="string", dest="vicon_file", default=None, help="Vicon file")
parser.add_option("-l", "--line",
                      type="int", dest="line", default=0, help="Read ahead this many lines in Vicon file (default-0)")
parser.add_option("-t", "--game-time", type="int", dest="game_time", default=30, help="Game time in seconds (default=30)")
parser.add_option("-k", "--fake-cam",
                  action="store_true", dest="fake", default=False, help="Fake ip camera with local webcam (default=False)")
parser.add_option("-b", "--boy-girl-mode", action="store_true",
                  dest="boygirl", default=False, help="Boys vs. Girls")
parser.add_option("-x", "--camera-width",
                      action="store", type="int", dest="camerawidth", default=640, help="IP cam width (default=640)")
parser.add_option("-y", "--camera-height",
                      action="store", type="int", dest="cameraheight", default=480, help="IP cam height (default=480)")
parser.add_option("-m", "--not-mirrored",
                  action="store_false", dest="mirrored", default=True, help="Remove mirroring")

(options,args) = parser.parse_args()

if options.boygirl:
    # override
    options.numteams = 2
    maleimage = pygame.image.load('male.jpg')
    femaleimage = pygame.image.load('female.jpg')
    

g = Game()
g.run()


