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

class Camera:
    """Represents an IP cam."""
    def __init__(self,cameraname,img_size=(640,480),fps=30,mirrored=True):
        print "Initializing %s" % cameraname
        self.name=cameraname
        self.mirrored=mirrored # display camera as mirrored (i.e. flip x)
        capturestring = "http://root:gr33nd0t@%s.cs.nyu.edu/axis-cgi/mjpg/video.cgi?resolution=%dx%d&fps=%d&clock=0&.mjpg" % (cameraname,img_size[0],img_size[1],fps)

        # from where to read camera parameters
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
        timer_w=640
        w,h = (cam_w+timer_w,cam_h)
        pygame.init()
        self.cam_w = cam_w
        self.cam_h = cam_h
        self.sounds = Sounds()
        #ISPIRO: Load goal image. Pygame handles png transparency gracefully
        self.goal_image = pygame.image.load("goal.png")
        self.goal_w = self.goal_image.get_width()
        self.goal_h = self.goal_image.get_height()
    

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
                  "Possession will hang here if you are not connected to a Vicon Proxy server"
 
        window = pygame.display.set_mode((w,h), DOUBLEBUF)

        # surface representing cam view
        self.camsurface = pygame.Surface((cam_w,cam_h))

        #ISPIRO: Need a mask image for blitting circles
        self.mask = pygame.Surface((cam_w,cam_h))
        # surface representing entire display
        self.screen = pygame.display.get_surface()

        self.cameras = []
        for c in options.cameras:
            self.cameras.append(Camera(c,fps=self.TARGET_FPS))


        if self.simulation_mode:
            self.f = open(options.vicon_file,'r')
            # should we read ahead?
            for i in xrange(options.line):
                self.f.readline()
        else:
            # Initialize the object...
            print "Waiting for Vicon..."
            self.vp = ViconProxy()

        self.balls = []
        for o in options.objects:
            self.balls.append(Ball(self.f if self.simulation_mode else self.vp,o) )

        # set up team colors
        self.teamcolors = [cm.jet(x) for x in np.linspace(0.2,0.8,options.numteams)]
        # set up object colors (using a different colormap)
        self.objectcolors = [cm.spring(x) for x in np.linspace(0,1,len(options.objects))]
        
        self.score = [0]*options.numteams
        #self.update_score()

        
        self.accumulator = 0
        self.digit = options.game_time
        self.clock = pygame.time.Clock()
        #self.update_digit()
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

    def reset(self):
        """Choose a new random camera and random target."""
        cam = self.get_cam()
        self.mask = pygame.Surface((self.cam_w,self.cam_h))
        return cam

    def tick(self,timeChange):
        """Updates timer display."""
        self.accumulator += timeChange
        if self.accumulator > 1000:
            self.accumulator = self.accumulator - 1000
            self.digit-=1
            self.update_digit()

    def update_score(self):
        # want to blank out the score area
        # so we are not writing scores on top of scores
        screen_w,screen_h = self.screen.get_size()
        cam_w,cam_h = self.camsurface.get_size()
        score_rec = pygame.Rect(cam_w,(2.0/3)*screen_h,(screen_w-cam_w),(1.0/3)*screen_h)
        self.screen.fill((0,0,0),score_rec)

        #font = pygame.font.Font(None,440)
        font = pygame.font.SysFont("Menlo",150)

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
        font = pygame.font.SysFont("Menlo",300)

        # now draw the new digit
        font_color = (255,255,255)
        image = font.render(str(self.digit), True, font_color)
        rect = image.get_rect()
        x = cam_w + (screen_w-cam_w)/2
        y = screen_h/3
        rect.center = (x,y)
        self.screen.blit(image,rect)

    def draw(self,frame,cam):
        m = cv.GetMat(frame)
        arr = np.asarray(m)
        pygame.surfarray.blit_array(self.camsurface,arr.transpose(1,0,2)[:,:,::-1])

        #ISPIRO: Make a second copy of the image on the right half for masking out
        self.screen.blit(pygame.transform.flip(self.camsurface,True,False),(640,0))
        rad = 50

        # draw markers
        cam_w,cam_h = self.camsurface.get_size()
        for j,b in enumerate(self.balls):
            ball_color = (np.array(self.objectcolors[j])*255).tolist()
            x,y = (cam.image_points[j,0],cam.image_points[j,1])
            # note use of half radius (display size) to prevent circle going into clock display

            #ISPIRO: Blit out png graphic
            self.camsurface.blit(self.goal_image, (x-self.goal_w/2,y-self.goal_h/2))

#            pygame.draw.circle(self.camsurface, ball_color,
#                              (x,y), rad, 0)
#          pygame.draw.circle(self.camsurface, (0,0,0),
#                            (x,y), rad, 3)


            #ISPIRO: Draw a white circle on the mask image
            pygame.draw.circle(self.mask, (255, 255, 255),
                               (x,y), rad, 0)
        # optionally, flip the camera view (and targets,balls) horizontally to get mirror
        # then blit to screen

        #ISPIRO: Use BLEND_MULT to quickly produce the masked ipcam image
        if cam.mirrored:
            self.screen.blit(pygame.transform.flip(self.camsurface,True,False),(0,0))
            self.screen.blit(pygame.transform.flip(self.mask,True,False),(640,0),None,pygame.BLEND_MULT)
        else:
            self.screen.blit(self.camsurface,(0,0))
            self.screen.blit(self.mask,(640,0),None,pygame.BLEND_MULT)
                                                              
        pygame.display.flip()


    def run(self):

        mode = WAITING

        cam = self.reset()
        
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


            # update screen
            if mode==GAMEON:
                self.tick(timeChange)
            self.draw(frame,cam)


            if mode==GAMEON and self.digit==0:
                print "GAME OVER"
                self.sounds.gameover_sound.play()
                mode=WAITING
            

            for event in pygame.event.get():
                if event.type == pygame.QUIT: sys.exit()

                if (event.type == KEYUP) or (event.type == KEYDOWN):
                    print event
                    if (event.type == KEYDOWN and event.key == K_ESCAPE):
                        if self.simulation_mode:
                            self.f.close()
                        else:
                            self.vp.close()
                        done = True
                    if (event.type == KEYDOWN and event.key == K_SPACE):
                        # select new camera, target
                        cam = self.reset()

                    if (event.type == KEYDOWN and event.key == K_g):
                        print "GO!"
                        mode=GAMEON
                        self.score = [0]*options.numteams
                        self.update_score()
                        self.clock = pygame.time.Clock()
                        self.digit = options.game_time
                        #self.update_digit()
 
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
(WAITING,GAMEON)=range(0,2) #like an Enum

parser = OptionParser()

parser.add_option("-o", "--object",
                      action="append", type="string", dest="objects", help="Add Vicon object")

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


(options,args) = parser.parse_args()


g = Game()
g.run()


