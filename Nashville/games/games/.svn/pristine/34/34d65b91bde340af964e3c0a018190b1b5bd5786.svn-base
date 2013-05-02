#!/usr/bin/env python
#
# To kill windows:
# cv.DestroyWindow("Histogram")
# cv.DestroyWindow("CamShiftDemo")

import cv
import time
import pygame
from pygame.locals import *
import numpy as np

# import Image

use_grabframe = 1 #use GrabFrame/RetrieveFrame instead of QueryFrame

class DisplayCam:

    def __init__(self):


        #on NYU network
        #self.capture = cv.CreateFileCapture("http://root:gr33nd0t@ipcam6.cs.nyu.edu/axis-cgi/mjpg/video.cgi?resolution=640x480&fps=30&clock=1&.mjpg") #low-res, low fps motion jpg -- seems to be responsive with calibration

        #local net (technical test)
        #self.capture = cv.CreateFileCapture("http://root:gr33nd0t@192.168.1.104/axis-cgi/mjpg/video.cgi?resolution=640x480&fps=5&clock=1&.mjpg") #low-res, low fps motion jpg -- seems to be responsive with calibration

        #rtsp will work only if anonymous access (i.e. no user:pass) is enabled
        #self.capture = cv.CreateFileCapture("rtsp://ipcam3.cs.nyu.edu/axis-media/media.amp?resolution=640x480")

        self.capture = cv.CaptureFromCAM(0)

        self.w = cv.GetCaptureProperty(self.capture,cv.CV_CAP_PROP_FRAME_WIDTH)
        self.h = cv.GetCaptureProperty(self.capture,cv.CV_CAP_PROP_FRAME_HEIGHT)
       
        cv.NamedWindow( "DisplayCam", 1 )
     
        print( "Keys:\n"
            "    ESC - quit the program\n" )


        pygame.init()
 
        window = pygame.display.set_mode((self.w,self.h), DOUBLEBUF)
        self.screen = pygame.display.get_surface()


    def run(self):

        while True:
            #time.sleep(0.1)

            if use_grabframe:
                cv.GrabFrame( self.capture )
                frame = cv.RetrieveFrame( self.capture )
            else:                
                frame = cv.QueryFrame( self.capture )

            if not frame:
                print "Frame is None"
                break;

            #self.ocr(frame)

            #Uncomment this to test lag
            #[ok,corners] = cv.FindChessboardCorners( frame, self.board_size, cv.CV_CALIB_CB_ADAPTIVE_THRESH )

            cv.ShowImage("DisplayCam", frame)


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
            pygame.surfarray.blit_array(self.screen,arr.transpose(1,0,2)[:,:,::-1])
            pygame.display.flip()

            # parr = pygame.surfarray.pixels3d(self.screen)

            # cv.ShowImage("DisplayCam", cv.fromarray(arr))

            ## pi = Image.fromstring("RGB", cv.GetSize(frame), frame.tostring())
            ## pg_img = pygame.image.frombuffer( pi.tostring(), pi.size, pi.mode )
            ## self.screen.blit( pg_img, (0,0) )
            ## pygame.display.flip()

            c = cv.WaitKey(7) % 0x100
            if c == 27:
                cv.DestroyWindow("DisplayCam")
                del self.capture
                break

if __name__=="__main__":
    demo = DisplayCam()
    demo.run()
