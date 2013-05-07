#/usr/bin/python

# Author: Piotr Zielinski (http://www.cl.cam.ac.uk/~pz215/)
# Licence: GPLv2

# This is a proof-of-concept prototype of 2D dasher, it absolutely
# does not qualify for production use.  You are welcome to improve or,
# better, completely rewrite the code

# you are also advised to have a fast computer ...

import time
# import psyco
# psyco.full()

import math
import gobject
import pygtk
pygtk.require('2.0')
import gtk
import pango
import random
from languagemodel import *
import gtk.glade
import os.path
import os
import struct
# import mmap

from optparse import OptionParser

parser = OptionParser()
parser.add_option("-a", "--auto-resize", action="store_true", dest="autoresize",
                  default=False, help="Independent horizontal/vertical zoom")
parser.add_option("-t", "--tolerance", action="store", type="float",
                  dest="tolerance", default=1.0, help="The maximum ratio between the lenghts of the sides of a rectangle (default=1.0, square)")
parser.add_option("-c", "--crop", action="store_true", 
                  dest="cropsize", default=True)
parser.add_option("-d", "--decrop", action="store_false", 
                  dest="cropsize", default=True)
parser.add_option("-i", "--hilbert", action="store_true", dest="hilbert", default=False)
parser.add_option("-p", "--peano", action="store_true", dest="peano", default=False)
parser.add_option("-v", "--adaptive", action="store_true", dest="adaptive", default=False)

options, args = parser.parse_args()


names = [" "] + [chr(ord("a") + i) for i in range(26)]
pivotcount = 4
pivots = [len(names) * i / pivotcount for i in range(pivotcount+1)]

# probs = [random.random() for x in names]

def firstlarger(number, list):
    for i in range(len(list)):
        if number < list[i]: 
            return i
    return len(list)


def lastword(text):
    return text[text.rfind(" ") + 1:]

def logscale(number, scale):
    return (number - 1.0) * scale + 1.0

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def abs(self):
        return math.hypot(self.x, self.y)

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)

    def __mul__(self, scale):
        if isinstance(scale, Point):
            return Point(self.x * scale.x, self.y * scale.y)
        else:
            return Point(self.x * scale, self.y * scale)

    def logscale(self, scale):
	return Point(logscale(self.x, scale), logscale(self.y, scale))

    def scaledaround(self, source, dest, scale):
        return (self - source) * scale + dest

    def between(self, other, time):
        return self * (1-time) + other * time

    def isinside(self, rect):
        return (self.x >= rect.x) and (self.y >= rect.y) and \
           (self.x < rect.x + rect.sizex) and (self.y < rect.y + rect.sizey)

    def __str__(self):
        return str((self.x, self.y))

    def inverse(self):
        return Point(1.0 / self.x, 1.0 / self.y)

class Rectangle:
    def __init__(self, x, y, sizex, sizey, xtoy):
        self.x = x
        self.y = y
        self.sizex = sizex
        self.sizey = sizey
        self.xtoy = xtoy

    def scaledaround(self, source, dest, scale):
        return Rectangle((self.x - source.x) * scale.x + dest.x,
                         (self.y - source.y) * scale.y + dest.y,
                         self.sizex * scale.x,
                         self.sizey * scale.y,
                         self.xtoy * scale.x / scale.y)

    def center(self):
        return Point(self.x + self.sizex / 2, self.y + self.sizey / 2)

    def scaledwithrects(self, srect, drect):
        xscale = drect.sizex / srect.sizex
        yscale = drect.sizey / srect.sizey
        return Rectangle((self.x - srect.x) * xscale + drect.x,
                         (self.y - srect.y) * yscale + drect.y,
                         self.sizex * (drect.sizex / srect.sizex),
                         self.sizey * (drect.sizey / srect.sizey))

    def scaled(self, scale):
        center = self.center()
        return self.scaledaround(center, center, scale)

    def __str__(self):
        return str((self.x, self.y, self.sizex, self.sizey, self.xtoy))

    def empty(self):
        return (self.sizex <= 0) or (self.sizey <= 0)

    def __add__(self, point):
        return Rectangle(self.x + point.x, self.y + point.y,
                         self.sizex, self.sizey, self.xtoy)
    
    def gethighx(self):
        return self.x + self.sizex

    def gethighy(self):
        return self.y + self.sizey

    def frompoints(x1, y1, x2, y2):
        return Rectangle(x1, y1, x2 - x1, y2 - y1)

    frompoints = staticmethod(frompoints)

    def intersect(self, other):
        return Rectangle.frompoints(max(self.x, other.x),
                                    max(self.y, other.y),
                                    min(self.gethighx(), other.gethighx()),
                                    min(self.gethighy(), other.gethighy()))

    def union(self, other):
        return Rectangle.frompoints(min(self.x, other.x),
                                    min(self.y, other.y),
                                    max(self.gethighx(), other.gethighx()),
                                    max(self.gethighy(), other.gethighy()))

    def distance(self, point):
        return max(self.x - point.x, 0) + \
               max(point.x - self.x - self.sizex, 0) + \
               max(self.y - point.y, 0) + \
               max(point.y - self.y - self.sizey, 0)

    def normalize(self):
        if self.sizex < 0:
            self.x += self.sizex
            self.sizex = -self.sizex
        if self.sizey < 0:
            self.y += self.sizey
            self.sizey = -self.sizey
        return self
        
        

class NameTracker:
    def __init__(self):
        self.name = ""
        self.times = []
        self.currenttime = 0

    def setcurrent(self, name, time=None):
        self.currenttime = self.currenttime + 1
        time = time or self.currenttime
        
        common = len(os.path.commonprefix(self.name, name))

        for i in range(common, len(name)):
            self.times[i] = time
        
        self.name = name

def sums(list):
    sums = []
    last = 0
    for item in list:
        sums.append(last)
        last = last + item
    return sums

def fullsums(list):
    sums = [0]
    for item in list:
        sums.append(sums[-1] + item)
    return sums

def normalize(list):
    total = float(sum(list))
    return [x / total for x in list]


# todo: introduce short spaces among the rectangles

def makerectangles1(rect, probs, horizontal=True):
    if len(probs) == 1:
        return [rect]

    half = sum(probs) / 2.0
    costs = [abs(s - half) for s in sums(probs)]
    mincost = min(costs)
    pivot = costs.index(mincost)
    pivotvalue = sum(probs[:pivot]) / float(sum(probs))

    pivotfrac = max(0.00 * ((min(rect.sizex, rect.sizey) - 50) /
                            (rect.sizex + rect.sizey)), 0)
    pivot1 = pivotvalue - pivotfrac
    pivot2 = 1 - pivotvalue - pivotfrac

    if options.adaptive:
        next = (rect.xtoy > 1)
    else:
        next = not horizontal

    if next:
        left = makerectangles1(
            Rectangle(rect.x, rect.y, rect.sizex * pivot1, rect.sizey,
                      rect.xtoy * pivotvalue),
            probs[:pivot], next)
        right = makerectangles1(
            Rectangle(rect.x + rect.sizex * (1-pivot2), rect.y,
                      rect.sizex * pivot2, rect.sizey,
                      rect.xtoy * (1-pivotvalue)),
            probs[pivot:], next)
    else:
        left = makerectangles1(
            Rectangle(rect.x, rect.y, rect.sizex, rect.sizey  * pivot1,
                      rect.xtoy / pivotvalue),
            probs[:pivot], next)
        right = makerectangles1(
            Rectangle(rect.x, rect.y + rect.sizey * (1-pivot2), 
                      rect.sizex, rect.sizey * pivot2,
                      rect.xtoy / (1-pivotvalue)),
            probs[pivot:], next)

    return left + right[::-1]

def makerectangles3(rect, probs, horizontal=True):
    "distribute rectangles along the (adaptive) Peano curve"
    if len(probs) == 0:
        return []
    
    if len(probs) == 1:
        return [rect.normalize()]

    # hmm, what if there are exactly 2?

    break1 = sum(probs) * 0.333
    break2 = sum(probs) * 0.667
    costs1 = [abs(s - break1) for s in sums(probs)]
    costs2 = [abs(s - break2) for s in sums(probs)]
    pivot1 = costs1.index(min(costs1))
    pivot2 = costs2.index(min(costs2))

    firstvalue = sum(probs[:pivot1]) / float(sum(probs))
    lastvalue = sum(probs[:pivot2]) / float(sum(probs))

#     pivotfrac = max(0.00 * ((min(rect.sizex, rect.sizey) - 50) /
#                             (rect.sizex + rect.sizey)), 0)
    value1 = firstvalue
    value2 = lastvalue - firstvalue
    value3 = 1 - lastvalue

#    if rect.xtoy > 1:

    if options.adaptive:
        next = (rect.xtoy > 1)
    else:
        next = not horizontal

    if next:
        sub1 = makerectangles3(
            Rectangle(rect.x, rect.y, rect.sizex * value1, rect.sizey,
                      rect.xtoy * value1),
            probs[:pivot1], next)
        sub2 = makerectangles3(
            Rectangle(rect.x + rect.sizex * firstvalue,
                      rect.y + rect.sizey,
                      rect.sizex * value2, -rect.sizey,
                      rect.xtoy * value2),
            probs[pivot1:pivot2], next) # simplify!
        sub3 = makerectangles3(
            Rectangle(rect.x + rect.sizex * lastvalue, rect.y,
                      rect.sizex * value3, rect.sizey,
                      rect.xtoy * value3),
            probs[pivot2:], next)
    else:
        sub1 = makerectangles3(
            Rectangle(rect.x, rect.y, rect.sizex, rect.sizey  * value1,
                      rect.xtoy / (value1+0.0001)),
            probs[:pivot1], next)
        sub2 = makerectangles3(
            Rectangle(rect.x + rect.sizex, rect.y + rect.sizey * firstvalue,
                      -rect.sizex, rect.sizey  * value2,
                      rect.xtoy / (value2+0.0001)),
            probs[pivot1:pivot2], next) # simplify!
        sub3 = makerectangles3(
            Rectangle(rect.x, rect.y + rect.sizey * lastvalue, 
                      rect.sizex, rect.sizey * value3,
                      rect.xtoy / (value3+0.0001)),
            probs[pivot2:], next)

    return sub1 + sub2 + sub3


def sizelargest(sortedlist, maxx):
#    print sortedlist, maxx
    origsum = leftspace = leftprob = sum(sortedlist)
    for item in sortedlist:
        if item * leftspace > maxx * leftprob:
            leftprob -= item
            leftspace -= maxx
        else:
#            print "result:", item, leftspace / leftprob
            return item, leftspace / leftprob
    else:
        return 0.0, 1.0

def croplargest(sizelist, maxx):
    biggest, mult = sizelargest(sorted(sizelist, reverse=True), maxx)
#    mult = between(1.0, mult, 0.3)
    return [item > biggest and maxx or item*mult for item in sizelist]

def makerectangles2(rect, probs):
    cumprobs = fullsums(normalize(probs))
    yoffsets = [rect.sizey * cumprobs[pivots[row]]
                for row in range(pivotcount+1)]

    aspect = hello.scale.y / hello.scale.x
    rectlist = []
    for row in range(pivotcount):
        rowheight = yoffsets[row+1] - yoffsets[row]
#        tolerance = max(1.0, 2.0 - rowheight / 200.0)
        maxprob = rowheight / rect.sizex * options.tolerance * aspect
        rowprobs = normalize(probs[pivots[row]:pivots[row+1]])
        # fixme: speed this up, another algorithm required
        if options.cropsize:
            rowprobs = normalize([min(maxprob, prob) for prob in rowprobs])
        else:
            rowprobs = croplargest(rowprobs, maxprob)
        rowcumprobs = fullsums(normalize(rowprobs))
        xoffsets = [rect.sizex * rowcumprobs[col]
                    for col in range(len(rowcumprobs))]

        for col in range(len(rowprobs)):
#             size = min(xoffsets[col+1] - xoffsets[col],
#                        yoffsets[row+1] - yoffsets[row])
            rectlist.append(
                Rectangle(rect.x + xoffsets[col],
                          rect.y + yoffsets[row], 
                          xoffsets[col+1] - xoffsets[col],
                          yoffsets[row+1] - yoffsets[row], rect.xtoy))
                
    return rectlist


def between(x, y, t):
    return x * (1-t) + y * t


class Calibrator:
    def __init__(self):
        self.centerestimate = Point(0,0)
        self.averagepoint = Point(0,0)

    def correctpoint(self, point):
        return point + self.centerestimate

    def updateestimates(self, point):
        self.averagepoint += (point - self.averagepoint) * 0.01
        self.centerestimate += self.averagepoint * 0.01
    

class HelloWorld:

    # This is a callback function. The data arguments are ignored
    # in this example. More on callbacks below.
    def virtualtoscreen(self, rectangle):
        return rectangle.scaledaround(self.vcenter, self.scenter, self.scale)
    
    def screentovirtual(self, point):
        return point.scaledaround(self.scenter, self.vcenter,
                                  self.scale.inverse())
    
    def rescale(self, factor, newrect):
        print "before:", self.virtualtoscreen(newrect)
        self.scale.x = self.scale.x / factor.x # do something here
        self.scale.y = self.scale.y / factor.y # do something here
        self.mainrect = newrect.scaledaround(self.vcenter, self.vcenter, factor)
        print "after:", self.virtualtoscreen(self.mainrect)
        self.currentrectangle = self.currentrectangle.scaledaround(self.vcenter, self.vcenter, factor)
        self.mainlevel = self.mainlevel + 1


    def getrawpointer(self):
#         spoint = Point(*struct.unpack("ii", self.mousefile[0:8]))
#         if spoint.x == 3:
#             return Point(256, 256)
#         else:
#             print spoint
#             return spoint * 1.5 + self.scenter
         return Point(*self.drawing.get_pointer())

    def getpointer(self):
        return self.getrawpointer()
        xpoint = self.calibrator.correctpoint(self.getrawpointer() -
                                              Point(256,256))
        if xpoint.abs() < 200:
            self.calibrator.updateestimates(xpoint)
        return xpoint + Point(256, 256)


    def timertick(self):
        self.tickcount = self.tickcount + 1
        thistime = time.time()
        deltatime = thistime - self.lasttime
        self.lasttime = thistime
        
        if self.stop:
            self.drawing.queue_draw()
            return True

        spoint = self.getrawpointer()


        # compute movement factor

        distance = (spoint - self.prevpointer).abs()
        self.movement = 0.9 * self.movement + distance
        self.prevpointer = spoint
        mfactor = min(self.movement / 2000.0, 1.0)
        origvelocity = between(self.origvelocity, 1.0, mfactor)

#        origvelocity = self.origvelocity
        
        sspoint = spoint # + Point(10, 0)
        # + Point(35, 35)

        width, height = self.drawing.window.get_size()

        # test whether mouse pointer is inside
        if (spoint.x < 0) or (spoint.x >= width) \
               or (spoint.y < 0) or (spoint.y >= height):
            return True


        vpoint = self.screentovirtual(spoint)
        vvpoint = self.screentovirtual(sspoint)

        srect = self.virtualtoscreen(self.currentrectangle)

        if max(srect.sizex, srect.sizey) > 10000:
            self.drawing.queue_draw()
            return True
            

        org = 1
#        org = between(0.97, self.origvelocity, 0.3)







	# dasher dynamics: find the linear transformation that
	# translates refrect into stargetrect in 1 / vel steps

        # refrect = self.sinterect
	refrect = srect

#        refrect = Rectangle(-10, -10, 20, 20) + spoint + Point(5,0)
	 
# 	vel = 0.09
# 	stargetrect = self.screenrect

# 	velocity = Point(stargetrect.sizex / refrect.sizex, 
# 			 stargetrect.sizey / refrect.sizey)

# 	factor = max(velocity.x, velocity.y) / 3.0

# 	if factor >= 1.0:
# 	    vel = vel / factor

# 	velocity = velocity.logscale(vel * 0.2)
#       self.scale = self.scale * self.velocity;

# 	self.vcenter = self.vcenter.between(vvpoint, 3 * vel)

        
#        velocity = Point(origvelocity, origvelocity)
        if options.autoresize:
            velocity = Point(between(org, origvelocity,
                                     between(min(refrect.sizey / refrect.sizex, 1), 1, 0)),
                             between(org, origvelocity,
                                     between(min(refrect.sizex / refrect.sizey, 1), 1, 0)))
        else:
            velocity = Point(origvelocity, origvelocity)

        myfact = max(self.focusrect.sizex, self.focusrect.sizey) / 300.0 - 1.5

#         if myfact > -0.35:
#             self.lostcounter = self.lostcounter + 1
#         else:
#             if self.lostcounter > 0:
#                 print "not lost"
#             self.lostcounter = 0

#         if (myfact > 0) and (self.lostcounter > 5):
#             print "lost (%i)" % self.lostcounter
#             velocity = velocity.logscale(max(1.0 - myfact, -0.2))

        self.velocity = Point.between(self.velocity, velocity, 0.2)

        realvelocity = between(Point(1.0, 1.0), self.velocity, 10 * deltatime)
 	
        if self.zoomin:
            self.scale = self.scale * realvelocity;

            vvvpoint = vvpoint.scaledaround(self.vcenter, self.vcenter, 1.5)

            self.vcenter = self.vcenter.scaledaround(vvvpoint, vvvpoint,
                                                     realvelocity.inverse())
            self.vcenter = Point.between(self.vcenter, vvpoint, 0.02)
            
        else:
            print realvelocity
            self.scale = self.scale * (1/between(1,origvelocity,0.5)) # realvelocity * (1/origvelocity) * 0.98
            self.vcenter = Point.between(self.vcenter, vpoint, 0.04)
            
        if self.rescalenow:
            self.rescalenow = False
            # find the currect rectangle
            currentrect = None
            probs = self.language.getprobs(lastword(self.gettext()), names)
            for rectangle, name in zip(self.makerects(self.mainrect, probs),
                                       names):
                if vpoint.isinside(rectangle):
                    sfactor = max(width, height) / \
                              max(rectangle.sizex, rectangle.sizey)

                    if options.autoresize:
                        factor = Point(width / rectangle.sizex,
                                       height / rectangle.sizey)
                    else:
                        factor = Point(sfactor, sfactor)

                    print "rescaling:", factor
                    print "movement factor:", mfactor
                    print "fps:", self.tickcount / (thistime - self.starttime)
                    self.rescale(factor, rectangle)
                    self.textbuf.insert(self.textbuf.get_end_iter(), name)
                    self.saylastword()
                    break

        self.drawing.queue_draw()
        return True                     # don't stop callbacks

    def speedup(self):
        self.origvelocity = self.origvelocity + 0.01
        print "speed:", self.origvelocity

    def slowdown(self):
        self.origvelocity = max(1.00, self.origvelocity - 0.01)
        print "speed:", self.origvelocity

    def on_drawingarea_button_press_event(self, widget, event, data=None):
        self.zoomin = False
        if self.stop:
            self.stop = False
        else:
            if event.type == gtk.gdk._2BUTTON_PRESS: 
                self.stop = True       # double-click: stop
            if event.button == 1:
                self.stop = True

    def on_drawingarea_scroll_event(self, widget, event, data=None):
        if event.direction == gtk.gdk.SCROLL_UP:
            self.speedup()
        elif event.direction == gtk.gdk.SCROLL_DOWN:         # speed down
            self.slowdown()

    def on_drawingarea_key_press_event(self, widget, event, data=None):
        for key in event.string:
            if key == " ":
                self.stop = not self.stop
            elif key == ",":
                self.slowdown()
            elif key == ".":
                self.speedup()

    def on_drawingarea_button_release_event(self, widget, event, data=None):
        self.zoomin = True

    def on_mainwindow_delete_event(self, widget, event, data=None):
        gtk.main_quit()

    def on_drawingarea_expose_event(self, widget, event, data=None):
#        print "draw", self.drawcount
        self.drawcount = self.drawcount + 1
        gc = widget.get_style().fg_gc[gtk.STATE_NORMAL]
        x, y, width, height = event.area
        
#        spoint = Point(*widget.get_pointer()) #  + Point(5, 0) # (5,0)
        spoint = self.getpointer()

        sspoint = spoint + Point(20, 0)
#        spointrect = Rectangle(spoint.x - 10, spoint.y - 10, 20, 20)
#        self.sinterect = spointrect

	self.pixmap.draw_rectangle(widget.get_style().white_gc, True,
                                   0, 0, width, height)


        def drawrectangle(vrectlist, srectlist, level, prefix):
            srect = srectlist[-1]
            distance = srect.distance(spoint)

            if srect.sizex + srect.sizey < 10:
                return False

            if distance > 1000:
                  return False

            # sometimes this method rescales a different rectangle,
            # because the mouse pointer moves between expose_event and
            # tick
            
            hasrectangle = (srect.sizex + srect.sizey >= 80) 

            if len(srectlist) > 1:
                sparent = srectlist[-2]
                pinside = spoint.isinside(sparent)
                parenthasrectangle = (sparent.sizex + sparent.sizey >= 100)
            else:
                pinside = True
                parenthasrectangle = True

#             if (not pinside) and (sparent.sizex + sparent.sizey >
#                                   5*(srect.sizex + srect.sizey)):
#                 return False

#             if (srect.x >= x + width) or (srect.y >= y + width) or \
#                (x >= srect.x + srect.sizex) or (y >= srect.y + srect.sizey):
#                 return False

            inside = spoint.isinside(srect)
            minsize = min(srect.sizex, srect.sizey)
            
            if inside and (len(srectlist) == 2) and (minsize > 512):
                print "rescale", srectlist[0], srectlist[1]
                self.rescalenow = True

            parentinside = (len(srectlist) < 2) or spoint.isinside(srectlist[-2])
#             intersects = srect.intersect(spointrect).empty()

#             if intersects and (min(srect.sizex, srect.sizey) < 15):
#                 self.sinterect = self.sinterect.union(srect)



#             if inside:
#                 gc.foreground = fillcolor
#                 self.pixmap.draw_rectangle(gc, True,
#                                            int(srect.x), int(srect.y),
#                                            int(srect.sizex), int(srect.sizey))

#             if inside and (len(srectlist) >= 2):
#                 sparent = srectlist[-2]
#                 gc.foreground = self.grey
#                 if sparent.x + 10 < srect.x:
#                     self.pixmap.draw_line(gc, int(sparent.x + 10),
#                                           int(sparent.y + sparent.sizey/2),
#                                           int(srect.x),
#                                           int(srect.y + srect.sizey/2))


            color = self.cfore[(self.mainlevel + level) % len(self.cfore)]

            if parenthasrectangle: # and (minsize > 20):
                filllevel = firstlarger(minsize, [50, 80, 120, 200]);
                fillcolor = self.fill[filllevel][(self.mainlevel + level) % len(self.fill[filllevel])]
                gc.foreground = fillcolor
                self.pixmap.draw_rectangle(gc, True,
                                           int(srect.x), int(srect.y),
                                           int(srect.sizex),
                                           int(srect.sizey))
                    
            gc.foreground = color


            if hasrectangle and parentinside:
                if minsize < 70:
                    gc.foreground = self.grey2
                    if inside:
                        gc.foreground = self.grey2alert
                elif minsize < 100:
                    gc.foreground = self.grey
                    if inside:
                        gc.foreground = self.greyalert
                else:
                    gc.foreground = color
                    if inside:
                        gc.foreground = self.alert
                    
                self.pixmap.draw_rectangle(gc, False,
                                           int(srect.x), int(srect.y),
                                           int(srect.sizex), int(srect.sizey))




            if minsize < 5:
                return False

            layout = None
            sizelist = [7, 12, 20, 30, 50, 70, 100, 120, 150, 200, 300, 100000]
            for i in range(len(sizelist)):
                if minsize < sizelist[i]: 
                    layout = self.layouts[i+5]
                    break

            if minsize >= sizelist[-1]:
                raise "impossible"

            if (minsize >= 7) and parentinside and parenthasrectangle: 
                gc.foreground = color
            else:
                gc.foreground = self.grey
                
#            gc.line_width = int(3 / (level + 1))


            if (len(vrectlist) > 1) and inside and \
                   (max(srect.sizex, srect.sizey) > 50):
                self.currentrectangle = vrectlist[-1]

            if inside:
                self.focusrect = srect

            if inside:
                gc.foreground = self.alert
                

            if len(prefix) > 0:
                layx, layy = layout.get_pixel_size()

                centery = int(srect.y + (srect.sizey - layy)/2)

#                 if inside and hasrectangle:
#                     layout = self.boldlayout

                if (not pinside) and (prefix[-1] == " "):
                    return False
                todraw = prefix[-1].replace(" ", "$")
#                todraw = prefix[-1]
                layout.set_text(todraw)
                self.pixmap.draw_layout(gc, int(srect.x), centery, layout)

            if (max(srect.sizex, srect.sizey) < 20):
                return False

#            ssinside = sspoint.isinside(srect)

#            return srect.distance(spoint)

    
            return (distance < 30) or inside

        def recursivedraw(rectangle, prefix, srectlist=[], vrectlist=[]):
            level = len(srectlist)
            vrectlist = vrectlist + [rectangle]
            srectlist = srectlist + [self.virtualtoscreen(rectangle)]
#             print "prefix: '%s'" % prefix
            probs = self.language.getprobs(lastword(prefix), names)

            # todo: add here the code that follows not only the
            # current tree but also other branches with high relative
            # probability (say > 0.5 or something).

            # todo: for boxes far away from the point, use logarithmic
            # model, so that the inside boxes seem bigger (because
            # only big words are displayed)

            if drawrectangle(vrectlist, srectlist, level, prefix):
                rectangles = self.makerects(rectangle, probs)
                
                for subrect, name in zip(rectangles, names):
                    recursivedraw(subrect, lastword(prefix) + name,
                                  srectlist, vrectlist)
                

        recursivedraw(self.mainrect, lastword(self.gettext()))
        gc.foreground = self.alert
        self.pixmap.draw_rectangle(gc, True, spoint.x-2, spoint.y-2, 4, 4);

        widget.window.draw_drawable(gc, self.pixmap, x, y, x, y, width, height)
                    
        return False                    # why?

    def on_drawingarea_realize(self, drawing):
        cmap = drawing.window.get_colormap()

        # the background and foreground colors
        self.grey = cmap.alloc_color(30000, 30000, 30000)
        self.grey2 = cmap.alloc_color(50000, 50000, 50000)
        self.greyalert = cmap.alloc_color(60000, 30000, 30000)
        self.grey2alert = cmap.alloc_color(60000, 50000, 50000)
        self.fillcolor = cmap.alloc_color(60000, 55000, 55000)
        self.alert = cmap.alloc_color(60000, 0, 0)

#         colors = [(random.random()* 20000,
#                    random.random()* 40000,
#                    random.random()* 40000) for x in names]

        colors = [(30000, 30000, 10000),
                  (0, 20000, 0),
                  (0, 0, 20000)]

        self.cfore = [cmap.alloc_color(*colors[i])
                      for i in range(len(colors))]

        self.fill = [[cmap.alloc_color
                     (*[between(comp1, between(comp2, 65535, rate2), rate1)
                                for comp1, comp2 in
                                zip(colors[i],
                                    colors[(i+len(colors)-1) % len(colors)])])
                     for i in range(len(colors))]
                     for rate1, rate2 in
                     zip([0.97, 0.93, 0.90, 0.85, 0.8],
                         [1.0, 1.0, 1.0, 1.0, 1.0])]

        self.fillcolors = []
        for i in range(5):
            color = 50000 + 2000 * i
            self.fillcolors.append(cmap.alloc_color(color, color, color))
            
    def on_drawingarea_configure_event(self, widget, event, data=None):
        print "configuring"
        window = widget.window
	sizex, sizey = window.get_size()
        
	self.pixmap = gtk.gdk.Pixmap(window, sizex, sizey)
        self.screenrect = Rectangle(0.0, 0.0, float(sizex), float(sizey),
                                    float(sizex) / float(sizey))
	self.mainrect = self.screenrect
        self.focusrect = self.screenrect
        self.currentrectangle = self.mainrect
        self.vcenter = self.mainrect.center()
        self.scenter = self.vcenter
#        self.sinterect = self.mainrect  # dubious
        
	return True

    def makerects(self, origrect, probs):
#         topmargin = min(4 / self.scale.y, 0.02 * origrect.sizey)
#         bottommargin = min(4 / self.scale.y, 0.02 * origrect.sizey)
#         sidemargin = min(20 / self.scale.x, 0.08 * origrect.sizex)
        topmargin = 0.02 * origrect.sizey
        bottommargin = 0.02 * origrect.sizey
        sidemargin = 0.08 * origrect.sizex

        newrect = Rectangle(origrect.x + sidemargin,
                            origrect.y + topmargin,
                            origrect.sizex - sidemargin,
                            origrect.sizey - bottommargin - topmargin,
                            origrect.xtoy)

#        corner = Point(rect.x + rect.sizex, rect.y + rect.sizey)
        newlist = []
#        tolerance = 1.5
        overflow = 1.0
        for rect in self.rectanglemodel(newrect, probs):
            center = rect.center()
#            rect = rect.scaledwithrects(origrect, origrect)
#            srect = self.virtualtoscreen(rect)
            srect = rect.scaled(self.scale)
            factor = srect.sizey / srect.sizex
#            tolerance = max(1.0, 2.0 - max(srect.sizex, srect.sizey) / 200.0)
#            overflow = max(1.0, 1.5 - max(srect.sizex, srect.sizey) / 50.0)
            scale = Point(min(options.tolerance * factor, overflow),
                          min(options.tolerance / factor, overflow))
#            scale = Point(1.0, 1.0)
            newlist.append(rect.scaledaround(center, center, scale))

        return newlist

    def gettext(self):
        return self.textbuf.get_text(self.textbuf.get_start_iter(),
                                     self.textbuf.get_end_iter())


    def widget(self, name):
        return gtk.glade.XML.get_widget(self.tree, name)

    def say(self, text):
        pass
# 	try:
#             self.speech.write('(SayText "%s")' % text)
#             self.speech.flush()
# 	except:
# 	    pass

    def saylastword(self):
        text = self.gettext()
        words = text.split()
        if (len(words) > 0) and (text[-1] == " "):
            self.say(words[-1])

    def __init__(self):
#         fd = os.open("/tmp/gaze-mouse", os.O_RDWR)
#         print fd
#         self.mousefile = mmap.mmap(fd, 8)
#         print "mapped"

        self.calibrator = Calibrator()
        self.drawcount = 0
        self.rescalenow = False
        self.zoomin = True
        self.stop = False
        self.maxrlevel = 2
        self.origvelocity = 1.01
        self.language = LanguageModel()
        if options.peano:
            self.rectanglemodel = makerectangles3
        elif options.hilbert:
            self.rectanglemodel = makerectangles1
        else:
            self.rectanglemodel = makerectangles2
            
        self.scale = Point(0.5, 0.5)
        self.velocity = Point(1.0, 1.0)
        self.prevpointer = Point(0, 0)
        self.movement = 0.0
        self.lostcounter = 0
        self.lasttime = time.time()
        self.starttime = self.lasttime
        self.tickcount = 0
        self.mainlevel = 0

        self.speech = os.popen("festival --pipe", "w")
        self.tree = gtk.glade.XML("bidasher.glade")
                
        self.window = self.widget("mainwindow")
        self.drawing = self.widget("drawingarea")
        self.textview = self.widget("textview")

        

        self.textbuf = gtk.TextBuffer()
        self.textview.set_buffer(self.textbuf)

        self.layouts = [None] * 20
        for i in range(5, 20):
            fontdesc = pango.FontDescription('Sans %i' % i)
            self.layouts[i] = self.drawing.create_pango_layout("")
            self.layouts[i].set_font_description(fontdesc)

        fontdesc = pango.FontDescription('Sans 9')
        self.layout = self.drawing.create_pango_layout("")
        self.layout.set_font_description(fontdesc)

        fontdesc = pango.FontDescription('Sans 6')
        self.smalllayout = self.drawing.create_pango_layout("")
        self.smalllayout.set_font_description(fontdesc)

        fontdesc = pango.FontDescription('Sans 16')
        self.boldlayout = self.drawing.create_pango_layout("")
        self.boldlayout.set_font_description(fontdesc)

        self.tree.signal_autoconnect(self)
        self.window.show_all()

        gobject.timeout_add(100, self.timertick)


        

    def main(self):
        # All PyGTK applications must have a gtk.main(). Control ends here
        # and waits for an event to occur (like a key press or mouse event).
        gtk.main()

# If the program is run directly or passed as an argument to the python
# interpreter then create a HelloWorld instance and show it
if __name__ == "__main__":
    hello = HelloWorld()
    hello.main()
