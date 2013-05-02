## figure(1)
## ax = plt.axes()
## ax.set_xlim(options.xmin,options.xmax)
## ax.set_ylim(options.ymin,options.ymax)
## ax.set_autoscale_on(False)

## r = matplotlib.patches.Rectangle((0,0),5000,5000)
## p = PatchCollection([r])


## #colors = 100*pylab.rand(len(patches))
## #p = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.4)
## #p.set_array(pylab.array(colors))
## ax.add_collection(p)
## plt.draw()


import matplotlib
from matplotlib.patches import Circle, Wedge, Polygon, Rectangle
from matplotlib.collections import PatchCollection
import pylab

fig = pylab.figure()
ax1 = fig.add_subplot(111)

#ax1.set_xlim(options.xmin,options.xmax)
#ax1.set_ylim(options.ymin,options.ymax)
#ax1.set_autoscale_on(False)
ax1.set_xlim(options.xmin,options.xmax)
ax1.set_ylim(options.ymin,options.ymax)
xl=options.xmax-options.xmin
yl=options.ymax-options.ymin
w=0.5*(xl)
h=0.5*(yl) 
rec1 = Rectangle((options.xmin,options.ymin+0.5*yl),w,h)
rec2 = Rectangle((options.xmin+0.5*xl,options.ymin+0.5*yl),w,h)
rec3 = Rectangle((options.xmin,options.ymin),w,h)
rec4 = Rectangle((options.xmin+0.5*xl,options.ymin),w,h)

patches = [rec1,rec2,rec3,rec4]
#patches = [rec1,rec2]
colors = 100*pylab.rand(len(patches))
#colors = linspace(0,1,len(patches))
print colors
p = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=1)
p.set_array(pylab.array(colors))
ax1.add_collection(p)






fig = pylab.figure()
ax1 = fig.add_subplot(111)

#ax1.set_xlim(options.xmin,options.xmax)
#ax1.set_ylim(options.ymin,options.ymax)
#ax1.set_autoscale_on(False)
ax1.set_xlim(options.xmin,options.xmax)
ax1.set_ylim(options.ymin,options.ymax)

ax1.set_aspect('equal') # true to room
ax1.set_axis_off() # no labeling of axis
xl=options.xmax-options.xmin
yl=options.ymax-options.ymin
w=0.5*(xl)
h=0.5*(yl) 

patches=[]
for p in g.r.partitions:
    #walk through Room partitions and plot a colored rectangle for each
    w = p.xmax-p.xmin
    h = p.ymax-p.ymin
    patches.append(Rectangle((p.xmin,p.ymin),w,h))

colors = 100*pylab.rand(len(patches))
#colors = linspace(0,1,len(patches))
print colors
p = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=1)
p.set_array(pylab.array(colors))
ax1.add_collection(p)
plot([0],[-3000],mfc='w',mec='k',mew=5,marker='o',ms=30,ls='None')
# pylab.show()


fig = pylab.figure()
#ax2 = fig.add_subplot(111)
ax2 = fig.add_axes([0,0,1,1]) # tight
#ax2.set_xlim(options.xmin,options.xmax)
#ax2.set_ylim(options.ymin,options.ymax)
#ax2.set_autoscale_on(False)
ax2.set_xlim(options.xmin,options.xmax)
ax2.set_ylim(options.ymin,options.ymax)

ax2.set_aspect('equal') # true to room
ax2.set_axis_off() # no labeling of axis


facecolors = [cm.jet(x) for x in pylab.rand(len(g.r.partitions))]
for i,p in enumerate(g.r.partitions):
    #walk through Room partitions and plot a colored rectangle for each
    w = p.xmax-p.xmin
    h = p.ymax-p.ymin
    
    ax2.add_patch(Rectangle((p.xmin,p.ymin),w,h,fc=facecolors[i]))

#colors = 100*pylab.rand(len(patches))
#colors = linspace(0,1,len(patches))
#print colors
#p = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=1)
#p.set_array(pylab.array(colors))
#ax2.add_collection(p)
plot([0],[-3000],mfc='w',mec='k',mew=5,marker='o',ms=30,ls='None')
