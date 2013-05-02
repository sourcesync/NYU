import cv
import subprocess
from subprocess import Popen


cv.NamedWindow("Status",1)

#p1 = Popen(["python PySpell.py"], stdin=p1.stdout, stdout=subprocess.PIPE)
#p1 = Popen(["python", "PySpell.py"], stdin=subprocess.PIPE)
p1 = Popen(["python", "PySpell.py"], stdin=subprocess.PIPE)
while True:

    #look for escape
    key = cv.WaitKey(1) % 0x100
    if key == 27:
        cv.DestroyAllWindows()
        break


    if key==ord('a'):
        # pipe
        print "sending a"
        #output = p1.communicate("a")[0]
        output = p1.stdin.write("a")
        print output



    if key==ord('c'):
        print key

        

    
