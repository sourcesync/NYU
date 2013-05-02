""" Shows a list of images from command line.
Space bar toggles images
Example: python show_image.py image001.jpg image002.jpg

"""
import pygame
import sys



def show_image(image):

    # show first image
    image_w = image.get_width()
    image_h = image.get_height()


    resolution = (image_w,image_h)
    if fullscreen == 1:
        s = pygame.display.set_mode(resolution, pygame.FULLSCREEN)
    else:
        s = pygame.display.set_mode(resolution)

    screen = pygame.display.get_surface()

    screen.blit(image,(0,0))
    pygame.display.flip()

## print len(sys.argv)
## print sys.argv[1]
## print pygame.image.get_extended()

pygame.init()
pygame.mouse.set_visible(False)

fullscreen=1

images = [] # list of surfaces
for i in xrange(len(sys.argv)-1):
    # images in command line

    image = pygame.image.load(sys.argv[i+1])
    images.append(image)

assert(len(images)>0)

current_image = 0

show_image(images[current_image])


quit = False
while not quit:
    #pygame.display.flip()

    # fullscreen = (pygame.display.get_surface().get_flags() & pygame.FULLSCREEN) == pygame.FULLSCREEN
    # if not fullscreen:
    #     print "LOST FULLSCREEN"

    for event in pygame.event.get():
        if hasattr(event, 'key') and event.type == pygame.KEYDOWN:
            quit = event.key == pygame.K_ESCAPE

        if hasattr(event, 'key') and event.type == pygame.KEYDOWN:
            if event.key == pygame.K_f:
                print "f"
                if fullscreen:
                    fullscreen=0
                    show_image(images[current_image])
                else:
                    fullscreen=1
            if event.key == pygame.K_SPACE:
                current_image+=1
                print "Changing image to %d" % current_image
                if current_image>len(images)-1:
                    current_image=0
                print "Now image is to %d" % current_image
                show_image(images[current_image])


        

print "Quitting"
pygame.quit()
