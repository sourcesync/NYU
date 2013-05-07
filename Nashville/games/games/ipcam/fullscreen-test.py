import pygame

pygame.init()
pygame.mouse.set_visible(False)

# resolution = (int(pygame.display.Info().current_w), int(pygame.display.Info().current_h)) #current screen resolution
# smallresolution = (int(pygame.display.Info().current_w*2/3), int(pygame.display.Info().current_h*2/3))

# Don't read in resolution automatically
# Suddenly this is causing crashes on my machine
resolution = (1280,1024)
smallresolution = (1024,768)
s = pygame.display.set_mode(resolution, pygame.FULLSCREEN)
fullscreen = 1

quit = False
while not quit:
    pygame.display.flip()

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
                    s = pygame.display.set_mode(smallresolution)
                else:
                    fullscreen=1
                    s = pygame.display.set_mode(resolution, pygame.FULLSCREEN)

print "Quitting"
pygame.quit()
