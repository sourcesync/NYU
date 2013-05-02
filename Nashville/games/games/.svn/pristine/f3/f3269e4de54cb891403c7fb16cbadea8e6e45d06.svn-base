import pygame
import os
from random import randint

UP = 3
DOWN = 7
RIGHT = 5
LEFT = 9
EXEC_DIR = os.path.dirname(__file__)

class BouncingImage(pygame.sprite.Sprite):
    """ An image that bounces against the sides of the screen """
    def __init__(self, initial_position, initial_direction, path_to_image, bounds_x, bounds_y):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load(path_to_image)
        self.bounds_x, self.bounds_y = bounds_x, bounds_y
        self.rect = self.image.get_rect()
        self.rect.topleft = initial_position
        self.next_update_time = 0
        self.bottom = self.rect.bottom
        self.top = self.rect.top
        self.right = self.rect.right
        self.left = self.rect.left
        self.direction = initial_direction
        self.path_to_image = path_to_image
        self.speed = 5

    def update(self):
        self.top = self.rect.top
        self.left = self.rect.left
        self.right = self.rect.right
        self.bottom = self.rect.bottom
        if self.direction == RIGHT:
            self.rect.left += 1 * self.speed
            if self.right > bounds:
                self.reverse()
        elif self.direction == LEFT:
            self.rect.left -= 1 * self.speed
            if self.left < 0:
                self.reverse()
        #elif plane == 'vertical':
        #    if self.direction == UP:
        #        self.rect.top -= 1 * self.speed
        #        if self.top < 30:
        #            self.reverse()
        #    elif self.direction == DOWN:
        #        self.rect.top += 1 * self.speed
        #        if self.bottom > bounds:
        #            self.reverse()

    def reverse(self):
        if self.direction == RIGHT:
            self.direction = LEFT
        elif self.direction == LEFT:
            self.direction = RIGHT
        elif self.direction == UP:
            self.direction = DOWN
        elif self.direction == DOWN:
            self.direction = UP

