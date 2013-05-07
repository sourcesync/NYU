#! /usr/bin/python
'''
A fancy looking metronome in under 200 lines of code.
'''

import pygame
import time
import math
from pygame.locals import *

RESOLUTION = (480,480)
TARGET_FPS = 60

accumulator = 0

last_press = None
last_deltas = None

#------------------------------------------------------------------------------
def sample_beat():
    global last_deltas
    global last_press

    now = time.time()*1000
    if not last_press:
        last_press = now
        last_deltas = []
    else:
        delta = now - last_press
        last_press = now
        last_deltas.append(delta)
        if len(last_deltas) > 5:
            last_deltas.pop(0)
        
#------------------------------------------------------------------------------
class Dial:
    size = 600
    @classmethod
    def draw(cls, screen, amount, full):
        screen_rect = screen.get_rect()
        percent = float(amount)/full
        rads = math.pi*2*percent

        color = (255,255,255)

        half = cls.size/2
        start_angle = math.pi/2
        endpos = (half+half*math.cos(rads-start_angle),
                  half+half*math.sin(rads-start_angle))

        image = pygame.Surface((cls.size,cls.size))
        rect = image.get_rect()
        pygame.draw.aaline(image, color, rect.center, endpos)

        rect.center = screen_rect.center

        screen.blit(image, rect)

#------------------------------------------------------------------------------
class Numeral:
    beat_index = 0
    beats = ['1', '2', '3', '4']
    color = (255,255,255)
        
    @classmethod
    def beat(cls):
        cls.beat_index += 1
        cls.beat_index %= len(cls.beats)

    @classmethod
    def draw(cls, screen, amount, full):
        screen_rect = screen.get_rect()
        # font = pygame.font.Font( None, 440 )
        font = pygame.font.SysFont( "Menlo", 440)
        image = font.render( cls.beats[cls.beat_index], True, cls.color )
        rect = image.get_rect()
        rect.center = screen_rect.center
        screen.blit( image, rect )

#------------------------------------------------------------------------------
class ThrobOnHalfBeat:
    @classmethod
    def draw(cls, screen, amount, full):
        screen_rect = screen.get_rect()

        if amount > full:
            amount = full

        percent = float(amount)/full
        percent_distance_to_half = 1-2*abs(0.5-percent)
        size_intensity = 20*percent_distance_to_half

        rect = pygame.Rect(0, 0, int(0.05*screen_rect.width), 60)

        inflation = cls.calc_inflation(percent_distance_to_half)
        rect.inflate_ip(inflation, inflation)
        cls.calc_pos(rect, screen_rect)

        image = pygame.Surface((rect.width, rect.height))
        image.fill( cls.calc_color(percent_distance_to_half) )
        screen.blit( image, rect )

    @classmethod
    def calc_pos(cls, rect, screen_rect):
        rect.bottom = screen_rect.bottom - 60
        rect.centerx = screen_rect.centerx

    @classmethod
    def calc_color(cls, percent_distance_to_half):
        color_intensity = 255 * percent_distance_to_half
        color = (color_intensity,10,color_intensity)
        return color

    @classmethod
    def calc_inflation(cls, percent_distance_to_half):
        inflation = 20*percent_distance_to_half
        return inflation

#------------------------------------------------------------------------------
class ThrobOnBeat(ThrobOnHalfBeat):
    @classmethod
    def calc_pos(cls, rect, screen_rect):
        rect.top = screen_rect.top + 60
        rect.centerx = screen_rect.centerx

    @classmethod
    def calc_color(cls, percent_distance_to_half):
        color_intensity = 255 - (255 * percent_distance_to_half)
        color = (color_intensity,10,color_intensity)
        return color

    @classmethod
    def calc_inflation(cls, percent_distance_to_half):
        inflation = 20*(1-percent_distance_to_half)
        return inflation

#------------------------------------------------------------------------------
class Block:
    color = (255,210,10)

    @classmethod
    def draw(cls, screen, amount, full):
        screen_rect = screen.get_rect()
        percent = float(amount)/full

        rect = pygame.Rect(20,0, int(0.05*screen_rect.width), 40)
        rect.centery = screen_rect.bottom - int(screen_rect.height*percent)

        image = pygame.Surface((rect.width, rect.height))
        image.fill( cls.color )
        screen.blit( image, rect )

#------------------------------------------------------------------------------
def Tick(timeChange, screen):
    global accumulator

    screen.fill((0,0,0))

    if last_deltas:
        avg_delta = sum(last_deltas)/len(last_deltas)
    else:
        avg_delta = 1000

    accumulator += timeChange
    if accumulator > avg_delta:
        accumulator = accumulator - avg_delta
        Numeral.beat()

    #Dial.draw(screen, accumulator, avg_delta)
    Numeral.draw(screen, accumulator, avg_delta)
    #Block.draw(screen, accumulator, avg_delta)
    #ThrobOnBeat.draw(screen, accumulator, avg_delta)
    #ThrobOnHalfBeat.draw(screen, accumulator, avg_delta)

#------------------------------------------------------------------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode(RESOLUTION)
    pygame.display.set_caption( 'Metronome' )

    clock = pygame.time.Clock()

    while True:
        timeChange = clock.tick(TARGET_FPS)

        remainingEvents = pygame.event.get()
        for event in remainingEvents:
            if event.type == QUIT:
                return
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                return
            elif event.type == KEYDOWN:
                sample_beat()

        Tick( timeChange, screen )
        pygame.display.flip()

if __name__ == '__main__':
	main()
