#!/usr/bin/env python
from __future__ import print_function
from __future__ import unicode_literals
#### REMEMBER TO USE
#### str.format() instead of %s
#### Python 3 often returns iterables where Python 2 returned lists.
#### This is usually fine, but if a list is realy needed, use the list() factory
###### function.
#### For example, given dictionary, d, list(d.keys()) returns its keys as a list.
#### Affected functions and methods include dict.items(), 
####### dict.keys(), dict.values(), filter(), map(), range(), and zip().
import os
import pygame
import sys
EXEC_DIR = os.path.dirname(__file__)

class Word(object):
    def __init__(self, word):
        if sys.platform == 'darwin':
            self.word_image = os.path.join('word_files', word)
        else:
            self.word_image = os.path.join(EXEC_DIR, "word_files", word)
        self.word = word
        self.image = pygame.image.load(self.word_image)
        self.rect = self.image.get_rect()
        self.spelling_word = self.word.split('.')[0]
        self.letters = list(self.spelling_word)
        self.width = self.image.get_width()
        self.length = len(self.letters)
        
    def draw(self, screen, x, y):
        screen.blit(self.image, [x, y])
