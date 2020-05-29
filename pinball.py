import os
import pygame as pg
from pygame.compat import geterror
import math

img_dir = "img"

def load_image(name, colorkey=None):
    fullname = os.path.join(img_dir, name)
    try:
        image = pg.image.load(fullname)
    except pg.error:
        print("Cannot load image:", fullname)
        raise SystemExit(str(geterror()))
    
    if image.get_alpha() is None:
        image = image.convert()
    else:
        image = image.convert_alpha()

    if colorkey is not None:
        if colorkey == -1:
            colorkey = image.get_at((0, 0))
        image.set_colorkey(colorkey, pg.RLEACCEL)
    return image, image.get_rect()

def degreeToRad(degree):
    return degree * math.pi / 180

class leftBu(pygame.sprite.Sprite):
    def __init__(self, name):
        pygame.sprite.Sprite.__init__(self)
        self.image, self.rect = load_image(name)
        self.max_angle = 179;
        self.min_angle = 60;
        self.cur_angle = 60;
        self.controlled = False
        self.transforming = False

    def up(self):
        self.cur_angle += 4
        if self.cur_angle > self.max_angle:
            self.cur_angle = self.max_angle
            self.controlled = False
        rotatebu = pygame.transform.rotate(bo, self.cur_angle)
        if self.cur_angle > 90:
            return rotatebu, (0, - math.sin(degreeToRad(cur_angle - 90)) * self.rect.height)
        else:
            return rotatebu, (0, 0)

    def down(self):
        self.cur_angle -= 4
        if self.cur_angle < self.min_angle
            self.cur_angle = self.min_angle
        rotatebu = pygame.transform.rotate(bo, self.cur_angle)
        self.transforming = False
        if self.cur_angle > 90:
            return rotatebu, (0, - math.sin(degreeToRad(cur_angle - 90)) * self.rect.height)
        else:
            return rotatebu, (0, 0)

    def update(self):
        if self.controlled:
            return self.up()
        else:
            return self.down()

class rightBu(pygame.sprite.Sprite):
    def __init__(self, name):
        pygame.sprite.Sprite.__init__(self)
        self.image, self.rect = load_image(name)
        self.max_angle = 300;
        self.min_angle = 181;
        self.cur_angle = 300;
        self.stable = True
        self.controlled = False
        self.transforming = False

    def up(self):
        self.cur_angle -= 4
        if self.cur_angle < self.min_angle:
            self.cur_angle = self.min_angle
            self.controlled = False
        rotatebu = pygame.transform.rotate(bo, self.cur_angle)
        if self.cur_angle < 270:
            return rotatebu, ((1 - math.cos(degreeToRad(270 - self.cur_angle))) * self.rect.height, -math.sin(degreeToRad(270 - self.cur_angle)) * self.rect.height)
        else:
            return rotatebu, ((1 - math.cos(degreeToRad(self.cur_angle - 270))) * self.rect.height, math.sin(degreeToRad(self.cur_angle - 270)) *self.rect.height)

    def down(self):
        self.cur_angle += 4
        if self.cur_angle > self.max_angle
            self.cur_angle = self.max_angle
        rotatebu = pygame.transform.rotate(bo, self.cur_angle)
        self.transforming = False
        if self.cur_angle < 270:
            return rotatebu, ((1 - math.cos(degreeToRad(270 - self.cur_angle))) * self.rect.height, -math.sin(degreeToRad(270 - self.cur_angle)) * self.rect.height)
        else:
            return rotatebu, ((1 - math.cos(degreeToRad(self.cur_angle - 270))) * self.rect.height, math.sin(degreeToRad(self.cur_angle - 270)) *self.rect.height)

    def update(self):
        if self.controlled:
            return self.up()
        else:
            return self.down()

class ball(pygame.sprite.Sprite):
    def __init__(self, name):
        self.img, self.rect = load_image(name)
        self.

    def calcnewpos(self, rect, vector):
        (angle, z) = vector
        (dx, dy) = (z*math.cos(angle), z*math.sin(angle))
        return rect.move(dx, dy)




def main():
    pygame.init()
    screen = pygame.display.set_mode((640,480))
    pygame.display.set_caption('basic pygame program')

    background = pygame.Surface(screen.get_size())
    background = background.convert()
    background.fill((0,0,0))

    

    screen.blit(background,(0,0))
    pygame.display.flip()

    while 1:
        for event in pygame.event.get():
            if event.type == QUIT:
                return 
        screen.blit(background, (0,0))
        pygame.display.flip()

if __name__ == '__main__':
    main()