import pygame
from pygame import Color, Rect, sprite, Surface


class Paddle(sprite.Sprite):
    color: Color
    width: float
    height: float
    rect: Rect
    velocity: float
    x: float
    y: float

    def __init__(self, color: Color, width: float, height: float, velocity: float, x: float, y: float):
        """
        :param color: The color of the paddle
        :param width: The width of the paddle
        :param height: The height of the paddle
        :param velocity: The velocity of the paddle
        :param x: The x-coordinate of the ball
        :param y: The y-coordinate of the ball
        """
        sprite.Sprite.__init__(self)

        self.color = color
        self.width = width
        self.height = height
        self.rect = pygame.Rect(x, y, width, height)
        self.velocity = velocity
        self.x = x
        self.y = y

    def draw(self, surface: Surface):
        pygame.draw.rect(surface, self.color, self.rect)

    def update(self):
        """
        Updates the position of the paddle
        :return:
        """
        self.y += self.velocity
        self.rect = Rect(self.x, self.y, self.width, self.height)
