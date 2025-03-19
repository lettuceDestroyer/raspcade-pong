import pygame
from pygame import Color, Rect, sprite, Surface


class Ball(sprite.Sprite):
    color: Color
    radius: float
    rect: Rect
    x: float
    y: float
    x_velocity: float
    y_velocity: float

    def __init__(self, color: Color, radius: float, x: float, y: float, x_velocity: float, y_velocity: float):
        """
        :param color: The color of the ball
        :param radius: The radius of the ball
        :param x: The x-coordinate of the ball
        :param y: The y-coordinate of the ball
        :param x_velocity: The velocity of the ball in the x-direction
        :param y_velocity: The velocity of the ball in the y-direction
        """
        sprite.Sprite.__init__(self)

        self.color = color
        self.radius = radius
        self.rect = Rect(x, y, radius * 2, radius * 2)
        self.x = x
        self.y = y
        self.x_velocity = x_velocity
        self.y_velocity = y_velocity

    def draw(self, surface: Surface):
        pygame.draw.circle(surface, self.color, (self.x + self.radius, self.y + self.radius), self.radius)

    def update(self):
        """
        Updates the position of the ball
        :return:
        """
        self.x += self.x_velocity
        self.y += self.y_velocity
        self.rect = Rect(self.x, self.y, self.radius * 2, self.radius * 2)
