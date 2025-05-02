import os
import sys
import torch
import pygame
import pygame.camera
from src.classes.Ball import Ball
from src.classes.Paddle import Paddle

# Constants
WIDTH, HEIGHT = 1000, 600
MODEL_PATH = "./model.pth"

# Variables
window = pygame.display.set_mode((WIDTH, HEIGHT))
is_game_running: bool = True
is_game_over: bool = False

left_paddle = Paddle(pygame.Color("red"), 20, 120, 0, 80, (HEIGHT - 120) / 2)
right_paddle = Paddle(pygame.Color("blue"), 20, 120, 0, WIDTH - 80 - 20, (HEIGHT - 120) / 2)
ball = Ball(pygame.Color("white"), 15, WIDTH / 2 - 15, HEIGHT / 2 - 15, 0.5, 0.5)

game_over_font: pygame.font.Font

def init():
    pygame.init()
    pygame.camera.init()
    pygame.display.set_caption("PingPong")

def game_over():
    window.fill(pygame.Color("black"))
    game_over_text = game_over_font.render("Game over!", False, pygame.Color("red"))
    window.blit(game_over_text, ((WIDTH - game_over_text.get_width()) / 2, (HEIGHT - game_over_text.get_height()) / 2))
    pygame.display.update()

def load_fonts():
    try:
        game_over_font = pygame.font.Font(os.path.join("res", "fonts", "faster_one", "faster_one_regular.ttf"), 80)
    except FileNotFoundError:
        game_over_font = pygame.font.SysFont(None, 80)

def main():
    global is_game_over

    while True:
        if is_game_over:
            game_over()

        if not is_game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.camera.quit()
                    pygame.quit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        right_paddle.velocity = -0.7
                    if event.key == pygame.K_DOWN:
                        right_paddle.velocity = 0.7
                elif event.type == pygame.KEYUP:
                    right_paddle.velocity = 0
                    right_paddle.velocity = 0

            # ball movement controls
            if ball.x <= 0 or ball.x + ball.radius * 2 >= WIDTH:
                is_game_over = True

            if ball.y <= 0 or ball.rect.y >= HEIGHT - ball.radius:
                ball.y_velocity *= -1

            # make sure paddle does not go outside the window
            if left_paddle.y + left_paddle.height >= HEIGHT:
                left_paddle.y = HEIGHT - left_paddle.height
            if left_paddle.y < 0:
                left_paddle.y = 0

            if right_paddle.y + right_paddle.height > HEIGHT:
                right_paddle.y = HEIGHT - right_paddle.height
            if right_paddle.y < 0:
                right_paddle.y = 0

            # check if paddle and ball collide
            if pygame.sprite.collide_rect(left_paddle, ball):
                ball.x_velocity *= -1

            if pygame.sprite.collide_rect(ball, right_paddle):
                ball.x_velocity *= -1

            # move left paddle based on ball location
            if ball.x <= WIDTH / 2:
                if left_paddle.y + left_paddle.height / 2 < ball.y:
                    left_paddle.velocity = 0.9
                else:
                    left_paddle.velocity = -0.9
            else:
                left_paddle.velocity = 0

            window.fill(pygame.Color("black"))

            # update positions
            ball.update()
            left_paddle.update()
            right_paddle.update()

            # draw objects
            ball.draw(window)
            left_paddle.draw(window)
            right_paddle.draw(window)

            # update screen
            pygame.display.update()


if __name__ == '__main__':
    main()