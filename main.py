import os
import pygame
from pygame.font import FontType

pygame.init()

# INITIALS
WIDTH, HEIGHT = 1000, 600
pygame.display.set_caption("PingPong")
window = pygame.display.set_mode((WIDTH, HEIGHT))
is_game_running: bool = True
is_game_over: bool = False

# paddle dimensions
paddle_width, paddle_height = 20, 120
left_paddle_y = right_paddle_y = HEIGHT / 2 - paddle_height / 2
left_paddle_x, right_paddle_x = 100 - paddle_width / 2, WIDTH - (100 - paddle_width / 2)
left_paddle_vel, right_paddle_vel = 0, 0

# for the ball
radius = 15
ball_x, ball_y = WIDTH / 2 - radius, HEIGHT / 2 - radius
ball_vel_x, ball_vel_y = 0.7, 0.7
game_over_font: FontType

try:
    game_over_font = pygame.font.Font(os.path.join("res", "fonts", "faster_one", "faster_one_regular.ttf"), 80)
except FileNotFoundError:
    game_over_font = pygame.font.SysFont(None, 80)

def game_over():
    global is_game_running
    window.fill(pygame.Color("black"))
    game_over_text = game_over_font.render("Game over!", False, pygame.Color("red"))
    window.blit(game_over_text, ((WIDTH - game_over_text.get_width()) / 2, (HEIGHT - game_over_text.get_height()) / 2))
    pygame.display.update()

def main():
    global is_game_over
    global is_game_running
    global  ball_x
    global ball_y
    global ball_vel_x
    global ball_vel_y
    global right_paddle_vel
    global left_paddle_vel
    global left_paddle_x
    global left_paddle_y
    global right_paddle_x
    global right_paddle_y

    while is_game_running:
        if is_game_over:
            game_over()

        if not is_game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    is_game_running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        right_paddle_vel = -0.9
                    if event.key == pygame.K_DOWN:
                        right_paddle_vel = 0.9
                elif event.type == pygame.KEYUP:
                    right_paddle_vel = 0
                    left_paddle_vel = 0

            # ball movement controls
            if ball_x <= 0 + radius or ball_x >= WIDTH - radius:
                is_game_over = True

            if ball_y <= 0 + radius or ball_y >= HEIGHT - radius:
                ball_vel_y *= -1

            # paddle movement controls
            if left_paddle_y >= HEIGHT - paddle_height:
                left_paddle_y = HEIGHT - paddle_height
            if left_paddle_y <= 0:
                left_paddle_y = 0

            if right_paddle_y >= HEIGHT - paddle_height:
                right_paddle_y = HEIGHT - paddle_height
            if right_paddle_y <= 0:
                right_paddle_y = 0

            # paddle collisions
            # left paddle
            if left_paddle_x <= ball_x <= left_paddle_x + paddle_width:
                if left_paddle_y <= ball_y <= left_paddle_y + paddle_height:
                    ball_vel_x *= -1

            # right paddle
            if right_paddle_x <= ball_x <= right_paddle_x + paddle_width:
                if right_paddle_y <= ball_y <= right_paddle_y + paddle_height:
                    ball_vel_x *= -1

            # movements
            ball_x += ball_vel_x
            ball_y += ball_vel_y
            right_paddle_y += right_paddle_vel
            left_paddle_y += left_paddle_vel

            # move left paddle based on ball location
            if left_paddle_y + paddle_height / 2 <= ball_y:
                if ball_x <= WIDTH / 2:
                    left_paddle_vel = 0.9
            elif left_paddle_y + paddle_height / 2 >= ball_y:
                if ball_x <= WIDTH / 2:
                    left_paddle_vel = -0.9

            window.fill(pygame.Color("black"))

            # OBJECTS
            pygame.draw.circle(window, pygame.Color("white"), (ball_x, ball_y), radius)
            pygame.draw.rect(window, pygame.Color("red"),
                             pygame.Rect(right_paddle_x, right_paddle_y, paddle_width, paddle_height))
            pygame.draw.rect(window, pygame.Color("red"),
                             pygame.Rect(left_paddle_x, left_paddle_y, paddle_width, paddle_height))
            pygame.display.update()

if __name__ == '__main__':
    main()