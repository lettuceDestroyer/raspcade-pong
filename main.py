import os
import sys
import torch
from torchvision import transforms
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

model: any
camera: pygame.camera.Camera
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

def load_model():
    model = torch.load(MODEL_PATH, weights_only=False)
    model.eval()

def init_camera():
    camera_list = pygame.camera.list_cameras()
    if not camera_list:
        raise ValueError("Sorry, no cameras detected.")
    camera = pygame.camera.Camera(camera_list[0])
    camera.start()

def resize_image_and_bbox(img, bboxes, new_height, new_width):
    original_width, original_height = img.size

    height_factor = original_height / new_height
    width_factor = original_width / new_width

    transform = transforms.Compose([
        transforms.Resize((new_height, new_width)),
        transforms.ToTensor()
    ])
    img_as_tensor = transform(img)

    # Scale x coordinates
    bboxes[:, [0, 2]] = bboxes[:, [0, 2]] / width_factor
    # Scale y coordinates
    bboxes[:, [1, 3]] = bboxes[:, [1, 3]] / height_factor

    return img_as_tensor, bboxes

def surfact_to_tensor(surface: pygame.Surface):
    # Use surfarray to get the pixel data
    pixel_array = pygame.surfarray.pixels3d(surface)
    # Convert the numpy array to a PyTorch tensor
    tensor = torch.tensor(pixel_array, dtype=torch.float32)
    # If needed, permute the dimensions to match the typical (C, H, W) format
    tensor = tensor.permute(2, 0, 1)
    # Should print torch.Size([3, .., ..])
    print(tensor.shape)
    return tensor

def main():
    global is_game_over

    init()
    init_camera()
    load_fonts()
    load_model()

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

            image_as_surface = camera.get_image()
            image_as_tensor = surfact_to_tensor(image_as_surface)
            predictions = model(image_as_tensor)

            boxes = predictions[0]['boxes']
            labels_as_tensor = predictions[0]['labels']
            scores = predictions[0]['scores']


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
