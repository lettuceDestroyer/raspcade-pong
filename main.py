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
score: int
high_score: int

model: any
camera: pygame.camera.Camera
game_over_font: pygame.font.Font
default_font: pygame.font.Font

def init():
    global score
    global high_score

    pygame.init()
    pygame.camera.init()
    pygame.display.set_caption("Raspcade Pong")

    score = 0
    high_score = 0

def game_over():
    window.fill(pygame.Color("black"))
    game_over_text = game_over_font.render("Game over!", False, pygame.Color("red"))
    window.blit(game_over_text, ((WIDTH - game_over_text.get_width()) / 2, (HEIGHT - game_over_text.get_height()) / 2))
    pygame.display.update()

def reload_game():
    global score

    for count in range(3, 0, -1):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        window.fill(pygame.Color("black"))

        # Render the main text
        main_text = default_font.render("Hand recognised. Game starting in", True, pygame.Color("white"))
        window.blit(main_text, ((WIDTH - main_text.get_width()) / 2, (HEIGHT - main_text.get_height()) / 2))

        # Render the countdown number
        countdown_text = default_font.render(str(count), True, pygame.Color("white"))
        window.blit(countdown_text, ((WIDTH - main_text.get_width()) / 2, ((HEIGHT - main_text.get_height()) / 2)) + main_text.get_height() + 10)

        pygame.display.flip()
        pygame.time.wait(1000)

    score = 0
    main()

def load_fonts():
    global game_over_font
    global default_font

    try:
        game_over_font = pygame.font.Font(os.path.join("res", "fonts", "faster_one", "faster_one_regular.ttf"), 80)
    except FileNotFoundError:
        game_over_font = pygame.font.SysFont(None, 80)

    try:
        default_font = pygame.font.Font(os.path.join("res", "fonts", "robo_condensed", "RobotoCondensed-Regular.ttf"), 20)
    except FileNotFoundError:
        default_font = pygame.font.SysFont(None, 80)

def update_score(score: int):
    if score > high_score:
        high_score = score
    
    score_text = default_font.render(f"high score: {high_score} | score: {score}", True, pygame.Color("white"))
    window.blit(score_text, (0,0))

def load_model():
    model = torch.load(MODEL_PATH, weights_only=False)
    model.eval()

def init_camera():
    camera_list = pygame.camera.list_cameras()

    print(f"camera_list: {camera_list}")

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

def translate_box(bbox: torch.Tensor):

    xmin = bbox[0] * HEIGHT/640
    ymin = bbox[1] * HEIGHT/640
    xmax = bbox[2] * HEIGHT/640
    ymax = bbox[3] * HEIGHT/640

    return torch.Tensor([xmin, ymin, xmax, ymax])

def main():
    global is_game_over

    init()
    init_camera()
    load_fonts()
    load_model()

    while True:
        image_as_surface = camera.get_image()
        image_as_tensor = surfact_to_tensor(image_as_surface)
        predictions = model(image_as_tensor)

        bboxes = predictions[0]['boxes']
    
        if is_game_over and len(bboxes) <= 0:
            game_over()
        elif is_game_over and len(bboxes) >= 1:
            reload_game()
        else:
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



            bbox: torch.Tensor = None
            if len(bboxes) > 0:
                bbox = translate_box(bboxes[0])

                y = (bbox[1] + bbox[3])/2
                left_paddle.y = y - left_paddle.height / 2

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
                score += 1
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

            # update score
            update_score()

            # update screen
            pygame.display.update()


if __name__ == '__main__':
    main()