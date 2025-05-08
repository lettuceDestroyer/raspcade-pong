import torch.multiprocessing as multiprocessing
from src.classes.Ball import Ball
from src.classes.Paddle import Paddle
from src.modules.predictor import predict_bbox
from src.modules.utils import translate_bbox
import torchvision
import numpy
import os
import pygame
import pygame.camera
import torch
import PIL

# Constants
MODEL_PATH = "./model.pth"

IMAGE_WIDTH = 800
IMAGE_HEIGHT = 640

class Game():
    def __init__(self):
        pygame.init()
        pygame.camera.init()
        pygame.display.set_caption("Raspcade Pong")
        
        self.clock = pygame.time.Clock()
        self.window = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        self.width = self.window.get_width()
        self.height = self.window.get_height()
        
        self.left_paddle = Paddle(pygame.Color("red"), 20, 120, 0, 80, (self.height - 120) / 2)
        self.right_paddle = Paddle(pygame.Color("blue"), 20, 120, 0, self.width - 80 - 20, (self.height - 120) / 2)
        self.ball = Ball(pygame.Color("white"), 15, self.width / 2 - 15, self.height / 2 - 15, 0.5, 0.5)

        self.score = 0
        self.high_score = 0
        self.is_game_over = False

        self.camera = None
        self.model = None
        self.queue = multiprocessing.Queue()
        self.process = None

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
            torchvision.transforms.ToTensor()
        ])

    def game_over(self):
        self.window.fill(pygame.Color("black"))
        game_over_text = game_over_font.render("Game over!", False, pygame.Color("red"))
        self.window.blit(game_over_text, ((self.width - game_over_text.get_width()) / 2, (self.height - game_over_text.get_height()) / 2))
        pygame.display.update()

    def reload_game(self):
        self.window.fill(pygame.Color("black"))

        # Render the main text
        main_text = default_font.render("Hand recognised. Game starting in", True, pygame.Color("white"))
        self.window.blit(main_text, ((self.width - main_text.get_width()) / 2, (self.height - main_text.get_height()) / 2))

        # Render the countdown number
        for count in range(3, 0, -1):
            countdown_text = default_font.render(str(count), True, pygame.Color("white"))
            x = (self.width - countdown_text.get_width()) / 2
            y = (self.height - main_text.get_height()) / 2 + main_text.get_height() + 10

            self.window.blit(countdown_text, (x, y))
            pygame.display.flip()
            pygame.time.wait(1000)

        self.score = 0
        self.run_game()

    def load_fonts(self):
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

    def update_score(self, score: int):
        if self.score > self.high_score:
            self.high_score = score

        score_text = default_font.render(f"  high score: {self.high_score} | score: {score}", True, pygame.Color("white"))
        self.window.blit(score_text, (0,0))

    def load_model(self):
        if torch.cuda.is_available():
            self.model = torch.load(MODEL_PATH, weights_only=False)
        else:
           self.model = torch.load(MODEL_PATH, weights_only=False, map_location=torch.device("cpu"))
        self.model.eval()

    def init_camera(self):
        camera_list = pygame.camera.list_cameras()
        if not camera_list:
            raise ValueError("Sorry, no cameras detected.")
        self.camera = pygame.camera.Camera(camera_list[0])
        self.camera.start()

    def tensor_to_surface(self, tensor: torch.Tensor):
        # Ensure it's a uint8 tensor in (3, H, W) format
        if tensor.dtype != torch.uint8:
            tensor = (tensor.clamp(0, 1) * 255).to(torch.uint8)
        # Convert to (H, W, 3) for NumPy
        pixel_array = tensor.permute(1, 2, 0).cpu().numpy()
        # Transpose to (W, H, 3) for pygame
        pixel_array = numpy.transpose(pixel_array, (1, 0, 2))
        # Create a surface
        surface = pygame.surfarray.make_surface(pixel_array.copy())
        return surface

    def take_image(self):
        image_as_surface = self.camera.get_image()
        return image_as_surface

    def run_game(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.camera.quit()
                    pygame.quit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.camera.quit()
                        pygame.quit()
                    if event.key == pygame.K_UP:
                        self.right_paddle.velocity = -0.7
                    if event.key == pygame.K_DOWN:
                        self.right_paddle.velocity = 0.7
                elif event.type == pygame.KEYUP:
                    self.right_paddle.velocity = 0

            if self.process is None or not self.process.is_alive():
                img = self.take_image()
                img = pygame.transform.rotate(img, 90)
                img = pygame.surfarray.array3d(img)
                img = PIL.Image.fromarray(img)
                img_as_tensor = self.transform(img)
                self.process = multiprocessing.Process(target=predict_bbox, args=(img_as_tensor, self.model, self.queue))
                self.process.start()

            bbox = None

            if not self.queue.empty():
                bbox = self.queue.get()

            if self.is_game_over and bbox is None:
                self.game_over()
            elif self.is_game_over and bbox is not None:
                self.reload_game()
            else:
                if bbox is not None:
                    bbox = translate_bbox(bbox, self.height)
                    y = int(bbox[1])
                    print(f"y: {y}")
                    self.right_paddle.y = y

                # ball movement controls
                if self.ball.x <= 0 or self.ball.x + self.ball.radius * 2 >= self.width:
                    self.is_game_over = True

                if self.ball.y <= 0 or self.ball.rect.y >= self.height - self.ball.radius:
                    self.ball.y_velocity *= -1

                # make sure paddle does not go outside the self.window
                if self.left_paddle.y + self.left_paddle.height >= self.height:
                    self.left_paddle.y = self.height - self.left_paddle.height
                if self.left_paddle.y < 0:
                    self.left_paddle.y = 0

                if self.right_paddle.y + self.right_paddle.height > self.height:
                    self.right_paddle.y = self.height - self.right_paddle.height
                if self.right_paddle.y < 0:
                    self.right_paddle.y = 0

                # check if paddle and ball collide
                if pygame.sprite.collide_rect(self.left_paddle, self.ball):
                    
                    if self.ball.x_velocity > 0:
                        self.ball.x_velocity += 0.05
                    else:
                        self.ball.x_velocity -= 0.05

                    if self.ball.y_velocity > 0:
                        self.ball.y_velocity += 0.05
                    else:
                        self.ball.y_velocity -= 0.05

                    self.ball.x_velocity *= -1

                if pygame.sprite.collide_rect(self.ball, self.right_paddle):
                    self.score += 1

                    if self.ball.x_velocity > 0:
                        self.ball.x_velocity += 0.05
                    else:
                        self.ball.x_velocity -= 0.05

                    if self.ball.y_velocity > 0:
                        self.ball.y_velocity += 0.05
                    else:
                        self.ball.y_velocity -= 0.05

                    self.ball.x_velocity *= -1

                # move left paddle based on ball location
                if self.ball.x <= self.width / 2:
                    if self.left_paddle.y + self.left_paddle.height / 2 < self.ball.y:
                        self.left_paddle.velocity = 0.9
                    else:
                        self.left_paddle.velocity = -0.9
                else:
                    self.left_paddle.velocity = 0

                self.window.fill(pygame.Color("black"))

                # update positions
                self.ball.update()
                self.left_paddle.update()
                self.right_paddle.update()

                # draw objects
                self.ball.draw(self.window)
                self.left_paddle.draw(self.window)
                self.right_paddle.draw(self.window)

                # update score
                self.update_score(self.score)

            # limit FPS to 120
            self.clock.tick(120)

            # update screen
            pygame.display.update()