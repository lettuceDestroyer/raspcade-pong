import PIL
import torch
import torchvision

# Constants
IMAGE_WIDTH = 800
IMAGE_HEIGHT = 640

def translate_bbox(bbox: torch.Tensor, new_height):
    xmin = bbox[0] * new_height / 640
    ymin = bbox[1] * new_height / 640
    xmax = bbox[2] * new_height / 640
    ymax = bbox[3] * new_height / 640

    return torch.Tensor([xmin, ymin, xmax, ymax])

def surface_to_tensor(surface: PIL.Image):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        torchvision.transforms.ToTensor()
    ])

    return transform(surface)