import torch
import torchvision

IMAGE_WIDTH = 800
IMAGE_HEIGHT = 640

def translate_bbox(bbox: torch.Tensor, new_height):
    xmin = bbox[0] * new_height / 640
    ymin = bbox[1] * new_height / 640
    xmax = bbox[2] * new_height / 640
    ymax = bbox[3] * new_height / 640

    return torch.Tensor([xmin, ymin, xmax, ymax])

def img_to_tensor(img):
    """
    Converts a PIL Image tensor and scales the image accordingly.
    :param img: a PIL Image
    :return a tensor
    """
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        torchvision.transforms.ToTensor()
    ])

    return transform(img)