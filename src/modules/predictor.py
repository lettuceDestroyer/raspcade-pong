import torch
import torch.multiprocessing as multiprocessing

def predict_bbox(img: torch.Tensor, model: any, queue: multiprocessing.Queue):
    with torch.no_grad():
        predictions = model([img])
        bboxes = predictions[0]['boxes']

        if len(bboxes) > 0:
            queue.put(bboxes[0])