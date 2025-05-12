from multiprocessing.connection import Connection
import torch


class Predictor:
    def __init__(self, parent_connection: Connection, child_connection: Connection, model):
        self.parent_connection = parent_connection
        self.child_connection = child_connection
        self.model = model
        self.should_run = True

    def predict_bbox(self, img: torch.Tensor):
        with torch.no_grad():
            predictions = self.model([img])
            bboxes = predictions[0]['boxes']

            if len(bboxes) > 0:
                return bboxes[0]
            else:
                return None

    def start(self):
        while self.should_run:
            if self.parent_connection.poll():
                img = self.parent_connection.recv()
                bbox = self.predict_bbox(img)
                self.parent_connection.send(bbox)

    def stop(self):
        self.should_run = False