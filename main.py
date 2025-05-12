from src.classes.Game import Game
from src.classes.Predictor import Predictor
import torch.multiprocessing as multiprocessing

def main():
    parent_connection, child_connection = multiprocessing.Pipe()

    predictor = Predictor(parent_connection, child_connection)
    game = Game(child_connection, parent_connection)

    predictor_process = multiprocessing.Process(target=predictor.start)

    predictor_process.start()
    game.start()

if __name__ == '__main__':
    main()