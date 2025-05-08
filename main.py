from src.classes.Game import Game

def main():
    game = Game()
    game.init_camera()
    game.load_fonts()
    game.load_model()
    game.run_game()

if __name__ == '__main__':
    main()