from easyAI import TwoPlayerGame, Negamax, AI_Player, Human_Player
import numpy as np


class Oware(TwoPlayerGame):
    def __init__(self, players=None):
        for i, player in enumerate(players):
            player.score = 0
        self.players = players
        self.board = np.array([4] * 12)
        self.current_player = 1

    def possible_moves(self):
        moves = []
        if self.current_player == 1:
            if max(self.board[6:12]) == 0:
                moves = [i for i in range(6) if self.board[i] >= (6 - i)]
            if not moves:
                moves = [i for i in range(6) if self.board[i] != 0]
        else:
            if max(self.board[:6]) == 0:
                moves = [i for i in range(6, 12) if self.board[i] >= (6 - (i % 6))]
            if not moves:
                moves = [i for i in range(6, 12) if self.board[i] != 0]
        return moves

    def make_move(self, move):
        move = int(move)
        seeds_to_sow = self.board[move]
        self.board[move] = 0
        current_position = move

        while seeds_to_sow > 0:
            current_position = (current_position + 1) % 12
            if current_position != move:
                self.board[current_position] += 1
                seeds_to_sow -= 1

        self.collect_seeds(current_position)
        start_index = (self.opponent_index - 1) * 6
        if max(self.board[start_index: start_index + 6]) == 0:
            self.players[self.current_player - 1].score += sum(
                self.board[(self.current_player - 1) * 6: (self.current_player - 1) * 6 + 6])

    def collect_seeds(self, position):
        if self.current_player == 1 and 6 <= position <= 11 and self.board[position] in [2, 3]:
            while 6 <= position <= 11 and self.board[position] in [2, 3]:
                self.players[0].score += self.board[position]
                self.board[position] = 0
                position = (position - 1) % 12
        elif self.current_player == 2 and 0 <= position <= 5 and self.board[position] in [2, 3]:
            while 0 <= position <= 5 and self.board[position] in [2, 3]:
                self.players[1].score += self.board[position]
                self.board[position] = 0
                position = (position - 1) % 12

    def lose(self):
        return self.opponent.score > 24

    def is_over(self):
        return self.lose() or sum(self.board) < 7 or (
                (max(self.board[:6]) == 0 and self.current_player == 1)
                or (max(self.board[6:12]) == 0 and self.current_player == 2))

    def scoring(self):
        return 48 - self.players[self.opponent_index - 1].score

    def show(self):
        print("Plansza:")
        print("Gracz 2 ->", self.board[6:12][::-1], " suma:", self.players[1].score)
        print("Gracz 1 ->", self.board[0:6], " suma:", self.players[0].score)


if __name__ == "__main__":
    ai_algo = Negamax(6)
    ai_algo2 = Negamax(6)
    game = Oware([AI_Player(ai_algo), AI_Player(ai_algo2)])

    game.play()
    if game.players[0].score > game.players[1].score:
        print("Gracz 1 wygrywa.")
    elif game.players[0].score < game.players[1].score:
        print("Gracz 2 wygrywa.")
    else:
        print("Remis")
