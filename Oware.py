from easyAI import TwoPlayerGame, Negamax, AI_Player, Human_Player
import numpy as np


class Oware(TwoPlayerGame):
    def __init__(self, players=None):
        for i, player in enumerate(players):
            player.score = 0
        self.players = players
        self.board = np.array([4] * 6 + [4] * 6)  # Początkowy stan planszy
        self.current_player = 2  # Gracz 1 rozpoczyna

    def possible_moves(self):
        if self.current_player == 1:
            # return [str(i) for i in range(6) if self.board[i] != 0]
            if (max(self.board[:6]) == 0): return ['none']
            return [i for i in range(6) if self.board[i] != 0]
        else:
            if (max(self.board[6:12]) == 0): return ['none']
            return [i for i in range(6, 12) if self.board[i] != 0]

    def make_move(self, move):
        if (move == 'none'):
            if self.current_player == 1:
                self.players[1].score += sum(self.board[0:6])
            else:
                self.players[0].score += sum(self.board[6:12])
            return

        move = int(move)
        seeds_to_sow = self.board[move]
        self.board[move] = 0
        current_position = move

        while seeds_to_sow > 0:
            current_position = (current_position + 1) % 12
            if current_position != move:
                self.board[current_position] += 1
                seeds_to_sow -= 1

        self.collect_seads(current_position)

    def collect_seads(self, position):
        if self.current_player == 1 and 6 <= position <= 11 and self.board[position] in [2, 3]:
            while 6 <= position <= 11 and self.board[position] in [2, 3]:
                self.player.score += self.board[position]
                self.board[position] = 0
                position = (position - 1) % 12
        elif self.current_player == 2 and 0 <= position <= 5 and self.board[position] in [2, 3]:
            while 0 <= position <= 5 and self.board[position] in [2, 3]:
                self.player.score += self.board[position]
                self.board[position] = 0
                position = (position - 1) % 12

    def lose(self):
        return self.opponent.score > 24

    def is_over(self):
        return self.lose() or sum(self.board) < 7

    def scoring(self):
        return 48 - self.opponent.score

    def show(self):
        print("Plansza:")
        print("Gracz 2 ->", self.board[6:13][::-1], " suma :", self.players[1].score)
        print("Gracz 1 ->", self.board[0:6], " suma :", self.players[0].score)


if __name__ == "__main__":
    ai_algo = Negamax(6)  # Algorytm AI (zmień głębokość według potrzeb)
    game = Oware([AI_Player(ai_algo), AI_Player(ai_algo)])

    game.play()
    if (game.player.score > game.opponent.score):
        print("Gracz ", {game.current_player}, " wins.")
    elif game.player.score < game.opponent.score:
        print("Gracz ", {game.opponent_index}, " wins.")
    else:
        print("Remis")
