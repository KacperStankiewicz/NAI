import numpy as np
from easyAI import TwoPlayerGame, Negamax, AI_Player


class Oware(TwoPlayerGame):
    """
    Klasa implementująca klasę TwoPlayerGame. Umożliwia zagranie w grę Oware

        Atrybuty:
            - players (array) : tablica graczy
            - board (array) : tablica z aktualnym stanem gry
            - current_player (int) : numer gracza aktualnie wykonującego ruch (1,2)
            - player (Player) : gracz aktualnie wykonujący ruch
            - opponent_index (int) : indeks przeciwnika (0,1)

        Metody:
                - __init__(self, players) : Inicjalizacja gry Oware
                - possible_moves(self) : Zwrócenie możliwych ruchów
                - make_move(self, move) : Modyfikacja planszy zgodnie z wykonanym ruchem
                - collect_seeds(self, position) : Zebranie nasion z kubeczków i zwiększenie punktacji gracza
                - lose(self) : Czy aktualny gracz przegrał
                - is_over(self) : Sprawdzenie czy gra została zakończona
                - scoring(self) : Metoda oceniająca jakość gry gracza (AI)
                - show(self) : Wyświetlenie aktualnego stanu gry
    """

    def __init__(self, players=None):
        """
        Inicjalizacja gry Oware, ustawienie początkowych parametrów takich jak punktacja, plansza oraz numer gracza
        rozpoczynającego

            :parameter:
                players (array): Tablica dwóch graczy, gdzie każdy gracz jest obiektem typu Player (lub po nim dziedziczy).
                            Domyślnie ustawiona na None. Gracze muszą być zdefiniowani przed rozpoczęciem gry.

            :returns:
                Oware : obiekt klasy Oware
        """

        for i, player in enumerate(players):
            player.score = 0
        self.players = players
        self.board = np.array([4] * 12)
        self.current_player = 1

    def possible_moves(self):
        """
        Ta metoda oblicza i zwraca listę możliwych ruchów dla obecnego gracza. Ruchy to indeksy kubeczków,
        w których gracz może zacząć ruch.

            :returns:
                array: Lista indeksów kubeczków, które gracz może wybrać aby rozpocząć ruch.
        """
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
        """
        Modyfikuje planszę (tabelę) gry zgodnie z jej zasadami. Gracz podaje indeks kubeczka z którego zabiera
        nasiona, które następnie rozkłada po następnych kubeczkach w kierunku odwrotnym niż ruch wskazówek zegara
        z pominięciem kubeczka od którego zaczął. Po wykonaniu ruchu aktualizowana jest punktacja gracza.

            :parameter:
                move (int): Indeks kubeczka, w którym gracz rozpoczyna ruch.

            :returns:
                void
        """

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
        opponents_first_index = (self.opponent_index - 1) * 6
        if max(self.board[opponents_first_index: opponents_first_index + 6]) == 0:
            self.players[self.current_player - 1].score += sum(
                self.board[(self.current_player - 1) * 6: (self.current_player - 1) * 6 + 6])

    def collect_seeds(self, position):
        """
        Usuwa z planszy zebrane nasiona, i dodaje je do punktacji odpowiedniego gracza. Usuwanie następuje tylko
        jeśli w kubeczku o indeksie /position/ znajdują się 2 lub 3 nasiona. Jeśli w kubeczkach obok
        (zgodnie z ruchem wskazówek zegara) znajdują się 2 lub 3 nasiona, zostają one również zebrane.
        Zbieranie kończy się w momencie napotkania pierwszego kubeczka z inną ilością nasion niż 2 lub 3.

            :parameter
                position (int): Indeks kubeczka, w którym zakończony został ruch

            :returns
                void
        """

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
        """
        Sprawdza, czy obecny gracz przegrał grę.
        Jeśli przeciwnik zebrał więcej niż połowę wszystkich nasion drugi gracz nie ma możliwości na wygraną.

            :returns:
                bool : True jeśli przeciwnik zebrał więcej niż 24 nasiona, False w przeciwnym wypadku
        """

        return self.opponent.score > 24

    def is_over(self):
        """
        Sprawdza, czy gra została zakończona.
        Gra może zostać zakończona w jednym z trzech przypadków:
        1. Jeden z graczy zebrał więcej niż 24 nasiona.
        2. Suma nasion na planszy jest mniejsza niż 7.
        3. Wszystkie kubeczki obecnego gracza są puste.

            :returns:
                bool: True, jeśli gra Oware została zakończona; w przeciwnym razie False.
        """

        return self.lose() or sum(self.board) < 7 or (
                (max(self.board[:6]) == 0 and self.current_player == 1)
                or (max(self.board[6:12]) == 0 and self.current_player == 2))

    def scoring(self):
        """
        Ocena jakości gry. Tę metodę wykorzystuje algorytm AI w celu wybrania najlepszych ruchów

            :returns:
                int: Punktacja gracza obliczona według wzoru : 48 - ilość_punktów_przeciwnika
        """
        return 48 - self.players[self.opponent_index - 1].score

    def show(self):
        """
            Wyświetla stan planszy i punktację obu graczy.

            :returns:
                void
            Przykład:

            Plansza:
            Gracz 2 -> [4, 4, 4, 4, 4, 4] suma: 15
            Gracz 1 -> [4, 4, 4, 4, 4, 4] suma: 20
        """
        print("Plansza:")
        print("Gracz 2 ->", self.board[6:12][::-1], " suma:", self.players[1].score)
        print("Gracz 1 ->", self.board[0:6], " suma:", self.players[0].score)

if __name__ == "__main__":
    ai_algo = Negamax(1)
    ai_algo2 = Negamax(6)
    game = Oware([AI_Player(ai_algo), AI_Player(ai_algo2)])

    game.play()
    if game.players[0].score > game.players[1].score:
        print("Gracz 1 wygrywa.")
    elif game.players[0].score < game.players[1].score:
        print("Gracz 2 wygrywa.")
    else:
        print("Remis")
