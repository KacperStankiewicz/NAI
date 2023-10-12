from easyAI import TwoPlayerGame, Negamax, AI_Player, Human_Player
import numpy as np


class Oware(TwoPlayerGame):
    def __init__(self, players=None):
        for i, player in enumerate(players):
            player.score = 0
        self.players = players
        self.board = np.array([4] * 12)
        self.current_player = 1
        """
            Inicjalizacja gry Oware.

            Parametry:
            players (list): Lista dwóch graczy, gdzie każdy gracz jest obiektem typu Player (lub jego dziedzicem). 
                            Domyślnie ustawiona na None. Gracze muszą być zdefiniowani przed rozpoczęciem gry.

            Opis:
            Ta metoda jest wywoływana podczas inicjalizacji obiektu gry Oware. Inicjalizuje ona początkowy stan gry, 
            w tym ustawia początkową punktację dla graczy, stan planszy i określa, który gracz rozpoczyna grę.

            Argument 'players' jest opcjonalny i służy do zdefiniowania dwóch graczy, którzy będą uczestniczyć w grze. 
            Każdy gracz musi być reprezentowany jako obiekt typu Player lub jego dziedzic. Gracze są przypisywani do 
            swoich indeksów w liście, gdzie gracz 1 znajduje się pod indeksem 0, a gracz 2 pod indeksem 1.

            Gra rozpoczyna się z początkową planszą, w której w każdym z dwunastu kubeczków znajduje się po 4 nasiona.
            Początkowo punktacja obu graczy wynosi 0.

            Zmienna 'current_player' wskazuje, który gracz jest obecnie odpowiedzialny za ruch. Domyślnie rozpoczyna gracz 1.

            Przykład użycia:
            >>> player1 = Human_Player("Gracz 1")
            >>> player2 = Human_Player("Gracz 2")
            >>> game = Oware([player1, player2])
            >>> game.play()
            """


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

    """
       Zwraca listę możliwych ruchów dla obecnego gracza w grze Oware.

       Opis:
       Ta metoda oblicza i zwraca listę możliwych ruchów dla obecnego gracza. Ruchy to indeksy kubeczków, 
       w których gracz może zacząć ruch. Lista jest zwracana jako lista liczb całkowitych.

       Jeśli obecny gracz to gracz 1, metoda sprawdza, czy dla kubeczków gracza 2 (indeksy 6-11) nie ma
       możliwości ruchu, w których zostawione jest wystarczająco nasion, aby gracz 1 mógł kontynuować ruch. 
       Jeśli takie ruchy istnieją, to lista ruchów ograniczona jest do tych kubeczków, aby nie pozostawić 
       przeciwnikowi możliwości ruchu.

       Następnie, jeśli lista ruchów jest nadal pusta, metoda dodaje do niej kubeczki, w których są nasiona.

       Jeśli obecny gracz to gracz 2, metoda analogicznie sprawdza ruchy dla kubeczków gracza 1 (indeksy 0-5). 
       Jeśli możliwe ruchy zostawiają graczowi 1 możliwość ruchu, to lista ruchów jest ograniczona do tych kubeczków.

       Jeśli obecny gracz nie ma możliwych ruchów, zostaje zwrócona pusta lista ['none'].

       Zwracane wartości:
       list: Lista indeksów kubeczków, w których gracz może rozpocząć ruch, lub ['none'] w przypadku braku ruchów.

       Przykład użycia:
       >>> game = Oware([player1, player2])
       >>> game.possible_moves()
       [0, 1, 2, 3, 4, 5]  # Lista możliwych ruchów dla gracza 1
       >>> game.make_move(2)
       >>> game.possible_moves()
       [0, 1, 3, 4, 5]  # Po wykonaniu ruchu przez gracza 1
       >>> game.make_move(5)
       >>> game.possible_moves()
       ['none']  # Brak możliwych ruchów dla gracza 2
       """

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

    """
        Wykonuje ruch w grze Oware.

        Parametry:
        move (int): Indeks kubeczka, w którym gracz rozpoczyna ruch.

        Opis:
        Ta metoda wykonuje ruch w grze Oware, przenosząc nasiona z wybranego kubeczka do innych kubeczków 
        zgodnie z zasadami gry. Ruch rozpoczyna się w kubeczku o indeksie 'move', a nasiona są przenoszone 
        zgodnie z ruchem wskazówek zegara.

        Gracz zbiera nasiona i punktację, jeśli spełnione są pewne warunki, a także wykonuje zbieranie nasion 
        w kubeczkach przeciwnika, jeśli spełnione są odpowiednie warunki.

        Po zakończeniu ruchu, metoda sprawdza, czy kubeczki przeciwnika zostały opróżnione, a następnie 
        aktualizuje punktację obecnego gracza.

        Parametr 'move' to indeks kubeczka, w którym gracz rozpoczyna ruch.

        Przykład użycia:
        >>> game = Oware([player1, player2])
        >>> game.make_move(2)  # Gracz wykonuje ruch rozpoczynając od kubeczka o indeksie 2
        >>> game.make_move(5)  # Kolejny ruch gracza
        >>> game.make_move(0)  # Ruch innego kubeczka
        """

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
    """
        Zbiera nasiona z kubeczków zgodnie z zasadami gry Oware.

        Parametry:
        position (int): Indeks kubeczka, w którym zakończony został ruch, i zbierane są nasiona zgodnie 
                        z zasadami gry.

        Opis:
        Ta metoda zbiera nasiona z kubeczków zgodnie z zasadami gry Oware, jeśli spełnione są odpowiednie warunki. 
        Ruch kończy się w kubeczku o indeksie 'position', i jeśli w tym kubeczku znajdują się 2 lub 3 nasiona, 
        są one zbierane przez obecnego gracza i punktowane. Nasiona w innych kubeczkach są usuwane.

        Funkcja uwzględnia zasady zbierania nasion przez obu graczy. Jeśli gracz 1 zakończy ruch w kubeczku od 6 do 11,
        to zbierane są nasiona, jeśli spełnione są odpowiednie warunki. Jeśli gracz 2 zakończy ruch w kubeczku od 0 do 5,
        to również zbierane są nasiona, jeśli spełnione są odpowiednie warunki.

        Parametr 'position' to indeks kubeczka, w którym zakończony został ruch.

        Przykład użycia:
        >>> game = Oware([player1, player2])
        >>> game.collect_seeds(8)  # Gracz 1 zakończył ruch w kubeczku o indeksie 8
        >>> game.collect_seeds(3)  # Gracz 2 zakończył ruch w kubeczku o indeksie 3
        >>> game.collect_seeds(11)  # Gracz 1 zakończył ruch w kubeczku o indeksie 11
        """
    def lose(self):
        return self.opponent.score > 24

    """
        Sprawdza, czy obecny gracz przegrał grę.

        Opis:
        Ta metoda sprawdza, czy obecny gracz (gracz 1 lub gracz 2) przegrał grę Oware. Gracz przegrywa, jeśli 
        punktacja przeciwnika przekroczy 24 punkty, co oznacza, że przeciwnik wygrał w zgodzie z zasadami gry.

        Zasady wygranej w grze Oware zakładają, że gracz, który zdobył więcej niż połowę wszystkich nasion na planszy,
        wygrywa grę.

        Zwracane wartości:
        bool: True, jeśli obecny gracz przegrał grę; w przeciwnym razie False.

        Przykład użycia:
        >>> game = Oware([player1, player2])
        >>> game.players[0].score = 26  # Gracz 1 przegrał grę, gdy punktacja przekroczyła 24
        >>> game.lose()
        True
        >>> game.players[1].score = 15  # Gracz 1 nie przegrał gry
        >>> game.lose()
        False
        """

    def is_over(self):
        return self.lose() or sum(self.board) < 7 or (
                (max(self.board[:6]) == 0 and self.current_player == 1)
                or (max(self.board[6:12]) == 0 and self.current_player == 2))

    """
        Sprawdza, czy gra Oware została zakończona.

        Opis:
        Ta metoda sprawdza, czy gra Oware została zakończona. Gra może zostać zakończona w jednym z trzech przypadków:
        1. Obecny gracz przegrał grę, czyli punktacja przeciwnika przekroczyła 24 punkty (warunek lose()).
        2. Suma nasion na planszy jest mniejsza niż 7, co oznacza, że nie można już kontynuować rozgrywki (warunek sum(self.board) < 7).
        3. Obecny gracz nie ma możliwości ruchu, ponieważ nie ma dostępnych kubeczków do zbierania nasion (warunek braku dostępnych ruchów).

        Zwracane wartości:
        bool: True, jeśli gra Oware została zakończona; w przeciwnym razie False.

        Przykład użycia:
        >>> game = Oware([player1, player2])
        >>> game.is_over()
        False  # Gra nie jest jeszcze zakończona
        >>> game.players[0].score = 26  # Gracz 1 przegrał grę
        >>> game.is_over()
        True  # Gra została zakończona, ponieważ gracz 1 przegrał
        >>> game.players[0].score = 15  # Gracz 1 nie przegrał, ale na planszy pozostało mniej niż 7 nasion
        >>> game.is_over()
        True  # Gra została zakończona z powodu małej liczby nasion na planszy
        >>> game.players[0].score = 10
        >>> game.players[1].score = 10
        >>> game.is_over()
        True  # Gra została zakończona z powodu braku dostępnych ruchów
        """

    def scoring(self):
        return 48 - self.players[self.opponent_index - 1].score

    """
       Oblicza punktację dla obecnego gracza w grze Oware.

       Opis:
       Ta metoda oblicza punktację dla obecnego gracza w grze Oware. Punkty są obliczane na podstawie punktacji 
       przeciwnika, co oznacza, że punktacja obecnego gracza jest równa 48 pomniejszona o punktację przeciwnika.

       Zasady wygranej w grze Oware zakładają, że gracz, który zdobył więcej niż połowę wszystkich nasion na planszy,
       wygrywa grę. Obliczanie punktów gracza w ten sposób ma na celu ocenę wygranej i porażki w grze.

       Zwracane wartości:
       int: Punkty obecnego gracza obliczone na podstawie punktacji przeciwnika.

       Przykład użycia:
       >>> game = Oware([player1, player2])
       >>> game.players[0].score = 20
       >>> game.players[1].score = 15
       >>> game.scoring()
       33  # Punkty gracza 1 obliczone na podstawie punktacji gracza 2
       >>> game.players[1].score = 30
       >>> game.scoring()
       18  # Punkty gracza 1 obliczone na podstawie punktacji gracza 2
       """

    def show(self):
        print("Plansza:")
        print("Gracz 2 ->", self.board[6:12][::-1], " suma:", self.players[1].score)
        print("Gracz 1 ->", self.board[0:6], " suma:", self.players[0].score)

"""
    Wyświetla stan planszy i punktację obu graczy w grze Oware.

    Opis:
    Ta metoda służy do wyświetlania aktualnego stanu planszy gry Oware oraz punktacji obu graczy. Wyświetlane są
    informacje o ilości nasion w poszczególnych kubeczkach planszy oraz o punktacji graczy.

    Plansza jest podzielona na dwie części: dla gracza 2 (indeksy 6-11) i gracza 1 (indeksy 0-5). Na planszy
    wyświetlane są ilości nasion w każdym z kubeczków. Dodatkowo wyświetlane są punktacje obu graczy.

    Przykład użycia:
    >>> game = Oware([player1, player2])
    >>> game.players[0].score = 20
    >>> game.players[1].score = 15
    >>> game.show()
    Plansza:
    Gracz 2 -> [4, 4, 4, 4, 4, 4] suma: 15
    Gracz 1 -> [4, 4, 4, 4, 4, 4] suma: 20
    """

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
