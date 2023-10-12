# Gra Oware

## Autorzy: 
  Kacper Stankiewicz, Mikołaj Prętki
  
## Instrukcja przygotowania środowiska:
  Należy zainstalować na sowim komputerze pythona
    Windows: https://docs.python.org/3/using/windows.html
    Mac: https://docs.python.org/3/using/mac.html
    Unix: https://docs.python.org/3/using/unix.html
  Przyda się też IDE - Pycharm : https://www.jetbrains.com/help/pycharm/installation-guide.html

  Należy również zainstalować paczkę numpy (https://numpy.org/install/), oraz easyAI (https://zulko.github.io/easyAI/installation.html)

## Zasady gry: 
  Oware rozgrywane jest na płaszczyźnie składającej się z dwóch rzędów zawierających po sześć wgłębień.
  Przygotowując się do gry umieszczamy cztery nasiona w każdym z dwunastu wgłębień planszy. Celem gry jest schwytanie większej liczby nasion niż przeciwnik.
  W swej turze gracz wybiera jedno z wypełnionych wgłębień w rzędzie przed sobą i wyjmuje z niego wszystkie obecne tam nasiona. 
  Następnie gracz umieszcza po jednym z nich w kolejnych wgłębieniach w kolejności przeciwnej do ruchu wskazówek zegara z pominięciem wgłębienia od którego zaczął.
  Jeśli ostatnie nasiono zostaje umieszczone w rzędzie przeciwnika i miejsce, do którego zostało wrzucone zawiera 2 lub 3 nasionka, nasionka te zostają schwytane przez rozgrywającego turę.
  Jeśli pole poprzedzające również zawiera 2 lub 3 nasiona one również zostają przejęte przez gracza i tak dalej aż do momentu, gdy wgłębienie zawiera inną liczbę nasion lub dotrzemy do końca rzędu przeciwnika.
  Jeśli gracz nie może wykonać ruchu, bo wszystkie pola w jego rzędzie są puste, rozgrywka kończy się, a wszystkie nasiona w drugim rzędzie zostają przejęte przez przeciwnika.
  Jednak gracz nie może wymusić takiej sytuacji, tj. gdy rozgrywa swą turę, a rząd przeciwnika jest już pusty, musi rozegrać ją tak, by zasiać przynajmniej jedno z nasion w rzędzie oponenta.
  Dozwolone jest oczywiście takie poprowadzenie własnej rozgrywki, by zebrać wszystkie nasiona przeciwnika pozostawiając go bez możliwości wykonania ruchu.
  Gdy jeden z graczy schwyta więcej niż 24 nasiona, rozgrywka kończy się i zostaje on zwycięzcą.
