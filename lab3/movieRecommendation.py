"""
Problem:
Na podstawie mojej oceny filmów chciałbym dostać 5 rekomendacji filmów które powinienem obejrzeć (powinny mi się spodobać), oraz 5
których nie powinienem oglądać (raczej mi się nie spodobają)

Autor:
Kacper Stankiewicz

============================
=Silnik rekomendacji filmów=
============================

Aby uruchomić program należy zainstalować następujące biblioteki
pip install numpy
pip install tmdbv3api
pip install unidecode

Należy również uzyskać klucz dostępowy do API TMDB https://developer.themoviedb.org/reference/intro/getting-started
Jego wartość umieścić w zmiennej środowiskowej o nazwie TMDB_SECRET

Silnik uruchamiamy poprzez wywołanie komendy w konsoli z parametrami:
- target : imie i nazwisko użytkownika dla którego chcemy uzyskac rekomendacje
- score-type : metryka podobieństwa która ma być użyta [Euclidean,Pearson]

Przykładowe wywołanie:
 python .\movieRecommendation.py --target 'Paweł Czapiewski' --score-type 'Pearson'

"""

import argparse
import json
import numpy as np
from tmdbv3api import TMDb
from tmdbv3api import Movie
import os
from unidecode import unidecode


def argsParser():
    """
    Funkcja służąca do zadeklarowania wymaganych argumentów, oraz
    wyciągania wartości z tych argumentów podanych na wejściu przez użytkownika

    Zdefiniowane są 2 argumenty:
        - targetUser
        - scoreType
    """
    parser = argparse.ArgumentParser(description='Compute similarity score')
    parser.add_argument('--target', dest='targetUser', required=True,
                        help='Target user')
    parser.add_argument("--score-type", dest="scoreType", required=True,
                        choices=['Euclidean', 'Pearson'], help='Similarity metric to be used')
    return parser


def euclideanScore(dataset, user1, user2):
    """
    Funkcja licząca współczynnik podobieństwa metrykją Euklidesową
    :param dataset (dictionary): oceny filmów wszystkich osób
    :param user1 (dictionary): oceny filmów użytkownika pierwszego
    :param user2 (dictionary): oceny filmów użytkownika drugiego
    :return: (float) współczynnik podobieństwa z zakresu 0 - 1.0
    """
    common_movies = {}

    for item in dataset[user1]:
        if item in dataset[user2]:
            common_movies[item] = 1

    if len(common_movies) == 0:
        return 0

    squared_diff = []

    for item in dataset[user1]:
        if item in dataset[user2]:
            squared_diff.append(np.square(dataset[user1][item] - dataset[user2][item]))

    return 1 / (1 + np.sqrt(np.sum(squared_diff)))


def pearsonScore(dataset, user1, user2):
    """
    Funkcja licząca współczynnik podobieństwa metryką Pearson'a
    :param dataset (dictionary): oceny filmów wszystkich osób
    :param user1 (dictionary): oceny filmów użytkownika pierwszego
    :param user2 (dictionary): oceny filmów użytkownika drugiego
    :return: (float) współczynnik podobieństwa z zakresu 0 - 2.0
    """
    common_movies = {}

    for item in dataset[user1]:
        if item in dataset[user2]:
            common_movies[item] = 1

    num_ratings = len(common_movies)

    if num_ratings == 0:
        return 0

    user1_sum = np.sum([dataset[user1][item] for item in common_movies])
    user2_sum = np.sum([dataset[user2][item] for item in common_movies])

    user1_squared_sum = np.sum([np.square(dataset[user1][item]) for item in common_movies])
    user2_squared_sum = np.sum([np.square(dataset[user2][item]) for item in common_movies])

    sum_of_products = np.sum([dataset[user1][item] * dataset[user2][item] for item in common_movies])

    Sxy = sum_of_products - (user1_sum * user2_sum / num_ratings)
    Sxx = user1_squared_sum - np.square(user1_sum) / num_ratings
    Syy = user2_squared_sum - np.square(user2_sum) / num_ratings

    if Sxx * Syy == 0:
        return 0

    return 1 - (Sxy / np.sqrt(Sxx * Syy))


def calculateScores(dataset, targetUser, correlation_method):
    """
    Funkcja sterująca. Oblicza współczynniki podobieństwa dla danego użytkownika, oraz sortuje wyniki od największego
    :param dataset (dictionary): oceny filmów wszystkich osób
    :param targetUser (dictionary): oceny filmów użytkownika dla którego chcemy poznać rekomendacje
    :param correlation_method: funkcja licząca współczynnik [pearsonScore,euclideanScore]
    :return: posortowana tablica zawierająca imię i nazwisko osoby, oraz policzony współczynnik
    """
    if targetUser not in dataset:
        raise TypeError('Cannot find ' + user1 + ' in the dataset')

    scores = {}
    filtered_dataset = (user for user in dataset if user != targetUser)
    for user in filtered_dataset:
        scores[user] = correlation_method(dataset, targetUser, user)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def getRecommendations(dataset, scores, targetUser):
    """
    Funkcja wybierająca filmy na podstawie ocen najlepiej dopasowanych użytkowników do celu (osoby)
    :param dataset: oceny filmów wszystkich osób
    :param scores: tablica zawierająca imię i nazwisko osoby, oraz policzony współczynnik
    :param targetUser: oceny filmów użytkownika dla którego chcemy poznać rekomendacje
    :return: 5 tytułów filmowych (polecanych)
    """
    recommendations = []
    for elem in scores:
        neighbour = elem[0]
        neighbourMovies = [item[0] for item in
                           sorted(dataset[neighbour].items(), key=lambda item: item[1], reverse=True)]
        for movie in neighbourMovies:
            if movie not in [item for item in dataset[targetUser]]:
                recommendations.append(movie)
                if len(recommendations) == 5:
                    return recommendations


def getAntiRecommendations(dataset, scores, targetUser):
    """
    Funkcja wybierająca filmy na podstawie ocen najlepiej dopasowanych użytkowników do celu (osoby)
    :param dataset: oceny filmów wszystkich osób
    :param scores: tablica zawierająca imię i nazwisko osoby, oraz policzony współczynnik
    :param targetUser: oceny filmów użytkownika dla którego chcemy poznać rekomendacje
    :return: 5 tytułów filmowych (niepolecanych)
    """
    anti_recommendations = []
    for elem in scores:
        neighbour = elem[0]
        neighbourMovies = [item[0] for item in
                           sorted(dataset[neighbour].items(), key=lambda item: item[1])]
        for movie in neighbourMovies:
            if movie not in [item for item in dataset[targetUser]]:
                anti_recommendations.append(movie)
                if len(anti_recommendations) == 5:
                    return anti_recommendations


def printMovies(movies):
    """
    Funkcja wyświetlająca oryginalne tytuły filmów wraz z krótkikm opisem z The Movie DB
    :param movies: tytuły filmów
    :return: void
    """
    tmdb = TMDb()
    tmdb.api_key = os.environ.get('TMDB_SECRET')
    movie = Movie()
    for title in movies:
        m = movie.search(unidecode(title))
        if len(m['results']) > 0:
            print(m[0].title)
            print("\t" + m[0].overview)
            print("-----------------------")


if __name__ == '__main__':
    args = argsParser().parse_args()
    targetUser = args.targetUser
    scoreType = args.scoreType
    ratings_file = './data/ratings.json'

    with open(ratings_file, 'r', encoding='utf-8') as f:
        data = json.loads(f.read())

    if scoreType == 'Euclidean':
        scores = calculateScores(data, targetUser, euclideanScore)
    else:
        scores = calculateScores(data, targetUser, pearsonScore)

    recommendations = getRecommendations(data, scores, targetUser)
    antiRecommendations = getAntiRecommendations(data, scores, targetUser)

    print(f"TOP 5 movies {targetUser} should watch:")
    printMovies(recommendations)
    print(f"\nTOP 5 movies {targetUser} should NOT watch:")
    printMovies(antiRecommendations)
