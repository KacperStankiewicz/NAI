import argparse
import json
import numpy as np
from tmdbv3api import TMDb
from tmdbv3api import Movie
import os
from unidecode import unidecode


def argsParser():
    parser = argparse.ArgumentParser(description='Compute similarity score')
    parser.add_argument('--target', dest='targetUser', required=True,
                        help='Target user')
    parser.add_argument("--score-type", dest="scoreType", required=True,
                        choices=['Euclidean', 'Pearson'], help='Similarity metric to be used')
    return parser


def euclideanScore(dataset, user1, user2):
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

    return Sxy / np.sqrt(Sxx * Syy)


def calculateScores(dataset, targetUser, correlation_method):
    if targetUser not in dataset:
        raise TypeError('Cannot find ' + user1 + ' in the dataset')

    scores = {}
    filtered_dataset = (user for user in dataset if user != targetUser)
    for user in filtered_dataset:
        scores[user] = correlation_method(dataset, targetUser, user)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def getRecommendations(dataset, scores, targetUser):
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

    tmdb = TMDb()
    tmdb.api_key = os.environ.get('TMDB_SECRET')

    movie = Movie()
    print(f"TOP 5 movies {targetUser} should watch:")
    for title in recommendations:
        m = movie.search(unidecode(title))
        if len(m['results']) > 0:
            print(m[0].title)
            print("\t" + m[0].overview)

    print(f"\nTOP 5 movies {targetUser} should NOT watch:")
    for title in antiRecommendations:
        m = movie.search(unidecode(title))
        if len(m['results']) > 0:
            print(m[0].title)
            print("\t" + m[0].overview)
