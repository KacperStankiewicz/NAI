import argparse
import json
import numpy as np


def build_arg_parser():
    parser = argparse.ArgumentParser(description='Compute similarity score')
    parser.add_argument('--target', dest='targetUser', required=False,
                        help='Target user')
    parser.add_argument("--score-type", dest="score_type", required=False,
                        choices=['Euclidean', 'Pearson'], help='Similarity metric to be used')
    return parser


def euclidean_score(dataset, user1, user2):
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


def pearson_score(dataset, user1, user2):
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


def calucate_scores(dataset, targetUser, correlation_method):
    if targetUser not in dataset:
        raise TypeError('Cannot find ' + user1 + ' in the dataset')

    scores = {}
    filtered_dataset = (user for user in dataset if user != targetUser)
    for user in filtered_dataset:
        scores[user] = correlation_method(dataset, targetUser, user)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def get_recommendations(dataset, scores, targetUser):
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


def get_anti_recommendations(dataset, scores, targetUser):
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
    args = build_arg_parser().parse_args()
    targetUser = args.targetUser
    score_type = args.score_type

    ratings_file = './data/ratings.json'

    with open(ratings_file, 'r', encoding='utf-8') as f:
        data = json.loads(f.read())

    if score_type == 'Euclidean':
        scores = calucate_scores(data, targetUser, euclidean_score)
    else:
        scores = calucate_scores(data, targetUser, pearson_score)

    print("TOP 5 movies to watch:")
    print(get_recommendations(data, scores, targetUser))
    print("TOP 5 movies NOT to watch:")
    print(get_anti_recommendations(data, scores, targetUser))
