import warnings
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style

style.use('fivethirtyeight')


def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')

    distances = []
    print distances
    for group in data:
        print group
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclidean_distance, group])
            print "euclid: " + str(euclidean_distance)
            print distances

    votes = [i[1] for i in sorted(distances)[:k]]
    print votes
    vote_result = Counter(votes).most_common(1)[0][0]
    print vote_result

    return vote_result


dataset = {'k': [[1, 2], [2, 3], [3, 1], [5, 5], [7, 7.5], [8.5, 9]], 'r': [[6, 5], [7, 7], [8, 6], [3, 2]]}
new_features = [5, 7]
[[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
# same as:
##for i in dataset:
##    for ii in dataset[i]:
##        plt.scatter(ii[0],ii[1],s=100,color=i)

plt.scatter(new_features[0], new_features[1], s=100)

result = k_nearest_neighbors(dataset, new_features)
plt.scatter(new_features[0], new_features[1], s=50, color=result)
plt.show()