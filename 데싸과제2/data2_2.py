import csv

matrix = []

f = open('data2.csv', 'rt')
reader = csv.reader(f)



for line in reader:
    if line[5] == 'fdust':
        continue
    if line[6] == 'ozone':
        continue
    if line[7] == 'nd':
        continue
    if line[8] == 'cm':
        continue
    if line[9] == 'sgas':
        continue
    if line[10] == 'airquality':
        continue
    else:
        matrix.append(([float(line[5]), float(line[6]), float(line[7]), float(line[8]), float(line[9])], line[10]))



from typing import TypeVar
X = TypeVar('X')  # generic type to represent a data point


import math
from collections import Counter
from typing import TypeVar, List, Tuple
Vector = List[float]

def vector_subtract(v, w):
    """subtracts two vectors componentwise"""
    return [v_i - w_i for v_i, w_i in zip(v,w)]

def dot(v:Vector, w:Vector) -> float:
    assert len(v) == len(w)
    return sum(v_i * w_i for v_i, w_i in zip(v,w))

# 각 성분의 제곱 합
def sum_of_squares(v:Vector) -> float:
    return dot(v,v)

def squared_distance(v, w):
    return sum_of_squares(vector_subtract(v, w))

def distance(v, w):
   return math.sqrt(squared_distance(v, w))

def raw_majority_vote(labels):
    votes = Counter(labels)
    winner, _ = votes.most_common(1)[0]
    return winner

def majority_vote(labels):
    """assumes that labels are ordered from nearest to farthest"""
    vote_counts = Counter(labels)
    winner, winner_count = vote_counts.most_common(1)[0]
    num_winners = len([count
                       for count in vote_counts.values()
                       if count == winner_count])

    if num_winners == 1:
        return winner                     # unique winner, so return it
    else:
        return majority_vote(labels[:-1]) # try again without the farthest


def knn_classify(k, labeled_points, new_point):
    """each labeled point should be a pair (point, label)"""

    # order the labeled points from nearest to farthest
    by_distance = sorted(labeled_points,
                         key=lambda point_label: distance(point_label[0], new_point))

    # find the labels for the k closest
    k_nearest_labels = [label for _, label in by_distance[:k]]

    # and let them vote
    return majority_vote(k_nearest_labels)

def split_data(data: List[X], prob: float) -> Tuple[List[X], List[X]]:
    """Split data into fractions [prob, 1 - prob]"""
    data = data[:]                    # Make a shallow copy
    cut = int(len(data) * prob)       # Use prob to find a cutoff
    return data[:cut], data[cut:]     # and split the shuffled list there.

train, test = split_data(matrix, 0.7)


def precision(tp, fp, fn, tn):
    if tp+ fp == 0:
        return 0
    else:
        return tp / (tp + fp)

def recall(tp, fp, fn, tn):
    if tp + fn == 0:
        return 0
    else:
        return tp / (tp + fn)

def f1_score(tp, fp, fn, tn):
    p = precision(tp, fp, fn, tn)
    r = recall(tp, fp, fn, tn)
    if(p + r == 0):
        return 0
    else:
        return 2 * p * r / (p + r)



def testing (k:int, arr1:List, arr2:List, name):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for testi in arr1:
        others, actual = testi
        predict = knn_classify(k, arr2, others)
        if ((predict == name) & (predict == actual)):
            tp+=1
        elif ((predict == name) & (predict != actual)):
            tn+=1
        elif ((predict != name) & (predict != actual)):
            fp+=1
        elif ((predict != name) & (predict == actual)):
            fn+=1
        else:
            print("error")
    print("k:", k, " precision:", precision(tp, tn, fp, fn), " recall:",recall(tp, tn, fp, fn), " f1_score:" ,f1_score(tp, tn, fp, fn))

print("--------'최고'레벨 예측--------")
testing(1, test, train, "최고")
testing(3, test, train, "최고")
testing(5, test, train, "최고")
testing(7, test, train, "최고")

print("--------'좋음'레벨 예측--------")
testing(1, test, train, "좋음")
testing(3, test, train, "좋음")
testing(5, test, train, "좋음")
testing(7, test, train, "좋음")

print("--------'양호'레벨 예측--------")
testing(1, test, train, "양호")
testing(3, test, train, "양호")
testing(5, test, train, "양호")
testing(7, test, train, "양호")

print("--------'보통'레벨 예측--------")
testing(1, test, train, "보통")
testing(3, test, train, "보통")
testing(5, test, train, "보통")
testing(7, test, train, "보통")

print("--------'나쁨'레벨 예측--------")
testing(1, test, train, "나쁨")
testing(3, test, train, "나쁨")
testing(5, test, train, "나쁨")
testing(7, test, train, "나쁨")

print("--------'상당히나쁨'레벨 예측--------")
testing(1, test, train, "상당히나쁨")
testing(3, test, train, "상당히나쁨")
testing(5, test, train, "상당히나쁨")
testing(7, test, train, "상당히나쁨")

print("--------'매우나쁨'레벨 예측--------")
testing(1, test, train, "매우나쁨")
testing(3, test, train, "매우나쁨")
testing(5, test, train, "매우나쁨")
testing(7, test, train, "매우나쁨")

print("--------'최악'레벨 예측--------")
testing(1, test, train, "최악")
testing(3, test, train, "최악")
testing(5, test, train, "최악")
testing(7, test, train, "최악")