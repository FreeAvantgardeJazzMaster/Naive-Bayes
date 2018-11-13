import numpy as np
import math
from scipy.stats import norm
from sklearn.base import BaseEstimator
from functools import reduce
from collections import Counter

def mean(vector):
    return sum(vector)/len(vector)


def stdev(vector):
    average = mean(vector)
    return math.sqrt((sum((x-average)**2 for x in vector))/max(1, len(vector)-1))


def divide_by_class(X, Y):
    result = dict()
    for x, y in zip(X, Y):
        if y in result:
            result[y] = np.append(result[y], [x], axis=0)
        else:
            result[y] = np.array([x])
    return result



class NaiveBayesNominal:
    def __init__(self):
        self.classes_ = None
        self.model = dict()
        self.y_prior = []

    def fit(self, X, y):
        self.probabilities={}
        self.possible_results=[]
        self.y_probabilities={}
        divided_by_class=divide_by_class(X,y)
        for key,values in divided_by_class.items():
            self.y_probabilities[key]=len(values)/len(X)
            x_prob=[]
            self.possible_results.append(key)
            # x_prob = [{key:item/len(line) for key,item in Counter(line).items()} for line in values.T]
            for line in values.T:
                x_prob.append({key : item/len(line) for key, item in Counter(line).items()})
            self.probabilities[key] = x_prob
        self.possible_results.sort()

    def predict_proba(self, X):
        result = []
        for x in X:
            probabilities = []
            for y in self.possible_results:
                P = self.y_probabilities[y]
                for i, x_i in enumerate(x):
                    prob_i = self.probabilities[y][i][x_i]
                    P = P * prob_i
                probabilities.append(P)
            result.append(probabilities)
        return np.array(result)

    def predict(self, X):
        x = self.predict_proba(X)
        return [list(row).index(max(row)) for row in x]

class NaiveBayesGaussian:
    def __init__(self):
        pass

    def fit(self, X, y):
        self.mean_coef = {}
        self.stdev_coef = {}
        self.possible_results=[]
        self.y_probabilities={}
        divided_by_class=divide_by_class(X, y)
        for key,values in divided_by_class.items():
            self.y_probabilities[key] = len(values)/len(X)
            means = []
            stdevs = []
            self.possible_results.append(key)
            for line in values.T:
                means.append(mean(line))
                stdevs.append(stdev(line))
            self.mean_coef[key] = means
            self.stdev_coef[key] = stdevs
        self.possible_results.sort()
        return self


    def predict_proba(self, X):
        result=[]
        for x in X:
            probabilities=[]
            for y in self.possible_results:
                values = norm.pdf(x, loc=self.mean_coef[y], scale=self.stdev_coef[y])
                P=self.y_probabilities[y]*reduce(lambda x, y : x*y, values)
                probabilities.append(P)
            result.append(probabilities)
        return np.array(result)

    def predict(self, X):
        x = self.predict_proba(X)
        return [list(row).index(max(row)) for row in x]


class NaiveBayesNumNom(BaseEstimator):
    def __init__(self, is_cat=None, m=0.0):
        raise NotImplementedError

    def fit(self, X, yy):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError
