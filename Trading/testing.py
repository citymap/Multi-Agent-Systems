from urllib.request import urlopen
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import math
import pandas as pd
import random

def get_coeffs(n):
    taylor_table = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            taylor_table[j,n-i-1] = (-(i))**j
    taylor_derivatives = np.zeros((n,1))
    taylor_derivatives[n-2] = 1
    coeffs = np.dot(np.linalg.inv(taylor_table),taylor_derivatives)
    return coeffs

coeffs = get_coeffs(3)
df1 = pd.read_csv("MSFT.csv")['open']*100

max_gain = 0
stock2 = list(df1[-1000:])
for i in range(10,len(stock2)):
    if stock2[i] > stock2[i-1]:
        max_gain += stock2[i] - stock2[i-1]
print(max_gain)

plt.plot(df1)
plt.show()

class Brain:
    def __init__(self, model):
        if model==0:
            self.model =  tf.keras.models.clone_model(model_base)
            input = np.asarray([list(np.zeros(i_n))])
            self.model.predict(input, 1)[0]

        else:
            self.model = tf.keras.models.clone_model(model, input_tensors=None)

    def crossover(self, genes_1, genes_2):

        weights_hidden = (genes_1[0]+genes_2[0])/2
        biases_hidden = (genes_1[1]+genes_2[1])/2
        weights_outputs = (genes_1[2]+genes_2[2])/2
        biases_outputs = (genes_1[3]+genes_2[3])/2
        self.weights = [weights_hidden, biases_hidden, weights_outputs, biases_outputs]
        self.model.set_weights(self.weights)


    def mutate(self):
        self.weights = self.model.get_weights()
        w1 = np.random.randn(i_n,30)
        r = np.random.rand(i_n,30)
        w1 = np.where(r>0.7,w1,0)

        b1 = np.random.randn(30)
        r = np.random.rand(30)
        b1 = np.where(r> 0.7, b1, 0)

        w2 = np.random.randn(30,1)
        r = np.random.rand(30, 1)
        w2 = np.where(r > 0.7, w2, 0)

        b2 = np.random.randn(1)/2
        r = np.random.rand(1)
        b2 = np.where(r > 0.7, b2, 0)
        self.weights[0] += w1
        self.weights[1] += b1
        self.weights[2] += w2
        self.weights[3] += b2
        self.model.set_weights(self.weights)


    def create(self):
        self.model.set_weights(self.weights)




class Vehicle:
    def __init__(self, model=0):
        self.scoretrack = []
        self.capitaltrack = []
        self.score = 0
        self.fitness = 0
        self.brain = Brain(model)
        self.capital = 1000



    def cash_in(self, last_10):
        n_derivatives =[]
        for i in range(2,len(last_10)):
            n_derivatives.append(np.dot(last_10[-i-1:],get_coeffs(i+1))[0])
        input = np.asarray([n_derivatives+[last_10[-1], last_10[-2]]])
        output = self.brain.model.predict(input, 1)[0]
        self.percentage_invested = output[0]
        money_in = min(self.capital*self.percentage_invested, last_10[-1])
        self.capital -= money_in
        self.actions = money_in /(last_10[-1])
        self.score -= last_10[-1] * self.percentage_invested

    def cash_out(self,fst, last):
        self.score += last*self.percentage_invested
        self.capital += self.actions * last
        self.scoretrack.append(self.score + fst-last)
        self.capitaltrack.append(self.capital)

    def reset(self):
        self.scoretrack = []
        self.capitaltrack = []
        self.score = 0
        self.capital = 1000
        self.fitness = 0

population_size = 10
vehicles = []
last_vehicles =[]
counter = 0
histograms = []
medians = []
means = []
mins = []
maxs = []
i_n = 5

model_base = tf.keras.models.Sequential()
input_layer = tf.keras.layers.Flatten()
hidden_layer = tf.keras.layers.Dense(units=30, input_shape=[i_n], activation='sigmoid')
output_layer = tf.keras.layers.Dense(units=1, input_shape=[30], activation='sigmoid')
model_base.add(input_layer)
model_base.add(hidden_layer)
model_base.add(output_layer)
input = np.asarray([list(np.zeros(i_n))])
start = time.time()
model_base.predict(input, verbose=0)

capital = 1000
for n in range(population_size):
    vehicles.append(Vehicle())
    vehicles[-1].brain.mutate()

for generation in range(1,100):
    st = random.randint(0,len(df1)-2001)
    print(st)
    stock = list(df1[st:st+1000])

    i = 10
    for i in range(10,len(stock)-1):
        for vehicle in vehicles:
            vehicle.cash_in(stock[i-i_n:i])
        for vehicle in vehicles:
            vehicle.cash_out(stock[10], stock[i])
        # print(i)
    collective_score = 0
    scores = []

    # plt.plot(vehicles[0].capitaltrack)
    plt.show()
    for vehicle in vehicles:

        vehicle.score = vehicle.score + stock[0] - stock[len(stock)-1]
        collective_score += vehicle.score
        scores.append(vehicle.score)

    # plt.show()
    median_score = np.median(scores)
    for vehicle in vehicles:
        vehicle.fitness = vehicle.score/collective_score
    p1s = []
    p2s = []

    print('------------')
    print(collective_score, median_score, np.mean(scores))
    for vehicle in vehicles:
        index = 0
        r = np.random.uniform(0,1)
        while r > 0:
            r-=vehicles[index].fitness
            index +=1
        index -=1
        # print(vehicles[index].brain.model.get_weights[0])
        p1s.append(vehicles[index].brain.model.get_weights())

        index = 0
        r = np.random.uniform(0, 1)
        while r > 0:
            r -= vehicles[index].fitness
            index += 1
        index -= 1
        # print(vehicles[index].brain.model.get_weights[0])
        p2s.append(vehicles[index].brain.model.get_weights())

    for v in range(len(vehicles)):
        vehicles[v].reset()

    st = random.randint(0, len(df1) - 2001)

    stock2 = list(df1[4000:5000])

    i = 10
    for i in range(10, len(stock2) - 1):
        for vehicle in vehicles:
            vehicle.cash_in(stock2[i - i_n:i])
        for vehicle in vehicles:
            vehicle.cash_out(stock2[10], stock2[i])

    plt.plot(stock2)
    for vehicle in vehicles:
        plt.plot(vehicle.scoretrack)
    plt.show()

    collective_score = 0
    scores = []
    for vehicle in vehicles:
        collective_score += vehicle.score + stock2[0] - stock2[len(stock2)-1]
        scores.append(vehicle.score)

    histograms.append(scores)
    medians.append(np.median(scores))
    means.append(np.mean(scores))
    maxs.append(np.max(scores))
    mins.append(np.min(scores))
    fig = plt.figure()
    plt.plot(np.arange(0, generation), means, label='Mean')
    plt.plot(np.arange(0, generation), medians, label='Median')
    plt.plot(np.arange(0, generation), mins, label='Min')
    plt.plot(np.arange(0, generation), maxs, label='Max')
    plt.legend(loc='upper left')
    plt.xlabel('Generation [-]')
    plt.ylabel('Score [-]')
    fig.savefig('progression/___progress_' + str(generation) + '.png')

    # plt.show()
    fig = plt.figure()
    plt.hist(histograms[-1], bins=np.linspace(0, 100, 40))
    plt.xlim(0, 1000)
    plt.ylim(0, population_size)
    plt.xlabel('Score [-]')
    plt.ylabel('Frequence [-]')
    plt.title('Generation ' + str(generation))
    fig.savefig('histograms/___histogram_' + str(generation) + '.png')

    print('-----2-------')
    print(collective_score, np.median(scores), np.mean(scores))

    for v in range(len(vehicles)):
        child = vehicles[v]
        child.reset()
        child.brain.crossover(p1s[v],p2s[v])
        child.brain.mutate()
        child.brain.create()
        vehicles[v] = child


