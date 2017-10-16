#!/usr/bin/env python3

import math
import random


class GenerateData():

    def __init__(self, size, parameter_count, output_count = 1):
        self.parameter_count = parameter_count
        self.points = size
        self.outputs = output_count
        self.data = []
        self.target_vector = []

    def stratified_sample(self, interval = 10):
        ppi = float(self.points / (interval * 2))
        step = -interval
        for point in range(self.points):
            self.data.append([])
            self.data[point].append(random.uniform(step, step + 1))
            for d in range(1, self.parameter_count):
                self.data[point].append(random.uniform(-interval, interval))
            if len(self.data) % ppi == 0:
                step += 1
            for o in range(self.outputs):
                self.target_vector.append([])
                self.target_vector[point].append(self.rosenbrock(self.data[point]))

    def simple_random_sample(self, lower = -2, upper = 2):
        for point in range(self.points):
            self.data.append([])
            for d in range(self.parameter_count):
                self.data[point].append(random.uniform(lower, upper))
            for o in range(self.outputs):
                self.target_vector.append([])
                self.target_vector[point].append(self.rosenbrock(self.data[point]))

    def rosenbrock(self, x):
        rosenbrock_sum = 0
        for i in range(len(x) - 1):
            rosenbrock_sum += (((1 - x[i])**2) + (100 * ((x[i + 1] - (x[i]**2))**2)))
        return rosenbrock_sum

    def get_target_vector(self):
        return self.target_vector

    def get_data(self):
        return self.data

