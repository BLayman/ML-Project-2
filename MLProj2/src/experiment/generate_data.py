#!/usr/bin/env python3

import math
import random


class GenerateData():

    def __init__(self, size, parameter_count):
        self.parameter_count = parameter_count
        self.points = size
        self.data = []
        self.target_vector = []

    def calculate_parameters(self):
        points_per_axis = int(math.pow(self.points, (1 / self.parameter_count)))
        range_exp = round(points_per_axis / 2)
        if range_exp > 11:
            self.generate_data(10, points_per_axis)
        else:
            self.generate_data(range_exp, points_per_axis)

    def stratified_sample(self, interval = 10):
        ppi = int(self.points / (interval * 2))
        step = -interval
        for point in range(self.points):
            self.data.append([])
            self.data[point].append(random.uniform(step, step + 1))
            for d in range(1, self.parameter_count):
                self.data[point].append(random.uniform(-interval, interval))
            if len(self.data) % ppi == 0:
                step += 1
            self.target_vector.append(self.rosenbrock(self.data[point]))

    def simple_random_sample(self, lower = -2, upper = 2):
        for point in range(self.points):
            self.data.append([])
            for d in range(self.parameter_count):
                self.data[point].append(random.uniform(lower, upper))
            self.target_vector.append(self.rosenbrock(self.data[point]))

    def rosenbrock(self, x):
        rosenbrock_sum = 0
        for i in range(len(x) - 1):
            rosenbrock_sum += (((1 - x[i])**2) + (100 * ((x[i + 1] - (x[i]**2))**2)))
        return rosenbrock_sum

    def get_target_vector(self):
        return self.target_vector

    def get_data(self):
        return self.data

