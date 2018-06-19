import Gnomon_model as gm
import model
import numpy as np
import math
import DormanPrins_lab2 as dp
from abc import *
import matplotlib.pyplot as plt
import matplotlib as mpl


def getsize(a):
    size = 0
    for i in range(len(a)):
        size += 1
    return size


def degrees_from_minutes(grad, minutes):
    result = grad + minutes/60
    return result


def scalar_product(a, b):
    c = 0
    for i in range(len(a)):
        c += a[i] * b[i]
    return c


class Gnomon(gm.MyModel):

    earth_radius = 6371  # earth radius km
    omega = 7.292115e-5 * 60  # angular velocity of earth rotation
    h = 0.1  # 10 meters
    r0 = np.empty(3)
    control_var = np.empty(0)  # result of scalar product between rg and re

    def __init__(self, t0, t, h, n, phi, lamda, day):  # phi: latitude   lambda: longitude
        super().__init__(t0, t, h, n)
        rows = self.result.shape[0]
        # takes the result of previous integration
        for i in range(n):
            self.x0[i] = self.result[rows-1][i+1]
        # setting the result array to zeros
        Gnomon.result = np.empty((0, 0))
        self.phi = math.radians(phi)  # latitude
        self.lamda = math.radians(lamda)  # longitude
        self.day1 = day  # current day (result of julian date function)
        self.sg0 = gm.siderial_time_sg0(day)
        self.rg = np.empty(3)
        self.re = np.zeros(3)
        self.re_z = np.zeros(3)
        self.rsh = np.zeros(3)
        self.rsht = np.zeros(3)

    def siderial_time(self, t):
        sg = self.sg0 + Gnomon.omega * t + self.lamda
        return sg

    def add_result(self, a, t):
        rows = Gnomon.result.shape[0]
        Gnomon.result = np.resize(Gnomon.result, (rows + 1, getsize(self.x0) + 1))
        Gnomon.result[rows][0] = t

        for i in range(1, getsize(self.x0) + 1):
            Gnomon.result[rows][i] = a[i - 1]

        self.count_re(a)
        self.count_rg(t)
        self.check(t)
        self.count_rsh()
        self.count_rsht(t)

    # The vector rg is a vector from center of earth to point on surface
    def count_rg(self, t):
        # radius (x, y, z)
        Gnomon.r0[0] = (Gnomon.earth_radius + Gnomon.h) * math.cos(math.radians(self.phi)) * math.cos(self.siderial_time(t))
        Gnomon.r0[1] = (Gnomon.earth_radius + Gnomon.h) * math.cos(math.radians(self.phi)) * math.sin(self.siderial_time(t))
        Gnomon.r0[2] = (Gnomon.earth_radius + Gnomon.h) * math.sin(self.phi)
        module_r = math.sqrt(Gnomon.r0[0]*Gnomon.r0[0] + Gnomon.r0[1]*Gnomon.r0[1] + Gnomon.r0[2]*Gnomon.r0[2])
        self.rg = np.empty(3)

        for i in range(3):
            self.rg[i] = Gnomon.r0[i] / module_r * self.h

    def count_re(self, a):
        self.re = np.array([[a[0]], [a[1]], [a[2]]])
        module_re = math.sqrt(self.re[0] * self.re[0] + self.re[1] * self.re[1] + self.re[2] * self.re[2])

        for i in range(3):
            self.re[i] = self.re[i] / module_re

        c = scalar_product(self.re, self.rg)
        self.re_z = - 1 / c * self.re

    def check(self, t):
        c = scalar_product(self.re, self.rg)

        row = Gnomon.result.shape[0]
        Gnomon.control_var = np.resize(Gnomon.control_var, row + 1)
        Gnomon.control_var[t] = c[0]

    def count_rsh(self):
        for i in range(3):
            self.rsh[i] = self.rg[i] + self.re_z[i]

    def count_rsht(self, t):
        self.rsht = np.zeros(3)

        matrix_a = np.array([[-math.sin(self.phi) * math.cos(self.siderial_time(t)), -math.sin(self.phi) * math.sin(self.siderial_time(t)), math.cos(self.phi)],
                             [math.cos(self.phi) * math.cos(self.siderial_time(t)), math.cos(self.phi) * math.sin(self.siderial_time(t)), math.sin(self.phi)],
                             [-math.sin(self.siderial_time(t)), math.cos(self.siderial_time(t)), 0]])

        for i in range(3):
            for j in range(3):
                self.rsht[j] += matrix_a[i][j] * self.rsh[i]


def run():
    dorm__prins = dp.TDP()
    dorm__prins.geps = 1e-20

    day1 = gm.julian_date(26, 7, 2018, 0, 0, 0)
    day2 = gm.julian_date(1, 1, 2018, 0, 0, 0)
    day3 = (day1 - day2) * 1440

    model__one = gm.MyModel(0, day3, day3, 6)
    dorm__prins.run(model__one)

    latitude = degrees_from_minutes(55, 45)
    longitude = degrees_from_minutes(37, 37)

    t = 1440
    gnom = Gnomon(0, t, 1, 6, latitude, longitude, day1)
    dorm__prins.run(gnom)

    result_for_print = np.empty(t+2)
    for i in range(t+2):
        result_for_print[i] = i
    result_for_print_2 = Gnomon.control_var
    mpl.pyplot.scatter(result_for_print, result_for_print_2, marker='.', linewidths=1, label=r'$\Check x$')
    plt.grid(True)
    mpl.pyplot.show()


run()
