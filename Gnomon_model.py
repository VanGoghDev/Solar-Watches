import model as model
import numpy as np
import math
import DormanPrins_lab2 as dp


def getsize(a):
    size = 0
    for k in range(len(a)):
        size += 1
    return size


# star time
def julian_date(dd, mm, yy, hour, minute, second):  # day, month, year, hour, minute and second of current date
    a = (14 - mm) // 12
    m = mm + 12 * a - 3
    y = yy + 4800 - a
    jdn = dd + ((153 * m + 2) // 5) + 365 * y + (y // 4) - (y // 100) + (y // 400) - 32045
    jd = jdn + (hour - 12) / 24 + minute / 1440 + second / 86400
    return jd


# A star time is actually a time angle of the dot of vernal equinox (точка весеннего равноденствия)
# In order to use this function correctly it would be nice to call it after the declaration of variable
# equals to julian_date(...)

# !!!!!!!!!!!!!!!!!!!!!!!JD SHOULD BE ONLY THE RESULT OF JULIAN_DATE!!!!!!!!!!!!!!!!!!!!!!!
def siderial_time_sg0(jd):  # where jd is julian date for chosen date
    j2000 = 2451544.5  # julian date for 0:00 of 01.01.2000
    d = jd - j2000  # number of days since 01.01.2000
    tt = d / 36525  # number of centuries since 01.01.2000
    s = 24110.54841 + 8640184.812866 * tt + 0.093104 * math.pow(tt, 2) - 6.2 * math.pow(10, -6) * math.pow(tt, 3)
    # in order to succeed in this tough game we need to count an angle witch is corresponds to period 0 - 2pi
    # in relation to the direction of vernal equinox in radians, we should use the equation:
    s_rad = 2 * math.pi / 86400 * (s % 86400)
    return s_rad


# noinspection PyCompatibility
class MyModel(model.Model):

    # initializing starting conditions in constructor
    # we have a system of 6 equations for Vx, Vy, Vz and Vx/dx, Vy/dy, Vz/dz
    def __init__(self, t0, t, h, n):  # n =6
        super().__init__(t0, t, h, n)
        # below ephemeris of Earth is initialized
        self.x0[0] = -2.594245439450025e7
        self.x0[1] = 1.336562360362486e8
        self.x0[2] = 5.792094070218258e7
        self.x0[3] = -2.980665847911364e1 * 60
        self.x0[4] = -4.963978559743379 * 60
        self.x0[5] = -2.151503497654563 * 60

    # The core of Dorman-Prins method
    # Here we are solving the system of equations
    def get_right(self, a, t):
        mu_s = 132712.43994e6 * 3600  # heliocentric gravitational constant
        module_x = math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])  # length of a vector
        result1 = np.empty(getsize(self.x0))
        result1[0] = a[3]  # Vx = dx/dt
        result1[1] = a[4]  # Vy = dy/dt
        result1[2] = a[5]  # Vz = dz/dt
        result1[3] = -mu_s * a[0] / math.pow(module_x, 3)
        result1[4] = -mu_s * a[1] / math.pow(module_x, 3)
        result1[5] = -mu_s * a[2] / math.pow(module_x, 3)
        return result1
