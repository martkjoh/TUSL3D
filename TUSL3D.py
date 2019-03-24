import numpy as np
from numpy import pi, exp, cos, sin
from math import factorial
from matplotlib import cm
from matplotlib import animation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def diff(f, x, dx = 10e-5):
    dx = np.ones_like(x) * dx
    return (-f(x + 2*dx) + 8*f(x + dx) - 8*f(x - dx) + f(x - 2*dx)) / (12*dx)
    return (f(x + dx / 2) - f(x - dx / 2)) / dx

def higerOrderDiff(f, x, n, dx = 10e-2):
    delx = np.ones_like(x) * dx
    if n == 0:
        return f(x)
    elif n == 1:
        return diff(f, x, dx=dx)
    else:
        lhs = higerOrderDiff(f, x + delx / 2, n - 1, dx = dx)
        rhs = higerOrderDiff(f, x - delx / 2, n - 1, dx = dx)
        return (lhs - rhs) / delx

def legendrePolynomials(x, l, m):
    g = lambda y : (y**2 - np.ones_like(y))**l
    h = lambda y : (-1)**m * np.sqrt((np.ones_like(y) - y**2))**m
    return 1 / (2**l * factorial(l)) * h(x) * higerOrderDiff(g, x, l + m)

def sphericalHarmonics(phi, theta, l, m):
    return legendrePolynomials(np.cos(theta), l, m) *np.exp( - m * phi * 1j)


theta = np.linspace(0, pi, 1000)
phi = np.linspace(0, 2 * pi, 1000)
phi, theta = np.meshgrid(phi, theta)

x = sin(phi) * cos(theta)
y = sin(phi) * sin(theta)
z = cos(phi)

n = 2
kws = {"projection" : "3d"}
fig, axs = plt.subplots(n, n, subplot_kw = kws)
surf = np.zeros_like(axs)

Writer = animation.writers['ffmpeg']
writer = Writer(fps = 15, metadata = dict(artist = 'Me'), bitrate = 1800)

for l in range(n):
    for m in range(n):
        if m <= l:
            f = abs((sphericalHarmonics(phi, theta, l, m)))**2
            f = (f) / (np.max(f) - np.min(f))

            surf[l][m] = axs[l][m].plot_surface(x, y, z, facecolors = cm.RdYlBu(f))

        axs[l][m].set_axis_off()

def update(angle):
    for l in range(n):
        for m in range(n):
            axs[l][m].view_init(10, angle * 5)
    
    return surf

anim = animation.FuncAnimation(fig, update, interval = 200, frames = 2)
anim.save("TUSL3D.mp4", extra_args=['-vcodec', 'libx264'])
