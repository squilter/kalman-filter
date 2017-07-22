from math import sqrt

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

#from sebekf import ekf

'''
The three plots represent position, velocity and accelaration. Each plot has real, measured and estimated values.
'''

'''
np.diag([3]*3)

pure = np.linspace(-1, 1, 100)
noise = np.random.normal(0, 1, 100)
signal = pure + noise
'''


class SubplotAnimation(animation.TimedAnimation):
    def __init__(self):
        fig = plt.figure()
        ax1 = fig.add_subplot(3, 1, 1)
        ax2 = fig.add_subplot(3, 1, 2)
        ax3 = fig.add_subplot(3, 1, 3)

        self.t = np.linspace(0, 100, 400)
        self.pos = np.cos(2 * np.pi * self.t / 10.)
        self.vel = np.sin(2 * np.pi * self.t / 10.)
        self.acc = np.cos(2 * np.pi * self.t / 10.)

        self.pos_noise = np.random.normal(0, sqrt(POSITION_NOISE_VARIANCE), 100) # mean, std.dev, length
        self.vel_noise = np.random.normal(0, sqrt(VELOCITY_NOISE_VARIANCE), 100) # mean, std.dev, length
        self.acc_noise = np.random.normal(0, sqrt(ACCELERATION_NOISE_VARIANCE), 100) # mean, std.dev, length

        ax1.set_ylabel('position')
        self.line1 = Line2D([], [], color='black')
        self.line1a = Line2D([], [], color='red', linewidth=2)
        self.line1e = Line2D(
            [], [], color='red', marker='o', markeredgecolor='r')
        ax1.add_line(self.line1)
        ax1.add_line(self.line1a)
        ax1.add_line(self.line1e)
        ax1.set_xlim(0, 100)
        ax1.set_ylim(-2, 2)
        ax1.set_aspect('equal', 'datalim')

        ax2.set_ylabel('velocity')
        self.line2 = Line2D([], [], color='black')
        self.line2a = Line2D([], [], color='red', linewidth=2)
        self.line2e = Line2D(
            [], [], color='red', marker='o', markeredgecolor='r')
        ax2.add_line(self.line2)
        ax2.add_line(self.line2a)
        ax2.add_line(self.line2e)
        ax2.set_xlim(0, 100)
        ax2.set_ylim(-1.5, 1.5)

        ax3.set_ylabel('acceleration')
        self.line3 = Line2D([], [], color='black')
        self.line3a = Line2D([], [], color='red', linewidth=2)
        self.line3e = Line2D(
            [], [], color='red', marker='o', markeredgecolor='r')
        ax3.add_line(self.line3)
        ax3.add_line(self.line3a)
        ax3.add_line(self.line3e)
        ax3.set_xlim(0, 100)
        ax3.set_ylim(-1.5, 1.5)

        animation.TimedAnimation.__init__(self, fig, interval=50, blit=True)

    def _draw_frame(self, framedata):
        i = framedata
        head = i - 1
        head_slice = (self.t > self.t[i] - 1.0) & (self.t < self.t[i])

        self.line1.set_data(self.t[:i], self.vel[:i])
        self.line1a.set_data(self.t[head_slice], self.vel[head_slice])
        self.line1e.set_data(self.t[head], self.vel[head])

        self.line2.set_data(self.vel[:i], self.acc[:i])
        self.line2a.set_data(self.vel[head_slice], self.acc[head_slice])
        self.line2e.set_data(self.vel[head], self.acc[head])

        self.line3.set_data(self.pos[:i], self.acc[:i])
        self.line3a.set_data(self.pos[head_slice], self.acc[head_slice])
        self.line3e.set_data(self.pos[head], self.acc[head])

        self._drawn_artists = [self.line1, self.line1a, self.line1e,
                               self.line2, self.line2a, self.line2e,
                               self.line3, self.line3a, self.line3e]

    def new_frame_seq(self):
        return iter(range(self.t.size))

    def _init_draw(self):
        lines = [self.line1, self.line1a, self.line1e,
                 self.line2, self.line2a, self.line2e,
                 self.line3, self.line3a, self.line3e]
        for l in lines:
            l.set_data([], [])

SubplotAnimation()
plt.show()
