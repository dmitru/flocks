__author__ = 'dmitru'

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import numpy as np
import scipy
from Utils import FormationsUtil

class ModelAnimator(object):
    """An animated scatter plot using matplotlib.animations.FuncAnimation.
    Params: T - time to animate, dt - time step to show"""
    def __init__(self, model, T=10, dt=0.01, multiplier=10, field_to_show=(-10, 10, -10, 10)):
        self.data = model.compute(T, dt)
        self.model = model
        self.N = model.N
        self.pause = False
        self.steps = self.data.shape[0]
        self.multiplier = multiplier
        self.field_to_show = field_to_show
        print("steps: %d" % self.steps)
        self.stream = self.data_stream()

        # Setup the figure and axes...
        self.fig = plt.figure(figsize=(18, 16))
        self.fig.canvas.mpl_connect('button_press_event', self.onClick)
        self.anim_ax = self.fig.add_subplot(121, aspect='equal')
        self.plot_ax = self.fig.add_subplot(122)
        # Then setup FuncAnimation.
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=0.03, frames=self.steps,
                                           init_func=self.setup_plot, blit=True, repeat=True)

    def setup_plot(self):
        step, xy, quality = next(self.stream)
        self.quality_y = []

        self.s = [50 for _ in range(self.N)]
        self.c = [i for i in range(self.N)]
        self.step_text = self.anim_ax.text(0.02, 0.95, '', transform=self.anim_ax.transAxes)
        self.quality_text = self.plot_ax.text(0.02, 0.95, '', transform=self.plot_ax.transAxes)
        self.scat = self.anim_ax.scatter(xy[0::4], xy[2::4], s=self.s, c=self.c, animated=True)

        self.formation_quality, = self.plot_ax.plot(self.quality_y, c='b')
        self.anim_ax.axis(self.field_to_show)
        self.anim_ax.grid()
        return self.scat, self.step_text, self.formation_quality, self.quality_text

    def onClick(self, event):
        self.pause ^= True

    def data_stream(self):
        cnt = 0
        while True:
            yield cnt, self.data[cnt,:], FormationsUtil.compute_closeness(self.model.h, self.data[cnt,:])
            if not self.pause:
                if cnt + 1 * self.multiplier < self.steps:
                    cnt += 1 * self.multiplier

    def update(self, i):
        step, data, quality = next(self.stream)
        if not self.pause:
            self.quality_y.append(quality)
        self.formation_quality, = self.plot_ax.plot(self.quality_y, c='b')
        self.scat = self.anim_ax.scatter(data[::4], data[2::4], s=self.s, c=self.c, animated=True)
        if step % 8 == 0:
            self.step_text.set_text('step: %d / %d' % (step, self.steps))
            self.quality_text.set_text('%f' % (quality))
        return self.scat, self.step_text, self.formation_quality, self.quality_text

    def show(self):
        plt.show()