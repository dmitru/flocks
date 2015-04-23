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
    def __init__(self, model, dt=0.01, draw_each_kth_frame=10, field_size=10):
        self.model = model
        self.N = model.N
        self.dt = dt
        self.pause = False
        self.draw_each_kth_frame = draw_each_kth_frame
        self.field_size = field_size
        self.stream = self.data_stream()

        # Setup the figure and axes...
        self.fig = plt.figure(figsize=(18, 16))
        self.fig.canvas.mpl_connect('button_press_event', self.onClick)
        self.fig.canvas.mpl_connect('key_press_event', self.onKeyPress)
        self.anim_ax = self.fig.add_subplot(121, aspect='equal')
        self.plot_ax = self.fig.add_subplot(122)
        # Then setup FuncAnimation.
        self.model.simulation_start()
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=0.03,
                                           init_func=self.setup_plot, blit=True, repeat=True)

    def setup_plot(self):
        step, xy, quality = next(self.stream)
        self.quality_y = []

        self.s = [50 for _ in range(self.N)]
        self.c = [i for i in range(self.N)]
        self.step_text = self.anim_ax.text(0.02, 0.95, '', transform=self.anim_ax.transAxes)
        self.model_info_text = self.anim_ax.text(0.02, 0.92, '', transform=self.anim_ax.transAxes)
        self.quality_text = self.plot_ax.text(0.02, 0.95, '', transform=self.plot_ax.transAxes)
        self.scat = self.anim_ax.scatter(xy[0::4], xy[2::4], s=self.s, c=self.c, animated=True)
        self.anim_ax.get_xaxis().set_animated(True)
        self.anim_ax.get_yaxis().set_animated(True)
        self.anim_ax.get_xaxis().set_ticklabels([])
        self.anim_ax.get_yaxis().set_ticklabels([])

        self.formation_quality, = self.plot_ax.plot(self.quality_y, c='b')
        self.update_draw_bounds(new_bounds=(-self.field_size, self.field_size, -self.field_size, self.field_size))
        self.anim_ax.grid()
        return self.anim_ax.get_xaxis(), self.scat, self.step_text, self.model_info_text, self.formation_quality, self.quality_text

    def onClick(self, event):
        self.pause ^= True

    def onKeyPress(self, event):
        if event.key == 'left':
            self.model.set_k(self.model.k + 0.2)
        elif event.key == 'right':
            self.model.set_k(self.model.k - 0.2)

    def data_stream(self):
        cnt = 0
        while True:
            for _ in range(self.draw_each_kth_frame):
                ys = self.model.simulation_step(self.dt)
            cnt += 1
            yield cnt, ys, self.model.compute_formation_quality(ys, self.dt)

    def update(self, i):
        step, data, quality = next(self.stream)
        if quality is not None:
            while len(self.quality_y) < len(quality):
                self.quality_y.append([])
            for i in range(len(quality)):
                self.quality_y[i].append(quality[i])
        self.update_draw_bounds(data)
        self.scat = self.anim_ax.scatter(data[::4], data[2::4], s=self.s, c=self.c, animated=True)
        t = list(zip(self.quality_y))
        plot_args = []
        for i in t:
            plot_args.append(i[0])
            plot_args.append('r')
        self.formation_quality = self.plot_ax.plot(*plot_args)
        if step % 4 == 0:
            #self.model_info_text.set_text('k = %.3f' % self.model.k)
            self.model_info_text.set_text('k = %.3f' % self.model.k)
            self.step_text.set_text('Step %d\n%s' % (step, self.model.text_to_show))
            self.quality_text.set_text('%f' % (quality[0]))
        return (self.anim_ax.get_xaxis(), self.scat, self.step_text, self.model_info_text, self.quality_text) + tuple(self.formation_quality)

    def update_draw_bounds(self, data=None, new_bounds=None):
        assert data is not None or new_bounds is not None
        if not new_bounds:
            xs = data[0::4]
            ys = data[2::4]
            max_x = max(xs)
            min_x = min(xs)
            max_y = max(ys)
            min_y = min(ys)

            old_bounds = list(self.anim_ax.axis())
            if min_x < old_bounds[0] + self.field_size / 8:
                delta_x = old_bounds[0] + self.field_size / 8 - min_x
                old_bounds[0] -= delta_x
                old_bounds[1] -= delta_x
            if max_x > old_bounds[1] - self.field_size / 8:
                delta_x = max_x - (old_bounds[1] - self.field_size / 8)
                old_bounds[0] += delta_x
                old_bounds[1] += delta_x
            if min_y < old_bounds[2] + self.field_size / 8:
                delta_y = old_bounds[2] + self.field_size / 8 - min_y
                old_bounds[2] -= delta_y
                old_bounds[3] -= delta_y
            if max_y > old_bounds[3] - self.field_size / 8:
                delta_y = max_y - (old_bounds[3] - self.field_size / 8)
                old_bounds[2] += delta_y
                old_bounds[3] += delta_y

            new_bounds = old_bounds
        self.anim_ax.axis(new_bounds)

    def show(self):
        plt.show()