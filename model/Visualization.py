import time

__author__ = 'dmitru'

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import collections  as mc
import networkx as nx
import numpy as np
import scipy
from Utils import FormationsUtil, position_vector, get_cmap


class ModelAnimator(object):
    """An animated scatter plot using matplotlib.animations.FuncAnimation.
    Params: T - time to animate, dt - time step to show"""
    def __init__(self, model, dt=0.01, draw_each_kth_frame=10, field_size=10, trace_size=200, trace_step=2):
        self.model = model
        self.N = model.N
        self.dt = dt
        self.pause = False
        self.draw_each_kth_frame = draw_each_kth_frame
        self.field_size = field_size
        self.stream = self.data_stream()
        self.trace_size = trace_size
        self.trace_step = trace_step
        self.trace_positions = []

        # Setup the figure and axes...
        self.fig = plt.figure(figsize=(18, 16))
        self.fig.canvas.mpl_connect('button_press_event', self.onClick)
        self.fig.canvas.mpl_connect('key_press_event', self.onKeyPress)
        self.anim_ax = self.fig.add_subplot(121, aspect='equal')
        #self.anim_ax_2 = self.fig.add_subplot(223, aspect='equal')
        self.plot_ax = self.fig.add_subplot(122)
        self.old_edges_collection = None
        # Then setup FuncAnimation.
        self.model.simulation_start()
        self.set_up_called = False
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=0.03,
                                           init_func=self.setup_plot, blit=True, repeat=True)

    def setup_plot(self):
        if self.set_up_called:
            return ()

        self.set_up_called = True
        step, xy, rel_h, quality = next(self.stream)
        self.quality_y = []

        self.s = [50 for _ in range(self.N)]
        self.cmap = get_cmap(self.N)
        self.c = [self.cmap(i) for i in range(self.N)]
        self.step_text = self.anim_ax.text(0.02, 0.95, '', transform=self.anim_ax.transAxes)
        self.model_info_text = self.anim_ax.text(0.02, 0.92, '', transform=self.anim_ax.transAxes)
        self.quality_text = self.plot_ax.text(0.02, 0.85, '', transform=self.plot_ax.transAxes)
        self.anim_ax.get_xaxis().set_animated(True)
        self.anim_ax.get_yaxis().set_animated(True)
        self.anim_ax.get_xaxis().set_ticklabels([])
        self.anim_ax.get_yaxis().set_ticklabels([])
        self.scat = self.anim_ax.scatter(xy[0::4], xy[2::4], s=self.s, c=self.c, animated=True)
        self.edges = None
        self.update_called_one_time = False

        self.formation_quality, = self.plot_ax.plot(self.quality_y, c='b')
        self.update_draw_bounds(new_bounds=(-self.field_size, self.field_size, -self.field_size, self.field_size))
        self.anim_ax.grid()
        #self.anim_ax_2.grid()
        return tuple()
        return self.anim_ax.get_xaxis(), self.anim_ax_2.get_xaxis(), self.step_text, self.model_info_text, self.formation_quality, self.quality_text

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
                ys, rel_h = self.model.simulation_step(self.dt)
            cnt += 1
            yield cnt, ys, rel_h, self.model.compute_formation_quality(ys, self.dt)

    def update(self, i):
        step, data, rel_h, quality = next(self.stream)
        for c in self.anim_ax.collections:
            self.anim_ax.collections.remove(c)

        if quality is not None:
            while len(self.quality_y) < len(quality):
                self.quality_y.append([])
            for i in range(len(quality)):
                self.quality_y[i].append(quality[i])

        self.update_draw_bounds(data)
        self.scat = self.anim_ax.scatter(data[::4], data[2::4], s=self.s, c=self.c, animated=True)
        self.scat2 = self.anim_ax.scatter(rel_h[::2], rel_h[1::2], s=self.s * 3, c=self.c, animated=True, alpha=0.3, marker='s')

        #self.scat_h = self.anim_ax_2.scatter(self.model.h[::4], self.model.h[2::4], s=self.s, c='blue', animated=True)
        #self.scat_h_2 = self.anim_ax_2.scatter(self.model.rel_h[::2], self.model.rel_h[1::2], s=self.s, c='red', animated=True)
        #self.scat_h_3 = self.anim_ax_2.scatter(self.model.current_direction[0], self.model.current_direction[1], s=self.s, c='green', animated=True)

        segments = []
        for edge in self.model.CG.edges():
            segments.append([(data[4*edge[0]], data[4*edge[0] + 2]), (data[4*edge[1]], data[4*edge[1] + 2])])
        edges_collection = mc.LineCollection(segments, colors=[(1,0,0,0.3) for _ in range(len(segments))])
        if self.update_called_one_time:
            self.edges = self.anim_ax.add_collection(edges_collection)
        self.old_edges_collection = edges_collection

        if step % self.trace_step == 0:
            self.trace_positions.append(data)
            if len(self.trace_positions) > self.trace_size:
                self.trace_positions.pop(0)
        trace_segments = []
        trace_colors = []
        for i in range(self.model.N):
            for j in range(len(self.trace_positions) - 1):
                cur = self.trace_positions[j + 1]
                prev = self.trace_positions[j]
                trace_segments.append(((cur[4*i], cur[4*i+2]), (prev[4*i], prev[4*i+2])))
                color = list(self.c[i][:])
                color[-1] = max(0.4, float(j) / len(self.trace_positions))
                trace_colors.append(color)
        trace_segment_collection = mc.LineCollection(trace_segments, colors=trace_colors)
        if self.update_called_one_time:
            self.traces = self.anim_ax.add_collection(trace_segment_collection)

        t = list(zip(self.quality_y))
        plot_args = []
        for i in t:
            plot_args.append(i[0])
            plot_args.append('r')
        self.formation_quality = self.plot_ax.plot(*plot_args)
        if step % 4 == 0:
            self.model_info_text.set_text('k = %.3f' % (self.model.k))
            self.step_text.set_text('Step %d' % (step))
            self.quality_text.set_text(self.model.text_to_show)
        if not self.update_called_one_time:
            self.update_called_one_time = True
            return tuple()
        return (self.anim_ax.get_xaxis(),
                #self.anim_ax_2.get_xaxis(), self.scat_h, self.scat_h_2, self.scat_h_3,
                edges_collection,
                self.scat, self.scat2, self.step_text, self.model_info_text, trace_segment_collection,
                self.quality_text) + tuple(self.formation_quality)

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