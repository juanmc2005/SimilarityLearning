#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def visualize(feat, labels, epoch, bound=5):
    plt.ion()
    c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
         '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    plt.clf()
    for i in range(10):
        plt.plot(feat[labels == i, 0], feat[labels == i, 1], '.', c=c[i])
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc = 'upper right')
    plt.xlim(left=-bound,right=bound)
    plt.ylim(bottom=-bound,top=bound)
    plt.text(-bound+0.2,bound-0.7,"epoch=%d" % epoch)
    plt.savefig('./images/epoch=%d.jpg' % epoch)
    plt.draw()
    plt.pause(0.001)


def visualize3d(feat, labels, epoch, bound=5):
    plt.ion()
    c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
         '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    plt.clf()
    fig = plt.figure()
    ax = Axes3D(fig)
    for i in range(10):
        ax.scatter(feat[labels == i, 0], feat[labels == i, 1], feat[labels == i, 2], c=c[i], marker='.')
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc = 'upper right')
    #plt.text(-bound+0.2,bound-0.7,"epoch=%d" % epoch)
    plt.savefig('./images/epoch=%d.jpg' % epoch)
    plt.draw()
    plt.pause(0.001)
