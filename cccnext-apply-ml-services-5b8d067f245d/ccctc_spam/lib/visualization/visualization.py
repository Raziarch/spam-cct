################################################################
# Copyright (C) 2017 SeyVu, Inc; support@seyvu.com
#
# The file contents can not be copied and/or distributed
# without the express permission of SeyVu, Inc
################################################################

#########################################################################################################
#  Description: Collection of functions for various visualization needs
#########################################################################################################
import logging
import logging.config

# Python libraries
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.dates as mdates
import matplotlib.patches as patches
import math
import re


#########################################################################################################
# Global variables
__author__ = "DataCentric1"
__pass__ = 1
__fail__ = 0

#########################################################################################################
# Setup logging

log = logging.getLogger("info")

#########################################################################################################


def plot_series(s, kind='bar', title='', xlabel='', ylabel='', loc=''):
    plt.switch_backend('agg')
    ax = s.plot(kind=kind)
    ax.set_title(title, fontsize=16, color='darkblue')
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    plt.tight_layout()
    filename = os.path.join(loc, re.sub('[^A-Za-z0-9 ]', '', title).replace(' ', '_').lower() + '.png')
    plt.savefig(filename)


class Plots:
    def __init__(self):
        # Create a list of frequently used colors
        self.color_list = list(["blue", "green", "red", "cyan", "magenta", "yellow", "black", "white"])

    # Create plot from given x and multiple y datasets
    def basic_2d_plot(self, x, y=(), legends=(), title="", xaxis_label="", yaxis_label="", marker="o-",
                      filename="temp_2d_scatter_plot.png"):
        plt.figure()
        plt.title(title)
        plt.xlabel(xaxis_label)
        plt.ylabel(yaxis_label)

        plt.grid()

        index = 0
        color_index = 0

        if len(y):
            for yval in y:
                # Recycle if we reach end of unique colors that can be used
                # Hopefully we don't have so many unique "y" datasets that we need to do this
                if color_index > len(self.color_list):
                    color_index = 0

                color = colors.cnames[self.color_list[color_index]]

                plt.plot(x, yval, marker, color=color, label=legends[index])

                color_index += 1
                index += 1

            plt.legend(loc="best")
        else:
            color = colors.cnames[self.color_list[0]]

            plt.plot(x, marker=marker, color=color, label=legends[index])

            color_index += 1
            index += 1

            plt.legend(loc="best")

        plt.savefig(os.path.join(os.getcwd(), filename))

        return None

    def bar_plot(self, x=(), y=(), num_x_axis_groups=1, width=0.3, color=(), legend=(), legend_loc="", title="",
                 xlabel="", ylabel="", xaxis_isdate=False, xticklabels=(), set_plot_size_inches=(),
                 filename="temp_bar_plot.png"):
        """
        Wrapper for creating bar plots

        :param x: x-axis values (can be 1 list of values for all y-groups or
                                 provided as a tuple of lists for multiple y-groups)
        :param y: y-axis values (provided as a tuple of lists for multiple y-groups)
        :param num_x_axis_groups: Number of unique groups in x-axis
        :param width: bar width, will be auto-adjusted depending on total number of bars in plot
        :param color: color schemes (provided as a tuple for multiple y-groups, length must match y)
        :param legend: Provided as a tuple for multiple y-groups, length must match y
        :param legend_loc: Location of legend
        :param title: Title of plot
        :param xlabel: X-axis label
        :param ylabel: Y-axis label
        :param xticklabels: Labels for the x-axis groups
        :param filename: Filenmae for plot to be saved as
        :return: None
        """

        # adjust width / fontsize according to total number of bars needed
        if num_x_axis_groups * len(y) < 20:
            width = width
            fontsize = 12
        elif num_x_axis_groups * len(y) < 100:
            width = 0.3
            fontsize = 10
        elif num_x_axis_groups * len(y) < 200:
            width = 0.2
            fontsize = 8
        else:
            width = 0.2
            fontsize = 8

        fig = plt.figure()
        if set_plot_size_inches:
            fig.set_size_inches(set_plot_size_inches)
        ax = fig.add_subplot(111)

        # the bars
        for i in range(len(y)):

            if len(x) > 1:  # Each y-group has corresponding x-axis values
                if xaxis_isdate:
                    # Convert your sequence of datetimes into matplotlib's float-based date format.
                    x_values = mdates.date2num(x[i]) + width*i
                else:
                    x_values = x[i] + width*i
            else:  # Use same x-values for all y groups
                if xaxis_isdate:
                    # Convert your sequence of datetimes into matplotlib's float-based date format.
                    x_values = mdates.date2num(x[0]) + width*i
                else:
                    x_values = x[0] + width*i

            if color:
                if len(color) == len(y):
                    ax.bar(x_values, y[i], width=width, color=color[i])
                else:
                    raise ValueError("Not enough colors provided for all groups!")
            elif len(self.color_list) <= len(y):  # use pre-defined color scheme
                ax.bar(x_values, y[i], width=width, color=self.color_list[i])
            else:  # use default color scheme
                ax.bar(x_values, y[i], width=width)

        # axes and labels
        if not xaxis_isdate:
            ax.set_xticks(np.arange(num_x_axis_groups)+width)

            if not xticklabels:  # Group labels numerically
                ax.set_xticklabels(np.arange(num_x_axis_groups), fontsize=fontsize)
            else:  # use provided labels
                if num_x_axis_groups < 10:
                    xtick_fontsize = 10
                elif num_x_axis_groups < 30:
                    xtick_fontsize = 8
                elif num_x_axis_groups < 50:
                    xtick_fontsize = 8
                else:
                    xtick_fontsize = 8

                xticknames = ax.set_xticklabels(xticklabels)
                if type(xticklabels) != int:
                    plt.setp(xticknames, rotation=45, fontsize=xtick_fontsize)
        else:
            # Tell matplotlib to interpret the x-axis values as dates
            ax.xaxis_date()
            # Make space for and rotate the x-axis tick labels
            fig.autofmt_xdate()

        ax.set_title(title)
        ax.set_xlabel(xlabel=xlabel)
        ax.set_ylabel(ylabel=ylabel)

        if legend:
            # add legends
            ax.legend(legend, loc=legend_loc, fontsize=fontsize)

        ax.autoscale(tight=True)

        plt.savefig(os.path.join(os.getcwd(), filename))

        return None

    @staticmethod
    # Function to test out code or sub-parts of any visualization routine
    def scratchpad():
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ratio = 1.0 / 3.0
        count = math.ceil(math.sqrt(len(colors.cnames)))
        x_count = count * ratio
        y_count = count / ratio
        x = 0
        y = 0
        w = 1 / x_count
        h = 4 / y_count

        for c in colors.cnames:
            pos = (x / x_count, y / y_count)
            ax.add_patch(patches.Rectangle(pos, w, h, color=c))
            ax.annotate(c, xy=pos)
            if y >= y_count-1:
                x += 1
                y = 0
            else:
                y += 1

        plt.show()
#########################################################################################################

if __name__ == "__main__":
    vis = Plots()
    vis.scratchpad()
