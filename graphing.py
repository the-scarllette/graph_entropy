import json
import os

from environments.fourroom import FourRoom
from environments.hanoi import HanoiEnvironment
from environments.taxicab import TaxiCab, code_to_state

import colorsys
import matplotlib.colors as mc
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import networkx as nx


def adjust_lightness(color, amount=0.5):
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def extract_data(folderpath):
    data = []

    filenames = os.listdir(folderpath)
    for filename in filenames:
        f = open(folderpath + '/' + filename)
        file_data = json.load(f)
        f.close()
        data.append(list(file_data.values()))
    return data


def get_averages(data):
    num_sets = len(data)
    n = min(len(elm) for elm in data)
    to_graph = []

    for i in range(n):
        to_graph.append(sum([elm[i] for elm in data]) / num_sets)
    return to_graph


def get_rolling_average(data, k):
    rolling_average = []
    current_total = 0
    for i in range(k):
        current_total += data[i]

    rolling_average.append(current_total / k)

    for i in range(k, len(data)):
        current_total -= data[i - k]
        current_total += data[i]
        rolling_average.append(current_total / k)
    return rolling_average


def get_ranking_colours(rankings, start_colour='darkred'):
    num_rankings = len(rankings)
    colour_adjust = 1 + (1.5 / num_rankings)
    ranking_colours = [start_colour]
    for i in range(1, num_rankings):
        ranking_colours.append(adjust_lightness(ranking_colours[i - 1], colour_adjust))
    return ranking_colours


def get_percentage_successes(data, last_n):
    n = len(data[0])
    num_sets = len(data)

    total_runs = last_n * num_sets

    current_sum = 0
    for k in range(last_n):
        for elm in data:
            current_sum += int(elm[k])

    to_graph = [(current_sum / total_runs) * 100]
    for i in range(last_n, n):
        for elm in data:
            current_sum -= int(elm[i - last_n])
            current_sum += int(elm[i])
        to_graph.append((current_sum / total_runs) * 100)
    return to_graph


def get_reward_per_timestep(data, window):
    reward_per_timestep = []
    current_sum = 0
    for i in range(len(data)):
        if (i + 1) % window == 0:
            reward_per_timestep.append(current_sum / window)
            current_sum = 0
        current_sum += data[i]
    return reward_per_timestep


def graph(y, name=None, label=None):
    x = list(range(len(y)))

    if label is None:
        plt.plot(x, y)
    else:
        plt.plot(x, y, label=label)

    if name is not None:
        plt.title(name)

    if label is not None:
        plt.legend()
    plt.show()

    if name is not None:
        plt.savefig(name + '.png')
    return


def graph_average(data, name=None, label=None):
    to_graph = get_averages(data)

    graph(to_graph, name=name, label=label)
    return


def graph_barchart(labels, values):
    plt.bar(labels, values)
    plt.show()
    return


def graph_reward_per_timestep(data, window=10, name=None, labels=None, x_label=None, y_label=None,
                              xlim=None, ylim=None):
    averaged_data = list(map(get_averages, data))
    reward_per_timestep_data = list(map(lambda x: get_reward_per_timestep(x, window), averaged_data))
    n = min([len(values) for values in reward_per_timestep_data])
    reward_per_timestep_data_trimmed = list(map(lambda x: x[:n], reward_per_timestep_data))
    x = [i * window for i in range(n)]

    graph_multiple(reward_per_timestep_data_trimmed, x=x,
                   name=name, labels=labels, x_label=x_label, y_label=y_label,
                   xlim=xlim, ylim=ylim)
    return


def graph_multiple(data, x=None, name=None, labels=None, x_label=None, y_label=None, xlim=None, ylim=None):
    data_len = len(data)

    if x is None:
        x = list(range(len(data[0])))

    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)

    if labels is None:
        for y in data:
            plt.plot(x, y)
    else:
        for i in range(data_len):
            plt.plot(x, data[i], label=labels[i])

        plt.legend()

    if name is not None:
        plt.title(name)
        plt.savefig(name)

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    plt.show()
    return


def get_ongoing_average(data):
    ongoing_average = []
    n = len(data)
    current_total = 0.0

    for i in range(n):
        current_total += data[i]
        ongoing_average.append(current_total / (i + 1))

    return ongoing_average


def graph_percentage_successes(data, last_n, name=None, label=None):
    to_graph = get_percentage_successes(data, last_n)

    graph(to_graph, name=name, label=label)
    return
