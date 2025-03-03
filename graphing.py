import json
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple


def extract_data(folderpath: str, filenames: List[str] | None=None) -> List[List[List[float]]]:
    data = []

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


def get_min_max(data):
    fill_data = []

    for y in data:
        y_min_max = [[], []]

        n = len(y[0])
        for i in range(n):
            data_along_i = [data_set[i] for data_set in y]
            y_min_max[0].append(min(data_along_i))
            y_min_max[1].append(max(data_along_i))

        fill_data.append(y_min_max)

    return fill_data


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


def get_standard_dev(data, average=None):
    if average is None:
        average = get_averages(data)[0]

    n = min(len(elm) for elm in data)
    num_sets = len(data)
    std_dev = []

    for i in range(n):
        square_sum = 0
        for j in range(num_sets):
            square_sum += (average[i] - data[j][i])**2
        std_dev.append(np.sqrt(square_sum / num_sets))
    return std_dev

def get_standard_error(data, average=None):
    standard_deviation = get_standard_dev(data, average)
    standard_error = []

    n = min(len(elm) for elm in data)
    num_sets = len(data)

    for i in range(n):
        standard_error.append(standard_deviation[i] / np.sqrt(num_sets))
    return standard_error


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


def graph_reward_per_epoch(data: List[List[List[float]]], graphing_window: int=10, evaluate_timesteps: int=1,
                           name: str="", labels: None | List[str]=None, colours: None | List[str]=None,
                           x_label: str="", y_label: str="",
                           x_lim: None | List[int]=None, y_lim: None | List[int]=None, error_bars: bool=False) -> None:
    num_agents = len(data)
    num_samples = min([len(agent_data) for agent_data in data])
    n = min([len(agent_data[i]) for agent_data in data for i in range(num_samples)])

    adjusted_data = [[[] for _ in range(num_samples)] for _ in range(num_agents)]
    for i in range(n):
        all_average = sum([data[agent][sample][i] for agent in range(num_agents)
                           for sample in range(num_samples)])
        all_average /= (num_samples * num_agents)

        for sample in range(num_samples):
            sample_average = sum([data[agent][sample][i] for agent in range(num_agents)])
            sample_average /= num_agents

            for agent in range(num_agents):
                adjusted_data[agent][sample].append(data[agent][sample][i] + all_average - sample_average)

    averaged_data = list(map(get_averages, adjusted_data))
    windowed_data = list(map(lambda x: get_reward_per_timestep(x, graphing_window), averaged_data))
    len_windowed_data = int(np.floor(n / graphing_window))

    fill_data = None
    if error_bars:
        error_values = [get_standard_error(adjusted_data[i], averaged_data[i]) for i in range(num_agents)]
        windowed_error = list(map(lambda x: get_reward_per_timestep(x, graphing_window), error_values))
        fill_data = []
        for i in range(num_agents):
            bottom_st = [windowed_data[i][j] - windowed_error[i][j] for j in range(len_windowed_data)]
            top_st = [windowed_data[i][j] + windowed_error[i][j] for j in range(len_windowed_data)]
            fill_data.append([bottom_st, top_st])

    x = [i * graphing_window for i in range(1, len_windowed_data + 1)]
    x = [i * evaluate_timesteps for i in x]

    graph_multiple(windowed_data, x,
                   name=name, labels=labels, colours=colours, x_label=x_label, y_label=y_label,
                   xlim=x_lim, ylim=y_lim, fill_data=fill_data)
    return


def graph_reward_per_timestep(data, window=10, evaluate_window=1, name=None, labels=None, x_label=None, y_label=None,
                              xlim=None, ylim=None, error_bars=None):
    averaged_data = list(map(get_averages, data))
    reward_per_timestep_data = list(map(lambda x: get_reward_per_timestep(x, window), averaged_data))
    n = min([len(values) for values in reward_per_timestep_data])
    num_sets = len(data)
    reward_per_timestep_data_trimmed = list(map(lambda x: x[:n], reward_per_timestep_data))

    fill_data = None
    if error_bars is not None:
        if error_bars == 'std':
            error_values = [get_standard_dev(data[i], averaged_data[i]) for i in range(num_sets)]
        elif error_bars == 'st_error':
            error_values = [get_standard_error(data[i], averaged_data[i]) for i in range(num_sets)]
        windowed_error = list(map(lambda x: get_reward_per_timestep(x, window), error_values))
        windowed_error_trimmed = list(map(lambda x: x[:n], windowed_error))
        fill_data = []
        for i in range(num_sets):
            bottom_std = [reward_per_timestep_data_trimmed[i][j] - windowed_error_trimmed[i][j] for j in range(n)]
            top_std = [reward_per_timestep_data_trimmed[i][j] + windowed_error_trimmed[i][j] for j in range(n)]
            fill_data.append([bottom_std, top_std])

    x = [i * window for i in range(1, n + 1)]
    x = [i * evaluate_window for i in x]

    graph_multiple(reward_per_timestep_data_trimmed, x,
                   name=name, labels=labels, x_label=x_label, y_label=y_label,
                   xlim=xlim, ylim=ylim, fill_data=fill_data)
    return reward_per_timestep_data_trimmed, x, fill_data


def graph_multiple(data, x=None, name=None, labels=None, x_label=None, y_label=None, xlim=None, ylim=None,
                   fill_data=None, colours: None | List[str]=None):
    plt.style.use('ggplot')

    data_len = len(data)

    if x is None:
        x = list(range(len(data[0])))

    fig, ax = plt.subplots()

    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)

    for i in range(data_len):
        y = data[i]

        colour = None
        if colours is not None:
            colour = colours[i]
        label = None
        if labels is not None:
            label = labels[i]

        ax.plot(x, y, label=label, color=colour)
        if fill_data is not None:
            ax.fill_between(x, fill_data[i][0], fill_data[i][1], color=colour, alpha=0.3)

    if labels is not None:
        ax.legend()

    if name is not None:
        ax.set_title(name)
        plt.savefig(name + '.png')

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    plt.show()
    return

def graph_multiple_stacked_barchart(data: List[Dict[str, np.ndarray]],
                                    labels: Tuple[str, ...],
                                    axes_labels: List[str],
                                    width: float=0.5,
                                    x_label: None|str=None, y_label: None|str=None, y_lim: None|List[int]=None,
                                    legend_axes: None|int=None, legend_location: str="upper right",
                                    name: None|str=None,
                                    colours: None|List[str]=None):
    plt.style.use('ggplot')
    num_plots = len(data)
    num_labels = len(labels)

    fig, axes = plt.subplots(nrows=1, ncols=num_plots, sharex=True, sharey=True)

    if x_label is not None:
        fig.supxlabel(x_label)
    if y_label is not None:
        fig.supylabel(y_label)
    for ax_position in range(num_plots):
        plot_data = data[ax_position]
        bottom = np.zeros(num_labels)
        j = 0
        for label, values in plot_data.items():
            colour = None
            if colours is not None:
                colour = colours[j]
            _ = axes[ax_position].bar(labels, values, width, label=label, bottom=bottom, color=colour)
            bottom += values
            j += 1
        axes[ax_position].set_title(axes_labels[ax_position])
        plt.setp(axes[ax_position].get_xticklabels(), rotation=90, ha='right')

        if y_lim is not None:
            axes[ax_position].set_ylim(y_lim)

    if legend_axes is not None:
        axes[legend_axes].legend(loc=legend_location)

    if name is not None:
        fig.suptitle(name)

    plt.tight_layout()
    plt.show()
    return

def graph_stacked_barchart(data: Dict[str, np.ndarray], labels: Tuple[str, ...],
                           threshold: None|float=None, threshold_key: None|str=None,
                           width: float=0.5,
                           x_label: None|str=None, y_label: None|str=None, y_lim: None|List[int]=None,
                           legend_location: None|str=None,
                           name: None|str=None,
                           colours: None|List[str]=None):
    plt.style.use('ggplot')

    fig, ax = plt.subplots()
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)

    num_labels = len(labels)
    bottom = np.zeros(num_labels)
    i = 0
    for label, values in data.items():
        colour = None
        if colours is not None:
            colour = colours[i]
        _ = ax.bar(labels, values, width, label=label, bottom=bottom, color=colour)
        bottom += values
        i += 1

    if threshold is not None:
        threshold_line = np.zeros(num_labels)
        threshold_line.fill(threshold)

        label = None
        if threshold_key is not None:
            label = threshold_key
        ax.plot([threshold, threshold], "k--", label=label)

    if y_lim is not None:
        ax.set_ylim(y_lim)

    if legend_location is not None:
        ax.legend(loc=legend_location)

    if name is not None:
        ax.set_title(name)

    plt.tight_layout()
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
