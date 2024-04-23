import os
import numpy as np
import matplotlib.pyplot as plt

def read_solution_trace_file(file_path):
    trace = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            running_time, value = map(float, line.strip().split(','))
            trace.append((running_time, value))
    
    return trace


def plot_boxplot(dir, algorithm, instance_indices, repetition, cutoff_time):
    running_times = np.zeros((repetition, len(instance_indices)))

    # Boxplot
    for instance in instance_indices:
        for i in range(repetition):
            filename = dir + f"large_{instance}_{algorithm}_{cutoff_time}_{i}.trace"
            trace = read_solution_trace_file(filename)
            running_times[i, instance_indices.index(instance)] = trace[-1][0]

    plt.boxplot(running_times)
    plt.ylabel('Running Time (s)')
    plt.xticks([i+1 for i in range(len(instance_indices))], labels = ['Large_' + i for i in instance_indices])
    plt.show()


trace_dir = os.getcwd() + '/DATASET/submission/large_scale_trace/'
instance_indices = ['1', '3']
repetition = 10
cutoff_time = 600
plot_boxplot(trace_dir, 'LS2', instance_indices, repetition, cutoff_time)

