import os
import matplotlib.pyplot as plt
import numpy as np

data_temp = []

def read_solution_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        optimal_value = float(lines[0])

    return optimal_value


def extract_data_from_trace_file(trace_file, time, q_star, optimal_quality):
    with open(trace_file, 'r') as file:
        for line in file:
            timestamp, quality = map(float, line.strip().split(','))
            if timestamp < time and (optimal_quality - quality)/quality <= q_star:
                data_temp.append((timestamp, (optimal_quality - quality)/quality))
                break

    return data_temp


instance = '3'
algorithm = 'LS2'
directory = os.getcwd() + '/DATASET/submission/large_scale_trace/'
optimal_quality = read_solution_file(os.getcwd() + '/DATASET/large_scale_solution/large_' + instance)
trace_files = [os.path.join(directory, file) for file in os.listdir(directory)
                        if file.startswith(f'large_{instance}_{algorithm}') and file.endswith("trace")]
solution_quality_thresholds = np.arange(0, 35) * 0.001
times = [0.02, 0.03, 0.04, 0.05, 0.1, 0.2]  # Large_1 & 3 of GA
fractions_of_successful_runs_sqd = {time: [] for time in times}

# Calculate fraction of successful runs*
for time in times:
    for q_star in solution_quality_thresholds:
        for file in trace_files:
            data_temp = extract_data_from_trace_file(file, time, q_star, optimal_quality)
        fraction_successful = len(data_temp) / len(trace_files)
        fractions_of_successful_runs_sqd[time].append((fraction_successful, q_star))
        data_temp = []

# Plot the SQD
plt.figure(figsize=(6, 6))
for time in times:
    x, y = [], []
    for item in fractions_of_successful_runs_sqd[time]:
        x.append(item[1]*100)
        y.append(item[0])
    plt.plot(x, y, label=f'{time}s')
plt.xlabel('Relative Solution Quality (%)')
plt.ylabel('P(solve)')
plt.title('SQDs for Various Run-Times')
plt.legend()
plt.grid(True)
plt.show()