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
if instance == '1':
    solution_quality_thresholds = [0, 0.005, 0.01, 0.023, 0.024, 0.03]  # Large_1 of GA
else:
    solution_quality_thresholds = [0, 0.03, 0.08, 0.1]  # Large_3 of GA
times = np.arange(0, 41) * 0.01
fractions_of_successful_runs_qrtd = {q_star: [] for q_star in solution_quality_thresholds}

for q_star in solution_quality_thresholds:
    # Calculate fraction of successful runs for each q*
    for time in times:
        for file in trace_files:
            data_temp = extract_data_from_trace_file(file, time, q_star, optimal_quality)
        fraction_successful = len(data_temp) / len(trace_files)
        fractions_of_successful_runs_qrtd[q_star].append((fraction_successful, time))
        data_temp = []

# Plot the QRTD
plt.figure(figsize = (6, 6))
for q_star in solution_quality_thresholds:
    x, y = [], []
    for item in fractions_of_successful_runs_qrtd[q_star]:
        x.append(item[1])
        y.append(item[0])
    plt.plot(x, y, label = f'q*={q_star*100}%')
plt.xlabel('Run-Time (CPU sec)')
plt.ylabel('P(solve)')
plt.title('QRTDs for Various Solution Qualities')
plt.legend()
plt.grid(True)
plt.show()

