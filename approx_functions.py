import math
import time

def read_knapsack_data(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()
        n, capacity = map(int, lines[0].split())  # Number of items and knapsack capacity
        items = [tuple(map(float, line.split())) for line in lines[1:]]  # (value, weight) pairs as floats
    return int(n), int(capacity), items

# Function to read OPT results
def read_OPT(filepath):
    with open(filepath, 'r') as file:
        # Read the first line and convert it to an integer
        OPT = float(file.readline().strip())
    return OPT

# The FPTAS for Knapsack: https://en.wikipedia.org/wiki/Knapsack_problem#Greedy_approximation_algorithm:~:text=%5B29%5D-,Fully%20polynomial%20time%20approximation%20scheme,-%5Bedit%5D
def knapsack_fptas_cutoff(epsilon, items, capacity, cutoff_time):
    start_time = time.time()  # Store the start time
    n = len(items)
    values = [item[0] for item in items]
    weights = [item[1] for item in items]

    # Scale factor for the weights to convert them to integers
    weight_scale = max(weights) / capacity
    scaled_weights = [math.ceil(w / weight_scale) for w in weights]

    P = max(values)
    value_scale = epsilon * P / n
    scaled_values = [math.floor(v / value_scale) for v in values]

    # Adjust capacity to be an integer suitable for the scaled weights
    integer_capacity = int(capacity / weight_scale)

    # Initialize the DP table
    dp = [[0 for _ in range(integer_capacity + 1)] for _ in range(n + 1)]

    cutoff_exceeded = False

    # Fill the DP table
    for i in range(1, n + 1):
        # Check if we've exceeded the cutoff time
        if time.time() - start_time > cutoff_time:
            cutoff_exceeded = True
            break
        for w in range(1, integer_capacity + 1):
            if scaled_weights[i - 1] <= w:
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - scaled_weights[i - 1]] + scaled_values[i - 1])
            else:
                dp[i][w] = dp[i - 1][w]

    # Find the items to include in the knapsack by tracing back through the table
    solution = []
    w = integer_capacity
    for i in range(n, 0, -1):
        if cutoff_exceeded and w == 0:
            break  # Break early if we've completed tracing back or cutoff time was exceeded
        if dp[i][w] != dp[i - 1][w]:
            solution.append(i - 1)
            w -= scaled_weights[i - 1]

    # Calculate the actual value from the original values
    actual_value = sum(values[i] for i in solution)
    running_time = time.time() - start_time  # Calculate the running time

    if cutoff_exceeded:
        print("Cutoff time exceeded")

    return actual_value, solution, running_time


def knapsack_fptas_cutoff_update(epsilon, items, capacity, cutoff_time):
    start_time = time.time()
    n = len(items)
    values = [item[0] for item in items]
    weights = [item[1] for item in items]

    # Scaling factor calculations
    P = max(values)
    value_scale = epsilon * P / n
    scaled_values = [math.floor(v / value_scale) for v in values]

    weight_scale = max(weights) / capacity
    scaled_weights = [math.ceil(w / weight_scale) for w in weights]

    integer_capacity = int(capacity / weight_scale)

    dp = [0 for _ in range(integer_capacity + 1)]
    cutoff_exceeded = False

    for i in range(n):
        if time.time() - start_time > cutoff_time:
            cutoff_exceeded = True
            break
        for w in range(integer_capacity, scaled_weights[i] - 1, -1):
            if scaled_weights[i] <= w:
                dp[w] = max(dp[w], dp[w - scaled_weights[i]] + scaled_values[i])

    # Traceback to find selected items
    solution = []
    w = integer_capacity
    for i in range(n - 1, -1, -1):
        if dp[w] == 0:
            break
        if i == 0 or dp[w] != dp[w - scaled_weights[i]] + scaled_values[i]:
            continue
        solution.append(i)
        w -= scaled_weights[i]

    # Reverse solution to match original item order
    solution.reverse()

    # Calculate the actual value from the original values
    actual_value = sum(values[i] for i in solution)
    running_time = time.time() - start_time

    if cutoff_exceeded:
        print("Cutoff time exceeded")

    return actual_value, solution, running_time

