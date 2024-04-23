import time
import os

def read_knapsack_data(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()
        n, capacity = map(int, lines[0].split())  # Number of items and knapsack capacity
        items = [tuple(map(float, line.split())) for line in lines[1:]]  # (value, weight) pairs
    return n, capacity, items


def read_solution_and_items(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()  # Read all lines from the file

        # Parse the best solution value from the first line
        best_value = int(lines[0].strip())  # Convert to integer after stripping any whitespace

        # Parse the item indices from the second line
        items = list(map(int, lines[1].strip().split(', ')))  # Split the line by commas and convert each to an integer

    return best_value, items


def calculate_bound(value, weight, capacity, items, level):
    if weight > capacity:
        return 0
    else:
        bound_value = value
        remaining_capacity = capacity - weight
        for i in range(level, len(items)):
            if items[i][1] <= remaining_capacity:
                bound_value += items[i][0]
                remaining_capacity -= items[i][1]
            else:
                bound_value += items[i][0] * (remaining_capacity / items[i][1])
                break
        return bound_value


def round_to_significant_digits(num, digits=2):
    if num == 0:
        return 0
    if num < 1:
        return float(f"{num:.{digits}g}")
    else:
        return round(num, digits)


def Rel_Err(solution, filepath):
  with open(filepath, 'r') as file:
    # Read the first line from the file
    line = file.readline()
    # Convert the line to an integer
    optimal_value = float(line.strip())
  err = abs(optimal_value - solution)/optimal_value
  round_err = round_to_significant_digits(err, 2)
  return round_err


def knapsack_bnb(filepath, trace_filepath, cutoff=600, bound=False, approx_file=None):
    n, capacity, items = read_knapsack_data(filepath)

    values = [item[0] for item in items]
    weights = [item[1] for item in items]

    # Calculate value-to-weight ratio for sorting
    ratios = [v / w for v, w in zip(values, weights)]

    # Get sorted indices based on ratios
    sorted_indices = sorted(range(n), key=lambda i: ratios[i], reverse=True)
    items = [items[i] for i in sorted_indices]

    # Initialize
    F = [(0, 0, 0, [])]  # Level, Value, Weight, Path
    B = (0, [])  # Best value and items picked

    if bound == True:
        B = read_solution_and_items(approx_file)

    start_time = time.time()
    # Open trace file for logging
    with open(trace_filepath, 'w') as trace_file:
        while F:
            current_time = time.time()

            if (current_time - start_time) >= cutoff:
                print("Time exceeds the limit.")
                break
            level, value, weight, path = F.pop(0)

            # Check if current node is better than B
            if value > B[0] and weight <= capacity:
                B = (value, path)
                current_timestamp = time.time() - start_time
                round_time = round_to_significant_digits(current_timestamp, 2)
                trace_file.write(f"{round_time}, {B[0]}\n")

            if level < n:
                if bound == True:
                    include_bound = calculate_bound(value + items[level][0], weight + items[level][1], capacity, items,
                                                    level + 1)
                    exclude_bound = calculate_bound(value, weight, capacity, items, level + 1)

                    if include_bound > B[0]:
                        F.append((level + 1, value + items[level][0], weight + items[level][1], path + [level]))
                    if exclude_bound > B[0]:
                        F.append((level + 1, value, weight, path))
                else:
                    # Include the next item
                    if weight + items[level][1] <= capacity:
                        F.append((level + 1, value + items[level][0], weight + items[level][1], path + [level]))

                    # Exclude the next item
                    F.append((level + 1, value, weight, path))

    runtime = time.time() - start_time
    round_time = round_to_significant_digits(runtime)
    # Reconstruct solution
    solution_items = [i for i in B[1]]
    return B[0], solution_items, round_time

