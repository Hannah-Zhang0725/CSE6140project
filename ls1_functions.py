import random
import time

# Function to read the dataset file
def read_dataset(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        # Extract number of items and knapsack capacity from the first line
        num_items, capacity = map(int, lines[0].split())
        # Extract item values and weights from subsequent lines
        items = []
        for line in lines[1:]:
            value, weight = map(float, line.split())
            items.append((value, weight))
    return num_items, capacity, items


def calculate_value_to_weight_ratio(items):
    """ Calculate value-to-weight ratio for each item. """
    return [(index, value / weight, weight, value) for index, (value, weight) in enumerate(items)]


# Function to generate a random initial solution
def generate_initial_solution(items, capacity):
# Generate an initial feasible solution using a greedy approach based on value-to-weight ratio.
# Calculate value-to-weight ratio and sort items by this ratio in descending order
    sorted_items = sorted(calculate_value_to_weight_ratio(items), reverse=True, key=lambda x: x[1])

    solution = [0] * len(items)  # Start with no items selected
    total_weight = 0

    for index, ratio, weight, value in sorted_items:

        if total_weight + weight <= capacity:
            solution[index] = 1
            total_weight += weight

    return solution


# Function to evaluate the total value of a solution
def evaluate_solution(items, solution):
    total_value = sum(item[0] * selected for item, selected in zip(items, solution))
    return total_value


# Function to check if a solution is feasible (within capacity)
def is_feasible(items, solution, capacity):
    total_weight = sum(item[1] * selected for item, selected in zip(items, solution))
    return total_weight <= capacity


def calculate_value(items, solution):
    return sum(item[0] * sol for item, sol in zip(items, solution))


def calculate_weight(items, solution):
    return sum(item[1] * sol for item, sol in zip(items, solution))


def calculate_total_weight(items, solution):
    """Helper function to calculate total weight of the current solution."""
    total_weight = sum(item[1] * sol for item, sol in zip(items, solution))
    return total_weight


def greedy_repair(items, solution, capacity):
    """Repair the solution greedily to ensure it remains feasible."""
    # Sort items inside the knapsack by value-to-weight ratio
    item_indices_in_knapsack = [i for i, sol in enumerate(solution) if sol == 1]
    sorted_items_by_ratio = sorted(item_indices_in_knapsack, key=lambda i: items[i][0] / items[i][1])

    # Remove items with the smallest ratio until the solution is feasible
    while calculate_total_weight(items, solution) > capacity and sorted_items_by_ratio:
        item_to_remove = sorted_items_by_ratio.pop(0)  # Get the item with the smallest ratio
        solution[item_to_remove] = 0  # Remove this item from the knapsack

    return solution


# Function to make a small adjustment to a solution
def perturb_solution(items, solution, capacity, num_changes):
    for _ in range(num_changes):
        index_to_perturb = random.randint(0, len(solution) - 1)

        solution[index_to_perturb] = 1 - solution[index_to_perturb]

    if calculate_total_weight(items, solution) > capacity:
        solution = greedy_repair(items, solution, capacity)

    return solution

    # return feasible_solution


# Function to perform hill climbing
def hill_climbing_knapsack(items, capacity, rand_seed, max_iterations, cutoff_time):
    if rand_seed is not None:
        random.seed(rand_seed)
    # sorted_items = sorted(calculate_value_to_weight_ratio(items), reverse=True, key=lambda x: x[0])
    start_time = time.time()
    current_solution = generate_initial_solution(items, capacity)
    current_value = evaluate_solution(items, current_solution)
    # print(current_value)
    best_solution = [i for i in current_solution]
    best_value = current_value
    trace_data = [(0, best_value)]  # Initial timestamp and quality
    for _ in range(max_iterations):

        if time.time() - start_time > cutoff_time:

            break  # Stop if cutoff time is reached
        temp = [i for i in current_solution]
        new_solution = perturb_solution(items, temp, capacity, 50)

        new_value = evaluate_solution(items, new_solution)

        if new_value > current_value and is_feasible(items, new_solution, capacity):
            current_solution = [i for i in new_solution]
            current_value = new_value

            if current_value > best_value:
                best_solution = [i for i in current_solution]

                best_value = current_value
                current_timestamp = time.time() - start_time
                trace_data.append((current_timestamp, best_value))

    return best_solution, best_value, trace_data

