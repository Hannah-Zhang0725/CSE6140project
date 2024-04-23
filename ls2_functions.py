import numpy as np
import random
import time
import os

def read_input_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        n, W = map(int, lines[0].strip().split())

        values = np.zeros(n)
        weights = np.zeros(n)
        i = 0
        for line in lines[1:(n+1)]:
            value, weight = map(float, line.strip().split())
            values[i] = value
            weights[i] = weight
            i += 1
        
    return n, W, values, weights


def read_solution_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        optimal_value = float(lines[0])

    return optimal_value


def write_solution_file(instance, method, cutoff_time, random_seed, best_solution, best_value):
    filename = f"{instance}_{method}_{cutoff_time}_{random_seed}" + ".sol"
    with open(filename, 'w') as file:
        file.write(str(int(best_value)) + "\n")
        selected_indices = [str(i+1) for i, selected in enumerate(best_solution) if selected == 1]
        file.write(",".join(selected_indices) + "\n")


def write_time_file(instance, method, complete_times):
    filename = f"{instance}_{method}_time" + ".txt"
    with open(filename, 'w') as file:
        for i, complete_time in enumerate(complete_times):
            file.write(f"{i+1},{complete_time}\n")


def write_solution_trace_file(instance, method, cutoff_time, random_seed, trace_data):
    filename = f"{instance}_{method}_{cutoff_time}_{random_seed}" + ".trace"
    with open(filename, 'w') as file:
        for timestamp, quality in trace_data:
            file.write(f"{timestamp},{quality}\n")


def calculate_fitness(weights, values, population, threshold):
    fitness = np.zeros(population.shape[0])
    for i in range(population.shape[0]):
        total_value = np.dot(population[i, :], values)
        total_weight = np.dot(population[i, :], weights)
        if total_weight <= threshold:
            fitness[i] = total_value  # Fitness is total value
        else :
            fitness[i] = 0

    return fitness


def selection(fitness, num_parents, population):
    parents = np.zeros((num_parents, population.shape[1]))
    for i in range(num_parents):
        max_fitness_index = np.argmax(fitness)
        parents[i, :] = population[max_fitness_index, :]
        fitness[max_fitness_index] = -1  # Negative fitness to rule out this solution

    return parents


def crossover(parents, num_offsprings, crossover_rate):
    offsprings = np.zeros((num_offsprings, parents.shape[1]))

    for i in range(num_offsprings):
        prob = random.random()
        if prob <= crossover_rate:
            # Single point crossover: Interchange between parents
            parents_indices = random.sample(range(num_offsprings), 2)
            crossover_point = random.randint(1, parents.shape[1]-2)
            offsprings[i, 0:crossover_point] = parents[parents_indices[0], 0:crossover_point]
            offsprings[i, crossover_point:] = parents[parents_indices[1], crossover_point:]
        else:
            offsprings[i, :] = parents[i, :]

    return offsprings


def mutation(offsprings, mutation_rate):
    for i in range(offsprings.shape[0]):
        prob = random.random()
        if prob <= mutation_rate:
            # Swap mutation: Randomly drop one item and pick another
            one_indices = [j for j, selected in enumerate(offsprings[i, :]) if selected == 1]
            zero_indices = [j for j, selected in enumerate(offsprings[i, :]) if selected == 0]
            mutation_position_1 = random.sample(one_indices, 1)
            mutation_position_0 = random.sample(zero_indices, 1)
            offsprings[i, mutation_position_1[0]] = 0
            offsprings[i, mutation_position_0[0]] = 1

    return offsprings


def optimize(cutoff_time, weights, values, threshold, population, num_generations, crossover_rate, mutation_rate):
    num_parents = int(population.shape[0]/2)
    num_offsprings = population.shape[0] - num_parents
    num_generations_same_value = int(num_generations/5)
    trace_data = [(0, np.dot(population[1, :], values))]
    best_values = []

    start_time = time.time()
    for _ in range(num_generations):
        # Cut-off
        if time.time() - start_time > cutoff_time:
            break

        fitness = calculate_fitness(weights, values, population, threshold)

        # Record trace
        current_weights = np.dot(population, weights)
        feasible_indices = [i for i, w in enumerate(current_weights) if w <= threshold]
        feasible_population = population[feasible_indices, :]
        if feasible_indices != []:
            current_best_value = np.max(np.dot(feasible_population, values))
        else:
            current_best_value = 0
        if current_best_value > trace_data[-1][1]:
            current_time = time.time() - start_time
            trace_data.append((current_time, current_best_value))

        # Early stop
        best_values.append(current_best_value)
        if best_values.count(current_best_value) >= num_generations_same_value:
            break

        # Crossover and mutation
        parents = selection(fitness, num_parents, population)
        offsprings = crossover(parents, num_offsprings, crossover_rate)
        mutants = mutation(offsprings, mutation_rate)
        population[0:parents.shape[0], :] = parents
        population[parents.shape[0]:, :] = mutants
        
    fitness_last_gen = calculate_fitness(weights, values, population, threshold)
    max_fitness_index = np.argmax(fitness_last_gen)
    best_solution = population[max_fitness_index, :]

    # complete_time = time.time() - start_time
    # print('Completion time: {} s'.format(complete_time))

    return best_solution, trace_data


def get_initial_by_greedy(values, weights, threshold):
    value_per_weight = [(idx, values/weights) for idx, (values, weights) in enumerate(zip(values, weights))]
    value_per_weight = sorted(value_per_weight, key = lambda x: x[1], reverse = True)

    current_weight = 0
    current_value = 0
    initial_solution = np.zeros((values.size))
    for item in value_per_weight:
        if current_weight + weights[item[0]] <= threshold:
            initial_solution[item[0]] = 1
            current_weight += weights[item[0]]
            current_value += values[item[0]]

    return initial_solution


def run_instance(file_paths, cutoff_time, random_seed, crossover_rate, mutation_rate):
    random.seed(random_seed)
    
    # File paths
    filename = file_paths[4]
    input_file_path = file_paths[0]
    solution_file_path = file_paths[1]
    output_file_path = os.path.join(file_paths[2], filename)
    trace_file_path = os.path.join(file_paths[3], filename)

    # Parameters
    n, W, values, weights = read_input_file(input_file_path)
    optimal_value = read_solution_file(solution_file_path)
    solutions_per_pop = max(4*n, 100)
    num_generations = 100 * n

    # Get initial solution by greedy
    initial_solution = get_initial_by_greedy(values, weights, W)
    initial_population = np.tile(initial_solution, (solutions_per_pop, 1))
            
    # Run GA and write .sol/.trace files
    best_solution, trace_data = optimize(cutoff_time, weights, values, W, initial_population, num_generations, crossover_rate, mutation_rate)
    best_value = np.dot(best_solution, values)
    write_solution_file(output_file_path, "LS2", cutoff_time, random_seed, best_solution, best_value)
    write_solution_trace_file(trace_file_path, "LS2", cutoff_time, random_seed, trace_data)

    best_relative_error = np.round((optimal_value-best_value)/optimal_value, 6)
    print('The best solution of', filename, 'is {}'.format(best_solution),
          '\nwith best value of {}'.format(best_value),
          '(optimal value: {} and relative error: {})'.format(optimal_value, best_relative_error),
          'and total weight of {}'.format(np.dot(best_solution, weights)),
          '(weight threshold: {})'.format(W))


'''
# Run all the instances in small/large-scale
def run_instances(scale, repetition, cutoff_time, crossover_rate, mutation_rate):
    input_dir = os.getcwd() + '/DATASET/' + scale + '_scale/'
    solution_dir = os.getcwd() + '/DATASET/' + scale + '_scale_solution/'
    output_dir = os.getcwd() + '/' + scale + '_solutions/'
    trace_dir = os.getcwd() + '/' + scale + '_traces/'
    time_dir = os.getcwd() + '/' + scale + '_solutions/'

    for input_filename in os.listdir(input_dir):
        input_file_path = os.path.join(input_dir, input_filename)
        solution_file_path = os.path.join(solution_dir, input_filename)
        output_file_path = os.path.join(output_dir, input_filename)
        trace_file_path = os.path.join(trace_dir, input_filename)
        # time_file_path = os.path.join(time_dir, input_filename)

        optimal_value = read_solution_file(solution_file_path)
        n, W, values, weights = read_input_file(input_file_path)
        solutions_per_pop = max(4*n, 100)
        num_generations = 100 * n

        initial_solution = get_initial_by_greedy(values, weights, W)
        best_solutions = np.zeros((repetition, n+1))
        # complete_times = np.zeros(repetition)
        for i in range(repetition):
            random.seed(i)
            initial_population = np.tile(initial_solution, (solutions_per_pop, 1))
            
            # Run GA
            best_solution, trace_data = optimize(cutoff_time, weights, values, W, initial_population, num_generations, crossover_rate, mutation_rate)
            best_value = np.dot(best_solution, values)
            best_solutions[i, 0] = best_value
            best_solutions[i, 1:] = best_solution
            
            write_solution_file(output_file_path, "LS2", cutoff_time, i, best_solution, best_value)
            write_solution_trace_file(trace_file_path, "LS2", cutoff_time, i, trace_data)
            # complete_times[i] = complete_time

        # write_time_file(time_file_path, "LS2", complete_times)
        best_index = np.argmax(best_solutions[:, 0])
        best_value = best_solutions[best_index, 0]
        average_value = np.average(best_solutions[:, 0])
        best_relative_error = np.round((optimal_value-best_value)/optimal_value, 6)
        average_relative_error = np.round((optimal_value-average_value)/optimal_value, 6)
        print('The best solution of', input_filename, 'is {} (# {} run)'.format(best_solutions[best_index, 1:], best_index+1),
              '\nwith best value of {}'.format(best_value),
              '(optimal value: {} and relative error: {})'.format(optimal_value, best_relative_error),
              'and total weight of {}'.format(np.dot(best_solutions[best_index, 1:], weights)),
              '(weight threshold: {})'.format(W))
        print('The average value of', input_filename, 'is {}'.format(average_value),
              '(optimal value: {} and RelErr: {})'.format(optimal_value, average_relative_error))
        print('-----------------------------------------------------------------------------')
'''

