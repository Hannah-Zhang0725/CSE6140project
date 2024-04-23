#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Team member: Zihan Zhang, Yuehan Zhang, Leyao Huang, Yifeng Wang

"""

import argparse
import os
from approx_functions import *
from bnb_functions import *
from ls1_functions import *
from ls2_functions import *


# Main Functions
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve the Knapsack problem with various algorithms.")
    parser.add_argument('-inst', type=str, help='File path to the instance file')
    parser.add_argument('-alg', choices=['BnB', 'Approx', 'LS1', 'LS2'], help='Algorithm to use')
    parser.add_argument('-time', type=int, default=600, help='Cutoff time in seconds')
    parser.add_argument('-seed', type=int, help='Random seed')
    
    args = parser.parse_args()
    HOME_FOLDER = os.getcwd()

    # Check if "-inst" argument contains "large_scale" or "small_scale"
    if "large" in args.inst:
        output_sol_path = os.path.join(HOME_FOLDER, 'DATASET', 'submission', 'large_scale_sol')
        output_trace_path = os.path.join(HOME_FOLDER, 'DATASET', 'submission', 'large_scale_trace')
        DATASET_PATH = os.path.join(HOME_FOLDER, 'DATASET', 'large_scale')
        OPT_PATH = os.path.join(HOME_FOLDER, 'DATASET', 'large_scale_solution')

    elif "small" in args.inst:
        output_sol_path = os.path.join(HOME_FOLDER, 'DATASET', 'submission', 'small_scale_sol')
        output_trace_path = os.path.join(HOME_FOLDER, 'DATASET', 'submission', 'small_scale_trace')
        DATASET_PATH = os.path.join(HOME_FOLDER, 'DATASET', 'small_scale')
        OPT_PATH = os.path.join(HOME_FOLDER, 'DATASET', 'small_scale_solution')

    else:
        output_sol_path = os.path.join(HOME_FOLDER, 'DATASET', 'submission', 'test_sol')
        output_trace_path = os.path.join(HOME_FOLDER, 'DATASET', 'submission', 'test_trace')
        DATASET_PATH = os.path.join(HOME_FOLDER, 'DATASET', 'test')
        OPT_PATH = os.path.join(HOME_FOLDER, 'DATASET', 'test_solution')
    
    os.makedirs(output_sol_path, exist_ok=True)
    os.makedirs(output_trace_path, exist_ok=True)
    
    # Execute the specified algorithm
    if args.alg == 'Approx':
        
        results = {}
        
        filename = os.path.basename(args.inst)
        input_file = os.path.join(DATASET_PATH, filename)
        n, capacity, items = read_knapsack_data(input_file)
        
        specific_instances = ['large_7', 'large_14', 'large_21']
        
        if filename in specific_instances:
            value, selected_items, execution_time = knapsack_fptas_cutoff_update(0.1, items, capacity, args.time)
        else:
            value, selected_items, execution_time = knapsack_fptas_cutoff(0.1, items, capacity, args.time)
            
        opt_file = os.path.join(OPT_PATH, filename)
        OPT = read_OPT(opt_file)
        RelErr = abs(value - OPT) / OPT

        results[filename] = {
            'Time (s)': round(execution_time, 8),
            'Value': value,
            'RelErr': round(RelErr, 10)
        }

        # Format the quality of the best solution found as an integer
        quality_of_solution = str(int(value)) + "\n"

        # Format the selected items indices as comma-separated values
        selected_items_indices = ", ".join(map(str, selected_items)) + "\n"

        # Prepare the filename to save the results
        new_filename = f"{filename}_Approx_{int(args.time)}.sol"

        # Prepare the filename to save the results
        output_file_path = os.path.join(output_sol_path, new_filename)

        # Prepare the content to write
        lines = [quality_of_solution, selected_items_indices]

        # Write to the file
        with open(output_file_path, 'w') as file:
            file.writelines(lines)

        print(f"{filename} done.")

        for filename, metrics in results.items():
            print(f"{filename}: {metrics}")
        
    elif args.alg == 'BnB':
        results = {}

        filename = os.path.basename(args.inst)
        input_file = os.path.join(DATASET_PATH, filename)
        trace_name = f"{filename}_BnB_{int(args.time)}.trace"
        trace_file = os.path.join(output_trace_path, trace_name)

        # initial upper bound for large scale dataset
        if "large" in args.inst:
            approx_name = f"{filename}_Approx_{int(args.time)}.sol"
            approx_file = os.path.join(output_sol_path, approx_name)
            value, selected_items, execution_time = knapsack_bnb(input_file, trace_file, bound = True, approx_file = approx_file)

        # naive upper bound for smaller scale dataset
        else:
            value, selected_items, execution_time = knapsack_bnb(input_file, trace_file)

        opt_file = os.path.join(OPT_PATH, filename)
        err = Rel_Err(value, opt_file)

        results[filename] = {
            'Time (s)': execution_time,
            'Value': value,
            'RelErr': err
        }

        # Format the quality of the best solution found as an integer
        quality_of_solution = str(int(value))

        # Prepare the filename to save the results
        new_filename = f"{filename}_BnB_{int(args.time)}.sol"

        # Prepare the filename to save the results
        output_file_path = os.path.join(output_sol_path, new_filename)

        # Write to the file
        with open(output_file_path, 'w') as file:
            # Line 1: Write the solution value
            file.write(f"{quality_of_solution}\n")

            # Line 2: Write the list of item indices, comma-separated
            items_str = ', '.join(map(str, selected_items))
            file.write(items_str + '\n')

        print(f"{filename} done.")

        for filename, metrics in results.items():
            print(f"{filename}: {metrics}")

    elif args.alg == 'LS1':
        def write_solution_file(instance, method, cutoff, rand_seed, best_solution, best_value):
            filename = os.path.basename(instance)
            filename = os.path.join(output_sol_path, filename)
            if rand_seed is not None:
                filename += f"_{method}_{cutoff}_{rand_seed}"
            filename += ".sol"
            with open(filename, 'w') as file:
                file.write(str(best_value) + "\n")
                selected_indices = [str(i) for i, selected in enumerate(best_solution) if selected == 1]
                file.write(",".join(selected_indices) + "\n")


        # Function to write solution trace files
        def write_solution_trace_file(instance, method, cutoff, rand_seed, trace_data):
            filename = os.path.basename(instance)
            filename = os.path.join(output_trace_path, filename)
            if rand_seed is not None:
                filename += f"_{method}_{cutoff}_{rand_seed}"
            filename += ".trace"
            with open(filename, 'w') as file:
                for timestamp, quality in trace_data:
                    file.write(f"{timestamp},{quality}\n")


        filename = os.path.basename(args.inst)
        input_file_path = os.path.join(DATASET_PATH, filename)
        solution_file_path = os.path.join(OPT_PATH, filename)
        num_items, capacity, items = read_dataset(input_file_path)
        best_solution, best_value, trace_data = hill_climbing_knapsack(items, capacity, args.seed,
                                                                       max_iterations=1000000,
                                                                       cutoff_time=args.time)
        print('Best solution: {} with best value: {} in {} s'.format(best_solution, best_value, trace_data[-1][0]))

        # Write solution files
        write_solution_file(args.inst, args.alg, args.time, args.seed, best_solution, best_value)

        # Write solution trace files
        write_solution_trace_file(args.inst, args.alg, args.time, args.seed, trace_data)
        
    elif args.alg == 'LS2':
        filename = os.path.basename(args.inst)
        input_file_path = os.path.join(DATASET_PATH, filename)
        solution_file_path = os.path.join(OPT_PATH, filename)
        file_paths = [input_file_path, solution_file_path, output_sol_path, output_trace_path, filename]

        # GA parameters
        crossover_rate = 0.9
        mutation_rate = 0.8

        # Run GA
        run_instance(file_paths, args.time, args.seed, crossover_rate, mutation_rate)

    else:
        print('Wrong alg, should be one of [BnB, Approx, LS1, LS2]')

