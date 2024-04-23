Instructions:

Please run “main.py” to test the codes. 

###########

For example, you can run the following commands in the terminal under the same path of "main.py".

"python main.py -inst small_5 -alg BnB -time 600 -seed 1" 
"python main.py -inst small_5 -alg Approx -time 600 -seed 1" 
"python main.py -inst small_5 -alg LS1 -time 600 -seed 1" 
"python main.py -inst small_5 -alg LS2 -time 600 -seed 1" 

###########

"bnb_functions.py" for Branch and Bound algorithm.
"approx_functions.py" for approximation algorithm.
"ls1_functions.py" for LS1 algorithm (Hill Climbing). 
"ls2_functions.py" for LS2 algorithm (Genetic Algorithm). To reproduce the results of LS2 in the report, random seeds should be 0,..,9 for large-scale and 0,...,19 for small-scale.

###########

Folder Structure:

Codes
│   README.txt
│   main.py
|   bnb_functions.py
|   approx_functions.py
|   ls1_functions.py
|   ls2_functions.py
|   plot_boxplot.py
|   QRTD.py
|   SQD.py
|  
└───DATASET
    │   large_scale
    │   large_scale_solution
    |   small_scale
    |   small_scale_solution
    |   test
    |   test_solution
    │
    └───submission
        │   large_scale_sol
        │   large_scale_trace
        │   small_scale_sol
        |   small_scale_trace
        |   small_scale_LS1
        └───large_scale_LS1


