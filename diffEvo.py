#https://machinelearningmastery.com/differential-evolution-from-scratch-in-python/
#https://medium.com/analytics-vidhya/optimization-modelling-in-python-multiple-objectives-760b9f1f26ee
#Issac Zachariah B922504
#Main file, containing the differential evolutionary algorithm
#Call the differential_evolution()

from numpy.random import rand
from numpy.random import choice
import numpy as np
import matplotlib.pyplot as plt
import random as rn



#mutation operation, a+F*(b-c), where F is the mutation scale factor
def mutation(x,F, bounds):
    mutated = x[0] + F*(x[1]-x[2])
    #check mutated vector is within bounds
    mutated_bound = [np.clip(mutated[i], bounds[i,0], bounds[i,1]) for i in range(len(bounds))]
    return mutated_bound


#crossover operation
def crossover(mutated, target, dims, cr):
    #generate random dimensional values
    p = rand(dims)
    #use binomilal crossover to generate trial vectors
    trial = [mutated[i] if p[i] < cr else target[i] for i in range(dims)]
    return np.asarray(trial)


#Estimate how close values are on Pareto front
def crowding_calculation(fitness_values):
    pop_size = len(fitness_values[:, 0])
    fitness_value_number = len(fitness_values[0, :])            
    matrix_for_crowding = np.zeros((pop_size, fitness_value_number)) 
    normalized_fitness_values = (fitness_values - fitness_values.min(0))/fitness_values.ptp(0)
    
    for i in range(fitness_value_number):
        crowding_results = np.zeros(pop_size)
        crowding_results[0] = 1 # extreme point has the max crowding distance
        crowding_results[pop_size - 1] = 1 # extreme point has the max crowding distance
        sorted_normalized_fitness_values = np.sort(normalized_fitness_values[:,i])
        sorted_normalized_values_index = np.argsort(normalized_fitness_values[:,i])
        
        # crowding distance calculation
        crowding_results[1:pop_size - 1] = (sorted_normalized_fitness_values[2:pop_size] - sorted_normalized_fitness_values[0:pop_size - 2])
        re_sorting = np.argsort(sorted_normalized_values_index)
        matrix_for_crowding[:, i] = crowding_results[re_sorting]
    
    crowding_distance = np.sum(matrix_for_crowding, axis=1)
    return crowding_distance    # array

#Crowding distance is used to maintain diversity
#Remove solutions that are clumped together
def remove_using_crowding(fitness_values, number_solutions_needed):
    pop_index = np.arange(fitness_values.shape[0])
    crowding_distance = crowding_calculation(fitness_values)
    selected_pop_index = np.zeros(number_solutions_needed)
    selected_fitness_values = np.zeros((number_solutions_needed, len(fitness_values[0, :])))
    
    for i in range(number_solutions_needed):
        pop_size = pop_index.shape[0]
        solution_1 = rn.randint(0, pop_size - 1)
        solution_2 = rn.randint(0, pop_size - 1)
        
        if crowding_distance[solution_1] >= crowding_distance[solution_2]:
            # solution 1 is better than solution 2
            selected_pop_index[i] = pop_index[solution_1]
            selected_fitness_values[i, :] = fitness_values[solution_1, :]
            pop_index = np.delete(pop_index, (solution_1), axis=0)
            fitness_values = np.delete(fitness_values, (solution_1), axis=0)
            crowding_distance = np.delete(crowding_distance, (solution_1), axis=0)
            
        else:
            # solution 2 is better than solution 1
            selected_pop_index[i] = pop_index[solution_2]
            selected_fitness_values[i, :] = fitness_values[solution_2, :]
            pop_index = np.delete(pop_index, (solution_2), axis=0)
            fitness_values = np.delete(fitness_values, (solution_2), axis=0)
            crowding_distance = np.delete(crowding_distance, (solution_2), axis=0)
    
    selected_pop_index = np.asarray(selected_pop_index, dtype=int)
    return selected_pop_index   # array


#find indicies of solutions that dominate others
def pareto_front_finding(fitness_values, pop_index):
    pop_size = fitness_values.shape[0]
    pareto_front = np.ones(pop_size, dtype=bool)    # all True
    for i in range(pop_size):
        for j in range(pop_size):
            if all(fitness_values[j] <= fitness_values[i]) and any(fitness_values[j] < fitness_values[i]):
                pareto_front[i] = 0 # i not in pareto front because dominated
                break

    return pop_index[pareto_front]  # array

# repeat Pareto front selection to build a population within defined size limits
def selection(pop, fitness_values, pop_size):
    pop_index_0 = np.arange(pop.shape[0]) # unselected pop ids
    pop_index = np.arange(pop.shape[0]) # all pop ids
    pareto_front_index = []
    
    while len(pareto_front_index) < pop_size:
        new_pareto_front = pareto_front_finding(fitness_values[pop_index_0, :], pop_index_0)
        total_pareto_size = len(pareto_front_index) + len(new_pareto_front)

        # check size of pareto front
        if total_pareto_size > pop_size:
            number_solutions_needed = pop_size - len(pareto_front_index)
            selected_solutions = remove_using_crowding(fitness_values[new_pareto_front], number_solutions_needed)
            new_pareto_front = new_pareto_front[selected_solutions]
        
        pareto_front_index = np.hstack((pareto_front_index, new_pareto_front))
        remaining_index = set(pop_index) - set(pareto_front_index)
        pop_index_0 = np.array(list(remaining_index))
        
    selected_pop = pop[pareto_front_index.astype(int)]
    return selected_pop     # array


#Pareto front visualisation
def plotPareto(pop, obj):
    # Pareto front visualization
    fitness_values = obj(pop)
    index = np.arange(pop.shape[0]).astype(int)

    """
    #Plot only the optimal solutions within the current population
    pareto_front_index = pareto_front_finding(fitness_values, index)
    pop = pop[pareto_front_index, :]
    fitness_values = fitness_values[pareto_front_index]
    """

    print("_________________")
    print("Optimal solutions:")
    print("       x1               x2")
    print(pop) # show optimal solutions
    print("______________")
    print("Fitness values:")
    print("  objective 1    objective 2")
    print(fitness_values)

    plt.scatter(fitness_values[:, 0],fitness_values[:, 1], label='Pareto optimal front')
    plt.legend(loc='best')
    plt.xlabel('Objective function F1')
    plt.ylabel('Objective function F2')
    plt.grid(b=1)
    plt.show()


#Main function
def differential_evolution(obj, pop_size, bounds, iter, F, cr):
    #init population, between bounds
    pop = bounds[:, 0] + (rand(pop_size, len(bounds)) * (bounds[:, 1] - bounds[:, 0]))
    #evaulating intial population
    obj_all = obj(pop)

    #run thru iterations
    for i in range (iter):
        print('iteration:', i)
        
        #iterate over candidates
        trialpop = [] #population of mutated and crossover'ed individuals
        for j in range (pop_size):
            #mutation process
            candidates = [candidate for candidate in range(pop_size) if candidate !=j]
            #3 random candidates are selected to generate a mutated vector
            a, b, c = pop[choice(candidates, 3, replace=False)]
            #mutation
            mutated = mutation([a,b,c],F, bounds)
            #crossover
            trial = crossover(mutated, pop[j], len(bounds), cr)
            trialpop.append(trial)
            
        #selection
        trialpop = np.asarray(trialpop)
        pop = np.append(pop, trialpop, axis=0)
        obj_all = obj(pop)
        pop = selection(pop, obj_all, pop_size)
        
    print("Algorithm finished")
    return pop

