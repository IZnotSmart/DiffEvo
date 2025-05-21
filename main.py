#Issac Zachariah B922504
#In this file is where the variables and objective function is changed
from diffEvo import * #contains the differential evolutionary algorithm
import numpy as np
import math

#https://en.wikipedia.org/wiki/Test_functions_for_optimization
def obj(pop):
    obj_all = np.zeros((pop.shape[0], 2))

    
    #Kursawe Function
    for i,x in enumerate(pop):
        f1 = 0
        for j in range(2):
            f1 += - 10*math.exp(-0.2*math.sqrt((x[j])**2 + (x[j+1])**2))

        f2 = 0
        for j in range(2):
            f2 += (abs(x[j]))**0.8 + 5*math.sin((x[j])**3)

        obj_all[i,0] = f1
        obj_all[i,1] = f2

    """

    for i, x in enumerate(pop):
        #Chankong and Haimes function
        #f1 = 2 + ((x[0]-2)**2) + ((x[1]-1)**2)
        #f2 = (9*x[0]) - ((x[1]-1)**2)

        #Binh and Korn
        f1 = (4*(x[0]**2))+(4*(x[1]**2))
        f2 = ((x[0]-5)**2)+((x[1]-5)**2)
        
        obj_all[i,0] = f1
        obj_all[i,1] = f2
       """

    return np.asarray(obj_all)




#run
#population size
pop_size = 150
#number of iterations
iter = 150
#Mutation scale factor
F = 0.9
#crossover rate
cr = 0.7
#define lower bound and upper bound
#number of variables are also determined by number of bounds
bounds = np.asarray([(-5, 5.0), (-5, 5.0), (-5, 5)])



#run differential evolution algorithm
solution = differential_evolution(obj, pop_size, bounds, iter, F, cr)


print(solution)
#plot graph
plotPareto(solution, obj)

print("end")
