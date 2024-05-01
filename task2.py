import numpy as np
import matplotlib.pyplot as plt


def initial_opinions(pop_size):
    return np.random.rand(pop_size)
'''
this function generates an array of random numbers representing
initial opinions of the population.
'''

def defuant_model(population, coupling = 0.2, threshold = 0.2, iterations = 10000):
    pop_history = [population.copy()]
    N = len(population)
    for j in range(iterations):
        i = np.random.randint(N)
        neighbour_choice = np.random.choice([i-1, i+1])
        if neighbour_choice < 0:
            neighbour_choice = N - 1
        elif neighbour_choice >= N:
            neighbour_choice = 0    # creating circular boundaries
        if abs(population[i] - population[neighbour_choice]) < threshold:
            mean_opinion = (population[i] + population[neighbour_choice]) / 2
            population[i] += coupling * (mean_opinion - population[i])
            population[neighbour_choice] += coupling * (mean_opinion - population[neighbour_choice])
        if j%100==0:
            pop_history.append(population.copy())
    distribution_opinions(population)
    plot_opinions(pop_history[:-1])
    return population
'''
This function updates opinions based on neighbours opinions by randomly selecting an individual
at each iteration, with circular boundary conditions. It then calculates difference in opinion.
If difference < threshold, it updates opinion towards the mean. It adds a copy to pop_history list
and plots the final opinions distribution.
'''

def distribution_opinions(population):
    plt.hist(population)
    plt.title('Opinion Distribution')
    plt.xlabel('Opinion')
    plt.ylabel('Frequency')
    plt.show()

'''
This function plots a histogram for distribution of opinions within population.
'''

def plot_opinions(pop_history):
    for i, population in enumerate(pop_history):
        plt.scatter([i]*len(population), population, c='r')
    plt.title('Opinions over time')
    plt.xlabel('Timestep')
    plt.ylabel('Opinion')
    plt.show()

'''
This function plots each persons opinion at each timestep, as it loops through.
'''

def test_defuant():
    population = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
    updated_pop = defuant_model(population, coupling = 0.1, threshold = 0.3)
    return updated_pop
'''
This test function tests the opinions_updated function updates correctly based on parameters.
'''

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Defuant Model Simulation") #command line arguments
    parser.add_argument("-pop_size", type=int, default=100)
    parser.add_argument("-coupling", type=float, default=0.2)
    parser.add_argument("-threshold", type=float, default=0.2)
    parser.add_argument("-test_defuant", action="store_true")
    args = parser.parse_args()

    if args.test_defuant:     #run test for defuant model
        test_defuant()
    else:                     #or run model with specified parameters
        population = initial_opinions(pop_size=args.pop_size)
        defuant_model(population, threshold=args.threshold, coupling=args.coupling)


