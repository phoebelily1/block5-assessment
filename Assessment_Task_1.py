import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm



'''
==============================================================================================================
This section contains code for the Ising Model - task 1 in the assignment
==============================================================================================================
'''


def calculate_agreement(population, row, col, external=0.0):
    '''
    This function should return the extent to which a cell agrees with its neighbours.
    Inputs: population (numpy array)
            row (int)
            col (int)
            external (float)
    Returns:
            change_in_agreement (float)
    '''

    m,n = population.shape
    neighbours = []
    if row > 0:
        neighbours.append(population[row + 1][col])
    if row < m:
        neighbours.append(population[row-1][col])
    if col > 0:
        neighbours.append(population[row][col +1])
    if col < n:
        neighbours.append(population[row][col - 1])



    return np.random.random() * population


def ising_step(population, external=0.0):
    '''
    This function will perform a single update of the Ising model
    Inputs: population (numpy array)
            external (float) - optional - the magnitude of any external "pull" on opinion
    '''

    n_rows, n_cols = population.shape
    row = np.random.randint(0, n_rows)
    col = np.random.randint(0, n_cols)

    agreement = calculate_agreement(population, row, col, external=0.0)

    if agreement < 0:
        population[row, col] *= -1


# Your code for task 1 goes here

def plot_ising(im, population):
    '''
    This function will display a plot of the Ising model
    '''


new_im = np.array([[255 if val == -1 else 1 for val in rows] for rows in population], dtype=np.int8)
im.set_data(new_im)
plt.pause(0.1)


def test_ising():
    '''
    This function will test the calculate_agreement function in the Ising model
    '''


print("Testing ising model calculations")
population = -np.ones((3, 3))
assert (calculate_agreement(population, 1, 1) == 4), "Test 1"

population[1, 1] = 1.
assert (calculate_agreement(population, 1, 1) == -4), "Test 2"

population[0, 1] = 1.
assert (calculate_agreement(population, 1, 1) == -2), "Test 3"

population[1, 0] = 1.
assert (calculate_agreement(population, 1, 1) == 0), "Test 4"

population[2, 1] = 1.
assert (calculate_agreement(population, 1, 1) == 2), "Test 5"

population[1, 2] = 1.
assert (calculate_agreement(population, 1, 1) == 4), "Test 6"

"Testing external pull"
population = -np.ones((3, 3))
assert (calculate_agreement(population, 1, 1, 1) == 3), "Test 7"
assert (calculate_agreement(population, 1, 1, -1) == 5), "Test 8"
assert (calculate_agreement(population, 1, 1, 10) == 14), "Test 9"
assert (calculate_agreement(population, 1, 1, -10) == -6), "Test 10"

print("Tests passed")


def ising_main(population, alpha=None, external=0.0):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    im = ax.imshow(population, interpolation='none', cmap='RdPu_r')

    # Iterating an update 100 times
    for frame in range(100):
        # Iterating single steps 1000 times to form an update
        for step in range(1000):
            ising_step(population, external)
        print('Step:', frame, end='\r')
        plot_ising(im, population)


'''
==============================================================================================================
This section contains code for the main function- you should write some code for handling flags here
==============================================================================================================
'''


def main():
    print('yes')

# You should write some code for handling flags here

if __name__ == "__main__":
    main()