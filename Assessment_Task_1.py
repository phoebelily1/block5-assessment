import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse



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

    m, n = population.shape  # obtains rows and columns of matrix
    neighbours = []
    if row > 0:
        neighbours.append(population[row - 1][col])  # if not on top row
    else:
        neighbours.append(population[-1][col])  # wraps around
    if row < m - 1:
        neighbours.append(population[row + 1][col])  # if not on bottom row
    else:
        neighbours.append(population[0])[col]
    if col > 0:
        neighbors.append(population[row][col - 1])  # if not in first col
    else:
        neighbours.append(population[row][-1])
    if col < n - 1:
        neighbours.append(population[row][col + 1])  # if not in last col
    else:
        neighbours.append(population[row][0])

    return np.sum([i*population[row][col] for i in neighbours]) + external * population[row][col] # uses formula for Di


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
    elif np.random.rand() < np.exp(-agreement / alpha):
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
    assert (calculate_agreement(population, 1, 1, -10) == 14), "Test 9"
    assert (calculate_agreement(population, 1, 1, 10) == -6), "Test 10"

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
    parser = argparse.ArgumentParser()
    parser.add_argument('-ising_model',action='store_true')
    parser.add_argument('-external',type=float,default=0.0)
    parser.add_argument('-test_ising',action='store_true')
    parser.add_argument('-alpha',type=float,default=1.0)
    args = parser.parse_args()
    if args.ising_model:
        population = np.random.choice([-1, 1], size=(10, 10))
        ising_main(population,args.external,args.alpha)
    if args.test_ising:
        test_ising()

# You should write some code for handling flags here

if __name__ == "__main__":
    main()