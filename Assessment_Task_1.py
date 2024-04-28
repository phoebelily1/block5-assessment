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
    This function should return the extent to which a cell agrees with its neighbours by providing a list of
    neighbouring (adajcent) people's opinions.
    Inputs: population (numpy array) - Matrix representing the population and their opinions.
            row (int) - Row index of current cell
            col (int) - Column index of current cell
            external (float) - External influence on the current cell's opinion (default is 0.0).
    Returns:
            change_in_agreement (float) - - Measure of agreement change in the cell's opinion.
    '''

    m, n = population.shape  # Obtains rows and columns of matrix
    neighbours = []
    # Checks neighbouring cells above and below
    if row > 0:
        neighbours.append(population[row - 1][col])  # If not on top row
    else:
        neighbours.append(population[-1][col])  # Wraps around to the bottom row
    if row < m - 1:
        neighbours.append(population[row + 1][col])  # If not on bottom row
    else:
        neighbours.append(population[0])[col] # Wraps around to the top row

    #Checks neighbouring cells to the left and right
    if col > 0:
        neighbors.append(population[row][col - 1])  # If not in first col
    else:
        neighbours.append(population[row][-1]) # Wraps around to the last col
    if col < n - 1:
        neighbours.append(population[row][col + 1])  # if not in last col
    else:
        neighbours.append(population[row][0]) # Wraps around to the first col

    return np.sum([i*population[row][col] for i in neighbours]) + external * population[row][col] # Uses formula for Di


def ising_step(population, external=0.0):
    '''
    This function will perform a single update of the Ising model
    Inputs: population (numpy array)
            external (float) - optional - the magnitude of any external "pull" on opinion
    '''

    # Extract dimensions of the population matrix
    n_rows, n_cols = population.shape

    # Randomly selects a cell in the population matrix
    row = np.random.randint(0, n_rows)
    col = np.random.randint(0, n_cols)

    #Calculate the agreement of the selected cell with its neighbours
    agreement = calculate_agreement(population, row, col, external=0.0)

    # If the agreement is negative, flip the opinion of the selected cell
    if agreement < 0:
        population[row, col] *= -1
    # If the agreement is positive, calculate a probability to accept the flip
    elif np.random.rand() < np.exp(-agreement / alpha):
        population[row, col] *= -1
    # Uses formula to accept negative flips, basically generating a random probability and then compares to formula


def plot_ising(im, population):
    '''
    This function will display a plot of the Ising model
    '''

    # Converts the Ising model representation to an image format
    # Assigns a value of 255 to -1 (spin down) and 1 to 1 (spin up)
    new_im = np.array([[255 if val == -1 else 1 for val in rows] for rows in population], dtype=np.int8)

    # Update the image data for the plot with the new Ising model representation
    im.set_data(new_im)

    # Pause for a short duration for plot rendering
    plt.pause(0.1)


def test_ising():
    '''
    This function will test the calculate_agreement function in the Ising model
    '''

    print("Testing ising model calculations")

    # Test for when the selected element is surrounded by -1s
    population = -np.ones((3, 3))
    assert (calculate_agreement(population, 1, 1) == 4), "Test 1"

    # Test for when the selected element is surrounded by 1s
    population[1, 1] = 1.
    assert (calculate_agreement(population, 1, 1) == -4), "Test 2"

    # Tests for when one of the adjacent elements is flipped to 1
    population[0, 1] = 1.
    assert (calculate_agreement(population, 1, 1) == -2), "Test 3"

    population[1, 0] = 1.
    assert (calculate_agreement(population, 1, 1) == 0), "Test 4"

    population[2, 1] = 1.
    assert (calculate_agreement(population, 1, 1) == 2), "Test 5"

    population[1, 2] = 1.
    assert (calculate_agreement(population, 1, 1) == 4), "Test 6"

    print("Testing external pull")
    # Testing the effect of an external pull on the selected element
    population = -np.ones((3, 3))
    assert (calculate_agreement(population, 1, 1, 1) == 3), "Test 7"
    assert (calculate_agreement(population, 1, 1, -1) == 5), "Test 8"
    assert (calculate_agreement(population, 1, 1, -10) == 14), "Test 9"
    assert (calculate_agreement(population, 1, 1, 10) == -6), "Test 10"

    print("Tests passed")


def ising_main(population, alpha=None, external=0.0):
    # Creates a figure for plotting
    fig = plt.figure()
    # Adds a subplot to the figure and turns of axis
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    #Displays initial state of the population matrix
    im = ax.imshow(population, interpolation='none', cmap='RdPu_r')

    # Iterating an update 100 times
    for frame in range(100):
        # Iterating single steps 1000 times to form an update
        for step in range(1000):
            # Performs 1 step of the update
            ising_step(population, external)
        # Prints the current step number
        print('Step:', frame, end='\r')
        # Updates the plot
        plot_ising(im, population)


'''
==============================================================================================================
This section contains code for the main function- you should write some code for handling flags here
==============================================================================================================
'''


def main():
    parser = argparse.ArgumentParser() # Create parser object

    parser.add_argument('-ising_model',action='store_true') # Add ising_model flag
    parser.add_argument('-external',type=float,default=0.0) # Add external flag
    parser.add_argument('-test_ising',action='store_true') # Add test_ising flag
    parser.add_argument('-alpha',type=float,default=1.0) # Add alpha flag
    args = parser.parse_args() # Obtains arguments

    # If given ising_model
    if args.ising_model:
        population = np.random.choice([-1, 1], size=(10, 10)) # Initalise random population
        ising_main(population,args.external,args.alpha) # Run ising_main

    # If test_ising flag is present
    if args.test_ising:
        # Run test_ising function
        test_ising()

if __name__ == "__main__":
    main()