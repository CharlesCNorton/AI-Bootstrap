import numpy as np
import matplotlib.pyplot as plt

def greedy_coloring(n, sequence_type='natural'):
    """
    Implements the deterministic greedy coloring algorithm.

    Parameters:
    - n: The maximum integer to consider in the sequence.
    - sequence_type: The type of sequence to use ('natural', 'odd', 'multiples_of_3', 'primes').

    Returns:
    - A list of colors ('red' or 'blue') corresponding to the assignment of each number in the sequence.
    """
    if sequence_type == 'natural':
        sequence = np.arange(2, n+1)
    elif sequence_type == 'odd':
        sequence = np.arange(3, n*2, 2)
        sequence = sequence[sequence <= n]
    elif sequence_type == 'multiples_of_3':
        sequence = np.arange(3, n+1, 3)
    elif sequence_type == 'primes':
        sequence = sieve_of_eratosthenes(n)
    else:
        raise ValueError("Invalid sequence type. Choose from 'natural', 'odd', 'multiples_of_3', 'primes'.")

    S_red = 0.0   # Sum of reciprocals in the red set
    S_blue = 0.0  # Sum of reciprocals in the blue set
    color_assignment = []  # Store the color for each number

    for i in sequence:
        recip = 1.0 / i
        # Compute potential discrepancies for both choices
        D_red = abs((S_red + recip) - S_blue)
        D_blue = abs(S_red - (S_blue + recip))
        # Assign to the color that minimizes the discrepancy
        if D_red <= D_blue:
            S_red += recip
            color_assignment.append('red')
        else:
            S_blue += recip
            color_assignment.append('blue')

    return color_assignment

def sieve_of_eratosthenes(max_num):
    """
    Generates all prime numbers up to max_num using the Sieve of Eratosthenes.

    Parameters:
    - max_num: The maximum number up to which to generate primes.

    Returns:
    - primes: A numpy array of prime numbers up to max_num.
    """
    is_prime = np.ones(max_num + 1, dtype=bool)
    is_prime[:2] = False
    for i in range(2, int(np.sqrt(max_num)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.nonzero(is_prime)[0]
    return primes

def visualize_grid(color_assignment, n):
    """
    Visualizes the coloring of numbers as a grid.

    Parameters:
    - color_assignment: A list of 'red' or 'blue' assignments for each number.
    - n: The maximum number in the sequence to visualize.
    """
    grid_size = int(np.sqrt(len(color_assignment)))  # Create a square grid
    grid = np.zeros((grid_size, grid_size, 3))  # RGB grid

    for idx, color in enumerate(color_assignment[:grid_size**2]):  # Limit to grid size
        row = idx // grid_size
        col = idx % grid_size
        if color == 'red':
            grid[row, col] = [1, 0, 0]  # Red
        else:
            grid[row, col] = [0, 0, 1]  # Blue

    plt.imshow(grid, interpolation='nearest')
    plt.title(f'Coloring Visualization (n = {n})')
    plt.axis('off')
    plt.show()

def menu():
    """
    Displays a terminal menu for selecting sequence types and N values.
    """
    print("Coloring Sequence Visualization")
    print("1. Natural Numbers")
    print("2. Odd Numbers")
    print("3. Multiples of 3")
    print("4. Prime Numbers")
    print("5. Exit")
    choice = input("Select a sequence type (1-5): ")
    return choice

def select_n_value():
    """
    Prompts the user to select a value for n.
    """
    while True:
        try:
            n = int(input("Enter the maximum value for N (e.g., 100, 1000, 10000): "))
            if n < 2:
                print("Please enter a number greater than 1.")
                continue
            return n
        except ValueError:
            print("Invalid input. Please enter a valid integer.")

def main():
    while True:
        choice = menu()

        if choice == '5':
            print("Exiting program.")
            break

        # Map the menu choice to the corresponding sequence type
        sequence_types = {
            '1': 'natural',
            '2': 'odd',
            '3': 'multiples_of_3',
            '4': 'primes'
        }

        if choice in sequence_types:
            sequence_type = sequence_types[choice]
            n = select_n_value()
            color_assignment = greedy_coloring(n, sequence_type)

            # Visualize the grid of colored numbers
            visualize_grid(color_assignment, n)
        else:
            print("Invalid choice. Please select a valid option.")

if __name__ == "__main__":
    main()
