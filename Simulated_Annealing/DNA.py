import random
import numpy as np
import matplotlib.pyplot as plt
import time

Alphabet = np.array(['a', 'c', 'g', 't'])


def initialize_state(n):
    """
    [inputs]
    length of the vector
    
    [outputs]
    returns a random vector with length n and uniform probability from Alphabet elements as a numpy array
    """
    return np.random.choice(Alphabet, size=n)

def calculate_evaluation(genomes, state):
    ##################################################################
    # [inputs]                                                       #
    # genomes is a 2D numpy array and represents the set of DNAs     #
    # state is a 1D numpy array which represents "a" in the equation  #
    # [outputs]                                                      #
    # the function returns the value of f(a)                         #
    ##################################################################
    # Convert genomes to 2D numpy array
    genomes_array = np.array(genomes)
    # Compute Hamming distance between state and each genome
    distances = np.sum(genomes_array != state, axis=1)
    # Return the maximum Hamming distance
    return np.max(distances)

def get_neighbor_state(state):
    ##################################################################
    # [inputs]                                                       #
    # state is a 1D numpy array which represents a state             #
    # [outputs]                                                      #
    # a neighboring state of input is returned                       #
    # [notice]                                                       #
    # note the the same state as the input should not be returned    #
    # and the neighbor should be selected uniformly at random        #
    ##################################################################
    # Create a copy of the state to avoid modifying the original
    neighbor = state.copy()
    # Randomly select an index to modify
    index = np.random.randint(0, len(state))
    # Get the current character at the selected index
    current_char = state[index]
    # Create a list of possible characters excluding the current one
    possible_chars = Alphabet[Alphabet != current_char]
    # Randomly select a new character from the possible characters
    new_char = np.random.choice(possible_chars)
    # Update the neighbor state with the new character at the selected index
    neighbor[index] = new_char
    return neighbor

def get_all_neighbors(state):
    ##################################################################
    # This function should generate and return a list of all         #
    # possible neighbors for a given state. A neighbor is defined    #
    # as a sequence that differs from the original state by exactly  #
    # one character.                                                 #
    # [inputs]                                                       #
    #  - state: a 1D numpy array representing the current state      #
    # [outputs]                                                      #
    #  - a list of all neighboring states                            #
    ##################################################################
    neighbors = []
    # Iterate over each position in the state
    for i in range(len(state)):
        # Get the current character at position i
        current_char = state[i]
        # Get all possible characters except the current one
        possible_chars = Alphabet[Alphabet != current_char]
        # For each possible character, create a new neighbor
        for char in possible_chars:
            # Create a copy of the state
            neighbor = state.copy()
            # Change the character at position i to the new character
            neighbor[i] = char
            # Add the neighbor to the list
            neighbors.append(neighbor)
    return neighbors

def hill_climbing_descent(genomes, initial_state, max_iterations=1000):
    ##################################################################
    # Implement the Hill Climbing (descent) algorithm.               #
    # [inputs]                                                       #
    #  - genomes: a 2D numpy array representing the set of DNAs      #
    #  - initial_state: the state to start with                      #
    #  - max_iterations: a limit to prevent infinite loops           #
    # [outputs]                                                      #
    #  - best state found as a numpy array                           #
    #  - best state's evaluation value                               #
    #  - all the evaluations of current state as a normal list       #
    ##################################################################
    current_state = initial_state
    evaluations = [] # Store the history of costs for plotting
    current_value = float('inf')
    
    for iteration in range(max_iterations):
        # Evaluate the current state
        current_value = calculate_evaluation(genomes, current_state)
        evaluations.append(current_value)
        
        # Get all neighbors of the current state
        neighbors = get_all_neighbors(current_state)
        
        # Find the neighbor with the minimum evaluation value
        best_neighbor_value = float('inf')
        best_neighbor = None
        for neighbor in neighbors:
            neighbor_value = calculate_evaluation(genomes, neighbor)
            if neighbor_value < best_neighbor_value:
                best_neighbor_value = neighbor_value
                best_neighbor = neighbor
        
        # If no better neighbor is found, stop
        if best_neighbor is None or best_neighbor_value >= current_value:
            break
        
        # Move to the better neighbor
        current_state = best_neighbor
        current_value = best_neighbor_value
    
    # Return the final state, its cost, and the history of costs
    return current_state, current_value, evaluations

def simulated_annealing(genomes, initial_state, alpha, initial_temp, max_iteration, min_temperature):
    ##############################################################################
    # [inputs]                                                                   #
    # genomes is a 2D numpy array and represents the set of DNAs                 #
    # initial state is the state to start with                                   #
    # alpha is the temperature decay rate                                        #
    # initial temp is T0                                                         #
    # max_iteration is the maximum number of iteration (termination condition)   #
    # min_temperature is the minimum temperature (termination condition)         #
    # [outputs]                                                                  #
    # best state found as a numpy array                                          #
    # best state's evaluation value                                              #
    # all the evaluations of current state as a normal list                      #
    ##############################################################################
    curr_state = initial_state
    i = 0
    temperature = initial_temp
    evaluations = []
    best_state = None
    best_value = float('inf')
    
    # Evaluate initial state
    curr_value = calculate_evaluation(genomes, curr_state)
    evaluations.append(curr_value)
    best_state = curr_state.copy()
    best_value = curr_value
    
    while i < max_iteration and temperature > min_temperature:
        # Generate a random neighbor
        neighbor_state = get_neighbor_state(curr_state)
        
        # Evaluate neighbor
        neighbor_value = calculate_evaluation(genomes, neighbor_state)
        
        # Calculate delta f (should be positive for worse moves)
        delta_f = neighbor_value - curr_value
        
        # Acceptance probability
        if delta_f < 0 or np.random.random() < np.exp(-delta_f / temperature):
            # Accept the move
            curr_state = neighbor_state
            curr_value = neighbor_value
        # else: reject the move (curr_state remains the same)
        
        # Update best solution
        if curr_value < best_value:
            best_state = curr_state.copy()
            best_value = curr_value
        
        # Record current evaluation
        evaluations.append(curr_value)
        
        # Update temperature (geometric cooling)
        temperature *= alpha
        
        # Increment iteration counter
        i += 1
    
    return best_state, best_value, evaluations

##############################################
# no need to the any thing                   #
##############################################

def go_to_next(number, alphabet_length):
    idx = len(number) - 1
    while idx >= 0:
        if number[idx] < alphabet_length - 1:
            number[idx] += 1
            return True
        else:
            number[idx] = 0
            idx -= 1
    return False


def brute_force(genomes):
    n = len(genomes[0])
    curr_state_index = np.zeros(n).astype(int)
    best_state = None
    best_value = float('inf')

    while True:
        new_result = calculate_evaluation(genomes, Alphabet[curr_state_index])

        if new_result < best_value:
            best_value = new_result
            best_state = Alphabet[curr_state_index]

        if not go_to_next(curr_state_index, len(Alphabet)):
            break

    return best_state, best_value

genomes_array = np.array([
                        [['g', 'c', 'a', 't', 'c'],
                         ['g', 'a', 'c', 't', 'c'],
                         ['c', 'a', 'c', 'g', 'c']],
                         [['a', 'c', 'g', 'g', 'g', 'a', 'c'],
                          ['a', 'g', 'g', 'c', 'g', 'a', 'g'],
                          ['c', 'g', 'g', 'g', 'g', 't', 'c']],
                        [['c', 'c', 'a', 'c', 't', 'a', 'g', 'c', 'a'],
                         ['c', 't', 'a', 'g', 't', 'c', 't', 'c', 't'],
                         ['c', 't', 'c', 'c', 't', 'c', 'c', 'c', 'g']]], dtype=object)

for genomes in genomes_array:
    initial_state = initialize_state(len(genomes[0]))
    
    # Run Simulated Annealing
    start_time_sa = time.time()
    result_dna_sa, result_value_sa, _ = simulated_annealing(genomes, initial_state, 0.9, 500, 1000, 1e-3)
    time_sa = time.time() - start_time_sa
    
    # Run Hill Climbing
    start_time_hc = time.time()
    result_dna_hc, result_value_hc, _ = hill_climbing_descent(genomes, initial_state)
    time_hc = time.time() - start_time_hc
    
    # Run Brute Force
    start_time_bf = time.time()
    result_dna_bf, result_value_bf = brute_force(genomes)
    time_bf = time.time() - start_time_bf

    print(f'Simulated Annealing found solution {result_dna_sa} with value {result_value_sa} in {time_sa*1000:.3f} milliseconds')
    print(f'Hill Climbing       found solution {result_dna_hc} with value {result_value_hc} in {time_hc*1000:.3f} milliseconds')
    print(f'Brute Force         found solution {result_dna_bf} with value {result_value_bf} in {time_bf*1000:.3f} milliseconds')
    print("-" * 50)