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
    # Compute Hamming distance between state and each genome
    distances = np.sum(genomes != state, axis=1)
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