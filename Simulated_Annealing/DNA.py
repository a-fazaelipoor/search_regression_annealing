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