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