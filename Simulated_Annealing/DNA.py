import random
import numpy as np
import matplotlib.pyplot as plt
import time

Alphabet = np.array(['a', 'c', 'g', 't'])

def initialize_state(n):
    """Generate a random DNA sequence of length n."""
    return np.random.choice(Alphabet, size=n)

def calculate_evaluation(genomes, state):
    genomes_array = np.vstack(genomes).astype(str)
    distances = np.sum(genomes_array != state, axis=1)
    return np.max(distances)

def get_neighbor_state(state):
    neighbor = state.copy()
    index = np.random.randint(0, len(state))
    current_char = state[index]
    possible_chars = Alphabet[Alphabet != current_char]
    neighbor[index] = np.random.choice(possible_chars)
    return neighbor

def get_all_neighbors(state):
    neighbors = []
    for i in range(len(state)):
        current_char = state[i]
        for char in Alphabet[Alphabet != current_char]:
            neighbor = state.copy()
            neighbor[i] = char
            neighbors.append(neighbor)
    return neighbors

def hill_climbing_descent(genomes, initial_state, max_iterations=1000):
    current_state = initial_state
    evaluations = []
    
    for _ in range(max_iterations):
        current_value = calculate_evaluation(genomes, current_state)
        evaluations.append(current_value)
        
        neighbors = get_all_neighbors(current_state)
        best_neighbor, best_value = None, float('inf')

        for neighbor in neighbors:
            value = calculate_evaluation(genomes, neighbor)
            if value < best_value:
                best_value = value
                best_neighbor = neighbor
        
        if best_neighbor is None or best_value >= current_value:
            break
        
        current_state = best_neighbor

    return current_state, calculate_evaluation(genomes, current_state), evaluations

def simulated_annealing(genomes, initial_state, alpha, initial_temp, max_iteration, min_temperature):
    curr_state = initial_state
    temperature = initial_temp
    evaluations = []

    curr_value = calculate_evaluation(genomes, curr_state)
    best_state, best_value = curr_state.copy(), curr_value
    evaluations.append(curr_value)

    for _ in range(max_iteration):
        if temperature <= min_temperature:
            break

        neighbor_state = get_neighbor_state(curr_state)
        neighbor_value = calculate_evaluation(genomes, neighbor_state)
        delta_f = neighbor_value - curr_value

        if delta_f < 0 or np.random.random() < np.exp(-delta_f / temperature):
            curr_state, curr_value = neighbor_state, neighbor_value

        if curr_value < best_value:
            best_state, best_value = curr_state.copy(), curr_value

        evaluations.append(curr_value)
        temperature *= alpha

    return best_state, best_value, evaluations

def go_to_next(number, alphabet_length):
    idx = len(number) - 1
    while idx >= 0:
        if number[idx] < alphabet_length - 1:
            number[idx] += 1
            return True
        number[idx] = 0
        idx -= 1
    return False

def brute_force(genomes):
    n = len(genomes[0])
    curr_state_index = np.zeros(n, dtype=int)
    best_state, best_value = None, float('inf')

    while True:
        value = calculate_evaluation(genomes, Alphabet[curr_state_index])
        if value < best_value:
            best_value, best_state = value, Alphabet[curr_state_index]
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
     ['c', 't', 'c', 'c', 't', 'c', 'c', 'c', 'g']]]
, dtype=object)

for genomes in genomes_array:
    initial_state = initialize_state(len(genomes[0]))

    start_time_sa = time.time()
    result_dna_sa, result_value_sa, _ = simulated_annealing(genomes, initial_state, 0.9, 500, 1000, 1e-3)
    time_sa = time.time() - start_time_sa

    start_time_hc = time.time()
    result_dna_hc, result_value_hc, _ = hill_climbing_descent(genomes, initial_state)
    time_hc = time.time() - start_time_hc

    start_time_bf = time.time()
    result_dna_bf, result_value_bf = brute_force(genomes)
    time_bf = time.time() - start_time_bf

    print(f'Simulated Annealing found solution {result_dna_sa} with value {result_value_sa} in {time_sa*1000:.3f} ms')
    print(f'Hill Climbing       found solution {result_dna_hc} with value {result_value_hc} in {time_hc*1000:.3f} ms')
    print(f'Brute Force         found solution {result_dna_bf} with value {result_value_bf} in {time_bf*1000:.3f} ms')
    print("-" * 50)
