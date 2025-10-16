from collections import deque
from env.env import play

def bfs(initial_state):
    # Node class to store state info
    class Node:
        def __init__(self, state, position, collected_stars, time, actions, cost):
            self.state = state
            self.position = position
            self.collected_stars = collected_stars
            self.time = time
            self.actions = actions
            self.cost = cost

    # Get enemy path and length
    enemy_path = initial_state.path_enemy_get()
    enemy_path_length = len(enemy_path) if enemy_path else 0

    # Initialize BFS queue and visited set
    queue = deque()
    visited = set()
    initial_position = initial_state.position_agent_get()
    all_stars = frozenset(initial_state.positions_targets_get())
    initial_node = Node(initial_state, initial_position, frozenset(), 0, [], 0)

    queue.append(initial_node)
    visited.add((initial_position, frozenset(), 0))

    while queue:
        node = queue.popleft()
        state, pos, stars, time, actions, cost = (
            node.state,
            node.position,
            node.collected_stars,
            node.time,
            node.actions,
            node.cost,
        )

        # Check if goal is reached
        if state.state_goal_is():
            return actions

        # Get enemy position
        enemy_pos = enemy_path[time % enemy_path_length] if enemy_path else None

        # Explore successors
        for action, move_cost, next_pos in state.successors_get():
            # Skip if move leads to enemy
            next_time = time + 1
            if enemy_path and next_pos == enemy_path[next_time % enemy_path_length]:
                continue

            # Update collected stars
            new_stars = stars | {next_pos} if next_pos in all_stars else stars

            # Skip visited states
            state_key = (next_pos, new_stars, next_time % (enemy_path_length or 1))
            if state_key in visited:
                continue

            # Create new node
            new_actions = actions + [action]
            new_cost = cost + move_cost
            new_node = Node(state, next_pos, new_stars, next_time, new_actions, new_cost)

            queue.append(new_node)
            visited.add(state_key)

    return []  # No solution found

if __name__ == "__main__":
    play("map_name", bfs, delay=500)