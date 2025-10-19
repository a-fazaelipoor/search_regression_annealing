from collections import deque

def get_dir(direction):
    if direction == (-1 , 0):
        return 'U'
    elif direction == (1 , 0):
        return 'D'
    elif direction == (0, -1):
        return 'L'
    else:
        return 'R'

def bfs(start_state):
    all_goals = frozenset(start_state.get_targets_positions())
    start_pos = start_state.get_agent_position()
    start_key = (start_pos, all_goals)

    # queue entries: (state_obj, remaining_goals)
    queue = deque()
    queue.append((start_state, all_goals))

    # parent map: key -> parent_key
    parents = {start_key: None}
    # map key -> actual state object (for expanding)
    visited = set([start_key])

    final_key = None

    while queue:
        state_obj, remaining_goals = queue.popleft()
        pos = state_obj.get_agent_position()
        curr_key = (pos, remaining_goals)

        # if standing on a goal, remove it
        if pos in remaining_goals:
            new_remaining = frozenset(g for g in remaining_goals if g != pos)
        else:
            new_remaining = remaining_goals

        # if now no remaining -> finished
        if not new_remaining:
            final_key = (pos, new_remaining)
            # ensure parent/key_to_state exist for final_key
            parents.setdefault(final_key, parents.get(curr_key))
            break

        # expand successors (treat each successor as one step)
        for action in state_obj.get_successors():
            next_state = action[2]
            next_pos = next_state.get_agent_position()

            next_remaining = new_remaining
            if next_pos in next_remaining:
                next_remaining = frozenset(g for g in next_remaining if g != next_pos)

            next_key = (next_pos, next_remaining)
            if next_key in visited or next_state.is_collision_state():
                continue

            visited.add(next_key)
            parents[next_key] = curr_key
            queue.append((next_state, next_remaining))

    # reconstruct path of positions from start to final_key (using parents on keys)
    path_keys = []
    k = final_key
    while k is not None:
        path_keys.append(k)
        k = parents.get(k)
    path_keys.reverse()

    # convert keys -> sequence of directions using positions
    actions = []
    for i in range(1, len(path_keys)):
        prev_pos = path_keys[i - 1][0]
        cur_pos = path_keys[i][0]
        direction = (cur_pos[0] - prev_pos[0], cur_pos[1] - prev_pos[1])
        actions.append(get_dir(direction))

    return actions