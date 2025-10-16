def getDir( direction):
    if direction == (-1 , 0):
        return 'U'
    elif direction == (1 , 0):
        return 'D'
    elif direction == (0, -1):
        return 'L'
    else:
        return 'R'

def dls(start_state , limit = 1000):


    all_goals = frozenset(start_state.get_targets_positions())

    path_set = set()

    def dfs(curr_state, remaining_goals, depth):
        curr_pos = curr_state.get_agent_position()
        key = (curr_pos, remaining_goals)

        # goal test
        if not remaining_goals:
            return []  # already satisfied — no more actions needed

        # depth cutoff
        if depth == 0:
            return None

        path_set.add(key)

        for action in curr_state.get_successors():
            next_state = action[2]
            next_pos = next_state.get_agent_position()

            if next_pos in remaining_goals:
                next_remaining = frozenset(g for g in remaining_goals if g != next_pos)
            else:
                next_remaining = remaining_goals

            next_key = (next_pos, next_remaining)

            # enemy detection and loop
            if next_key in path_set or next_state.is_collision_state():
                continue

            res = dfs(next_state, next_remaining, depth - 1)

            #if we had solution dfs return a list that may have nothing(last goal) or a path
            if res is not None:
                direction = (next_pos[0] - curr_pos[0], next_pos[1] - curr_pos[1])
                move = getDir(direction)
                path_set.remove(key)
                return [move] + res

        path_set.remove(key)
        return None

    return dfs(start_state, all_goals, limit)

