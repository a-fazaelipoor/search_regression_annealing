import heapq
import itertools

def getDir(direction):
    if direction == (-1, 0):
        return 'U'
    elif direction == (1, 0):
        return 'D'
    elif direction == (0, -1):
        return 'L'
    else:
        return 'R'
def astar(start_state):
    def manhattanDistance(pos1 , pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def heuristic(new_position , new_remaining_ , new_state ,wall_penalty = 4,step_cost_ = 5):
         c = float('inf')
         for goal in new_remaining_:
             distance = manhattanDistance(new_position , goal)
             punishment =0
             for wall in new_state.get_rocks_positions():
                 if (min(new_position[0],goal[0]) <= wall[0] <= max(new_position[0],goal[0])) and (min(new_position[1],goal[1]) <= wall[1] <= max(new_position[1],goal[1])) :
                     punishment += 1
             c = min(c, distance*step_cost_ + punishment*wall_penalty*step_cost_)
         if len(new_remaining_) == 0:c = 0
         return c*(len(new_remaining_)^3)


    all_goals = frozenset(start_state.get_targets_positions())
    start_pos = start_state.get_agent_position()
    start_key = (start_pos, all_goals)

    counter = itertools.count()
    # heap entries: (how many goal remaining,cost, tie, pos_tuple, remaining_goals, state_obj)
    #tie = when there is equal costs
    heap = [(len(all_goals),0, float('inf'),next(counter), start_pos, all_goals, start_state)]

    costs = {start_key: 0}
    parents = {start_key: None}  # parent maps key -> parent_key
    visited = set()

    final_key = None

    while heap:
        _,cost,her,steps, pos, remaining_goals, state_obj = heapq.heappop(heap)
        curr_key = (pos, remaining_goals)

        # skip if we already processed a better cost for this key
        if cost > costs.get(curr_key, float('inf')) and (curr_key in visited):
            continue

        visited.add(curr_key)

        # If agent is on a goal, remove it from remaining_goals
        if pos in remaining_goals:
            new_remaining = frozenset(g for g in remaining_goals if g != pos)
        else:
            new_remaining = remaining_goals

        # If now no remaining goals => finished
        if not new_remaining:
            final_key = (pos, new_remaining)
            # ensure parents/costs keyed for final_key; associate state_obj (current)
            costs[final_key] = cost
            parents[final_key] = parents.get(curr_key, None)
            break

        # expand successors using the actual state object
        # Note: ensure key_to_state[curr_key] exists (it should)
        actual_state = state_obj  # key_to_state.get(curr_key)

        for action in actual_state.get_successors():
            # assuming action is (action_obj, action_cost?, next_state_obj)
            # many APIs: action = (action_name, cost, next_state)
            next_state = action[2]
            next_pos = next_state.get_agent_position()
            next_remaining = new_remaining
            # make the remaining for next action
            if next_pos in next_remaining:
                next_remaining = frozenset(g for g in next_remaining if g != next_pos)

            next_key = (next_pos, next_remaining)

            if next_state.is_collision_state():
                continue

            step_cost = next_state.get_terrain_cost(next_pos)
            h_new = heuristic(next_pos, remaining_goals , next_state)
            new_cost = cost + step_cost +  h_new
            if new_cost < costs.get(next_key, float('inf')):
                costs[next_key] = new_cost
                parents[next_key] = curr_key
                heapq.heappush(heap, (len(next_remaining),new_cost,h_new, next(counter), next_pos, next_remaining, next_state))

    path_keys = []
    k = final_key
    while k is not None:
        path_keys.append(k)
        k = parents.get(k, None)
    path_keys.reverse()

    actions = []
    for i in range(1, len(path_keys)):
        prev_pos = path_keys[i - 1][0]
        cur_pos = path_keys[i][0]
        direction = (cur_pos[0] - prev_pos[0], cur_pos[1] - prev_pos[1])
        actions.append(getDir(direction))

    return actions
