import heapq
import itertools

def get_dir(direction):
    if direction == (-1, 0):
        return 'U'
    elif direction == (1, 0):
        return 'D'
    elif direction == (0, -1):
        return 'L'
    else:
        return 'R'

def get_dir_reverse(direction):
    if direction == 'U':
        return (-1, 0)
    elif direction == 'D':
        return (1, 0)
    elif direction == 'L':
        return (0, -1)
    else:
        return (0, 1)
def astar(start_state):
    wall_state = start_state

    def manhattan_distance(pos1 , pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def heuristic(new_position , new_remaining_ , new_state ,wall_penalty = 4,step_cost_ = 5):
         c = float('inf')
         for goal in new_remaining_:
             distance = manhattan_distance(new_position , goal)
             punishment =0
             for wall in new_state.get_rocks_positions():
                 if (min(new_position[0],goal[0]) <= wall[0] <= max(new_position[0],goal[0])) and (min(new_position[1],goal[1]) <= wall[1] <= max(new_position[1],goal[1])) :
                     punishment += 1
             c = min(c, distance*step_cost_ + punishment*wall_penalty*step_cost_)
         if len(new_remaining_) == 0:c = 0
         return c*(len(new_remaining_)^3)


    all_goals = frozenset(start_state.get_targets_positions())
    start_pos = start_state.get_agent_position()
    start_key = (start_pos, all_goals , ())

    counter = itertools.count()
    # heap entries: (how many goal remaining,actual cost, heuristic cost , tie, pos_tuple, remaining_goals, state_obj)
    #tie = when there is equal costs
    heap = [(len(all_goals),0, float('inf'),next(counter), start_pos, all_goals, start_state , ())]

    costs = {start_key: 0}
    parents = {start_key: None}  # parent maps key -> parent_key
    visited = set()

    final_key = None

    while heap:
        _,cost,her,steps, pos, remaining_goals, state_obj , wall_move = heapq.heappop(heap)
        curr_key = (pos, remaining_goals , wall_move)

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
            final_key = (pos, new_remaining , ())
            # ensure parents/costs keyed for final_key; associate state_obj (current)
            costs[final_key] = cost
            parents[final_key] = parents.get(curr_key, None)
            break

        # expand successors using the actual state object
        actual_state = state_obj

        for action in actual_state.get_successors():
            wall_move = False

            next_state = action[2]
            next_pos = next_state.get_agent_position()
            next_remaining = new_remaining
            # make the remaining for next action
            if next_pos in next_remaining:
                next_remaining = frozenset(g for g in next_remaining if g != next_pos)

            next_key = (next_pos, next_remaining , ())

            step_cost = next_state.get_terrain_cost(next_pos)
            h_new = heuristic(next_pos, remaining_goals , next_state)
            new_cost = cost + step_cost +  h_new
            if next_state.is_collision_state():
                wall_pos = ()
                list1 = []
                for g in actual_state.get_successors():
                    list1.append(g[2].get_agent_position())
                for g in actual_state.get_successors(toward_walls=True):
                    if g[2].get_agent_position() not in list1:
                        wall_state = g[2]
                        wall_pos = (g[2].get_agent_position()[0] + get_dir_reverse(g[0])[0],g[2].get_agent_position()[1] + get_dir_reverse(g[0])[1] )
                        wall_move = True

                        break
                wall_key = (pos , remaining_goals , wall_pos)
                if not wall_move or wall_key in visited:
                    continue
                costs[wall_key] = cost + wall_state.get_terrain_cost(wall_state.get_agent_position())
                parents[wall_key] = curr_key
                heapq.heappush(heap, (len(remaining_goals), costs[wall_key], her, next(counter), pos , remaining_goals ,
                                      wall_state , wall_pos))
                continue


            if new_cost < costs.get(next_key, float('inf')) and wall_move == False:
                costs[next_key] = new_cost
                parents[next_key] = curr_key
                heapq.heappush(heap, (len(next_remaining),new_cost,h_new, next(counter), next_pos, next_remaining, next_state , ()))

    path_keys = []
    k = final_key
    while k is not None:
        if len(k[2]) == 1:
            path_keys.append(k[2])
        else :
            path_keys.append(k[0])
        k = parents.get(k, None)
    path_keys.reverse()

    actions = []
    for i in range(1, len(path_keys)):
        prev_pos = path_keys[i - 1]
        cur_pos = path_keys[i]
        direction = (cur_pos[0] - prev_pos[0], cur_pos[1] - prev_pos[1])
        actions.append(get_dir(direction))

    return actions