# Node class for MCTS tree
class Node:
    def __init__(self, state: World = None, action = None, depth = 0, name = None):
        self.state = state
        self.action = action
        self.parent = None
        self.children = []
        self.visits = 0
        self.reward = 0
        self.depth = depth
        self.name = name
    
    def add_child(self, child):
        self.children.append(child)
        child.parent = self
    
    def update(self, reward):
        self.visits += 1
        self.reward += reward


class MCTS:

    def __init__(self, root: World = None):
        self.root = Node(state = root, name = "root")  # State to perform MTCS on
        self.depth = 0  # Depth of tree
        self.count = 0  # Number of root node visits
        # self.m = 20  # Number of simulations to run
        # self.c = 0  # Exploration constant
        self.tree = {}  # Dictionary of Tree of nodes with keys as parents and values as children
        self.state_i = 0
        self.action_i = 0

    def get_next_state(self, state: World, action):
        # Create a copy of the current state
        next_state = deepcopy(state)
        # Perform evacuation action
        next_state = next_state.evacWorld(action)
        # one time step worth of rain
        next_state = simRain(next_state, PRECIP_RATE)
        # one time step worth of flooding
        next_state = simFlow(next_state)
        # one time step worth of randomized drain clogging
        next_state = randDrainFail(next_state)
        # one time step worth of drainage
        next_state = simDrain(next_state)
        return next_state

    def narrow_action_space(self, state: World):
        # Get hexes with water level atleast 25% of flood level and not already flooded
        hexes = state.hexes
        hex_space = [h for h in hexes if (h.water_level > 0.25*FLOOD_LEVEL) and (not h.flood_flag) and (not h.evac_flag)]

        # Return x, y, coordinates of hexes in hex_space
        coord_space = [(h.x, h.y) for h in hex_space]
        return coord_space
    
    def calculate_reward(self, state: World, action):
        reward = 0

        for hex in state.hexes:
            x, y = hex.x, hex.y
            # cost for evacuating a hex
            if (x,y) in action:
                reward += R_EVAC*hex.population
            # reward if hex flooded and already evacuated
            if hex.flood_flag and hex.population==0:
                reward += R_FLOOD_EVAC
            # reward if hex flooded and not evacuated
            if hex.flood_flag and hex.population>0:
                reward += R_FLOOD_NO_EVAC*hex.population
        # for hex in state.hexes:
        #     x, y = hex.x, hex.y
        #     # evacuated and flooded
        #     if (x,y) in action and next_state.grid[x][y].flood_flag:
        #         reward += R_FLOOD_EVAC*hex.population
        #     # evacuated and not flooded
        #     if (x, y) in action and not next_state.grid[x][y].flood_flag:
        #         reward += R_DRY_EVAC*hex.population
        #     # not evacuated and flooded
        #     if (x,y) not in action and next_state.grid[x][y].flood_flag:
        #         reward += R_FLOOD_NO_EVAC*hex.population
        #     # not evacuated and not flooded
        #     if (x,y) not in action and not next_state.grid[x][y].flood_flag:
        #         reward += R_DRY_NO_EVAC*hex.population
        return reward
    
    def calculate_evac_reward(self, state: World, action):
        reward = 0
        for hex in state.hexes:
            x, y = hex.x, hex.y
            # cost for evacuating a hex
            if (x,y) in action:
                reward += R_EVAC*hex.population
        return reward
    
    def calculate_flood_reward(self, state):
        reward = 0
        for hex in state.hexes:
            x, y = hex.x, hex.y
            # reward if hex flooded and already evacuated
            if hex.flood_flag and hex.population==0:
                reward += R_FLOOD_EVAC
            # reward if hex flooded and not evacuated
            if hex.flood_flag and hex.population>0:
                reward += R_FLOOD_NO_EVAC*hex.population
        return reward
    
    def random_rollout(self, state: World, steps=ROLL_STEPS):
        # create a copy of our state after taking a current action
        rollout_state = deepcopy(state)
        # rollout for a certain number of states
        evac_reward = 0
        for i in range(steps):
            # select random action (i.e. select some number of random hexes to evacuate)
            action_space = self.narrow_action_space(rollout_state)
            if len(action_space) < MAX_EVAC_CELLS:
                evac_hexes = action_space
            else:
                evac_hexes = random.choices(action_space, k=MAX_EVAC_CELLS)
            # calculate reward for this action step and add to the total utility            
            evac_reward += self.calculate_evac_reward(rollout_state, evac_hexes)
            # get the next state and repeat
            rollout_state = self.get_next_state(rollout_state, evac_hexes)
            
        flood_reward = self.calculate_flood_reward(rollout_state)
        total_reward = evac_reward + flood_reward

        return total_reward

    def get_branched_actions(self, state, depth):
        # Get action space
        coord_space = self.narrow_action_space(state)
        # Create combinations of all actions
        action_space = []
        if len(coord_space) < MAX_EVAC_CELLS:
            for i in range(0, len(coord_space) + 1):
                for x in itertools.combinations(coord_space, i):
                    action_space.append(list(x))
        else:
            for i in range(0, MAX_EVAC_CELLS + 1):
                for x in itertools.combinations(coord_space, i):
                    action_space.append(list(x))

        # Randomly choose m actions from action space
        if depth == 0:
            max_actions = MAX_ACTION_SPACE
        else:
            max_actions = int(MAX_ACTION_SPACE/depth)
        # Calculate the max action for this depth
        if len(action_space) < max_actions:
            actions = action_space
        else:
            if MCTS_RUN:
                # sort the actions based on the average water level of each action if the action is not empty
                action_space.sort(key=lambda x: np.sum([state.grid[i[0]][i[1]].water_level for i in x]) if len(x) > 0 else 0, reverse=True)
                # choose the best k actions from the sorted list
                actions = action_space[:max_actions]
            else:
                actions = random.choices(action_space, k=max_actions)
        return actions

    def get_best_action(self, parent):
        # calculate UCB1 heuristic for each node in parent's children
        ucb1_values = [self.ucb1(child) for child in parent.children]
        # determine best action from index of maximum UCB1 value
        best_action = parent.children[ucb1_values.index(max(ucb1_values))]
        # print("Printing best action: ", best_action)
        # return best action
        return best_action
   
    # function to calulate the UCB1 exploration heuristic for a node
    def ucb1(self, node):
        if node.visits == 0:
            return float("inf")
        # otherwise, return the UCB1 heuristic for the node
        return (node.reward/node.visits) + (C/(node.depth)) * np.sqrt(np.log(node.parent.visits) / node.visits)
    
    # traverse the tree and return the state node to conduct rollout from
    def traverse(self):
        curr_node = self.root
        # print("Traversing tree...")
        # print("At root node...")
        # print("Action for root node: ", self.root.children[0].action)
        # While the current node has children or is an action node
        while curr_node.children or (curr_node.depth % 2) == 1:
            # print("Entering while loop...")
            # print("Current node depth: ", curr_node.depth)
            # Even depth nodes are state nodes, odd depth nodes are action nodes
            if curr_node.depth % 2 == 0:
                # Choose the best action node from current state
                curr_node = self.get_best_action(curr_node)
                # # Expand action node if it has no children
                # if not curr_node.children:
                #     return self.expand_action(curr_node)
            else:
                # Determine if we should branch or plunge
                branch = self.dpw_check(curr_node)
                # print("Branching? ", branch)
                # If we should branch, expand the action node
                if branch:
                    curr_node = self.expand_action(curr_node)
                    # print("Expanding Action Node...")
                else:
                    # Randomly choose a state to plunge into
                    curr_node = random.choice(curr_node.children)
                    # Expand the state node and get first action node
                    if not curr_node.children:
                        curr_node = self.expand_state(curr_node)
                    # print("Expanding State Node...")  

        # print("Current node depth: ", curr_node.depth)
        # print("Current node state: ", curr_node.state)
        # print('################################################')
        return curr_node
    

    # determine if branch or plunge returns true if branch to new state, false if plunge
    def dpw_check(self, action_node):
        branch = True
        N_action = action_node.visits
        # Select a state to plunge into
        limit = int(MAX_STATE_SPACE/action_node.depth)#(K/action_node.depth)*(N_action**ALPHA)
        if N_action > max(1,limit):
            branch = False
        return branch

    # expand the tree from an action node with one state node
    def expand_action(self, action_node):
        # Create new state node from action node
        state_node = Node(state = self.get_next_state(action_node.parent.state, action_node.action), name = "S"+str(self.state_i))
        # increase depth of state node
        state_node.depth = action_node.depth + 1
        # Add state node as child of action node
        action_node.add_child(state_node)
        # add child nodes to the corresponding parent key in the tree dictionary
        self.tree.setdefault(action_node,[]).append(state_node)
        self.state_i += 1
        return state_node
    
    # expand the tree from a state node
    def expand_state(self, state_node): 
        # Get action space
        actions = self.get_branched_actions(state_node.state, int(state_node.depth/2))
        # Create child nodes for each action
        for action in actions:
            child = Node(action = action, name = "A"+str(self.action_i))
            child.depth = state_node.depth + 1
            state_node.add_child(child)
            # add child nodes to the corresponding parent key in the tree dictionary
            self.tree.setdefault(state_node,[]).append(child)
            self.action_i += 1
        # return the first action of the children
        return state_node.children[0]

    # update the reward and visit count for a node
    def backpropogate(self, node: Node):
        curr_reward = node.reward*(DISCOUNT**node.depth)
        curr_depth = node.depth
        parent = node.parent
        while (curr_depth > 0):
            parent.update(curr_reward)
            curr_depth -= 1
            parent = parent.parent

    # def visualize(self):
    #     graph = gv.Digraph()
    #     for node in self.tree.keys():
    #         graph.node(node.name)
    #     for parent, children in self.tree.items():
    #         for child in children:
    #             graph.edge(parent.name, child.name)
    #     graph.render('./bin/tree_visual', format = 'png',view=True)


        
def simulate_dpw(tree: MCTS, t):
    # expand the root node once to get the child action nodes
    tree.expand_state(tree.root)
    for i in range(NUM_SIMS):
        # print("Simulation: ", i)
        # traverse the tree to get a state node for rollout
        rollout_node = tree.traverse()
        # rollout from the state node
        reward = tree.random_rollout(rollout_node.state, min(SIM_TIME-t,ROLL_STEPS))
        rollout_node.reward = reward
        # backpropogate the reward
        tree.backpropogate(rollout_node)
        # break if we have reached the maximum depth
        if rollout_node.depth >= MAX_DEPTH*2:
            break
    # get the best action from the root node of the tree
    best_action = tree.get_best_action(tree.root).action
    return best_action

def mcts_run(state: World):
    t = 0
    obj = MCTS(state)
    net_reward = 0
    while t < SIM_TIME:
        print("Outer loop: ",t)
        if t == 0:
            action = []
            obj.expand_state(obj.root)
        else:
            action = simulate_dpw(obj,t)
        # print(action)
        net_reward += obj.calculate_evac_reward(state, action)
        next_state = obj.get_next_state(state, action)
        state = next_state
        # print([child.name for child in obj.root.children[0].children])
        # obj.visualize()
        draw(state, 'test{}.png'.format(t), color_func_water, draw_edges=True)
        obj = MCTS(state)
        t += 1
    net_reward += obj.calculate_flood_reward(state)
    return net_reward, state.death_toll()


def RandAct(state: World, t):
    obj = MCTS(state)
    # get potential actions from given state
    action_space = obj.get_branched_actions(state, depth=0)
    # generate potential next states from each action
    if not action_space:
        return action_space
    else:
        next_states = [obj.get_next_state(state, action) for action in action_space]
    # generate random rollouts for each action
    rollout_results = [obj.random_rollout(state, t) for state in next_states]
    # determine best action index from rollout results
    best_action = action_space[rollout_results.index(max(rollout_results))]
    return best_action

def RandPolicy(state: World):
    t = 0
    obj = MCTS(state)
    net_reward = 0
    while t < SIM_TIME:
        print("Outer loop: ",t)
        action = RandAct(state, SIM_TIME - t)
        # print("action: ", action)
        net_reward += obj.calculate_evac_reward(state, action)
        next_state = obj.get_next_state(state, action)
        # draw(state, 'test{}.png'.format(t), color_func_water, draw_edges=True)
        state = next_state
        t += 1
        
    net_reward += obj.calculate_flood_reward(state)
    # return net reward and total death toll
    return net_reward, state.death_toll()

def main():
    # Create a world
    world = World(20, 20)
    # Create a MCTS object 
    if MCTS_RUN:
        reward, dead = mcts_run(world)
    else:
        reward, dead = RandPolicy(world)
    print(reward, dead)
    

if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))