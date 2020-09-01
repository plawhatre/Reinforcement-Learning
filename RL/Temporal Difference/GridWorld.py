class Grid:
    def __init__(self, row, col, start):
        self.row = row
        self.col = col
        self.i = start[0]
        self.j = start[1]
    
    def set_reward_action(self, rewards, actions):
        self.rewards = rewards
        self.actions = actions
    
    def set_state(self, s):
        self.i = s[0]
        self.j = s[1]
    
    def get_state(self):
        return (self.i, self.j)
      
    def get_next_state(self, s, a):
        i, j = s[0], s[1]
        if a in self.actions[(i,j)]:
            if a == 'U':
                i -= 1
            elif a == 'D':
                i += 1
            elif a == 'L':
                j -= 1
            elif a == 'R':
                j += 1
        return i,j
    
    def move(self, a):
        if a in self.actions[(self.i, self.j)]:
            if a == 'U':
                self.i -= 1
            elif a == 'D':
                self.i += 1
            elif a == 'L':
                self.j -= 1
            elif a == 'R':
                self.j += 1
        return self.rewards.get((self.i, self.j), 0)
    
    def undo_move(self, a):
        if a == 'U':
            self.i += 1
        elif a == 'D':
            self.i -= 1
        elif a == 'L':
            self.j += 1
        elif a == 'R':
            self.j -= 1
        # assert is for sanity checking and raises and error if condition is not satisfied
        assert(self.get_state() in self.all_state())
        
    def is_terminal(self,s):
        return s not in self.actions

    def game_over(self):
        return (self.i, self.j) not in self.actions
    
    def all_states(self):
        return set(self.actions.keys()) | set(self.rewards.keys())
##########################
action_space = ['L','R','U','D']
def standard_grid():
    g = Grid(3,4, (2,0))
    rewards = {(0,3):1,(1,3):-1}
    actions = {(0,0):('D','R'),
               (0,1):('L','R'),
               (0,2):('D','L','R'),
               (1,0):('U','D'),
               (1,2):('U','D','R'),
               (2,0):('U','R'),
               (2,1):('L','R'),
               (2,2):('U','L','R'),
               (2,3):('U', 'L'),}
    g.set_reward_action(rewards,actions)
    return g

def standard_neg_grid(step_cost = -0.1):
    g = Grid(3,4, (2,0))
    rewards = {(0,3):1,
    (0,0):step_cost,
    (0,1):step_cost,
    (0,2):step_cost,
    (1,0):step_cost,
    (1,2):step_cost,
    (2,0):step_cost,
    (2,1):step_cost,
    (2,2):step_cost,
    (2,3):step_cost,
    (1,3):-1,}
    actions = {(0,0):('D','R'),
               (0,1):('L','R'),
               (0,2):('D','L','R'),
               (1,0):('U','D'),
               (1,2):('U','D','R'),
               (2,0):('U','R'),
               (2,1):('L','R'),
               (2,2):('U','L','R'),
               (2,3):('U', 'L'),}
    g.set_reward_action(rewards,actions)
    return g

def print_values(V,g):
    for i in range(g.row):
        print('--------------------------------------')
        for j in range(g.col):
            v = V.get((i,j),0)
            if v >=0:
                print(' %.2f|'%v, end="")
            else:
                print('%.2f|'%v, end ="")
        print('')

def print_policy(P,g):
    for i in range(g.row):
        print('--------------------------------------')
        for j in range(g.col):
            p =P.get((i,j)," ")
            print(' %s |'%p, end='')
        print('')
