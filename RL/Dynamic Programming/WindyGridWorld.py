class WindyGrid:
    def __init__(self, row, col, start):
        self.row = row
        self.col = col
        self.i = start[0]
        self.j = start[1]

    def set_reward_action(self, rewards, actions, prob):
        self.rewards = rewards
        self.actions = actions
        self.prob = prob

    def set_state(self, s):
        set.i = s[0]
        set.j = s[1]

    def get_state(self):
        return (self.i, self.j)

    def move(self, a):
        s = (self.i, self.j)
        next_state_prob = self.prob[(s,a)]
        next_state = list(next_state_prob.keys())
        next_prob = list(next_state_prob.values())
        s2 = np.random.choice(next_state, p = next_prob)
        self.i, self.j =s2[0], s2[1]
        return self.rewards.get(s2,0)

    def is_terminal(self, s):
        return s not in self.actions

    def game_over(self):
        return (self.i, self.j) not in self.actions

    def all_states(self):
        return set(self.actions.keys()) | set(self.rewards.keys())
###########################
action_space = ['L','R','U','D']
def standard_grid():
    g = WindyGrid(3, 4, (2,0))
    rewards = {(0,3):1, (1,3):-1,}
    actions = {(0,0):('D','R'),
               (0,1):('L','R'),
               (0,2):('D','L','R'),
               (1,0):('U','D'),
               (1,2):('U','D','R'),
               (2,0):('U','R'),
               (2,1):('L','R'),
               (2,2):('U','L','R'),
               (2,3):('U', 'L'),}

    prob = {
    ((2, 0), 'U'): {(1, 0): 1.0},
    ((2, 0), 'D'): {(2, 0): 1.0},
    ((2, 0), 'L'): {(2, 0): 1.0},
    ((2, 0), 'R'): {(2, 1): 1.0},
    ((1, 0), 'U'): {(0, 0): 1.0},
    ((1, 0), 'D'): {(2, 0): 1.0},
    ((1, 0), 'L'): {(1, 0): 1.0},
    ((1, 0), 'R'): {(1, 0): 1.0},
    ((0, 0), 'U'): {(0, 0): 1.0},
    ((0, 0), 'D'): {(1, 0): 1.0},
    ((0, 0), 'L'): {(0, 0): 1.0},
    ((0, 0), 'R'): {(0, 1): 1.0},
    ((0, 1), 'U'): {(0, 1): 1.0},
    ((0, 1), 'D'): {(0, 1): 1.0},
    ((0, 1), 'L'): {(0, 0): 1.0},
    ((0, 1), 'R'): {(0, 2): 1.0},
    ((0, 2), 'U'): {(0, 2): 1.0},
    ((0, 2), 'D'): {(1, 2): 1.0},
    ((0, 2), 'L'): {(0, 1): 1.0},
    ((0, 2), 'R'): {(0, 3): 1.0},
    ((2, 1), 'U'): {(2, 1): 1.0},
    ((2, 1), 'D'): {(2, 1): 1.0},
    ((2, 1), 'L'): {(2, 0): 1.0},
    ((2, 1), 'R'): {(2, 2): 1.0},
    ((2, 2), 'U'): {(1, 2): 1.0},
    ((2, 2), 'D'): {(2, 2): 1.0},
    ((2, 2), 'L'): {(2, 1): 1.0},
    ((2, 2), 'R'): {(2, 3): 1.0},
    ((2, 3), 'U'): {(1, 3): 1.0},
    ((2, 3), 'D'): {(2, 3): 1.0},
    ((2, 3), 'L'): {(2, 2): 1.0},
    ((2, 3), 'R'): {(2, 3): 1.0},
    ((1, 2), 'U'): {(0, 2): 0.5, (1, 3): 0.5},
    ((1, 2), 'D'): {(2, 2): 1.0},
    ((1, 2), 'L'): {(1, 2): 1.0},
    ((1, 2), 'R'): {(1, 3): 1.0},
  }
    g.set_reward_action(rewards, actions, prob)
    return g

def print_values(V, g):
    for i in range(g.row):
        print('-----------------------')
        for j in range(g.col):
            s = (i,j)
            v = V.get(s,0)
            if v>= 0:
                print(' %.2f|'%v, end='')
            else:
                print('%.2f|'%v, end='')
        print('')

def print_policy(P, g):
    for i in range(g.row):
        print('-----------------------')
        for j in range(g.col):
            s = (i,j)
            p = P.get(s," ")
            print(' %s |'%p, end = "")
        print('')

def standard_grid_penalize(step_cost = -0.1):
    g = WindyGrid(3, 4, (2,0))
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

    prob = {
    ((2, 0), 'U'): {(1, 0): 1.0},
    ((2, 0), 'D'): {(2, 0): 1.0},
    ((2, 0), 'L'): {(2, 0): 1.0},
    ((2, 0), 'R'): {(2, 1): 1.0},
    ((1, 0), 'U'): {(0, 0): 1.0},
    ((1, 0), 'D'): {(2, 0): 1.0},
    ((1, 0), 'L'): {(1, 0): 1.0},
    ((1, 0), 'R'): {(1, 0): 1.0},
    ((0, 0), 'U'): {(0, 0): 1.0},
    ((0, 0), 'D'): {(1, 0): 1.0},
    ((0, 0), 'L'): {(0, 0): 1.0},
    ((0, 0), 'R'): {(0, 1): 1.0},
    ((0, 1), 'U'): {(0, 1): 1.0},
    ((0, 1), 'D'): {(0, 1): 1.0},
    ((0, 1), 'L'): {(0, 0): 1.0},
    ((0, 1), 'R'): {(0, 2): 1.0},
    ((0, 2), 'U'): {(0, 2): 1.0},
    ((0, 2), 'D'): {(1, 2): 1.0},
    ((0, 2), 'L'): {(0, 1): 1.0},
    ((0, 2), 'R'): {(0, 3): 1.0},
    ((2, 1), 'U'): {(2, 1): 1.0},
    ((2, 1), 'D'): {(2, 1): 1.0},
    ((2, 1), 'L'): {(2, 0): 1.0},
    ((2, 1), 'R'): {(2, 2): 1.0},
    ((2, 2), 'U'): {(1, 2): 1.0},
    ((2, 2), 'D'): {(2, 2): 1.0},
    ((2, 2), 'L'): {(2, 1): 1.0},
    ((2, 2), 'R'): {(2, 3): 1.0},
    ((2, 3), 'U'): {(1, 3): 1.0},
    ((2, 3), 'D'): {(2, 3): 1.0},
    ((2, 3), 'L'): {(2, 2): 1.0},
    ((2, 3), 'R'): {(2, 3): 1.0},
    ((1, 2), 'U'): {(0, 2): 0.5, (1, 3): 0.5},
    ((1, 2), 'D'): {(2, 2): 1.0},
    ((1, 2), 'L'): {(1, 2): 1.0},
    ((1, 2), 'R'): {(1, 3): 1.0},
  }
    g.set_reward_action(rewards, actions, prob)
    return g
