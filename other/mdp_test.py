import unittest

from mdp import Mdp, GridMdp, GridMove, ValueIterationAgent, PolicyIterationAgent, SampleQLearningAgent, best_policy

class TestSearch(unittest.TestCase):
    def test_value_iteration_values_2(self):
        mdp = GridMdp([[0,    0, 0, +1],
                       [0, None, 0, -1],
                       [0,    0, 0,  0]])
        agent = ValueIterationAgent(max_iterations=2)
        agent.learn(mdp)
        self.assertEqual(
            to_values_grid(mdp.grid, agent.values),
            [['0.00', '0.00', '0.72',  '1.00'],
             ['0.00',   None, '0.00', '-1.00'],
             ['0.00', '0.00', '0.00',  '0.00']]
        )

    def test_value_iteration_values_5(self):
        mdp = GridMdp([[0,    0, 0, +1],
                       [0, None, 0, -1],
                       [0,    0, 0,  0]])
        agent = ValueIterationAgent(max_iterations=5)
        agent.learn(mdp)
        self.assertEqual(
            to_values_grid(mdp.grid, agent.values),
            [['0.51', '0.72', '0.84', '1.00'],
             ['0.27', None, '0.55', '-1.00'],
             ['0.00', '0.22', '0.37', '0.13']]
        )

    def test_value_iteration_values_100(self):
        mdp = GridMdp([[0,    0, 0, +1],
                       [0, None, 0, -1],
                       [0,    0, 0,  0]])
        agent = ValueIterationAgent(max_iterations=100)
        agent.learn(mdp)
        self.assertEqual(
            to_values_grid(mdp.grid, agent.values),
            [['0.64', '0.74', '0.85', '1.00'],
             ['0.57', None, '0.57', '-1.00'],
             ['0.49', '0.43', '0.48', '0.28']]
        )

    def test_value_iteration_values_100_with_liveness_reward(self):
        mdp = GridMdp([[0,    0, 0, +1],
                       [0, None, 0, -1],
                       [0,    0, 0,  0]],
                       living_reward=-0.1)
        agent = ValueIterationAgent(max_iterations=100)
        agent.learn(mdp)
        self.assertEqual(
            to_values_grid(mdp.grid, agent.values),
            [['0.31', '0.51', '0.72',  '1.00'],
             ['0.15', None,   '0.36', '-1.00'],
             ['0.01', '0.01', '0.15', '-0.09']]
        )

    def test_value_iteration_q_values_100(self):
        mdp = GridMdp([[0,    0, 0, +1],
                       [0, None, 0, -1],
                       [0,    0, 0,  0]])
        agent = ValueIterationAgent(max_iterations=100)
        agent.learn(mdp)
        self.assertEqual(
            to_q_values_grid(agent, mdp, agent.values),
            [[['0.59', '0.64', '0.53', '0.57'], ['0.67', '0.74', '0.67', '0.60'], ['0.77',  '0.85', '0.57', '0.66'],                          ['1.00']],
             [['0.57', '0.51', '0.46', '0.51'],                             None, ['0.57', '-0.60', '0.30', '0.53'],                         ['-1.00']],
             [['0.49', '0.41', '0.44', '0.45'], ['0.40', '0.42', '0.40', '0.43'], ['0.48',  '0.29', '0.41', '0.40'], ['-0.65', '0.13', '0.27', '0.28']]]
        )

    def test_value_iteration_negative_rewards(self):
        mdp = GridMdp([[0,    0, 0, +1],
                       [0, None, 0, -1],
                       [0,    0, 0,  0]], 
                       living_reward=-0.04)
        agent = ValueIterationAgent(max_iterations=100)
        agent.learn(mdp)
        self.assertEqual(
            mdp.to_action_grid(best_policy(mdp, agent.values)),
            [['>', '>', '>', 'X'],
             ['^', '.', '^', 'X'],
             ['^', '>', '^', '<']]
        )

    def test_value_iteration_grid_mdp(self):
        mdp = GridMdp([[0,    0, 0, +1],
                       [0, None, 0, -1],
                       [0,    0, 0,  0]])
        agent = ValueIterationAgent(max_iterations=100)
        agent.learn(mdp)
        self.assertEqual(
            mdp.to_action_grid(best_policy(mdp, agent.values)),
            [['>', '>', '>', 'X'],
             ['^', '.', '^', 'X'],
             ['^', '<', '^', '<']]
        )

    def test_policy_iteration_grid_mdp(self):
        mdp = GridMdp([[0,    0, 0, +1],
                       [0, None, 0, -1],
                       [0,    0, 0,  0]])
        agent = PolicyIterationAgent(GridMove.TOP)
        agent.learn(mdp)
        self.assertEqual(
            mdp.to_action_grid(best_policy(mdp, agent.values)),
            [['>', '>', '>', 'X'],
             ['^', '.', '^', 'X'],
             ['^', '>', '^', '<']]
        )

    def test_policy_iteration_grid2_mdp(self):
        mdp = GridMdp([[-10, 100, -10],
                       [-10,   0, -10],
                       [-10,   0, -10],
                       [-10,   0, -10]])
        agent = PolicyIterationAgent(GridMove.TOP)
        agent.learn(mdp)
        self.assertEqual(
            mdp.to_action_grid(best_policy(mdp, agent.values)),
            [['X', 'X', 'X'],
             ['X', '^', 'X'],
             ['X', '^', 'X'],
             ['X', '^', 'X']]
        )

    def test_reinforcement_grid_agent(self):
        mdp = GridMdp([[0,    0, 0, +1],
                       [0, None, 0, -1],
                       [0,    0, 0,  0]])
        agent = SampleQLearningAgent(
            initial_state=(2, 0),
            samples=[
                [GridMove.TOP, GridMove.TOP, GridMove.RIGHT, GridMove.RIGHT, GridMove.RIGHT, GridMove.EXIT],
                [GridMove.TOP, GridMove.TOP, GridMove.RIGHT, GridMove.RIGHT, GridMove.RIGHT, GridMove.EXIT],
            ], 
            alpha=0.5,
            gamma=0.9)
        agent.learn(mdp)
        self.assertEqual(
            to_values_grid(mdp.grid, agent.values),
            [['0.00', '0.00', '0.23', '0.75'],
             ['0.00',   None,   None,   None],
             ['0.00',   None,   None,   None]]
        )

    def test_reinforcement_grid_agent2(self):
        mdp = GridMdp([[0,    0, 0, +1],
                       [0, None, 0, -1],
                       [0,    0, 0,  0]])
        agent = SampleQLearningAgent(
            initial_state=(2, 0),
            samples=[
                [GridMove.TOP, GridMove.TOP, GridMove.RIGHT, GridMove.RIGHT, GridMove.RIGHT, GridMove.EXIT],
                [GridMove.TOP, GridMove.TOP, GridMove.RIGHT, GridMove.RIGHT, GridMove.RIGHT, GridMove.EXIT],
                [GridMove.TOP, GridMove.TOP, GridMove.RIGHT, GridMove.RIGHT, GridMove.RIGHT, GridMove.EXIT],
                [GridMove.TOP, GridMove.TOP, GridMove.RIGHT, GridMove.RIGHT, GridMove.RIGHT, GridMove.EXIT],
                [GridMove.TOP, GridMove.TOP, GridMove.RIGHT, GridMove.RIGHT, GridMove.RIGHT, GridMove.EXIT],
                [GridMove.TOP, GridMove.TOP, GridMove.RIGHT, GridMove.RIGHT, GridMove.RIGHT, GridMove.EXIT],
                [GridMove.TOP, GridMove.TOP, GridMove.RIGHT, GridMove.RIGHT, GridMove.RIGHT, GridMove.EXIT],
                [GridMove.TOP, GridMove.TOP, GridMove.RIGHT, GridMove.RIGHT, GridMove.RIGHT, GridMove.EXIT],
                [GridMove.TOP, GridMove.TOP, GridMove.RIGHT, GridMove.RIGHT, GridMove.RIGHT, GridMove.EXIT],
                [GridMove.TOP, GridMove.TOP, GridMove.RIGHT, GridMove.RIGHT, GridMove.RIGHT, GridMove.EXIT],
                [GridMove.TOP, GridMove.TOP, GridMove.RIGHT, GridMove.RIGHT, GridMove.RIGHT, GridMove.EXIT],
                [GridMove.TOP, GridMove.TOP, GridMove.RIGHT, GridMove.RIGHT, GridMove.RIGHT, GridMove.EXIT],
                [GridMove.TOP, GridMove.TOP, GridMove.RIGHT, GridMove.RIGHT, GridMove.RIGHT, GridMove.EXIT],
            ], 
            alpha=0.5,
            gamma=0.9)
        agent.learn(mdp)
        self.assertEqual(
            to_values_grid(mdp.grid, agent.values),
            [['0.70', '0.80', '0.90', '1.00'],
             ['0.57',   None,   None,   None],
             ['0.42',   None,   None,   None]]
        )

    def test_reinforcement_grid_agent3(self):
        mdp = GridMdp([[0,    0, 0, +1],
                       [0, None, 0, -1],
                       [0,    0, 0,  0]])
        agent = SampleQLearningAgent(
            initial_state=(2, 0),
            samples=[
                [GridMove.TOP, GridMove.TOP, GridMove.RIGHT, GridMove.RIGHT, GridMove.RIGHT, GridMove.EXIT],
                # [GridMove.TOP, GridMove.TOP, GridMove.RIGHT, GridMove.RIGHT, GridMove.RIGHT, GridMove.EXIT],
                # [GridMove.TOP, GridMove.TOP, GridMove.RIGHT, GridMove.RIGHT, GridMove.RIGHT, GridMove.EXIT],
                # [GridMove.TOP, GridMove.TOP, GridMove.RIGHT, GridMove.RIGHT, GridMove.RIGHT, GridMove.EXIT],
                # [GridMove.TOP, GridMove.TOP, GridMove.RIGHT, GridMove.RIGHT, GridMove.RIGHT, GridMove.EXIT],
                # [GridMove.TOP, GridMove.TOP, GridMove.RIGHT, GridMove.RIGHT, GridMove.RIGHT, GridMove.EXIT],
                # [GridMove.TOP, GridMove.TOP, GridMove.RIGHT, GridMove.RIGHT, GridMove.RIGHT, GridMove.EXIT],
                # [GridMove.TOP, GridMove.TOP, GridMove.RIGHT, GridMove.RIGHT, GridMove.RIGHT, GridMove.EXIT],
                # [GridMove.TOP, GridMove.TOP, GridMove.RIGHT, GridMove.RIGHT, GridMove.RIGHT, GridMove.EXIT],
                # [GridMove.TOP, GridMove.TOP, GridMove.RIGHT, GridMove.RIGHT, GridMove.RIGHT, GridMove.EXIT],
                # [GridMove.TOP, GridMove.TOP, GridMove.RIGHT, GridMove.RIGHT, GridMove.RIGHT, GridMove.EXIT],
                # [GridMove.TOP, GridMove.TOP, GridMove.RIGHT, GridMove.RIGHT, GridMove.RIGHT, GridMove.EXIT],
                # [GridMove.TOP, GridMove.TOP, GridMove.RIGHT, GridMove.RIGHT, GridMove.RIGHT, GridMove.EXIT],
                [GridMove.TOP, GridMove.TOP, GridMove.RIGHT, GridMove.RIGHT, GridMove.BOTTOM, GridMove.RIGHT, GridMove.EXIT],
            ], 
            alpha=0.5,
            gamma=0.9)
        agent.learn(mdp)
        self.assertEqual(
            to_values_grid(mdp.grid, agent.values),
            [['0.00', '0.00', '0.00',  '0.50'],
             ['0.00',   None, '0.00', '-0.50'],
             ['0.00',   None,   None,    None]]
        )

def to_values_grid(grid, values):
    row_num, col_num = len(grid), len(grid[0])
    values_grid = [[None] * (col_num) for _ in range(row_num)]
    for row in range(row_num):
        for col in range(col_num):
            if (row, col) in values:
                values_grid[row][col] = "{:.02f}".format(values[(row, col)])

    return values_grid

def to_q_values_grid(agent, mdp, values):
    row_num, col_num = len(mdp.grid), len(mdp.grid[0])
    q_values_grid = [[None] * (col_num) for _ in range(row_num)]
    for row in range(row_num):
        for col in range(col_num):
            state = (row, col)
            if state in values:
                q_values_grid[row][col] = ["{:.02f}".format(agent.get_q_value(mdp, state, action)) for action in mdp.actions(state)]

    return q_values_grid


if __name__ == '__main__':
    unittest.main()