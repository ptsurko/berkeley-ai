import unittest

from mdp import CarRacingMdp, GridMdp, GridMove, value_iteration, policy_iteration, best_policy


class TestSearch(unittest.TestCase):
    # def test_car_mdp(self):
    #
    #     t = {
    #         "Cool": {
    #             "Slow": {
    #                 "Cool1": (1.0, 1),
    #             },
    #             "Fast": {
    #                 "Cool1": (0.5, 2),
    #                 "Warm1": (0.5, 2),
    #             },
    #         },
    #         "Warm": {
    #             "Slow": {
    #                 "Cool1": (0.5, 1),
    #                 "Warm1": (0.5, 1),
    #             },
    #             "Fast": {
    #                 "Overheated": (1.0, -10),
    #             },
    #         },
    #         "Cool1": {
    #             "Slow": {
    #                 "Cool2": (1.0, 1),
    #             },
    #             "Fast": {
    #                 "Cool2": (0.5, 2),
    #                 "Warm2": (0.5, 2),
    #             },
    #         },
    #         "Warm1": {
    #             "Slow": {
    #                 "Cool2": (0.5, 1),
    #                 "Warm2": (0.5, 1),
    #             },
    #             "Fast": {
    #                 "Overheated": (1.0, -10),
    #             },
    #         },
    #         "Cool2": {
    #             "exit": {
    #                 "End": (1, 0),
    #             }
    #         },
    #         "Warm2": {
    #             "exit": {
    #                 "End": (1, 0),
    #             }
    #         },
    #         "Overheated": {},
    #         "End": {}
    #     }
    #     mdp = CarRacingMdp(t)
    #     values = value_iteration(mdp)
    #
    # def test_car_mdp2(self):
    #
    #     t = {
    #         "Cool": {
    #             "Slow": {
    #                 "Cool": (1.0, 1),
    #             },
    #             "Fast": {
    #                 "Cool": (0.5, 2),
    #                 "Warm": (0.5, 2),
    #             },
    #         },
    #         "Warm": {
    #             "Slow": {
    #                 "Cool": (0.5, 1),
    #                 "Warm": (0.5, 1),
    #             },
    #             "Fast": {
    #                 "Overheated": (1.0, -10),
    #             },
    #         },
    #         "Overheated": {},
    #     }
    #     mdp = CarRacingMdp(t, terminals=["Overheated"], gamma=0.8)
    #     values = value_iteration(mdp)

    def test_grid_mdp(self):
        mdp = GridMdp([[-0.04, -0.04, -0.04, +1],
                       [-0.04, None, -0.04, -1],
                       [-0.04, -0.04, -0.04, -0.04]],
                      terminals=[(0, 3), (1, 3)], gamma=0.9)
        values = value_iteration(mdp)
        self.assertEqual(
            mdp.to_action_grid(best_policy(mdp, values)),
            [['>', '>', '>', '.'],
             ['^', '.', '^', '.'],
             ['^', '>', '^', '<']]
        )

    def test_value_iteration_grid_mdp(self):
        mdp = GridMdp([[0, 0, 0, +1],
                       [0, None, 0, -1],
                       [0, 0, 0, 0]],
                      terminals=[(0, 3), (1, 3)], gamma=0.9)
        values = value_iteration(mdp)
        self.assertEqual(
            mdp.to_action_grid(best_policy(mdp, values)),
            [['>', '>', '>', '.'],
             ['^', '.', '^', '.'],
             ['^', '<', '^', '<']]
        )

    def test_policy_iteration_grid_mdp(self):
        mdp = GridMdp([[0, 0, 0, +1],
                       [0, None, 0, -1],
                       [0, 0, 0, 0]],
                      terminals=[(0, 3), (1, 3)],
                      gamma=0.9)
        values = policy_iteration(mdp, GridMove.TOP)
        self.assertEqual(
            mdp.to_action_grid(best_policy(mdp, values)),
            [['>', '>', '>', '.'],
             ['^', '.', '^', '.'],
             ['^', '<', '^', '<']]
        )

    def test_policy_iteration_grid2_mdp(self):
        mdp = GridMdp([[-10, 100, -10],
                       [-10,   0, -10],
                       [-10,   0, -10],
                       [-10,   0, -10]],
                      terminals=[(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 2), (3, 0), (3, 2)],
                      gamma=0.9)
        values = policy_iteration(mdp, GridMove.TOP)
        self.assertEqual(
            mdp.to_action_grid(best_policy(mdp, values)),
            [['.', '.', '.'],
             ['.', '^', '.'],
             ['.', '^', '.'],
             ['.', '^', '.']]
        )


if __name__ == '__main__':
    unittest.main()