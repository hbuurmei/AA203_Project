import unittest
import numpy as np
from environment import WildFireEnv

class TestWildFireEnv(unittest.TestCase):
    def setUp(self):
        # Set up a test environment
        width, height = 10, 10
        N_agents, N_sats = 1, 0
        action_range = 5
        p_move = 1
        max_temp = 400
        init_positions = np.zeros((N_agents + N_sats, 2))
        init_mu = np.zeros((1, 2))
        init_sigma = np.array([[width/2, 0], [0, height/2]])
        init_state = np.vstack((init_positions, init_mu, init_sigma))
        max_steps = 100
        self.env = WildFireEnv(width, height, init_state, action_range, p_move, max_temp, N_agents, N_sats, max_steps)

    def test_get_temperatures(self):
        # Test that get_temperatures returns an array with the correct shape
        temperatures = self.env.get_temperatures(np.zeros((1, 2)))
        self.assertEqual(temperatures.shape, (1,))

    def test_step(self):
        # Test that step returns the correct observation, reward, and done values
        new_state, reward, done = self.env.step(action=24)
        self.assertEqual(new_state.shape, (8,))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)

    def test_reset(self):
        # Test that reset returns the state to the initial state
        self.env.reset()
        self.assertTrue(np.array_equal(self.env.state, self.env.init_state))

    def test_flatten_state(self):
        # Test that flatten_state returns a flattened array with the correct shape
        state = np.zeros((3, 2))
        flattened_state = self.env.flatten_state(state)
        self.assertEqual(flattened_state.shape, (6,))


if __name__ == '__main__':
    unittest.main()