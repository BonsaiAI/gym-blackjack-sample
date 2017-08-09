import gym
import time
from functools import reduce

import bonsai
from bonsai_gym_common import GymSimulator, logging_basic_config

ENVIRONMENT = 'Blackjack-v0'
RECORD_PATH = None
SKIPPED_FRAME = 1
RECORDING_TIME = 40*60*60


class BlackJackSimulator(GymSimulator):

    def __init__(self, env, skip_frame, record_path):
        GymSimulator.__init__(
            self, env, skip_frame=skip_frame,
            record_path=record_path)
        self._render_env = False

    def advance(self, actions):
        # Step 0: Check if we need to reset or move forward the episode.
        if self._terminal:
            self.reset()
            return

        # Step 1: Perform the action and update the game along with
        # the reward.
        average_reward = 0
        for i in range(self._skip_frame):
            observation, reward, done, info = self.env.step(
                self.get_gym_action(actions))
            self._frame_count += 1
            average_reward += int(reward)
            self.gym_total_reward += int(reward)

            # Step 2: Render the game.
            if self._render_env:
                self.env.render()

            if done:
                break
        self._reward = average_reward / (i + 1)

        time_from_start = time.time() - self._start_time
        if self._is_recording and (time_from_start > RECORDING_TIME):
            self.env.monitor.close()

        # Step 3: Get the current frames, and append it to deque to get current
        # state.
        current_frame = self.process_observation(observation)
        self._append_state(current_frame)

        # Step 4: Check if we should reset
        self._check_terminal(done)

    def get_state(self):
        # We append all of the states in the deque by adding the rows.
        current_state = reduce(
            lambda accum, state: accum + state, self._state_deque)

        # Convert the observation to an inkling schema.
        state_schema = self.get_state_schema(current_state)
        state_dict = {"current_sum": int(state_schema[0]),
                      "dealer_card": int(state_schema[1]),
                      "usable_ace": int(state_schema[2])}
        return bonsai.simulator.SimState(state_dict, self._terminal)


if __name__ == "__main__":
    logging_basic_config()
    env = gym.make(ENVIRONMENT)
    simulator = BlackJackSimulator(env, SKIPPED_FRAME, RECORD_PATH)
    bonsai.run_for_training_or_prediction("blackjack_simulator", simulator)
