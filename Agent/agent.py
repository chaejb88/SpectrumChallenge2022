from utility.environment_interface import EnvironmentInterface
import numpy as np
import itertools
import random
import torch
from torch import nn
from torch.distributions.categorical import Categorical
from pathlib import Path

class Agent:
    def __init__(self, environment, unit_packet_success_reward, unit_packet_failure_reward, discount_factor, dnn_learning_rate):
        super(Agent, self).__init__()
        self._env = environment
        self._unit_packet_success_reward = unit_packet_success_reward
        self._unit_packet_failure_reward = unit_packet_failure_reward
        self._discount_factor = discount_factor
        self._dnn_learning_rate = dnn_learning_rate
        self._num_freq_channel = 4
        self._max_num_unit_packet = 2
        self._sta_list = list(range(0,8))
        self._observation = np.zeros(self._num_freq_channel)

        self._num_freq_channel_combination = 2 ** self._num_freq_channel - 1
        self._num_action = self._num_freq_channel_combination * self._max_num_unit_packet + 1
        self._freq_channel_combination = [np.where(np.flip(np.array(x)))[0].tolist()
                                          for x in itertools.product((0, 1), repeat=self._num_freq_channel)][1:]

        self._actor = MLP(self._num_freq_channel, self._num_action, [128])
        self._critic = MLP(self._num_freq_channel, 1, [128])

        self._eps = 1e-25
        self._mse = torch.nn.MSELoss()
        self._total_param = list(self._actor.parameters()) + list(self._critic.parameters())
        self._optimizer = torch.optim.Adam(params=self._total_param, lr=self._dnn_learning_rate)
        self._model_path = str(Path(__file__).parent.resolve() / 'my_model')

    def set_init(self):
        initial_action = {'type': 'sensing'}
        observation_dict = self._env.step(initial_action)
        self._observation = self.convert_observation_dict_to_arr(observation_dict['observation'])

    def train(self, num_episode, run_time):
        self._env.disable_video_logging()
        self._env.disable_text_logging()
        for episode in range(num_episode):
            self._env.start_simulation(time_us=run_time)
            self.set_init()
            self.make_memory()
        self.save_model()

    def test(self, run_time: int):
        # self._env.disable_video_logging()
        # self._env.disable_text_logging()
        self._env.enable_video_logging()
        self._env.enable_text_logging()
        self._env.start_simulation(time_us=run_time)
        self.set_init()
        self.load_model()
        while True:
            action, _, _ = self.get_dnn_action_and_value(self._observation)
            action_dict = self.convert_action_index_to_dict(action)
            observation_dict = self._env.step(action_dict)
            if observation_dict == 0:
                break
            self._observation = self.convert_observation_dict_to_arr(observation_dict['observation'])
            print(f"{self._env.get_score()            }\r", end='', flush=True)

    def make_memory(self):
        done = False
        while not done:
            observation_dict = self._env.random_action_step()
            if observation_dict == {}:
                return True
            self._observation = self.convert_observation_dict_to_arr(observation_dict['observation'])
            action, _, _ = self.get_dnn_action_and_value(self._observation)
            action_dict = self.convert_action_index_to_dict(action)
            observation_dict = self._env.step(action_dict)
            if observation_dict == 0:
                return True
            else:
                reward = self.get_reward(action_dict, observation_dict['observation'])
                next_observation = self.convert_observation_dict_to_arr(observation_dict['observation'])

                self._observation = torch.Tensor(self._observation)
                next_observation = torch.Tensor(next_observation)

                # compute targets
                with torch.no_grad():
                    td_target = reward + self._discount_factor * self._critic(next_observation)
                    td_error = td_target - self._critic(self._observation)

                # compute log probabilities
                dist = Categorical(logits=self._actor(self._observation))
                prob = dist.probs.gather(0, action.long())

                # compute the values of current states
                v = self._critic(self._observation)

                loss = -torch.log(prob + self._eps) * td_error + self._mse(v, td_target)
                loss = loss.mean()

                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

    def convert_observation_dict_to_arr(self, observation):
        observation_type = observation['type']
        observation_arr = np.zeros(self._num_freq_channel)
        if observation_type == 'sensing':
            is_sensed = observation['sensed_freq_channel']
            observation_arr[is_sensed] = 1
        elif observation_type == 'tx_data_packet':
            observation_arr[:] = 2
            success_freq_channel_list = observation['success_freq_channel']
            observation_arr[success_freq_channel_list] = 3
        return observation_arr

    def get_dnn_action_and_value(self, observation):
        if observation.ndim == 1:
            observation = observation[np.newaxis, ...]
        observation = torch.Tensor(observation)

        with torch.no_grad():
            logits = self._actor(observation)
            dist = Categorical(logits=logits)
            action = dist.sample()  # sample action from softmax policy
        return action, dist, logits


    def convert_action_index_to_dict(self, action_index):
        action_index = action_index[0]
        if action_index == 0:
            action_dict = {'type': 'sensing'}
        else:
            num_unit_packet = int((action_index - 1) // self._num_freq_channel_combination + 1)
            freq_channel_combination_index = (action_index - 1) % self._num_freq_channel_combination
            freq_channel_list = self._freq_channel_combination[freq_channel_combination_index]
            sta_allocation_dict = {}
            for freq_channel in freq_channel_list:
                freq_channel = int(freq_channel)
                sta = random.choice(self._sta_list)
                sta_allocation_dict[freq_channel] = sta
            action_dict = {'type': 'tx_data_packet', 'sta_allocation_dict': sta_allocation_dict,
                           'num_unit_packet': num_unit_packet}
        return action_dict

    def get_reward(self, action, observation):
        observation_type = observation['type']
        reward = 0
        if observation_type == 'sensing':
            reward = 0
        elif observation_type == 'tx_data_packet':
            num_tx_packet = len(action['sta_allocation_dict'])
            num_success_packet = len(observation['success_freq_channel'])
            num_failure_packet = num_tx_packet - num_success_packet
            reward = num_success_packet * self._unit_packet_success_reward + num_failure_packet * self._unit_packet_failure_reward
        return reward

    def save_model(self):
        save_dict = {
            "Actor": self._actor.state_dict(),
            "Critic": self._critic.state_dict()
        }
        torch.save(save_dict, self._model_path)

    def load_model(self):
        saved_dict = torch.load(self._model_path)
        self._actor.load_state_dict(saved_dict["Actor"])
        self._critic.load_state_dict(saved_dict["Critic"])


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_neurons: list = [64, 32], hidden_act: str = 'ReLU',
                 out_act: str = 'Identity'):
        super(MLP, self).__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._num_neurons = num_neurons
        self._hidden_act = getattr(nn, hidden_act)()
        self._out_act = getattr(nn, out_act)()

        input_dims = [input_dim] + num_neurons
        output_dims = num_neurons + [output_dim]

        self._layers = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(input_dims, output_dims)):
            is_last = True if i == len(input_dims) - 1 else False
            self._layers.append(nn.Linear(in_dim, out_dim))
            if is_last:
                self._layers.append(self._out_act)
            else:
                self._layers.append(self._hidden_act)

    def forward(self, xs):
        for layer in self._layers:
            xs = layer(xs)
        return xs


if __name__ == "__main__":
    env = EnvironmentInterface()
    env.connect()
    agent = Agent(environment=env, unit_packet_success_reward=10, unit_packet_failure_reward=-40, discount_factor=0.9,
                  dnn_learning_rate=0.00005)

    # When submitting, the training part should be excluded,
    # and it should be submitted in a form that can be evaluated by loading the trained model like agent.test()
    agent.train(num_episode=10, run_time=100000)
    agent.test(100000)
