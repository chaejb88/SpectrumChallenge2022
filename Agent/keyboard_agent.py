from utility.environment_interface import EnvironmentInterface


class KeyboardAgent:
    def __init__(self, environment: EnvironmentInterface):
        self._env = environment
        self._freq_channel_list = []
        self._sta_list = []
        self._num_unit_packet_list = []
        self._num_freq_channel = 0
        self._num_sta = 0

    def run(self, run_time):
        self._env.start_simulation(time_us=run_time)
        self._freq_channel_list = self._env.freq_channel_list
        self._num_freq_channel = len(self._freq_channel_list)
        self._sta_list = self._env.sta_list
        self._num_sta = len(self._sta_list)
        self._num_unit_packet_list = self._env.num_unit_packet_list

        while True:
            action = self.keyboard_control()
            if 'num_step' in action:
                self._env.random_action_step(num_step=action['num_step'])
            else:
                print(f"Action: {action}")
                obs_rew = self._env.step(action=action)
                print(f"Observation: {obs_rew['observation']}")
                if 'reward' in obs_rew:
                    print("Reward: ", obs_rew['reward'])
                print(f"Score: {self._env.get_score()}\n")

    def keyboard_control(self):
        while True:
            action_type = input('Choose action (sensing: s, transmit: t, random: r): ')
            if action_type.lower() in ['s', 't', 'r']:
                break
        if action_type.lower() == 's':
            action = {'type': 'sensing'}
            return action
        elif action_type.lower() == 't':
            sta_allocation_dict = {}
            for ch in self._freq_channel_list:
                while True:
                    sta = eval(input(f'Choose a station among {self._sta_list} for channel {ch} '
                                     f'(choose -1 for not sending on this channel): '))
                    if (sta in self._sta_list) or sta == -1:
                        break
                if sta != -1:
                    sta_allocation_dict[ch] = sta
            while True:
                num_unit_packet = eval(input(f'Choose the number of unit packets '
                                             f'among {self._num_unit_packet_list}: '))
                if num_unit_packet in self._num_unit_packet_list:
                    break
            action = {'type': 'tx_data_packet', 'sta_allocation_dict': sta_allocation_dict,
                      'num_unit_packet': num_unit_packet}
            return action
        elif action_type.lower() == 'r':
            while True:
                num_step = int(input('Input number of steps: '))
                if num_step > 0:
                    break
            return {'num_step': num_step}


if __name__ == "__main__":
    env = EnvironmentInterface()
    env.connect()
    agent = KeyboardAgent(environment=env)
    agent.run(100000)
