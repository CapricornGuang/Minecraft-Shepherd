from __future__ import print_function
import os
import sys
import time
import datetime
import json
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.advanced_activations import PReLU

# Exploration factor
epsilon = 0.1


class Experience(object):
    def __init__(self, model, max_memory=100, discount=0.95):
        self.model = model
        self.max_memory = max_memory
        self.discount = discount
        self.memory = list()
        self.num_actions = model.output_shape[-1]

    def remember(self, episode):
        self.memory.append(episode)
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def predict(self, envstate):
        #防止卷积操作与全连接冲突，先进行判断
        if envstate.ndim == 1:
            envstate = np.array([envstate])
        return self.model.predict(envstate.reshape(-1,21,21,1))[0]

    def get_data(self, data_size=10):
        # envstate 1d size (1st element of episode
        env_size = self.memory[0][0].size
        mem_size = len(self.memory)
        data_size = min(mem_size, data_size)
        inputs = np.zeros((data_size, env_size))
        targets = np.zeros((data_size, self.num_actions))
        for i, j in enumerate(np.random.choice(range(mem_size), data_size, replace=False)):
            envstate, action, reward, envstate_next, game_over = self.memory[j]
            inputs[i] = envstate
            targets[i] = self.predict(envstate)
            #选择价值最大的动作
            Q_sa = np.max(self.predict(envstate_next))
            '''DQN核心代码'''
            if game_over:
                targets[i, action] = reward
            else:
                targets[i, action] = reward + self.discount * Q_sa
        return inputs, targets


def qtrain(model, world):
    n_epoch = 15000
    max_memory = 1000
    data_size = 50

    experience = Experience(model, max_memory=max_memory)
    win_history = []
    win_rate = 0.0
    epsilon = 0.3
    hsize = world.world.size // 4
    start_time = datetime.datetime.now()
    for epoch in range(n_epoch):
        loss = 0.0
        envstate = world.observe()
        n_episodes = 0
        world.reset()
        game_over = False

        while not game_over:
            prev_envstate = envstate
            '''应用epsilon-greddy策略进行策略搜索，保证搜索宽度'''
            if np.random.rand() < epsilon:
                action = random.choice(actions)
            else:
                action = np.argmax(experience.predict(prev_envstate))
            envstate, reward, game_status = world.act(action)
            '''为了避免没有意义的时间浪费在reward满足一定条件时（farmword）终止'''
            if game_status == 'win':
                win_history.append(1)
                game_over = True
            elif game_status == 'lose':
                win_history.append(0)
                game_over = True
            else:
                game_over = False
            #模型训练与策略更新
            episode = [prev_envstate, action, reward, envstate, game_over]
            experience.remember(episode)
            n_episodes += 1
            inputs, targets = experience.get_data(data_size=data_size)
            h = model.fit(
                inputs.reshape(-1,21,21,1),
                targets,
                epochs=8,
                batch_size=16,
                verbose=0,
            )
            loss = model.evaluate(inputs.reshape(-1,21,21,1), targets, verbose=0)
        if len(win_history) > hsize:
            win_rate = sum(win_history[-hsize:]) / hsize
        dt = datetime.datetime.now() - start_time
        t = format_time(dt.total_seconds())
        template = "Epoch: {:03d}/{:d} | Loss: {:.4f} | Episodes: {:d} | Win count: {:d} | Win rate: {:.3f} | time: {}"
        print(template.format(epoch, n_epoch-1, loss, n_episodes, sum(win_history), win_rate, t))
        # we simply check if training has exhausted all free cells and if in all
        # cases the agent won
        if win_rate > 0.9:
            epsilon = 0.05
        if sum(win_history[-hsize:]) == hsize and completion_check(model, qmaze):
            print("Reached 100%% win rate at epoch: %d" % (epoch,))
            break

    '''模型文件保存'''
    h5file = "model" + ".h5"
    json_file = "model" + ".json"
    model.save_weights(h5file, overwrite=True)
    with open(json_file, "w") as outfile:
        json.dump(model.to_json(), outfile)
    end_time = datetime.datetime.now()
    dt = datetime.datetime.now() - start_time
    seconds = dt.total_seconds()
    t = format_time(seconds)
    print('files: %s, %s' % (h5file, json_file))
    print("n_epoch: %d, max_mem: %d, data: %d, time: %s" %
          (epoch, max_memory, data_size, t))


def setupMission():
    mission_file = './farm.xml'
    global my_mission, my_mission_record
    with open(mission_file, 'r') as f:
        print("Loading mission from %s" % mission_file)
        mission_xml = f.read()
        my_mission = MalmoPython.MissionSpec(mission_xml, True)

    my_mission_record = MalmoPython.MissionRecordSpec()


# Attempt to start a mission:
def startMission():
    max_retries = 3
    for retry in range(max_retries):
        try:
            agent_host.startMission(my_mission, my_mission_record)
            break
        except RuntimeError as e:
            if retry == max_retries - 1:
                print("Error starting mission:", e)
                exit(1)
            else:
                time.sleep(2)

# Loop until mission starts:


def waitUntilMissionStart():
    print("Waiting for the mission to start ", end=' ')
    world_state = agent_host.getWorldState()
    while not world_state.has_mission_begun:
        print(".", end="")
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:", error.text)

    print()
    print("Mission running ", end=' ')


def missionLoop(model, world):
    world_state = agent_host.getWorldState()
    my_agent = MyAgent(world_state)
    while world_state.is_mission_running:
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
        my_agent.updateWorldState(world_state)
        if my_agent.takeAction():
            print(my_agent.takeAction())
            agent_host.sendCommand(my_agent.takeAction())
        for error in world_state.errors:
            print("Error:", error.text)
    print()
    print("Mission ended")
# Mission has ended.


if __name__ == "__main__":
    world = World()
    model = build_model(world.world)
    setupMission()
    startMission()
    waitUntilMissionStart()
    missionLoop(model, world)

if __name__ == "__main__":
    world = World()
    model = build_model(world.world)
    qtrain(model, world)
