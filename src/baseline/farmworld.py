import numpy as np
import json
import math
import time

GATE_COORDINATES = (3, -3)


class World:
    def __init__(self):
        self.reset()

    def reset(self):
        self.coords = (0, 0)
        self.prevCoords = (0, 0)
        self.state = (0, 0)
        self.total_reward = 0
        self.total_steps = 100
        self.sheeps = set()

        self.actions = 7
        self.prevAction = None
        self.world = np.zeros((21, 21))
        self.world_state = None
        self.shouldReturn = False
        self.holding_wheat = False

    # only allow agent to use the first 5 as actions
    def getValidActions(self):
        return [0, 1, 2, 3, 4]

    def game_status(self):
        if self.total_steps > 0:
            if self.total_reward > 200:
                return "win"
            else:
                return "playing"
        else:
            if self.total_reward > 0:
                return "win"
            else:
                return "lose"

    def observe(self):
        return self.world.reshape(-1)

    def agentInPen(self):
        x, z = self.state
        return 5 > x > 0 and -1 > z > -5

    def sheepInPen(self, x, z):
        return 6 > x > 0 and -1 > z > -5

    def returnToStart(self):
        x, z = self.state
        time.sleep(0.3)

        if self.agentInPen():
            if self.shouldReturn:
                self.shouldReturn = False
                return 5
            else:
                return 6

        if x > 9 and z < -1:
            return 3
        elif x < 8 and z > -3:
            return 2
        elif z > -3:
            return 0
        else:
            return 3

    def update_state(self, world_state, action, agent_host):
        self.total_steps -= 1
        self.world = np.zeros(self.world.shape)
        reward = -1

        if action not in self.getValidActions():
            if action == 5:
                print('Take Back Wheat')
            elif action == 6:
                print('Return To START Point')
            reward -= 500

        if world_state.number_of_observations_since_last_state > 0:
            '''首先对Agent的摆烂动作进行惩罚'''
            if action == 4:
                if self.prevAction == 4:    # 持续不断地拿出小麦，最终停留在原地
                    reward -= 200
                else:
                    reward += 50            
            elif self.coords == self.prevCoords:    # 沿着一条直线冲撞
                reward -= 200

            msg = world_state.observations[-1].text
            ob = json.loads(msg)
            self.world_state = ob

            for i in ob["entities"]:
                x = round(i["x"] - 0.5)
                z = round(i["z"] - 0.5)
                if i["name"] == "Agnis":
                    self.prevCoords = self.coords
                    self.coords = (i["x"], i["z"])
                    self.world[x][z] = 1
                    self.state = (x, z)
                elif i["name"] == "Sheep":
                    '''从Agent-Sheep的角度进行奖励函数设计'''
                    row, col = self.state
                    dist = (x-row)**2 + (z-col)**2
                    if dist <= 4:
                        #判断羊是否在羊圈中
                        if self.sheepInPen(x, z):
                            reward += 500
                        elif i["id"] not in self.sheeps:
                            self.sheeps.add(i["id"])
                            reward += 100
                            #触发作弊返回机制
                            self.shouldReturn = True
                        if action == 4:  # 靠近羊时拿出小麦
                            reward += 200
                    self.world[x][z] = 2
                    # Agent与sheep之间的动态距离惩罚
                    reward -= dist

                    '''从Sheep-Pen的角度进行奖励函数设计'''
                    dx = i["x"] - GATE_COORDINATES[0]
                    dz = i["z"] - GATE_COORDINATES[1]
                    dist2 = math.sqrt(dx**2 + dz**2)
                    #Sheep已经回到了羊圈之中
                    if dist2 < 50:
                        reward += 100
                    #Sheep没有回到羊圈时的动态惩罚
                    reward -= dist2
        self.prevAction = action
        self.total_reward += reward
        envstate = self.observe()
        status = self.game_status()
        return envstate, reward, status
