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
        self.preState = (0,0)
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
            if self.total_reward > 9000:
                return "win"
            else:
                return "playing"
        else:
            if self.total_reward > 400:
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
            '''?????????Agent???????????????????????????'''
            if action == 4:
                if self.prevAction == 4:    # ???????????????????????????????????????????????????
                    reward -= 200
                else:
                    reward += 50            
            elif self.coords == self.prevCoords:    # ????????????????????????
                reward -= 200

            msg = world_state.observations[-1].text
            ob = json.loads(msg)
            self.world_state = ob

            for i in ob["entities"]:
                #???????????????????????????
                x = round(i["x"] - 0.5)
                z = round(i["z"] - 0.5)
                if i["name"] == "Agnis":
                    #???????????????????????????????????????????????????
                    self.prevCoords = self.coords
                    x_pre = round(self.prevCoords[0]-0.5)
                    z_pre = round(self.prevCoords[1]-0.5)
                    self.coords = (i["x"], i["z"])
                    self.world[x][z] = 1
                    self.state = (x, z)
                    self.preState = (x_pre, z_pre)
                elif i["name"] == "Sheep":
                    '''???Agent-Sheep?????????????????????????????????'''
                    row, col = self.state
                    row_pre, col_pre = self.preState
                    dist = (x-row)**2 + (z-col)**2
                    pre_dist = (x-row_pre)**2 + (z-col_pre)**2

                    #Temporal Differential Reward
                    if pre_dist - dist >0:
                        reward += 20 + (pre_dist-dist)
                    else:
                        reward -= 20 + (dist-pre_dist)

                    if dist <= 4:
                        #???????????????????????????
                        if self.sheepInPen(x, z):
                            reward += 500
                        elif i["id"] not in self.sheeps:
                            self.sheeps.add(i["id"])
                            reward += 100
                            #????????????????????????
                            self.shouldReturn = True
                        if action == 4:  # ????????????????????????
                            reward += 200
                    self.world[x][z] = 2
                    # Agent???sheep???????????????????????????
                    reward -= dist

                    '''???Sheep-Pen?????????????????????????????????'''
                    dx = i["x"] - GATE_COORDINATES[0]
                    dz = i["z"] - GATE_COORDINATES[1]
                    dist2 = math.sqrt(dx**2 + dz**2)
                    #Sheep???????????????????????????
                    if dist2 < 50:
                        reward += 100
                    #Sheep????????????????????????????????????
                    reward -= dist2
        self.prevAction = action
        self.total_reward += reward
        envstate = self.observe()
        status = self.game_status()
        return envstate, reward, status
