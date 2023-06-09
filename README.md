# A Shepherd？DQN is All You Need!

## Task Analysis

The goal of this task is to train the Agent to lure sheep into the circle with objects. The Agent doesn't know the reward that his behavior in the world can get. The Agent can access the detailed 2D world grid, which specifies the location of each animal. The Agent will select the action to be performed according to the knowledge learned so far. These actions include the movement of the Agent (north, south, east, west) or taking out the wheat in the storage bar of the Agent. Successfully reaching the sheep will get a positive reward, while moving will get a negative reward (steps taken), and the negative gain will increase with the increase of the distance between the Agent and the sheep. The ultimate goal is to train an Agent that can lure more sheep into the sheepfold in the shortest time. 

## What is the world Like? 

This research is based on MalmoPython virtual experiment environment provided by MOJANG, and Minecraft carries out simulation experiments. The schematic diagram of our experimental environment is as follows

<p align="center">
<img src=".\.img/uTools_1641713915176.png" height = "200" alt="" align=center />
<br><br>
<b>Figure 1.</b> Virtual Shepherding Environment.
</p>

## How To Empoly DQN?

We mainly use Q-learning algorithm to solve this problem. Through Q-learning, we learn the value of **seven actions** corresponding to each state, and select the action with the highest value among the seven actions to execute. After getting a new state feedback from the environment,we  make the next round of action selection, and so on.
<p align="center">
<img src=".\.img/uTools_1641713987925.png" height = "200" alt="" align=center />
<br><br>
<b>Figure 1.</b> The DQN framework.
</p>
<p align="center">
<img src=".\.img/uTools_1641714117158.png" height = "200" alt="" align=center />
<br><br>
<b>Figure 1.</b> The illustration of Diviner framework.
</p>

