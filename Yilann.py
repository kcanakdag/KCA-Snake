import numpy as np
from collections import deque
import random
import pygame
import keyboard
import time
import math
import tensorflow as tf
import os

oldbatch = 0
newbatch = 0
Done = False
N_OF_EPISODES = 500000
batch_size = 64
output_dir = 'model_output/snakeCheckpts'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


class DQNAgent:

    def __init__(self):
        self.state_size = 5
        self.action_size = 4
        self.RewardState = deque(maxlen=128)
        self.memory = deque(maxlen=800)
        self.gamma = 0.99  # discount rate
        self.epsilon = 0.999  # exploration rate
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.001

        self.learning_rate = 0.0005

        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(30, input_dim=self.state_size))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
        model.add(tf.keras.layers.Dropout(0.15))
        model.add(tf.keras.layers.Dense(30))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
        model.add(tf.keras.layers.Dropout(0.15))
        model.add(tf.keras.layers.Dense(30))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
        model.add(tf.keras.layers.Dropout(0.15))
        model.add(tf.keras.layers.Dense(30))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
        model.add(tf.keras.layers.Dropout(0.15))

        model.add(tf.keras.layers.Dense(4))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.5))


        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))

        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def rememberApple(self, state, action, reward, next_state, done):
        self.RewardState.append((state, action, reward, next_state, done))

    def act(self, state):  # What action to take given the state
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Take a random action
        act_values = self.model.predict(state)
        # print(act_values)
        return np.argmax(act_values)

    def replay(self, batch_size):
        global oldbatch, newbatch
        minibatch = random.sample(self.memory, batch_size)
        # self.memory.reverse()
        # minibatch = self.memory
        howdied = self.memory[-1]
        minibatch.append(howdied)
        if len(self.RewardState) > 0:
            minibatch.append(self.RewardState[0])
        random.shuffle(minibatch)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)))
            target_f = self.model.predict(state)

            target_f[0][action] = target

            self.model.fit(state, target_f, epochs=2, verbose=0)

            # Decrease epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


class GameEnv:

    def __init__(self):
        self.widthWin = 800
        self.heightWin = 800
        self.rowsLines = 40
        self.length = 3
        self.xPlayer = 0
        self.yPlayer = 0
        self.xApple = 0
        self.yApple = 0
        self.reward = 0
        self.hunger = 1200
        self.DisBefore = 0
        self.sizeBetweenLines = self.widthWin // self.rowsLines
        self.sizeHead = self.sizeBetweenLines
        self.isInitial = True
        self.appleEaten = False
        self.BodyInit = True
        if self.isInitial:
            self.xHistory = []
            self.yHistory = []
            self.xPlayer = random.randrange(10, 30) * self.sizeBetweenLines
            self.yPlayer = random.randrange(10, 30) * self.sizeBetweenLines
            self.direction = random.randrange(0, 4)
            self.isInitial = False

        else:
            self.direction = GetKeyboardDirection(self)

    def getOutput(self):
        DistBtwApPl = getDistance(self.xPlayer, self.yPlayer, self.xApple, self.yApple)
        diffDist = DistBtwApPl - self.DisBefore
        self.DisBefore = DistBtwApPl
        # if diffDist > 0 and self.reward == 0:
        #     self.reward = -0.1
        # elif diffDist < 0 and self.reward == 0:  # if gets close to apple
        #     self.reward = 0.1
        UpColl, DownColl, RightColl, LeftColl = self.CheckCollisionAroundHead()

        AppleDirectionAngle = getRadianDir(self.xPlayer, self.xApple, self.yPlayer, self.yApple)
        AppleDirection = self.GetAppleDirection()
        state = [[UpColl, DownColl, RightColl, LeftColl, AppleDirection]]
        state = np.asarray(state)
        return state, self.reward

    def GetAppleDirection(self):
        appleDir = 0
        if self.xPlayer < self.xApple and self.yPlayer > self.yApple:
            appleDir = 1
        if self.xPlayer < self.xApple and self.yPlayer < self.yApple:
            appleDir = 2
        if self.xPlayer > self.xApple and self.yPlayer < self.yApple:
            appleDir = 3
        if self.xPlayer > self.xApple and self.yPlayer > self.yApple:
            appleDir = 4
        if self.xPlayer == self.xApple and self.yPlayer > self.yApple:
            appleDir = 5  # up
        if self.xPlayer == self.xApple and self.yPlayer < self.yApple:
            appleDir = 6  # down
        if self.xPlayer < self.xApple and self.yPlayer == self.yApple:
            appleDir = 7  # left
        if self.xPlayer > self.xApple and self.yPlayer == self.yApple:
            appleDir = 8  # right

        return appleDir

    def CheckCollisionAroundHead(self):
        UpColl = 0
        DownColl = 0
        LeftColl = 0
        RightColl = 0

        # UP head
        for i in range(self.length):

            try:
                if CollisionCheck(self.xPlayer, self.yPlayer - self.sizeBetweenLines, self.xHistory[-i - 2],
                                  self.yHistory[-i - 2]) or self.yPlayer == 0 or self.direction == 3:
                    UpColl = 1
                elif CollisionCheck(self.xPlayer, self.yPlayer - self.sizeBetweenLines, self.xApple,
                                  self.yApple):
                    UpColl = -1

            except:
                UpColl = 1

        # Down head

        for i in range(self.length):
            try:
                if CollisionCheck(self.xPlayer, self.yPlayer + self.sizeBetweenLines, self.xHistory[-i - 2],
                                  self.yHistory[
                                      -i - 2]) or self.yPlayer == self.widthWin - self.sizeBetweenLines or self.direction == 2:

                    DownColl = 1
                elif CollisionCheck(self.xPlayer, self.yPlayer + self.sizeBetweenLines, self.xApple,
                                  self.yApple):
                    DownColl = -1
            except:
                DownColl = 1
        # Left head

        for i in range(self.length):
            try:
                if CollisionCheck(self.xPlayer - self.sizeBetweenLines, self.yPlayer, self.xHistory[-i - 2],
                                  self.yHistory[-i - 2]) or self.xPlayer == 0 or self.direction == 0:
                    LeftColl = 1
                elif CollisionCheck(self.xPlayer - self.sizeBetweenLines, self.yPlayer, self.xApple,
                                  self.yApple):
                    LeftColl = -1
            except:
                LeftColl = 1
        # Right head

        for i in range(self.length):
            try:
                if CollisionCheck(self.xPlayer + self.sizeBetweenLines, self.yPlayer, self.xHistory[-i - 2],
                                  self.yHistory[
                                      -i - 2]) or self.xPlayer == self.widthWin - self.sizeBetweenLines or self.direction == 1:
                    RightColl = 1
                elif CollisionCheck(self.xPlayer + self.sizeBetweenLines, self.yPlayer, self.xApple,
                                  self.yApple):
                    RightColl = -1
            except:
                RightColl = 1
        return UpColl, DownColl, RightColl, LeftColl

    def getWindow(self):
        window = pygame.display.set_mode((self.widthWin, self.heightWin))
        pygame.display.set_caption("Yılan Adnan")  # title of window

        self.getPlayer(window)
        self.getApple(window)
        self.movePlayer()
        self.deathCheck()

        pygame.display.update()

    def getPlayer(self, window):
        self.getHead(window)
        self.getBody(window)

    def getHead(self, window):
        pygame.draw.rect(window, (255, 0, 0), (self.xPlayer, self.yPlayer, self.sizeHead, self.sizeHead))

    def getBody(self, window):

        if self.BodyInit:
            bodyDirWhenInit = self.direction
            for bodyPart in range(self.length):
                if bodyDirWhenInit == 0:
                    self.xHistory.append(self.xPlayer - ((bodyPart + 1) * self.sizeBetweenLines))
                    self.yHistory.append(self.yPlayer)
                if bodyDirWhenInit == 1:
                    self.xHistory.append(self.xPlayer + ((bodyPart + 1) * self.sizeBetweenLines))
                    self.yHistory.append(self.yPlayer)
                if bodyDirWhenInit == 2:
                    self.xHistory.append(self.xPlayer)
                    self.yHistory.append(self.yPlayer + ((bodyPart + 1) * self.sizeBetweenLines))
                if bodyDirWhenInit == 3:
                    self.xHistory.append(self.xPlayer)
                    self.yHistory.append(self.yPlayer - ((bodyPart + 1) * self.sizeBetweenLines))
            self.BodyInit = False
            self.xHistory.reverse()
            self.yHistory.reverse()
        if not self.BodyInit:
            self.xHistory.append(self.xPlayer)
            self.yHistory.append(self.yPlayer)
        for i in range(self.length):
            pygame.draw.rect(window, (255, 255, 255), (
                self.xHistory[-i - 2], self.yHistory[-i - 2], self.sizeBetweenLines - 1, self.sizeBetweenLines - 1))

    def movePlayer(self):
        global xPlayer, yPlayer
        if self.direction == 0 and self.xPlayer != self.widthWin - self.sizeBetweenLines:  # right
            self.xPlayer = self.xPlayer + self.sizeBetweenLines
        if self.direction == 1 and self.xPlayer != 0:  # left
            self.xPlayer = self.xPlayer - self.sizeBetweenLines
        if self.direction == 2 and self.yPlayer != 0:  # up
            self.yPlayer = self.yPlayer - self.sizeBetweenLines
        if self.direction == 3 and self.yPlayer != self.widthWin - self.sizeBetweenLines:  # down
            self.yPlayer = self.yPlayer + self.sizeBetweenLines
        self.hunger -= 1

    def getApplePos(self):
        xApple = random.randrange(0, 40) * self.sizeBetweenLines
        yApple = random.randrange(0, 40) * self.sizeBetweenLines
        for i in range(self.length):
            try:
                if CollisionCheck(xApple, yApple, self.xHistory[-i - 2], self.yHistory[-i - 2]):
                    xApple, yApple = self.getApplePos()
            except:
                xApple = random.randrange(0, 40) * self.sizeBetweenLines
                yApple = random.randrange(0, 40) * self.sizeBetweenLines
        return xApple, yApple

    def getApple(self, window):
        if not self.appleEaten:
            self.xApple, self.yApple = self.getApplePos()
            self.appleEaten = True

        if CollisionCheck(self.xApple, self.yApple, self.xPlayer, self.yPlayer):
            self.length += 1
            self.hunger += 500
            self.reward = 5
            self.xApple, self.yApple = self.getApplePos()

        pygame.draw.rect(window, (0, 255, 0), (self.xApple, self.yApple, self.sizeBetweenLines, self.sizeBetweenLines))

    def deathCheck(self):
        global Done
        DistBtwApPl = getDistance(self.xPlayer, self.yPlayer, self.xApple, self.yApple)
        #  Body Collision
        for i in range(self.length):
            try:
                if CollisionCheck(self.xHistory[-i - 2], self.yHistory[-i - 2], self.xPlayer,
                                  self.yPlayer) or CollisionCheck(self.xHistory[-1], self.yHistory[-1], self.xPlayer,
                                                                  self.yPlayer):
                    self.reward = -1
                    Done = True
            except:
                self.reward = -1
                Done = True
        if self.hunger <= 0:
            self.reward = 0
            Done = True


def getRadianDir(x1, y1, x2, y2):
    radian = (math.atan2(y2 - y1, x2 - x1))

    return radian


def CollisionCheck(x1, y1, x2, y2):
    if x1 == x2 and y1 == y2:
        return True
    else:
        return False


def getDistance(x1, y1, x2, y2):
    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist


def GetKeyboardDirection(game):
    if keyboard.is_pressed("right"):
        direction = 0

    elif keyboard.is_pressed("left"):
        direction = 1

    elif keyboard.is_pressed("up"):
        direction = 2

    elif keyboard.is_pressed("down"):
        direction = 3

    else:
        direction = game.direction

    return direction


LearningDecay = 0.999
maxLength = 0
agentGeneration = 1
agent = DQNAgent()
agent.model.load_weights(output_dir+'weights_0629.h5')
agent.epsilon = 0.001
Done = False
for e in range(N_OF_EPISODES):
    myGame = GameEnv()
    myGame.getWindow()
    state, _ = myGame.getOutput()
    myScore = 0
    Done = False

    while not Done:

        #time.sleep(0.01)
        # direction

        #direction = GetKeyboardDirection(myGame)
        direction = agent.act(state)
        # print(state)
        # print(agent.model.predict(state))
        myGame.direction = direction
        myGame.getWindow()
        next_state, reward = myGame.getOutput()
        # print(state)
        # print(agent.model.predict(state))
        if reward >= 1:
            agent.rememberApple(state, direction, reward, next_state, Done)
        elif reward <= 0:
            agent.rememberApple(state,direction,reward,next_state,Done)
        agent.remember(state, direction, reward, next_state, Done)
        myGame.reward = 0
        myScore += 1

        prevState = state
        state = next_state

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit()

        if Done:
            print('episode: {}/{}, score: {}, e:{:.2}, length : {}'.format(e, N_OF_EPISODES, myScore, agent.epsilon,
                                                                           myGame.length))
            break
    if len(agent.memory) >= batch_size:
        agentGeneration += 1
        print("Generation: {}".format(agentGeneration))
        if myGame.length > maxLength:
            maxLength = myGame.length
            agent.model.save_weights(output_dir + "weights_" + "{:04d}".format(agentGeneration-1) + ".h5")
            agent.learning_rate = agent.learning_rate * LearningDecay
            if agent.learning_rate < 0.000005:
                agent.learning_rate = 0.000005
        # agent.replay(batch_size)


print('DONE')
