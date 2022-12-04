# baselineTeam.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import contest.util as util

from contest.captureAgents import CaptureAgent
from contest.game import Directions
from contest.util import nearestPoint

# OUR IMPORTS
import os
import pickle
import sys


#################
# Team creation #
#################


def create_team(first_index, second_index, is_red,
                first='OffensiveQ', second='DefensiveReflexAgent', num_training=0):

    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}

class QCaptureAgent(CaptureAgent):

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

        self.alpha = 0.6
        self.discount = 0.9
        self.epsilon = 0.2

        self.qValues = util.Counter() 

        self.filename = "offensiveTraining.pkl"
        self.loadQValues(self.filename)

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        return self.qValues[(self.compute_position(state), action)]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """

        legalActions = state.get_legal_actions(self.index)

        policy = -99999.9               # -inf as we want to get max

        if len(legalActions) == 0:      # if no legal action, terminal state
            return 0.0

        for action in legalActions:
            policy = max(self.getQValue(state, action), policy)     # get maximum q value for state

        return policy

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """

        legalActions = state.get_legal_actions(self.index)
        actions = []

        if len(legalActions) == 0:      # if no legal action, terminal state
            return None
        
        value = self.getValue(state)

        for action in legalActions:
            if value == self.getQValue(state, action):  # if q value is equal to the q value for best action
                return action

    def choose_action(self, state):
        """
        Compute the action to take in the current state.  With
        probability self.epsilon, we should take a random action and
        take the best policy action otherwise.  Note that if there are
        no legal actions, which is the case at the terminal state, you
        should choose None as the action.
        HINT: You might want to use util.flipCoin(prob)
        HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = state.get_legal_actions(self.index)
        action = None

        if len(legalActions) == 0:    # no legal actions
            return None

        expProb = self.epsilon        # probability for random action

        if util.flipCoin(expProb):                  # chooses random actions an epsilon fraction of the time
            action = random.choice(legalActions)      # random action (not suboptimal)

        else:
            action = self.computeActionFromQValues(state)     # current best Q-value, current optimal

        # train de model
        nextState = self.get_successor(state, action)
        self.update(state, action, nextState)
        
        return action


    def update(self, state, action, nextState):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here
        """
        # implementation of q learning

        qState = self.qValues[(self.compute_position(state), action)]

        qNextState = self.computeValueFromQValues(nextState)

        alpha = self.alpha          # learning rate
        discount = self.discount    # discount
        reward = self.reward(state, nextState)

        newQ = qState + alpha * (reward + discount * qNextState - qState)

        self.qValues[(self.compute_position(state), action)] = newQ

        # update training file
        with open('./agents/edithAndCristina/offensiveTraining.pkl', 'wb') as f:       # THIS WORKS
            pickle.dump(self.qValues, f)

    def loadQValues(self, filename):

        with open('./agents/edithAndCristina/offensiveTraining.pkl', 'rb') as f:
            self.qValues = pickle.load(f)

    def compute_position(self, game_state):
        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()
        return my_pos

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)

    # OTHER FUNCTIONS

    def distToFood(self, gameState):
        # returns the nearest distance to opponents food
        foodList = self.get_food(gameState).as_list()

        if len(foodList) != 0:
            dist = 9999
            for food in foodList:
                tempDist = self.get_maze_distance(gameState.get_agent_position(self.index), food)

                if tempDist < dist:
                    dist = tempDist
                    temp = food

            return dist
        
        else:
            return 0

    def distToInvader(self, state):
        # returns the nearest distance to opponent pacmans
        myState = state.get_agent_state(self.index)
        myPos = myState.get_position()

        enemies = [state.get_agent_state(i) for i in self.get_opponents(state)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() != None]


        if len(invaders) != 0:
            dist = 9999

            for a in invaders:
                tempDist = self.get_maze_distance(myPos, a.get_position())

                if tempDist < dist:
                    dist = tempDist
                    temp = a

            return dist
        
        else:
            return 0


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  # self.getScore(successor)

        # Compute distance to the nearest food

        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'distance_to_food': -1}

class OffensiveQ(QCaptureAgent):

    def reward(self, game_state, nextState):
        reward = 0

        myPos = game_state.get_agent_position(self.index)

        # check if score already updated
        if self.get_score(nextState) > self.get_score(game_state):
            diff = self.get_score(nextState) - self.get_score(game_state)
            reward = diff * 10
        
        # check if food eaten in nextState
        foodList = self.get_food(game_state).as_list()
        distToFood = min([self.get_maze_distance(myPos, food) for food in foodList])

        #distance to food reward
        reward -= distToFood * 10

        #if 1 step away, will I be able to eat it?
        if distToFood == 1:
            nextFoods = self.get_food(nextState).as_list()
            if len(foodList) - len(nextFoods) == 1:     # we have been able to eat it
                reward += 10

        # check if capsule eaten in nextState
        capsulesList = self.get_capsules(game_state)
        distToCapsules = min([self.get_maze_distance(myPos, food) for food in foodList])

        # distance to capsule reward
        reward -= distToCapsules

        # if 1 step away, will I be able to eat it?
        if distToCapsules == 1:
            nextCapsules = self.get_capsules(nextState)
            if len(capsulesList) - len(nextCapsules) == 1:     # we have been able to eat it
                reward += 20

        # check if I am eaten in nextState
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        ghosts = [a for a in enemies if not a.is_pacman and a.get_position() != None]

        if len(ghosts) > 0:
            minDistGhost = min([self.get_maze_distance(myPos, g.get_position()) for g in ghosts])

            #distance to enemies reward
            if minDistGhost < 5:
                reward += minDistGhost

            #I die in the next state
            if minDistGhost == 1:
                nextPos = nextState.get_agent_position(self.index)
                if nextPos == self.start:
                    reward = -100

        # dist to other team side
        
        # 
        self.midWidth = game_state.data.layout.width / 2
        self.height = game_state.data.layout.height
        self.width = game_state.data.layout.width
        self.midHeight = game_state.data.layout.height / 2
        # 
        
        endMap = []

        if self.red:
            i = self.midWidth + 1
        else:
            i = self.midWidth - 1
        
        for j in range(self.height):
            endMap.append((i,j))

        validPositions = []

        for i, j in endMap:
            if not game_state.has_wall(int(i),int(j)):
                validPositions.append((i, j))

        dist = 9999
        for validPos in validPositions:
            tempDist = self.get_maze_distance(validPos, myPos)

            if tempDist < dist:
                dist = tempDist

        reward -= dist

        # if we have food go back
        foodLeft = len(self.get_food(game_state).as_list())

        if foodLeft < 20:
            distInit = self.get_maze_distance(myPos, self.start)
            reward -= distInit

        return reward

    
class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}

class DefensiveQ(QCaptureAgent):

    def dist(self, game_state, nextState):
        reward = 0

        agentPosition = game_state.get_agent_position(self.index)
        myFoods = self.get_food(game_state).as_list()
        distToFood = min(self.get_maze_distance(agentPosition, food) for food in myFoods)
        reward = -distToFood

        return reward

    def reward(self, game_state, nextState):
        reward = 0
        distance = self.distToInvader(game_state) 
        reward = -distance
        return reward