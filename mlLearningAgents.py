# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
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

# This template was originally adapted to KCL by Simon Parsons, but then
# revised and updated to Py3 for the 2022 course by Dylan Cope and Lin Li

from __future__ import absolute_import
from __future__ import print_function

import random

from pacman import Directions, GameState
from pacman_utils.game import Agent
from pacman_utils import util
from pacman_utils.util import manhattanDistance

class GameStateFeatures:
    """
    Wrapper class around a game state where you can extract
    useful information for your Q-learning algorithm

    WARNING: We will use this class to test your code, but the functionality
    of this class will not be tested itself
    """

    def __init__(self, state: GameState):
        """
        Initialize the GameStateFeatures object by extracting and storing
        relevant features from the provided GameState object.
        """
        # Example feature extraction (you'll need to customize these based on your assignment's requirements):
        self.pacmanPosition = state.getPacmanPosition()
        self.ghostPositions = state.getGhostPositions()
        self.numFood = state.getNumFood()
        self.foodPositions = [food for food in state.getFood().asList()]
        self.legalActions = state.getLegalPacmanActions()

    def getLegalPacmanActions(self):
        return self.legalActions

class QLearnAgent(Agent):

    def __init__(self,
                 alpha: float = 0.30,
                 epsilon: float = 0.1,
                 gamma: float = 0.8,
                 maxAttempts: int = 30,
                 numTraining: int = 30):
        """
        These values are either passed from the command line (using -a alpha=0.5,...)
        or are set to the default values above.

        The given hyperparameters are suggestions and are not necessarily optimal
        so feel free to experiment with them.

        Args:
            alpha: learning rate
            epsilon: exploration rate
            gamma: discount factor
            maxAttempts: How many times to try each action in each state
            numTraining: number of training episodes
        """
        super().__init__()
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.maxAttempts = int(maxAttempts)
        self.numTraining = int(numTraining)
        self.lastState = None
        self.lastAction = None
        self.lastStateFeatures = None
        # Count the number of games we have played
        self.episodesSoFar = 0
        self.qValues = {}  # Initialize qValues as an empty dictionary
        self.counts = {}  # Initialize counts as an empty dictionary if you plan to use it

    # Accessor functions for the variable episodesSoFar controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar += 1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
        return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value: float):
        self.epsilon = value

    def getAlpha(self) -> float:
        return self.alpha

    def setAlpha(self, value: float):
        self.alpha = value

    def getGamma(self) -> float:
        return self.gamma

    def getMaxAttempts(self) -> int:
        return self.maxAttempts

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    @staticmethod

    def computeReward(startState: GameState, endState: GameState) -> float:
        reward = 0

        # Reward for eating food
        if endState.getNumFood() < startState.getNumFood():
            reward += 250

        # Penalty for getting caught by a ghost
        if endState.isLose():
            reward -= 500

        # Reward for winning the game
        if endState.isWin():
            reward += 1000

        # Consider distance to the nearest food as an incentive to move closer
        foodGrid = endState.getFood()
        pacmanPos = endState.getPacmanPosition()
        foodPositions = foodGrid.asList()
        if foodPositions:
            distancesToFood = [manhattanDistance(pacmanPos, foodPos) for foodPos in foodPositions]
            nearestFoodDistance = min(distancesToFood)
            reward += max(10 - nearestFoodDistance, 0)  # Encourage Pacman to get closer to the nearest food

        # Penalty based on proximity to ghosts to encourage avoidance
        ghostStates = endState.getGhostStates()
        ghostPositions = [ghost.getPosition() for ghost in ghostStates if ghost.scaredTimer == 0]  # Only consider non-scared ghosts
        if ghostPositions:
            distancesToGhosts = [manhattanDistance(pacmanPos, ghostPos) for ghostPos in ghostPositions]
            nearestGhostDistance = min(distancesToGhosts)
            if nearestGhostDistance <= 1:  # Dangerously close
                reward -= (3 - nearestGhostDistance) * 100  # Increasing penalty the closer the ghost is

        return reward


    def __eq__(self, other):
        if isinstance(other, GameStateFeatures):
            return (self.pacmanPosition == other.pacmanPosition and
                    self.ghostPositions == other.ghostPositions and
                    self.numFood == other.numFood and
                    self.foodPositions == other.foodPositions)
        return False

    def __hash__(self):
        return hash((self.pacmanPosition, tuple(self.ghostPositions), self.numFood, tuple(self.foodPositions)))

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getQValue(self,
                  state: GameStateFeatures,
                  action: Directions) -> float:
        """
        Args:
            state: A given state
            action: Proposed action to take

        Returns:
            Q(state, action)
        """
        "*** YOUR CODE HERE ***"
        stateFeaturesTuple = (state.pacmanPosition, tuple(state.ghostPositions), state.numFood, tuple(state.foodPositions))
        return self.qValues.get((stateFeaturesTuple, action), 0.0)

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def maxQValue(self, state: GameStateFeatures) -> float:
        """
        Args:
            state: The given state

        Returns:
            q_value: the maximum estimated Q-value attainable from the state
        """
        "*** YOUR CODE HERE ***"
        legal = state.getLegalPacmanActions()
        if not legal:  # If there are no legal actions, return 0.0 as Q-value
            return 0.0
        return max([self.getQValue(state, action) for action in legal])

    def stateToTuple(self, stateFeatures: GameStateFeatures):
        """
        Converts GameStateFeatures to a tuple representation.
        """
        return (stateFeatures.pacmanPosition, tuple(stateFeatures.ghostPositions), stateFeatures.numFood, tuple(stateFeatures.foodPositions))

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def learn(self, state: GameStateFeatures, action: Directions, reward: float, nextState: GameStateFeatures):
        stateRep = self.stateToTuple(state)
        #nextState = self.stateToTuple(nextState)

        currentQValue = self.getQValue(state, action)
        futureQValue = self.maxQValue(nextState)

        sample = reward + self.gamma * futureQValue
        updatedQValue = (1 - self.alpha) * currentQValue + self.alpha * sample

        self.qValues[(stateRep, action)] = updatedQValue

        # Print the updated Q-value for debugging
        print(f"Updated Q-value: Q({stateRep}, {action}) = {updatedQValue}. Reward: {reward}, Future Q-value: {futureQValue}")
        return updatedQValue
    
    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def updateCount(self, state, action):
        stateRep = self.stateToTuple(state)
        if (stateRep, action) not in self.counts:
            self.counts[(stateRep, action)] = 1
        else:
            self.counts[(stateRep, action)] += 1

    def getCount(self, state, action):
        stateRep = self.stateToTuple(state)
        return self.counts.get((stateRep, action), 0)

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def explorationFn(self, utility, counts):
        """
        Computes the exploration function value.
        """
        # Example exploration strategy: epsilon-greedy
        return utility + self.alpha / (1 + counts)


    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getAction(self, state: GameState) -> Directions:
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        stateFeatures = GameStateFeatures(state)
        
        if random.random() < self.epsilon:
            self.epsilon -= (self.epsilon / self.getNumTraining()) if self.getEpisodesSoFar() < self.getNumTraining() else 0
            chosenAction = random.choice(legal)
            qValues = [self.getQValue(stateFeatures, action) for action in legal]
            counts = [self.getCount(stateFeatures, action) for action in legal]
            explorationValues = [self.explorationFn(utility, count) for utility, count in zip(qValues, counts)]
            maxExplorationValue = max(explorationValues)
            bestActions = [action for action, value in zip(legal, explorationValues) if value == maxExplorationValue]
            chosenAction = random.choice(bestActions)
            actionType = 'Exploration'
        else:
            qValues = [self.getQValue(stateFeatures, action) for action in legal]
            maxQValue = max(qValues)
            bestActions = [action for action, q in zip(legal, qValues) if q == maxQValue]
            chosenAction = random.choice(bestActions)
            actionType = 'Exploitation'

        # This is a stub for learning, which assumes you've somehow obtained the nextState and reward
        if self.lastState is not None and self.lastAction is not None:
            # Compute the reward using lastState (startState) and currentState (endState)
            reward = self.computeReward(self.lastState, state)

            # Learn from the last state, the action taken, the reward received, and the current state
            self.learn(self.lastStateFeatures, self.lastAction, reward, stateFeatures)

        # Store the current state and action to use in the next step
        self.lastState = state
        self.lastStateFeatures = stateFeatures
        self.lastAction = chosenAction

        chosenActionQValue = self.getQValue(stateFeatures, chosenAction)
        print(f"{actionType}: Chosen Action={chosenAction}, Q-value={chosenActionQValue}")
        return chosenAction



    def final(self, state: GameState):
        """
        Handle the end of episodes.
        This is called by the game after a win or a loss.

        Args:
            state: the final game state
        """
        print(f"Game {self.getEpisodesSoFar()} just ended!")

        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print('%s\n%s' % (msg, '-' * len(msg)))
            self.setAlpha(0)
            self.setEpsilon(0)
