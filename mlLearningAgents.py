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

        Args:
            state: A given game state object
        """
        # Initialize the Pac-Man features extracted from the state
        self.pacmanPosition = state.getPacmanPosition()
        self.ghostPositions = state.getGhostPositions()
        self.numFood = state.getNumFood()
        self.foodPositions = [food for food in state.getFood().asList()]
        self.legalActions = state.getLegalPacmanActions()

    # Legal actions for Pacman    
    def getLegalPacmanActions(self):
        return self.legalActions

class QLearnAgent(Agent):

    def __init__(self,
                 alpha: float = 0.01,
                 epsilon: float = 0.20,
                 gamma: float = 0.9,
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
        self.episodesSoFar = 0 # Count the number of games we have played
        self.qValues = {}  # Initialize qValues as an empty dictionary
        self.counts = {}  # Initialize counts as an empty dictionary if you plan to use it
        self.terminalRewards = [] # Store the terminal rewards for each game

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
    def computeReward(startState: GameState, 
                      endState: GameState) -> float:
        """
         Compute the reward for the given state transition from startState to endState.

            The reward is calculated based on the following criteria:
            - Eating food pellets: +200 points
            - Losing the game: -600 points
            - Winning the game: +1000 points
            - Moving closer to the nearest food pellet: Up to +10 points
            - Getting dangerously close to non-scared ghosts: Up to -300 points

           'The reward function provides feedback to the agent about the desirability of the state transitions 
           and guides the learning process towards optimal behavior.' (Sutton and Barto, 2018)
          
        Args:
            startState: A starting state
            endState: A resulting state

        Returns:
            The reward assigned for the given trajectory
        """
        reward = 0 # Initialize reward to 0
        # Reward for eating food
        if endState.getNumFood() < startState.getNumFood():
            reward += 100
        # Penalty for getting caught by a ghost
        if endState.isLose():
            reward -= 600
        # Reward for winning the game
        if endState.isWin():
            reward += 1000

        reward -= 1   # Slight penalty for each move to encourage efficiency
        # Encourage Pacman to move closer to the nearest food pellet
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
            if nearestGhostDistance <= 2:  # Dangerously close
                reward -= (3 - nearestGhostDistance) * 150  # Increasing penalty the closer the ghost is

        return reward

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getQValue(self,
                  state: GameStateFeatures,
                  action: Directions) -> float:
        """
        The Q-value represents the expected future reward for taking the specified action in the given state.
        It's used to guide the decision-making process of the Agent under Q-Learning.

        Args:
            state (GameStateFeatures): The current state of the game, represented by a GameStateFeatures object.
            action (Directions): The proposed action to take in the given state.

        Returns:
            float: The Q-value for the given state-action pair, Q(state, action).
        """
        stateFeaturesTuple = (state.pacmanPosition, tuple(state.ghostPositions), state.numFood, tuple(state.foodPositions))
        return self.qValues.get((stateFeaturesTuple, action), 0.0)

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def maxQValue(self, 
                  state: GameStateFeatures) -> float:
        """
        Compute the maximum estimated Q-value attainable from the given state.

        This function determines the maximum Q-value that can be obtained by taking
        any legal action from the current state. It iterates over all legal actions
        available to Pacman and computes the Q-value for each action using the
        getQValue function. The maximum Q-value among all actions is returned.

        In Q-Learning, the utility of the agent is determined by the maximum Q-value

        Args:
            state (GameStateFeatures): The current state of the game, represented by a GameStateFeatures object.

        Returns:
            Q-Value: the maximum estimated Q-value attainable from the state
        """
        # Get the legal actions available to Pacman in the current state
        legal = state.getLegalPacmanActions()
        # If there are no legal actions, return 0.0 as the Q-value
        if not legal: 
            return 0.0
        # Compute the Q-value for each legal action and return the maximum value
        return max([self.getQValue(state, action) for action in legal])

    def stateToTuple(self, 
                     state: GameStateFeatures):
        """
        Convert GameStateFeatures to a tuple representation.

        This function takes a GameStateFeatures object and converts it into a tuple
        representation. The tuple contains the following elements:
        - Pacman's position
        - Ghost positions as a tuple
        - Number of food pellets remaining
        - Food positions as a tuple

        Args:
            state (GameStateFeatures): The current state of the game, represented by a GameStateFeatures object.

        Returns:
            A tuple representation of the state, containing Pacman's position, ghost
            positions, number of food pellets, and food positions.
        """
        return (state.pacmanPosition, tuple(state.ghostPositions), 
                state.numFood, tuple(state.foodPositions))

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def learn(self, 
              state: GameStateFeatures, 
              action: Directions, 
              reward: float, 
              nextState: GameStateFeatures):
        """
        Update the Q-value for the given state-action pair based on the received reward and the maximum Q-value of the next state.

        This function implements the Q-learning update rule to learn the optimal Q-values for each state-action pair.
        It calculates the updated Q-value using the current Q-value, the received reward, and the maximum Q-value of the next state.
        The updated Q-value is a weighted average of the current Q-value and the sample (reward + discounted future Q-value),
        where the weights are determined by the learning rate (alpha).

        Args:
            state: The current state represented by GameStateFeatures.
            action: The action taken in the current state.
            reward: The reward received after taking the action.
            nextState: The next state reached after taking the action, represented by GameStateFeatures.

        Returns:
            The updated Q-value for the state-action pair.
        """
        # Get the tuple representation of the current state
        stateRep = self.stateToTuple(state)
        # Calculate the Q-learning update
        currentQValue = self.getQValue(state, action)
        futureQValue = self.maxQValue(nextState)
        sample = reward + self.gamma * futureQValue
        updatedQValue = (1 - self.alpha) * currentQValue + self.alpha * sample
        # Update the Q-value in the Q-value dictionary
        self.qValues[(stateRep, action)] = updatedQValue
        return updatedQValue # Return the updated Q-value
    
    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def updateCount(self, 
                    state: GameStateFeatures, 
                    action: Directions):
        """
        Update the count of the given state-action pair.

        This function increments the count of the state-action pair in the counts dictionary.
        If the state-action pair is encountered for the first time, it initializes the count to 1.

        Args:
            state: The current state.
            action: The action taken in the current state.
        """
        stateRep = self.stateToTuple(state)
        if (stateRep, action) not in self.counts:
            self.counts[(stateRep, action)] = 1
        else:
            self.counts[(stateRep, action)] += 1
    
    def getCount(self, 
                 state: GameStateFeatures, 
                 action:Directions)->int:
        """
        Get the count of the given state-action pair.

        This function retrieves the count of the state-action pair from the counts dictionary.
        If the state-action pair is not found, it returns a default count of 0.

        Args:
            state: The current state.
            action: The action taken in the current state.

        Returns:
            The count of the state-action pair, or 0 if the pair is not found.
        """
        stateRep = self.stateToTuple(state)
        return self.counts.get((stateRep, action), 0)

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def explorationFn(self, 
                      utility: float, 
                      counts: int)->float:
        """
        Compute the exploration function value using a count-based exploration strategy.

        This function calculates the exploration value based on the utility and count of a state-action pair.
        It combines the concepts of epsilon-greedy exploration and count-based exploration to balance the
        trade-off between exploring new actions and exploiting the current knowledge.

        The exploration value is computed as the sum of the utility and an exploration bonus. The exploration
        bonus is inversely proportional to the count of the state-action pair, encouraging the agent to explore
        less frequently visited state-action pairs. The `alpha` hyperparameter controls the balance between
        exploration and exploitation.

        Args:
            utility: The utility value of the state-action pair, representing the expected future rewards.
            counts: The number of times the state-action pair has been visited.

        Returns:
            The exploration function value, which balances exploration and exploitation based on the count-based
            exploration strategy.
        """
        # The exploration value is a combination of the utility and an exploration bonus
        exploration_value = utility + self.alpha / (1 + counts)
        return exploration_value


    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getAction(self, 
                  state: GameState) -> Directions:
        
        """
        Choose the best action for Pacman based on the current game state.

        This function combines exploration and exploitation strategies to select an action.
        It uses an epsilon-greedy approach to balance between random exploration and
        selecting the action with the highest Q-value. The exploration strategy is based
        on the count of state-action pairs, encouraging exploration of less visited pairs.

        Args:
            state: The current game state.

        Returns:
            The chosen action for Pacman.
        """
        # Get the legal actions for Pacman in the current state
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)
        # Convert the game state to a feature representation
        stateFeatures = GameStateFeatures(state)
       
        # Perform exploration with probability epsilon
        if random.random() < self.epsilon:
            # Decay epsilon during training
            self.epsilon -= (self.epsilon / self.getNumTraining()) if self.getEpisodesSoFar() < self.getNumTraining() else 0
            # Choose a random action for exploration
            chosenAction = random.choice(legal)
            # Calculate Q-values, counts, and exploration values for each action
            qValues = [self.getQValue(stateFeatures, action) for action in legal]
            counts = [self.getCount(stateFeatures, action) for action in legal]
            explorationValues = [self.explorationFn(utility, count) for utility, count in zip(qValues, counts)]
            # Select the action with the highest exploration value
            maxExplorationValue = max(explorationValues)
            bestActions = [action for action, value in zip(legal, explorationValues) if value == maxExplorationValue]
            chosenAction = random.choice(bestActions)

            actionType = 'Exploration'
        else:
            # Calculate Q-values for each action
            qValues = [self.getQValue(stateFeatures, action) for action in legal]
            # Select the action with the highest Q-value
            maxQValue = max(qValues)
            bestActions = [action for action, q in zip(legal, qValues) if q == maxQValue]
            chosenAction = random.choice(bestActions)

            actionType = 'Exploitation'
        # Update the count for the chosen state-action pair
        self.updateCount(stateFeatures, chosenAction)

        # Perform learning if the previous state and action are available
        if self.lastState is not None and self.lastAction is not None:
            # Compute the reward based on the previous state and current state
            reward = self.computeReward(self.lastState, state)

             # Learn the Q-value based on the previous state, action, reward, and current state
            self.learn(self.lastStateFeatures, self.lastAction, reward, stateFeatures)

        # Store the current state, features, and action for the next step
        self.lastState = state
        self.lastStateFeatures = stateFeatures
        self.lastAction = chosenAction

        # Print the chosen action and its Q-value for debugging
        chosenActionQValue = self.getQValue(stateFeatures, chosenAction)
        print(f"{actionType}: Chosen Action={chosenAction}, Q-value={chosenActionQValue}")

        return chosenAction

    def final(self, 
              state: GameState):
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

        # Compute the reward 
        if self.lastState is not None:
            terminalReward = self.computeReward(self.lastState, state)
            self.terminalRewards.append(terminalReward)
            print(f"Terminal reward for game {self.getEpisodesSoFar()}: {terminalReward}")

