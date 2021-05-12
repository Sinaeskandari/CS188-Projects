# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

#####
from pacman import GameState
from math import inf


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        newFoodasList = newFood.asList()
        ghostPos = successorGameState.getGhostPosition(1)
        score = successorGameState.getScore()
        score -= 20 * len(newFoodasList)
        distance_to_ghost = manhattanDistance(newPos, ghostPos)
        distance_to_foods = [manhattanDistance(newPos, newFoodasList[i]) for i in range(len(newFoodasList))]
        try:
            score -= 2 * min(distance_to_foods)
        except ValueError:
            pass
        score += distance_to_ghost
        return score


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        minimax = self.value(0, 0, gameState)
        action = minimax[1]
        return action

    def value(self, agentIndex, depth, gameState):
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return [self.evaluationFunction(gameState), ""]
        if agentIndex == 0:
            return self.max_value(gameState, depth, agentIndex)
        elif agentIndex > 0:
            if agentIndex == gameState.getNumAgents() - 1:
                depth += 1
            return self.min_value(gameState, depth, agentIndex)

    def min_value(self, gameState, depth, agentIndex):
        v = [inf, ""]
        for act in gameState.getLegalActions(agentIndex):
            v = min(v, [self.value((agentIndex + 1) % gameState.getNumAgents(), depth,
                                   gameState.generateSuccessor(agentIndex, act))[0], act])
        return v

    def max_value(self, gameState, depth, agentIndex):
        v = [-inf, ""]
        for act in gameState.getLegalActions(agentIndex):
            v = max(v, [self.value((agentIndex + 1) % gameState.getNumAgents(), depth,
                                   gameState.generateSuccessor(agentIndex, act))[0], act])
        return v


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        alphabetaminimax = self.value(0, 0, gameState, -inf, +inf)
        action = alphabetaminimax[1]
        return action

    def value(self, agentIndex, depth, gameState, alpha, beta):
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return [self.evaluationFunction(gameState), ""]
        if agentIndex == 0:
            return self.max_value(gameState, depth, agentIndex, alpha, beta)
        elif agentIndex > 0:
            if agentIndex == gameState.getNumAgents() - 1:
                depth += 1
            return self.min_value(gameState, depth, agentIndex, alpha, beta)

    def min_value(self, gameState, depth, agentIndex, alpha, beta):
        v = [inf, ""]
        for act in gameState.getLegalActions(agentIndex):
            v = min(v, [self.value((agentIndex + 1) % gameState.getNumAgents(), depth,
                                   gameState.generateSuccessor(agentIndex, act), alpha, beta)[0], act])
            if v[0] < alpha:
                return v
            beta = min(beta, v[0])
        return v

    def max_value(self, gameState, depth, agentIndex, alpha, beta):
        v = [-inf, ""]
        for act in gameState.getLegalActions(agentIndex):
            v = max(v, [self.value((agentIndex + 1) % gameState.getNumAgents(), depth,
                                   gameState.generateSuccessor(agentIndex, act), alpha, beta)[0], act])
            if v[0] > beta:
                return v
            alpha = max(alpha, v[0])
        return v


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        expectimax = self.value(0, 0, gameState)
        action = expectimax[1]
        return action

    def value(self, agentIndex, depth, gameState):
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return [self.evaluationFunction(gameState), ""]
        if agentIndex == 0:
            return self.max_value(gameState, depth, agentIndex)
        elif agentIndex > 0:
            if agentIndex == gameState.getNumAgents() - 1:
                depth += 1
            return self.exp_value(gameState, depth, agentIndex)

    def exp_value(self, gameState, depth, agentIndex):
        v = [0, ""]
        legal_acts = gameState.getLegalActions(agentIndex)
        for act in legal_acts:
            p = 1 / len(legal_acts)
            v = [v[0] + p * self.value((agentIndex + 1) % gameState.getNumAgents(), depth,
                                       gameState.generateSuccessor(agentIndex, act))[0], act]
        return v

    def max_value(self, gameState, depth, agentIndex):
        v = [-inf, ""]
        for act in gameState.getLegalActions(agentIndex):
            v = max(v, [self.value((agentIndex + 1) % gameState.getNumAgents(), depth,
                                   gameState.generateSuccessor(agentIndex, act))[0], act])
        return v


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    score = currentGameState.getScore()
    food_positions = currentGameState.getFood().asList()
    pacman_position = currentGameState.getPacmanPosition()
    ghost_states = currentGameState.getGhostStates()
    ghost_position = currentGameState.getGhostPositions()
    capsule_positions = currentGameState.getCapsules()
    scared_times = [g.scaredTimer for g in ghost_states]
    distance_to_foods = [manhattanDistance(pacman_position, food_positions[i]) for i in range(len(food_positions))]
    distance_to_ghost = manhattanDistance(pacman_position, ghost_position[0])
    score -= 10 * len(food_positions)
    score -= 100 * len(capsule_positions)
    try:
        score -= 2 * min(distance_to_foods)
    except ValueError:
        pass
    if scared_times[0]:
        score -= 1 / distance_to_ghost
    else:
        score += distance_to_ghost
    return score


# Abbreviation
better = betterEvaluationFunction
