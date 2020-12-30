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

    def evaluationFunction(self, currentGameState, action):
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
        newCapsules = successorGameState.getCapsules()
        newGhostStates = successorGameState.getGhostStates()
        newGhostPositions = successorGameState.getGhostPositions()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        "*** YOUR CODE HERE ***"

        if successorGameState.isWin():
            return 999999
        foodDistances = [util.manhattanDistance(food, newPos) for food in newFood.asList()]
        closestFood = sorted(foodDistances)[0]

        score = 1.0 / closestFood

        capsuleDistances = [util.manhattanDistance(capsule, newPos) for capsule in newCapsules]
        if capsuleDistances:
            closestCapsule = sorted(capsuleDistances)[0]
            score += 1.0 / closestCapsule

        ghostDistances = [util.manhattanDistance(ghostState.getPosition(), newPos) for ghostState in newGhostStates if
                          ghostState.scaredTimer == 0]
        if ghostDistances:
            closestGhost = sorted(ghostDistances)[0]
            if closestGhost <= 1:
                score = -500 / (closestGhost + 1)

        return successorGameState.getScore() + score


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

        # second argument is a list storing agent index and  depth(recursive steps)
        action, v = self.maxvalue(gameState, 0)
        return action
        util.raiseNotDefined()

    def minimax(self, state, depth, action):
        if state.isWin() or state.isLose() or depth == self.depth * state.getNumAgents():
            return action, self.evaluationFunction(state)
        agent = depth % state.getNumAgents()
        if agent == 0:
            tmp_action, v = self.maxvalue(state, depth)
        else:
            tmp_action, v = self.minvalue(state, depth)
        return tmp_action, v

    #
    def maxvalue(self, state, depth):
        best_action, v = "_", float('-Inf')
        agent = depth % state.getNumAgents()
        for action in state.getLegalActions(agent):
            new_action, new_v = self.minimax(state.generateSuccessor(agent, action), depth + 1, action)
            if new_v > v:
                best_action, v = action, new_v
        return best_action, v

    def minvalue(self, state, depth):
        best_action, v = "_", float('Inf')
        agent = depth % state.getNumAgents()
        for action in state.getLegalActions(agent):
            new_action, new_v = self.minimax(state.generateSuccessor(agent, action), depth + 1, action)
            if new_v <= v:
                best_action, v = action, new_v
        return best_action, v


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        action, v = self.maxvalue(gameState, 0, float('-Inf'), float('Inf'))
        return action
        util.raiseNotDefined()

    def minimax(self, state, depth, action, alpha, beta):
        if state.isWin() or state.isLose() or depth == self.depth * state.getNumAgents():
            return action, self.evaluationFunction(state)
        agent = depth % state.getNumAgents()
        if agent == 0:
            tmp_action, v = self.maxvalue(state, depth, alpha, beta)
        else:
            tmp_action, v = self.minvalue(state, depth, alpha, beta)
        return tmp_action, v

    #
    def maxvalue(self, state, depth, alpha, beta):
        best_action, v = "_", float('-Inf')
        agent = depth % state.getNumAgents()
        for action in state.getLegalActions(agent):
            new_action, new_v = self.minimax(state.generateSuccessor(agent, action), depth + 1, action, alpha, beta)
            if new_v > v:
                best_action, v = action, new_v
                if v > beta:
                    return best_action, v
                alpha = max(v, alpha)
        return best_action, v

    def minvalue(self, state, depth, alpha, beta):
        best_action, v = "_", float('Inf')
        agent = depth % state.getNumAgents()
        for action in state.getLegalActions(agent):
            new_action, new_v = self.minimax(state.generateSuccessor(agent, action), depth + 1, action, alpha, beta)
            if new_v <= v:
                best_action, v = action, new_v
                if v < alpha:
                    return best_action, v
                beta = min(v, beta)
        return best_action, v


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
        action, v = self.maxvalue(gameState, 0)
        return action
        util.raiseNotDefined()

    def expectimax(self, state, depth, action):
        if state.isWin() or state.isLose() or depth == self.depth * state.getNumAgents():
            return action, self.evaluationFunction(state)
        agent = depth % state.getNumAgents()
        if agent == 0:
            tmp_action, v = self.maxvalue(state, depth)
        else:
            tmp_action, v = self.excpectedvalue(state, depth)
        return tmp_action, v


    def maxvalue(self, state, depth):
        best_action, v = "_", float('-Inf')
        agent = depth % state.getNumAgents()
        for action in state.getLegalActions(agent):
            new_action, new_v = self.expectimax(state.generateSuccessor(agent, action), depth + 1, action)
            if new_v > v:
                best_action, v = action, new_v
        return best_action, v

    def excpectedvalue(self, state, depth):
        value, count = 0, 0
        agent = depth % state.getNumAgents()
        for action in state.getLegalActions(agent):
            new_action, new_value = self.expectimax(state.generateSuccessor(agent, action), depth + 1, action)
            value += new_value
            count += 1
        return None, value / count


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5). <3

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    score = currentGameState.getScore()

    position = currentGameState.getPacmanPosition()
    ghostStates = currentGameState.getGhostStates()

    ghostDistances = [util.manhattanDistance(ghostState.getPosition(), position) for ghostState in ghostStates if ghostState.scaredTimer == 0]
    scaredGhostDistances = [util.manhattanDistance(ghostState.getPosition(), position) for ghostState in ghostStates if ghostState.scaredTimer > 0]
    foodDistances = [util.manhattanDistance(food_pellet, position) for food_pellet in currentGameState.getFood().asList()]
    capsuleDistances = [util.manhattanDistance(capsule, position) for capsule in currentGameState.getCapsules()]

    for d in ghostDistances:
        score += 5 * d   # further the ghost better the score

    for d in scaredGhostDistances:
        score -= 10 * d  # further the scared ghost worse the score

    for d in foodDistances:
        score -=  d       # further the food worse the score

    for d in capsuleDistances:
        score -= d * 50     # further the capsule worse the score


    return score
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
