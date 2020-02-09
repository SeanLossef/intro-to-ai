# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
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
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

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
        newGhostStates = successorGameState.getGhostStates()
        curFood = currentGameState.getFood().asList()

        score = float("inf")

        # dont move into a spot where there is a ghost
        for ghostState in newGhostStates:
            ghostPos = ghostState.getPosition()
            if ghostPos == newPos:
                return float("-inf")

        # give higher score when closest to food
        for food in curFood:
            foodDist = manhattanDistance(food, newPos)
            score = min(score, foodDist)

        return 1.0 / (1.0 + float(score))


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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
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
        """
        
        self.numAgents = gameState.getNumAgents()

        return self.value(gameState, 0, self.depth)[1]

    def value(self, state, agent, depth):
        if depth <= 0 or state.isWin() or state.isLose():
            return [self.evaluationFunction(state)]

        if agent == 0:
            return self.maxvalue(state, agent, depth)
        else:
            return self.minvalue(state, agent, depth)

    def maxvalue(self, state, agent, depth):
        values = []
        actions = state.getLegalActions(agent)
        for action in actions:
            value = self.value(state.generateSuccessor(agent, action), agent+1, depth)[0]
            values.append([value, action])
        return max(values)

    def minvalue(self, state, agent, depth):
        values = []
        actions = state.getLegalActions(agent)
        if agent == state.getNumAgents() - 1:
            for action in actions:
                value = self.value(state.generateSuccessor(agent, action), 0, depth-1)[0]
                values.append([value, action])
        else:
            for action in actions:
                value = self.value(state.generateSuccessor(agent, action), agent+1, depth)[0]
                values.append([value, action])
        return min(values)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        self.numAgents = gameState.getNumAgents()

        return self.value(gameState, 0, self.depth, float("-inf"), float("inf"))[1]

    def value(self, state, agent, depth, alpha, beta):
        if depth <= 0 or state.isWin() or state.isLose():
            return [self.evaluationFunction(state)]

        if agent == 0:
            return self.maxvalue(state, agent, depth, alpha, beta)
        else:
            return self.minvalue(state, agent, depth, alpha, beta)

    def maxvalue(self, state, agent, depth, alpha, beta):
        values = []
        actions = state.getLegalActions(agent)
        for action in actions:
            value = self.value(state.generateSuccessor(agent, action), agent+1, depth, alpha, beta)[0]
            values.append([value, action])
            alpha = max(alpha, value)
            if value > beta:
                return [value, action]
        return max(values)

    def minvalue(self, state, agent, depth, alpha, beta):
        values = []
        actions = state.getLegalActions(agent)
        if agent == state.getNumAgents() - 1:
            for action in actions:
                value = self.value(state.generateSuccessor(agent, action), 0, depth-1, alpha, beta)[0]
                values.append([value, action])
                beta = min(beta, value)
                if beta < alpha:
                    return [value, action]
        else:
            for action in actions:
                value = self.value(state.generateSuccessor(agent, action), agent+1, depth, alpha, beta)[0]
                values.append([value, action])
                beta = min(beta, value)
                if value < alpha:
                    return [value, action]
        return min(values)

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
        self.numAgents = gameState.getNumAgents()

        return self.value(gameState, 0, self.depth)[1]

    def value(self, state, agent, depth):
        if depth <= 0 or state.isWin() or state.isLose():
            return [self.evaluationFunction(state)]

        if agent == 0:
            return self.maxvalue(state, agent, depth)
        else:
            return self.minvalue(state, agent, depth)

    def maxvalue(self, state, agent, depth):
        values = []
        actions = state.getLegalActions(agent)
        for action in actions:
            value = self.value(state.generateSuccessor(agent, action), agent+1, depth)[0]
            values.append([value, action])
        return max(values)

    def minvalue(self, state, agent, depth):
        expected = 0
        count = 0
        actions = state.getLegalActions(agent)
        if agent == state.getNumAgents() - 1:
            for action in actions:
                value = self.value(state.generateSuccessor(agent, action), 0, depth-1)[0]
                expected += value
                count += 1
        else:
            for action in actions:
                value = self.value(state.generateSuccessor(agent, action), agent+1, depth)[0]
                expected += value
                count += 1
        return [float(expected/count), "meow"]

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    foodPos = currentGameState.getFood().asList() 
    currentPos = list(currentGameState.getPacmanPosition()) 
    
    score = float("inf")

    # Find distance to nearest food
    for food in foodPos:
        foodDist = manhattanDistance(food, currentPos)
        score = min(score, foodDist)
        
    if score == float("inf"):
        score = 0

    return currentGameState.getScore() - score

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """

    def getAction(self, gameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

