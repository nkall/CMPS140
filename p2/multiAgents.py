# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

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
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.

    Print out these variables to see what you're getting, then combine them
    to create a masterful evaluation function.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    # Calculate average distance from food to detract from score
    foodList = oldFood.asList()
    foodDistances = []
    for food in foodList:
      foodDistances.append(manhattanDistance(newPos, food))
    avgFoodDist = sum(foodDistances) / len(foodList)

    # Reward Pacman for being further from ghosts
    ghostDistances = [manhattanDistance(newPos, ghostState.getPosition()) 
                        for ghostState in newGhostStates]
    # Double distance so Pacman values safety over pellets, but snap the
    # distance to limit so Pacman doesn't cower in a corner
    minGhostDistance = min(ghostDistances) * 2
    if minGhostDistance > 15:
      minGhostDistance = 15;
    return successorGameState.getScore() + minGhostDistance - avgFoodDist

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
    """
    (_, bestDirection) = self.runMinimax(gameState, 0, self.depth)
    return bestDirection

  # Returns tuples of (<score>, <action towards going down that path>)
  # In hindsight, there's no real need to keep track of anything but the 'top' action
  # I did something a bit more sensible with the AlphaBetaAgent
  def runMinimax(self, gameState, currAgentNum, depthRemaining):
    if depthRemaining is 0 or not gameState.getLegalActions(currAgentNum):
      return (self.evaluationFunction(gameState), Directions.STOP)

    nextAgentNum = currAgentNum + 1       # Number of agent to eval next
    nextDepthRemaining = depthRemaining   # Remaining depth after next eval
    # If done with all ghosts, decrease depth and go to Pacman
    if nextAgentNum >= gameState.getNumAgents():
      nextAgentNum = 0
      nextDepthRemaining -= 1

    chosenMini = None # 'Best' path tuple, either min or max
    if currAgentNum is 0:   # Evaluate max
      for action in gameState.getLegalActions(currAgentNum):
        successorState = gameState.generateSuccessor(currAgentNum, action)
        (currVal, _) = self.runMinimax(successorState, nextAgentNum, nextDepthRemaining)
        # If new score is the maximum, save that value and the action
        if (chosenMini is None) or (currVal > chosenMini[0]):
          chosenMini = (currVal, action)
    else:   # Evaluate min
      for action in gameState.getLegalActions(currAgentNum):
        successorState = gameState.generateSuccessor(currAgentNum, action)
        (currVal, _) = self.runMinimax(successorState, nextAgentNum, nextDepthRemaining)
        # If new score is the minimum, save that value and the action
        if (chosenMini is None) or (currVal < chosenMini[0]):
          chosenMini = (currVal, action)
    return chosenMini


class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """
  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    possibleActions = gameState.getLegalActions(0)
    if not possibleActions:
      return Directions.STOP

    # This time, since we only care about the first action in the tree,
    # we'll just run AlphaBeta for each initial action possibility
    bestAction = Directions.STOP
    bestScore = None
    for action in possibleActions:
      newState = gameState.generateSuccessor(0, action)
      newScore = self.runAlphaBeta(newState, 1, self.depth, 
                                   float("-inf"), float("inf"))
      if bestScore is None or newScore > bestScore:
        bestScore = newScore
        bestAction = action
    return bestAction

  # Returns score of going down a given path based on eval function
  def runAlphaBeta(self, gameState, currAgentNum, depthRemaining, alpha, beta):
    if depthRemaining is 0 or not gameState.getLegalActions(currAgentNum):
      return self.evaluationFunction(gameState)

    nextAgentNum = currAgentNum + 1       # Number of agent to eval next
    nextDepthRemaining = depthRemaining   # Remaining depth after next eval
    # If done with all ghosts, decrease depth and go to Pacman
    if nextAgentNum >= gameState.getNumAgents():
      nextAgentNum = 0
      nextDepthRemaining -= 1

    bestScore = None
    newAlpha = alpha
    newBeta = beta
    if currAgentNum is 0:   # Evaluate max
      for action in gameState.getLegalActions(currAgentNum):
        successor = gameState.generateSuccessor(currAgentNum, action)
        newScore = self.runAlphaBeta(successor, nextAgentNum,
                                     nextDepthRemaining, newAlpha, newBeta)
        # If new score is the best, set best
        if bestScore is None or newScore > bestScore:
          bestScore = newScore
        # If new score is more than alpha, change alpha
        if bestScore > newAlpha:
          newAlpha = bestScore
        # Stop searching nodes if not viable
        if newBeta <= newAlpha:
          break
    else:   # Evaluate min
      newBeta = beta
      for action in gameState.getLegalActions(currAgentNum):
        successor = gameState.generateSuccessor(currAgentNum, action)
        newScore = self.runAlphaBeta(successor, nextAgentNum, 
                                    nextDepthRemaining, newAlpha, newBeta)
        # If new score is the best, set best
        if bestScore is None or newScore < bestScore:
          bestScore = newScore
        # If new score is less than beta, change beta
        if bestScore < newBeta:
          newBeta = bestScore
        # Stop searching nodes if not viable
        if newBeta <= newAlpha and nextAgentNum is 0:
          break
    return bestScore


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
    util.raiseNotDefined()
'''
  def runExpectimax(self, gameState, currAgentNum, depthRemaining, alpha, beta):
    if depthRemaining is 0 or not gameState.getLegalActions(currAgentNum):
      return self.evaluationFunction(gameState)

    nextAgentNum = currAgentNum + 1       # Number of agent to eval next
    nextDepthRemaining = depthRemaining   # Remaining depth after next eval
    # If done with all ghosts, decrease depth and go to Pacman
    if nextAgentNum >= gameState.getNumAgents():
      nextAgentNum = 0
      nextDepthRemaining -= 1

    bestScore = None
    if currAgentNum is 0:   # Evaluate max
      for action in gameState.getLegalActions(currAgentNum):
        successor = gameState.generateSuccessor(currAgentNum, action)
        newScore = self.runExpectimax(successor, nextAgentNum,
                                     nextDepthRemaining)
        # If new score is the best, set best
        if bestScore is None or newScore > bestScore:
          bestScore = newScore
        # If new score is more than alpha, change alpha
        if bestScore > newAlpha:
          newAlpha = bestScore
        # Stop searching nodes if not viable
        if newBeta <= newAlpha:
          break
    else:   # Evaluate min
      
    return bestScore
'''

def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
  """
  "*** YOUR CODE HERE ***"
  util.raiseNotDefined()

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

