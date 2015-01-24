# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
In search.py, you will implement generic search algorithms which are called 
by Pacman agents (in searchAgents.py).
"""

import util
from util import Queue, PriorityQueue

class SearchProblem:
  """
  This class outlines the structure of a search problem, but doesn't implement
  any of the methods (in object-oriented terminology: an abstract class).
  
  You do not need to change anything in this class, ever.
  """
  
  def getStartState(self):
     """
     Returns the start state for the search problem 
     """
     util.raiseNotDefined()
    
  def isGoalState(self, state):
     """
       state: Search state
    
     Returns True if and only if the state is a valid goal state
     """
     util.raiseNotDefined()

  def getSuccessors(self, state):
     """
       state: Search state
     
     For a given state, this should return a list of triples, 
     (successor, action, stepCost), where 'successor' is a 
     successor to the current state, 'action' is the action
     required to get there, and 'stepCost' is the incremental 
     cost of expanding to that successor
     """
     util.raiseNotDefined()

  def getCostOfActions(self, actions):
     """
      actions: A list of actions to take
 
     This method returns the total cost of a particular sequence of actions.  The sequence must
     be composed of legal moves
     """
     util.raiseNotDefined()
           

def tinyMazeSearch(problem):
  """
  Returns a sequence of moves that solves tinyMaze.  For any other
  maze, the sequence of moves will be incorrect, so only use this for tinyMaze
  """
  from game import Directions
  s = Directions.SOUTH
  w = Directions.WEST
  return  [s,s,w,s,w,w,s,w]

def depthFirstSearch(problem):
  """
  Search the deepest nodes in the search tree first [p 85].
  
  Your search algorithm needs to return a list of actions that reaches
  the goal.  Make sure to implement a graph search algorithm [Fig. 3.7].
  
  To get started, you might want to try some of these simple commands to
  understand the search problem that is being passed in:
  
  print "Start:", problem.getStartState()
  print "Is the start a goal?", problem.isGoalState(problem.getStartState())
  print "Start's successors:", problem.getSuccessors(problem.getStartState())
  """
  startState = problem.getStartState()
  directionList = []  # Path to some path
  visitedList = []    # List of visited nodes (for cycle checking)
  solution = dfsRecurr(problem, startState, None, directionList, visitedList)
  return solution

def dfsRecurr(problem, state, direction, directionList, visitedList):
  # Build lists
  newDirList = list(directionList)
  if direction is not None:
    newDirList.append(direction)
  visitedList.append(state)

  # Success
  if problem.isGoalState(state):
    return newDirList

  # Move to further nodes
  successors = problem.getSuccessors(state)
  for (newState, newDirection, _) in successors:
    if newState not in visitedList:
      sol = dfsRecurr(problem, newState, newDirection, newDirList, visitedList)
      if sol is not None:
        return sol
  return None


def breadthFirstSearch(problem):
  "Search the shallowest nodes in the search tree first. [p 81]"
  succQueue = Queue() # Queue of nodes to visit next
  visitedList = []    # List of visited nodes (for cycle checking)

  currState = problem.getStartState()
  visitedList.append(currState)

  # Start with the start state's successors
  for successor in problem.getSuccessors(currState):
    #Queue has tuples of nodes and paths to that node
    visitedList.append(successor[0])
    succQueue.push((successor, [successor[1]]))

  while not succQueue.isEmpty():
    ((currState, currDir, _), currPath) = succQueue.pop()
    # Success
    if problem.isGoalState(currState):
      return currPath
    # Add successor nodes to queue
    for nextNode in problem.getSuccessors(currState):
      if nextNode[0] not in visitedList:
        visitedList.append(nextNode[0])
        succQueue.push((nextNode, currPath + [nextNode[1]]))
  return None

def uniformCostSearch(problem):
  "Search the node of least total cost first. "
  succQueue = PriorityQueue() # Queue of nodes to visit next, sorted by path cost
  visitedList = []            # List of visited nodes (for cycle checking)

  currState = problem.getStartState()
  visitedList.append(currState)

  # Start with the start state's successors
  for successor in problem.getSuccessors(currState):
    #Queue has triples of nodes, paths to that node, and cost
    #An object might be smarter to use here
    visitedList.append(successor[0])
    succQueue.push((successor, [successor[1]], successor[2]), successor[2])

  while not succQueue.isEmpty():
    ((currState, currDir, currCost), currPath, currTotalCost) = succQueue.pop()
    # Success
    if problem.isGoalState(currState):
      return currPath
    # Add successor nodes to queue ordered by cost
    for nextNode in problem.getSuccessors(currState):
      if nextNode[0] not in visitedList:
        visitedList.append(nextNode[0])
        succQueue.push((nextNode, currPath + [nextNode[1]],
                        currCost + currTotalCost), currCost + currTotalCost)
  return None

def nullHeuristic(state, problem=None):
  """
  A heuristic function estimates the cost from the current state to the nearest
  goal in the provided SearchProblem.  This heuristic is trivial.
  """
  return 0

def aStarSearch(problem, heuristic=nullHeuristic):
  "Search the node that has the lowest combined cost and heuristic first."
  succQueue = PriorityQueue() # Queue of nodes to visit next, sorted by cost + heuristic
  visitedList = []            # List of visited nodes (for cycle checking)

  currState = problem.getStartState()
  visitedList.append(currState)

  # Start with the start state's successors
  for successor in problem.getSuccessors(currState):
    #Queue has triples of nodes, paths to that node, and cost
    #An object might be smarter to use here
    visitedList.append(successor[0])
    succQueue.push((successor, [successor[1]], successor[2]),
                    successor[2] + heuristic(successor[0], problem))

  while not succQueue.isEmpty():
    ((currState, currDir, currCost), currPath, currTotalCost) = succQueue.pop()
    # Success
    if problem.isGoalState(currState):
      return currPath
    # Add successor nodes to queue ordered by cost + heuristic
    for nextNode in problem.getSuccessors(currState):
      if nextNode[0] not in visitedList:
        visitedList.append(nextNode[0])
        succQueue.push((nextNode, currPath + [nextNode[1]], currCost + currTotalCost),
                        currCost + currTotalCost + heuristic(nextNode[0], problem))
  return None
    
  
# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch