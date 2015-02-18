# valueIterationAgents.py
# -----------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
  """
      * Please read learningAgents.py before reading this.*

      A ValueIterationAgent takes a Markov decision process
      (see mdp.py) on initialization and runs value iteration
      for a given number of iterations using the supplied
      discount factor.
  """
  def __init__(self, mdp, discount = 0.9, iterations = 100):
    """
      Your value iteration agent should take an mdp on
      construction, run the indicated number of iterations
      and then act according to the resulting policy.
    
      Some useful mdp methods you will use:
          mdp.getStates()
          mdp.getPossibleActions(state)
          mdp.getTransitionStatesAndProbs(state, action)
          mdp.getReward(state, action, nextState)
    """
    self.mdp = mdp
    self.discount = discount
    self.iterations = iterations
    self.values = util.Counter() # A Counter is a dict with default 0

    states = self.mdp.getStates()
    for _ in range(self.iterations):
      # Credit to Anonymous on Piazza for suggesting I use self.values.copy()
      newValues = self.values.copy()
      # For each state, get the best possible q-value for each action from
      # that state and make it the new value of the state
      for state in states:
        possActions = self.mdp.getPossibleActions(state)

        # Generate list of Q-Values from each action possible from that state
        actionQValues = [self.getQValue(state, act) for act in possActions]

        # Set self.values of that state based on best computed value
        if actionQValues:
          newValues[state] = max(actionQValues)
      self.values = newValues

  def getValue(self, state):
    """
      Return the value of the state (computed in __init__).
    """
    return self.values[state]


  def getQValue(self, state, action):
    """
      The q-value of the state action pair
      (after the indicated number of value iteration
      passes).  Note that value iteration does not
      necessarily create this quantity and you may have
      to derive it on the fly.
    """
    transitions = self.mdp.getTransitionStatesAndProbs(state, action)
    qSum = 0

    for (nextState, prob) in transitions:
      # R(s,a,s')
      reward = self.mdp.getReward(state, action, nextState)
      # Gamma * V*(s')
      discountedValue = self.discount * self.getValue(nextState)
      qSum += prob * (reward + discountedValue)
    return qSum

  def getPolicy(self, state):
    """
      The policy is the best action in the given state
      according to the values computed by value iteration.
      You may break ties any way you see fit.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return None.
    """
    possActions = self.mdp.getPossibleActions(state)
    bestAction = None

    # If no legal actions, return None
    if not possActions:
      return bestAction

    bestQValue = None
    for action in possActions:
      currentQValue = self.getQValue(state, action)
      if bestQValue is None or currentQValue > bestQValue:
        bestQValue = currentQValue
        bestAction = action

    # Return action with the best Q-Value
    return bestAction

  def getAction(self, state):
    "Returns the policy at the state (no exploration)."
    return self.getPolicy(state)
  
