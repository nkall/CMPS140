# qlearningAgents.py
# ------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
  """
    Q-Learning Agent

    Functions you should fill in:
      - getQValue
      - getAction
      - getValue
      - getPolicy
      - update

    Instance variables you have access to
      - self.epsilon (exploration prob)
      - self.alpha (learning rate)
      - self.discount (discount rate)

    Functions you should use
      - self.getLegalActions(state)
        which returns legal actions
        for a state
  """
  def __init__(self, **args):
    "You can initialize Q-values here..."
    ReinforcementAgent.__init__(self, **args)

    # Dict of qValues with (state, action) tuples as keys
    self.qValues = util.Counter()

  def getQValue(self, state, action):
    """
      Returns Q(state,action)
      Should return 0.0 if we never seen
      a state or (state,action) tuple
    """
    return self.qValues[(state, action)]


  def getValue(self, state):
    """
      Returns max_action Q(state,action)
      where the max is over legal actions.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return a value of 0.0.
    """
    legalActions = self.getLegalActions(state)
    # No legal actions, then return 0.0
    if not legalActions:
      return 0.0

    # Return the value of the action with the best predicted outcome
    bestValue = None
    for action in legalActions:
      currentValue = self.getQValue(state, action)
      if bestValue is None or currentValue > bestValue:
        bestValue = currentValue
    return bestValue


  def getPolicy(self, state):
    """
      Compute the best action to take in a state.  Note that if there
      are no legal actions, which is the case at the terminal state,
      you should return None.
    """
    legalActions = self.getLegalActions(state)
    # No legal actions, then return None
    if not legalActions:
      return None

    # Return an action that has the best, or tied for the best (chosen
    # randomly), predicted outcomes
    bestValue = None
    bestActions = []
    for action in legalActions:
      currentValue = self.getQValue(state, action)
      # (Re)create bestActions if new best value
      if bestValue is None or currentValue > bestValue:
        bestValue = currentValue
        bestActions = [action]
      # Append all tied values to bestActions
      elif currentValue is bestValue:
        bestActions.append(action)
    return random.choice(bestActions)

  def getAction(self, state):
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
    legalActions = self.getLegalActions(state)
    action = None
    if legalActions:
      # Decide whether to go with a random choice or best choice
      shouldPickRandom = util.flipCoin(self.epsilon)
      # Pick random action or best choice, depending on shouldPickRandom
      if shouldPickRandom:
        action = random.choice(legalActions)
      else:
        action = self.getPolicy(state)
    return action

  def update(self, state, action, nextState, reward):
    """
      The parent class calls this to observe a
      state = action => nextState and reward transition.
      You should do your Q-Value update here

      NOTE: You should never call this function,
      it will be called on your behalf
    """
    # Update Q Value at old state and action based on the reward received at
    # the new state plus the discounted best-case value of future states
    self.qValues[(state, action)] = reward + (self.discount * 
                                              self.getValue(nextState))



class PacmanQAgent(QLearningAgent):
  "Exactly the same as QLearningAgent, but with different default parameters"

  def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
    """
    These default parameters can be changed from the pacman.py command line.
    For example, to change the exploration rate, try:
        python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

    alpha    - learning rate
    epsilon  - exploration rate
    gamma    - discount factor
    numTraining - number of training episodes, i.e. no learning after these many episodes
    """
    args['epsilon'] = epsilon
    args['gamma'] = gamma
    args['alpha'] = alpha
    args['numTraining'] = numTraining
    self.index = 0  # This is always Pacman
    QLearningAgent.__init__(self, **args)

  def getAction(self, state):
    """
    Simply calls the getAction method of QLearningAgent and then
    informs parent of action for Pacman.  Do not change or remove this
    method.
    """
    action = QLearningAgent.getAction(self,state)
    self.doAction(state,action)
    return action


class ApproximateQAgent(PacmanQAgent):
  """
     ApproximateQLearningAgent

     You should only have to overwrite getQValue
     and update.  All other QLearningAgent functions
     should work as is.
  """
  def __init__(self, extractor='IdentityExtractor', **args):
    self.featExtractor = util.lookup(extractor, globals())()
    PacmanQAgent.__init__(self, **args)

    # Initialize weights, a dict with features as keys
    self.weights = util.Counter()

  def getQValue(self, state, action):
    """
      Should return Q(state,action) = w * featureVector
      where * is the dotProduct operator
    """
    features = self.featExtractor.getFeatures(state, action)
    qSum = 0.0
    # Iterate through features dict for current state, action pair and
    # sum the weights associated with each feature
    for (feature, value) in features.iteritems():
      qSum += self.weights[feature] * value
    return qSum

  def update(self, state, action, nextState, reward):
    """
       Should update your weights based on transition
    """
    # 'Correction' very similar to our regular value iteration formula
    correction = ((reward + self.discount * self.getValue(nextState)) - 
                 self.getQValue(state, action))
    features = self.featExtractor.getFeatures(state, action)

    # Iterate through features dict for current state, action pair and
    # change associated weights based on learning rate and 'correction'
    for (feature, value) in features.iteritems():
      self.weights[feature] += self.alpha * correction * value

  def final(self, state):
    "Called at the end of each game."
    # call the super-class final method
    PacmanQAgent.final(self, state)

    # did we finish training?
    if self.episodesSoFar == self.numTraining:
      # you might want to print your weights here for debugging
      "*** YOUR CODE HERE ***"
      pass
