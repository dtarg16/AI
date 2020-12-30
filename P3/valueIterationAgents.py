# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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

import mdp, util
from collections import defaultdict

from learningAgents import ValueEstimationAgent
import collections


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for iteration in range(self.iterations):
            newValues = self.values.copy()
            for state in self.mdp.getStates():
                action = self.getAction(state)
                QValue = self.getQValue(state, action)
                newValues[state] = QValue;
            self.values = newValues

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        if action is None:
            return 0

        QValue = 0
        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            bellman = prob * (self.mdp.getReward(state, action, nextState) + self.values[nextState] * self.discount)
            QValue += bellman
        return QValue
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            return None
        bestAction, bestQValue = None, -99999;
        for action in self.mdp.getPossibleActions(state):
            QValue = self.getQValue(state, action)
            if QValue > bestQValue:
                bestAction, bestQValue = action, QValue
        return bestAction
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        for iteration in range(self.iterations):
            states = self.mdp.getStates()
            state = states[iteration % len(states)]
            action = self.getAction(state)
            QValue = self.getQValue(state, action)
            self.values[state] = QValue


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """

    def __init__(self, mdp, discount=0.9, iterations=100, theta=1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        self.predecessors = self.initializePredecessors(mdp)
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def initializePredecessors(self, mdp):
        "*** YOUR CODE HERE ***"
        predecessors = defaultdict(set)
        for state in mdp.getStates():
            for action in mdp.getPossibleActions(state):
                for nextState, prob in mdp.getTransitionStatesAndProbs(state, action):
                    if prob > 0:
                        predecessors[nextState].add(state)
        return predecessors

    def absoluteDifference(self, state):
        action = self.getAction(state)
        QValue = self.getQValue(state, action)
        return abs(self.values[state] - QValue)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        pq = util.PriorityQueue()

        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                diff = self.absoluteDifference(state)
                pq.push(state, -diff)

        for iteration in range(self.iterations):
            if not pq.isEmpty():
                state = pq.pop()
                action = self.getAction(state)
                QValue = self.getQValue(state, action)
                self.values[state] = QValue;

                for predecessor in self.predecessors[state]:
                    diff = self.absoluteDifference(predecessor)
                    if diff > self.theta:
                        pq.update(predecessor, -diff)
