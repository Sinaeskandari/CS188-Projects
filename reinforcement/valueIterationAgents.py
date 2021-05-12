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

from learningAgents import ValueEstimationAgent
import collections

import gridworld


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp: gridworld.Gridworld, discount=0.9, iterations=100):
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
        for _ in range(self.iterations):
            values = self.values.copy()
            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state):
                    # first state is 'TERMINAL_STATE'
                    continue
                q_values = []
                for action in self.mdp.getPossibleActions(state):
                    q_value = 0
                    for t in self.mdp.getTransitionStatesAndProbs(state, action):
                        q_value += (t[1] * (
                                self.mdp.getReward(state, action, t[0]) + (self.discount * self.values[t[0]])))
                    q_values.append(q_value)
                values[state] = max(q_values)
            self.values = values

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
        q = 0
        for t in self.mdp.getTransitionStatesAndProbs(state, action):
            q += (t[1] * (self.mdp.getReward(state, action, t[0]) + (self.discount * self.values[t[0]])))
        return q

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
            return
        q_vals = {}
        for action in self.mdp.getPossibleActions(state):
            q_vals[self.computeQValueFromValues(state, action)] = action
        return q_vals[max(q_vals)]

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
        state_count = len(self.mdp.getStates())
        for i in range(self.iterations):
            values = self.values.copy()
            state_index = i % state_count
            state = self.mdp.getStates()[state_index]
            if self.mdp.isTerminal(state):
                continue
            q_values = []
            for action in self.mdp.getPossibleActions(state):
                q_value = 0
                for t in self.mdp.getTransitionStatesAndProbs(state, action):
                    q_value += (t[1] * (
                            self.mdp.getReward(state, action, t[0]) + (self.discount * self.values[t[0]])))
                q_values.append(q_value)
            values[state] = max(q_values)
            self.values = values


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
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        predecessors = {}
        for state in self.mdp.getStates():
            if self.mdp.isTerminal(state):
                continue
            for action in self.mdp.getPossibleActions(state):
                for t in self.mdp.getTransitionStatesAndProbs(state, action):
                    try:
                        predecessors[t[0]].add(state)
                    except KeyError:
                        predecessors[t[0]] = set()
                        predecessors[t[0]].add(state)
        priority_queue = util.PriorityQueue()
        for state in self.mdp.getStates():
            if self.mdp.isTerminal(state):
                continue
            max_q_value = 0
            q_values = []
            actions = self.mdp.getPossibleActions(state)
            if len(actions) != 0:
                for action in actions:
                    q_values.append(self.computeQValueFromValues(state, action))
                max_q_value = max(q_values)
            diff = abs(self.values[state] - max_q_value)
            priority_queue.push(state, -diff)
        for i in range(self.iterations):
            if priority_queue.isEmpty():
                return
            state = priority_queue.pop()
            if not self.mdp.isTerminal(state):
                q_values = []
                for action in self.mdp.getPossibleActions(state):
                    q_values.append(self.computeQValueFromValues(state, action))
                self.values[state] = max(q_values)
            for p in predecessors[state]:
                if not self.mdp.isTerminal(p):
                    max_q_value = 0
                    q_values = []
                    actions = self.mdp.getPossibleActions(p)
                    if len(actions) != 0:
                        for action in actions:
                            q_values.append(self.computeQValueFromValues(p, action))
                        max_q_value = max(q_values)
                    diff = abs(self.values[p] - max_q_value)
                    if diff > self.theta:
                        priority_queue.update(p, -diff)
