from policy_network import PolicyNetwork, RolloutPolicyNetwork, ConvolutionPolicyNetwork
import chess_rule as rule
import numpy as np
import logging

logger = logging.getLogger('app')

HUMMAN = 'HUMMAN'
COMPUTER = 'COMPUTER'

class Player:
    def __init__(self, name, stone_val, signal, winner_text, clock, type_):
        self.name = name
        self.stone_val = stone_val
        self.signal = signal
        self.winner_text = winner_text
        self.clock = clock
        self.type_ = type_
        self.begin_time = 0
        self.total_time = 0

    def play(self, board):
        """
        返回(位置,动作)
        人类选手由界面操作，所以不执行任何动作
        """
        pass

    def is_humman(self):
        return self.type_ == HUMMAN

    def __str__(self):
        return self.name + ':' + self.type_

    def __repr__(self):
        return str(self)


class HummaPlayer(Player):
    def __init__(self, name, stone_val, signal, winner_text, clock):
        Player.__init__(self, name, stone_val, signal, winner_text, clock, type_=HUMMAN)


class ComputerPlayer(Player):
    def __init__(self, name, stone_val, signal, winner_text, clock):
        Player.__init__(self, name, stone_val, signal, winner_text, clock, type_=COMPUTER)


class PolicyNetworkPlayer(ComputerPlayer):
    def __init__(self, name, stone_val, signal, winner_text, clock, modelfile):
        ComputerPlayer.__init__(self, name, stone_val, signal, winner_text, clock)
        self.policy = PolicyNetwork.load(modelfile)

    def play(self, board):
        logger.info('%s play...', self.name)
        from_, action = self.policy.policy(board, self.stone_val)
        to_ = tuple(np.add(from_, rule.actions_move[action]))
        logger.info('from %s to %s', from_, to_)
        return from_, to_