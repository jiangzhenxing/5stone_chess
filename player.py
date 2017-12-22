import numpy as np
import logging
from multiprocessing import Process,Pipe
from threading import Thread
from policy_network import PolicyNetwork
from qlearning_network import DQN
import chess_rule as rule


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

    def opponent_play(self, board, from_, to_):
        """
        对手走的棋
        """
        pass

    def predict_opponent(self, board):
        """
        预测对手走的棋的概率或Q值
        """
        pass

    def start(self):
        """
        启动
        """
        pass

    def is_humman(self):
        return self.type_ == HUMMAN

    def stop(self):
        pass

    def __str__(self):
        return self.name + ':' + self.type_

    def __repr__(self):
        return str(self)


class HummaPlayer(Player):
    def __init__(self, name, stone_val, signal, winner_text, clock):
        Player.__init__(self, name, stone_val, signal, winner_text, clock, type_=HUMMAN)


class ComputerPlayer(Player):
    def __init__(self, name, stone_val, signal, winner_text, clock, play_func, modelfile):
        Player.__init__(self, name, stone_val, signal, winner_text, clock, type_=COMPUTER)
        self.play_func = play_func
        self.modelfile = modelfile
        self.model = self.load_model()
    def load_model(self):
        raise NotImplemented


class PolicyNetworkPlayer(ComputerPlayer):
    # def __init__(self, name, stone_val, signal, winner_text, clock, play_func, modelfile):
    #     ComputerPlayer.__init__(self, name, stone_val, signal, winner_text, clock, play_func,)
    #     self.model = PolicyNetwork.load(modelfile)
    def load_model(self):
        return PolicyNetwork.load(self.modelfile)

    def play0(self, board):
        logger.info('%s play...', self.name)
        from_, action, vp, p = self.model.policy(board, self.stone_val)
        to_ = tuple(np.add(from_, rule.actions_move[action]))
        logger.info('from %s to %s', from_, to_)
        return from_, to_, vp, p

    def play(self, board):
        logger.info('%s play...', self.name)
        board_self = rule.flip_board(board) if self.stone_val == -1 else board.copy()
        from_, action, vp, p = self.model.predict(board_self, self.stone_val)
        to_ = tuple(np.add(from_, rule.actions_move[action]))
        if self.stone_val == -1:
            from_ = rule.flip_location(from_)
            to_ = rule.flip_location(to_)
            # vp = rule.flip_action_probs(vp)
            p = rule.flip_action_probs(p)
        logger.info('from %s to %s', from_, to_)
        self.play_func(board, self.stone_val, from_, to_, p)

    def predict_opponent(self, board):
        """
        预测对手走的棋
        """
        from_, action, vp, p = self.model.policy(board, -self.stone_val)
        return p


class DQNPlayer(ComputerPlayer):
    # def __init__(self, name, stone_val, signal, winner_text, clock, modelfile):
    #     ComputerPlayer.__init__(self, name, stone_val, signal, winner_text, clock)
    #     self.model = DQN.load(modelfile)
    def load_model(self):
        return DQN.load(self.modelfile)

    def play0(self, board):
        logger.info('%s play...', self.name)
        from_, action, vp, p = self.model.policy(board, self.stone_val)
        to_ = tuple(np.add(from_, rule.actions_move[action]))
        logger.info('from %s to %s', from_, to_)
        # return from_, to_, vp, p
        self.play_func(board, self.stone_val, from_, to_, p)

    def play(self, board):
        logger.info('%s play...', self.name)
        board_self = rule.flip_board(board) if self.stone_val == -1 else board.copy()
        (from_, action), (valid, q) = self.model.predict(board_self, self.stone_val)
        logger.info('valid is:%s', valid)
        logger.info('q is:%s', q)
        logger.info('from:%s, action:%s', from_, action)
        to_ = tuple(np.add(from_, rule.actions_move[action]))
        q_table = np.zeros((5,5,4))
        for (f, a),q1 in zip(valid,q):
            q_table[f][a] = q1
        if self.stone_val == -1:
            from_ = rule.flip_location(from_)
            to_ = rule.flip_location(to_)
            q_table = rule.flip_action_probs(q_table)
        logger.info('from %s to %s', from_, to_)
        self.play_func(board, self.stone_val, from_, to_, q_table)

    def predict_opponent(self, board):
        """
        预测对手走的棋
        """
        (from_, action), (valid, q) = self.model.predict(board, -self.stone_val)
        q_table = np.zeros((5, 5, 4))
        for (f, a), q1 in zip(valid, q):
            q_table[f][a] = q1
        return q_table

class MCTSPlayer(ComputerPlayer):
    def __init__(self, name, stone_val, signal, winner_text, clock, play_func, modelfile, init_board, first_player):
        ComputerPlayer.__init__(self, name, stone_val, signal, winner_text, clock, play_func, modelfile)
        conn1, conn2 = Pipe()
        self.player_end = conn1
        self.mcts_end = conn2
        self.init_board = init_board
        self.first_player = first_player


    def load_model(self):
        return None

    def play(self, board):
        if self.stone_val == -1:
            board_self = rule.flip_board(board)
        else:
            board_self = board.copy()
        self.player_end.send((board_self, self.stone_val, None))
        def wait_result_to_play():
            (from_, action), q = self.player_end.recv()
            logger.info('resv: from:%s, action:%s', from_, action)
            to_ = tuple(np.add(from_, rule.actions_move[action]))
            q_table = np.zeros((5, 5, 4))
            for (f, a), q_ in q:
                q_table[f][a] = q_
            if self.stone_val == -1:
                from_ = rule.flip_location(from_)
                to_ = rule.flip_location(to_)
                q_table = rule.flip_action_probs(q_table)
            self.play_func(board, self.stone_val, from_, to_, q_table)
        Thread(target=wait_result_to_play).start()

    def opponent_play(self, board, from_, to_):
        if board is None:
            self.player_end.send((None, None, None))
            return
        player = board[from_]
        assert player == -self.stone_val, str(board) + '\nfrom:' + str(from_) + ' to:' + str(to_)
        act = tuple(np.subtract(to_, from_))
        a = rule.actions_move.index(act)
        action = (from_,a)
        if player == -1:
            board = rule.flip_board(board)
            action = rule.flip_action(action)
        self.player_end.send((board, player, action))

    def start(self):
        Process(target=self._start).start()

    def _start(self):
        logger.info('start...')
        from mcts import MCTSWorker
        board = self.init_board
        if self.first_player == -1:
            board = rule.flip_board(board)
        ts_worker = MCTSWorker(board, self.first_player, max_search=200, expansion_gate=10)
        if self.first_player != self.stone_val:
            # 对手走棋时，开始搜索
            ts_worker.begin_search()
        while True:
            board, player, action = self.mcts_end.recv()
            logger.info('\nrecv: \n%s\n player:%s action:%s', board, player, action)
            if board is None:
                ts_worker.stop()
                break
            if action is None:
                # 走棋
                a,q = ts_worker.predict(board, player)
                self.mcts_end.send((a,q))
                # 对手走棋时继续搜索
                ts_worker.begin_search()
            else:
                # 对手走棋，向下移动树
                logger.info('对手走棋:%s stop search', action)
                ts_worker.stop_search()
                logger.info('move down along %s', action)
                ts_worker.move_down(action)
        logger.info('...MCTS PROCESS ENDED...')

    def stop(self):
        self.player_end.send((None, None, None))