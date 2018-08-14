import numpy as np
import logging
from multiprocessing import Process,Pipe
from threading import Thread
from policy_network import PolicyNetwork
from qlearning_network import DQN
from value_network import ValueNetwork
from mcts0 import MCTSWorker
import chess_rule as rule


logger = logging.getLogger('app')

HUMMAN = 'HUMMAN'
COMPUTER = 'COMPUTER'

class Player:
    def __init__(self, name, stone_val, signal, winner_text, clock, init_board, first_player, type_):
        self.name = name
        self.stone_val = stone_val
        self.signal = signal
        self.winner_text = winner_text
        self.clock = clock
        self.init_board = init_board
        self.first_player = first_player
        self.type_ = type_
        self.begin_time = 0
        self.total_time = 0
        self.stopped = False

    def play(self, board):
        """
        返回(位置,动作)
        """
        raise NotImplementedError

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

    def start(self, init_board, first_player):
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


class HummanPlayer(Player):
    def __init__(self, name, stone_val, signal, winner_text, clock, init_board, first_player):
        Player.__init__(self, name, stone_val, signal, winner_text, clock, init_board=init_board, first_player=first_player, type_=HUMMAN)

    def play(self, board):
        """
        返回(位置,动作)
        人类选手由界面操作，所以不执行任何动作
        """
        pass


class ComputerPlayer(Player):
    def __init__(self, name, stone_val, signal, winner_text, clock, play_func, init_board, first_player, hidden_activation='relu', model_file=None, weights_file=None):
        Player.__init__(self, name, stone_val, signal, winner_text, clock, init_board, first_player, type_=COMPUTER)
        self.play_func = play_func
        self.hidden_activation = hidden_activation
        self.model_file = model_file
        self.weights_file = weights_file
        # self.model = self.load_model()
        self.play_process = None

    def load_model(self):
        raise NotImplemented

    def start(self, init_board, first_player):
        self.play_process = PlayProcess(model_fuc=self.load_model)
        self.play_process.start()
        self.stopped = False

    def stop(self):
        self.play_process.stop()
        self.stopped = True


class COMMAND:
    SEARCH = 'SEARCH'
    MOVE_DOWN = 'MOVE_DOWN'
    PREDICT = 'PREDICT'
    STOP = 'STOP'
    STOP_SEARCH = 'STOP_SEARCH'
    OPP_PLAY = 'OPP_PLAY'


class PlayProcess:
    def __init__(self, model_fuc):
        self.model_fuc = model_fuc
        conn1, conn2 = Pipe()
        self.player_end = conn1
        self.process_end = conn2
        self.stopped = False

    def start(self):
        p = Process(target=self._start)
        p.daemon = True
        p.start()
        self.stopped = False

    def _start(self):
        model = self.model_fuc()

        while True:
            command, args = self.process_end.recv()
            if command == COMMAND.STOP:
                model.close()
                self.process_end.send(0)
                break
            elif command == COMMAND.PREDICT:
                board, stone_val = args
                self.process_end.send(model.predict(board, stone_val))

    def predict(self, board, stone_val):
        self.player_end.send((COMMAND.PREDICT, (board.copy(), stone_val)))
        return self.player_end.recv()

    def stop(self):
        self.player_end.send((COMMAND.STOP, None))
        # self.player_end.recv()


class PolicyNetworkPlayer(ComputerPlayer):
    def load_model(self):
        return PolicyNetwork.load(self.model_file)

    def play0(self, board):
        logger.info('%s play...', self.name)
        from_, action, vp, p = self.play_process.predict(board, self.stone_val)
        to_ = tuple(np.add(from_, rule.actions_move[action]))
        logger.info('from %s to %s', from_, to_)
        return from_, to_, vp, p

    def play(self, board):
        logger.info('%s play...', self.name)
        board_self = rule.flip_board(board) if self.stone_val == -1 else board.copy()
        from_, action, vp, p = self.play_process.predict(board_self, self.stone_val)
        to_ = tuple(np.add(from_, rule.actions_move[action]))
        if self.stone_val == -1:
            from_ = rule.flip_location(from_)
            to_ = rule.flip_location(to_)
            # vp = rule.flip_action_probs(vp)
            p = rule.flip_action_probs(p)
        logger.info('from %s to %s', from_, to_)

        rule.move(board, from_, to_)
        opp_q_table = self.predict_opponent(board)
        logger.debug(opp_q_table)
        self.play_func(self.stone_val, from_, to_, p, opp_q_table)

    def predict_opponent(self, board):
        """
        预测对手走的棋
        """
        from_, action, vp, p = self.play_process.predict(board, -self.stone_val)
        return p


class DQNPlayer(ComputerPlayer):
    def load_model(self):
        return DQN(model_file=self.model_file, weights_file=self.weights_file)

    def play(self, board):
        logger.info('%s play...', self.name)
        board_self = rule.flip_board(board) if self.stone_val == -1 else board.copy()
        (from_, action), (valid, q) = self.play_process.predict(board_self, self.stone_val)
        logger.debug('valid is:%s', valid)
        logger.debug('q is:%s', q)
        logger.debug('from:%s, action:%s', from_, action)
        to_ = tuple(np.add(from_, rule.actions_move[action]))
        q_table = np.zeros((5,5,4))
        for (f, a),q1 in zip(valid,q):
            q_table[f][a] = q1
        if self.stone_val == -1:
            from_ = rule.flip_location(from_)
            to_ = rule.flip_location(to_)
            q_table = rule.flip_action_probs(q_table)
        logger.info('from %s to %s', from_, to_)

        rule.move(board, from_, to_)
        opp_q_table = self.predict_opponent(board)
        logger.debug(opp_q_table)
        self.play_func(self.stone_val, from_, to_, q_table, opp_q_table)

    def predict_opponent(self, board):
        """
        预测对手走的棋
        """
        (from_, action), (valid, q) = self.play_process.predict(board, -self.stone_val)
        q_table = np.zeros((5, 5, 4))
        for (f, a), q1 in zip(valid, q):
            q_table[f][a] = q1
        return q_table


class ValuePlayer(DQNPlayer):
    def load_model(self):
        return ValueNetwork(hidden_activation=self.hidden_activation, output_activation='sigmoid', model_file=self.model_file, weights_file=self.weights_file)


class MCTSPlayer(ComputerPlayer):
    def __init__(self, name, stone_val, signal, winner_text, clock, play_func, policy_model, value_model, init_board, first_player):
        ComputerPlayer.__init__(self, name, stone_val, signal, winner_text, clock, play_func, None, init_board, first_player)
        self.policy_model = policy_model
        self.value_model = value_model
        # self.mcts_process = MCTSProcess(policy_model, value_model, init_board, first_player, stone_val)

    def load_model(self):
        return None

    def play(self, board):
        if self.stone_val == -1:
            board_self = rule.flip_board(board)
        else:
            board_self = board.copy()
        def _play():
            action, q, opp_q = self.play_process.predict(board_self, self.stone_val)
            logger.info('resv: action:%s', action)
            if action is None:
                logger.info('_play thread stop...')
                return
            from_,act = action
            to_ = tuple(np.add(from_, rule.actions_move[act]))
            q_table = np.zeros((5, 5, 4))
            for (f, a), q_ in q:
                q_table[f][a] = q_
            if self.stone_val == -1:
                from_ = rule.flip_location(from_)
                to_ = rule.flip_location(to_)
                q_table = rule.flip_action_probs(q_table)
            # self.play_func(board, self.stone_val, from_, to_, q_table)
            opp_q_table = np.zeros((5, 5, 4))
            for (f, a), q_ in opp_q:
                opp_q_table[f][a] = q_
            self.play_func(self.stone_val, from_, to_, q_table, opp_q=opp_q_table)
        Thread(target=_play).start()

    def opponent_play(self, board, from_, to_):
        """
        对手走的棋
        :param board:   对手走棋之前的局面
        :param from_:
        :param to_:
        """
        player = board[from_]
        assert player == -self.stone_val, str(board) + '\nfrom:' + str(from_) + ' to:' + str(to_)
        act = tuple(np.subtract(to_, from_))
        a = rule.actions_move.index(act)
        action = (from_,a)
        if player == -1:
            board = rule.flip_board(board)
            action = rule.flip_action(action)
        self.play_process.opponent_play(board, player, action)

    def start(self, init_board, first_player):
        self.play_process = MCTSProcess(self.policy_model, self.value_model, init_board, first_player, self.stone_val)
        self.play_process.start()
        self.stopped = False

    def stop(self):
        self.play_process.stop()
        self.stopped = True


class MCTSProcess(PlayProcess):
    """
    MCTS进程
    该进程是非阻塞的，只是发送命令和接收结果
    具体搜索任务由另一个线程MCTSWorker完成
    """
    def __init__(self, policy_model, value_model, init_board, first_player, player):
        PlayProcess.__init__(self, model_fuc=None)
        self.policy_model = policy_model
        self.value_model = value_model
        self.first_player = first_player
        self.player = player
        if first_player == -1:
            init_board = rule.flip_board(init_board)
        self.init_board = init_board

    def _start(self):
        logger.info('start...')
        worker = MCTSWorker(self.player, self.init_board, self.first_player, self.policy_model, self.value_model,
                            predict_callback=lambda a, q, opp_q: self.process_end.send((a, q, opp_q)), min_search=500, min_search_time=10)
        worker.start()

        if self.first_player != self.player:
            worker.begin_search()

        while True:
            command, args = self.process_end.recv()
            logger.info('command:%s, args:\n%s', command, args)
            if command == COMMAND.STOP:
                worker.stop()
                self.process_end.send(0)
                break
            elif command == COMMAND.PREDICT:
                # 走棋
                worker.stop_search()
                board, player = args
                worker.predict(board, player)
            elif command == COMMAND.OPP_PLAY:
                # 对手走棋，先停止搜索，然后沿对手走棋的方向向下移动树
                board, player, action = args
                logger.info('对手走棋:%s stop search', action)
                worker.stop_search()
                # 将对手走的棋进行训练
                worker.opp_play(board, player, action)
                # logger.info('move down along %s', action)
                # self.worker.move_down(action)

        logger.info('...MCTS PROCESS ENDED...')

    def opponent_play(self, board, player, action):
        """
        对手走完棋:action,搜索树需要沿对手走棋的方向移动树的根结点
        """
        self.player_end.send((COMMAND.OPP_PLAY, (board, player, action)))
