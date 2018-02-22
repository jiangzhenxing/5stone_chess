import numpy as np
import time
import os
import chess_rule as rule
import util
import logging
import sys
import threading
from threading import Thread,Event
from multiprocessing import Process,Pipe
from queue import Queue
from qlearning_network import DQN
from value_network import ValueNetwork

logger = logging.getLogger('train')
logger_tree = logging.getLogger('tree')
sys.setrecursionlimit(10000) # 最大递归深度设置为一万


class Scene:
    TRAIN = 'TRAIN'
    PLAY = 'PLAY'


class Node:
    def __init__(self, board, player, tree, level=1, parent_edge=None, expanded=False, final=False, value_decay=1):
        self.board = board
        self.player = player
        self.tree = tree
        self.level = level
        self.parent_edge = parent_edge
        self.expanded = expanded
        self.final = final
        self.value_decay = value_decay
        self.board_str = ''.join(map(str, board.flatten()))
        self.sub_edge = []
        tree.set_depth(level)
        tree.n_node += 1

    def predict(self):
        """
        优势时不重覆走棋
        """
        edges = self.sub_edge.copy()
        me = max(edges, key=lambda e: e.n)
        '''
        while me.v > 0.5 and len(edges) > 1 and (self.board_str, *me.a) in self.tree.predicted:
            edges.remove(me)
            me_ = max(edges, key=lambda e:e.n)
            if me_.v > 0.5:
                me = me_
            else:
                break
        '''
        return me

    def search(self, walked):
        dis = 1
        q = 0
        if self.final:
            value = 0
            # q = 0
            player = self.player
        elif not self.expanded:
            self.expansion()
            e = self.selection(walked)
            value = e.v
            # q = e.q_
            player = self.player
            logger.debug('level:%s, value:%s, _q, player:%s', self.level, value, q, player)
        elif self.level > 1000:
            e = self.selection(walked)
            value = e.v
            # q = e.q_
            player = self.player
        else:
            e = self.selection(walked)
            # walked.add((self.board_str, *e.a))
            value, q, player, dis = e.down_node.search(walked)
        self.update(value, q, player, dis)
        return value, q, player, dis + 1

    def selection(self, walked):
        """
        对于树选手：
        返回Q值最大的一条边
        对于模拟走棋的对手：
        以epsilon的概率选择之前走的走法
        1-epsilon的概率选Q值最大的走法
        """
        if self.tree.scene == Scene.PLAY:
            if self.player == self.tree.player:
                # 选Q值最大的走法
                return self.selection_by_Q(walked)
            else:
                # 以epsilon的概率选择之前走的走法
                # 1-epsilon的概率选Q值最大的走法
                return self.selection_epsilon(walked)
        elif self.tree.scene == Scene.TRAIN:
            # 训练时按概率走棋
            return self.selection_by_probs()

    def selection_by_probs(self):
        """
        在训练的时候
        以softmax(N ** 1/t)的概率选择走法
        """
        N = [e.n ** (1/self.tree.t) for e in self.sub_edge]
        probs = util.softmax(N)
        return util.select_by_prob(self.sub_edge, probs)

    def selection_by_Q(self, walked):
        """
        返加Q值最大且之前未走过的一条边
        """
        edges = self.sub_edge.copy()
        me = max(edges, key=lambda e:e.q())
        '''
        while me.v > 0.5 and len(edges) > 1 and (self.board_str, *me.a) in walked:
            edges.remove(me)
            me_ = max(edges, key=lambda e:e.q())
            if me_.v > 0.5:
                me = me_
            else:
                break
        '''
        return me

    def selection_epsilon(self, walked):
        """
        以epsilon的概率选择之前走的边
        1-epsilon的概率选Q值最大的边
        """
        if self.board_str in self.tree.opp_actions:
            actions = self.tree.opp_actions[self.board_str]  # {action:n}
            epsilon = 0.7
            if np.random.random() < epsilon:
                a, n = zip(*actions.items())
                p = util.softmax(n)
                a = util.select_by_prob(a, p)
                for e in self.sub_edge:
                    if e.a == a:
                        return e
                raise ValueError(a) # 如果没找到就是不正常的
        return self.selection_by_Q(walked)

    def policy(self):
        pass

    def expansion(self):
        board,player = self.board, self.player
        actions = rule.valid_actions(board, player)
        # actions_ = list(filter(lambda a:(self.board_str, *a) not in walked, actions))
        # if len(actions_) == 0:
            # 全部已经走过，重新选
            # actions_ = actions
        if self.player == self.tree.player:
            with self.tree.value_model_lock:
                values = [self.tree.value_model.q(board, from_, act) for from_, act in actions]
        else:
            with self.tree.opp_value_model_lock:
                values = [self.tree.opp_value_model.q(board, from_, act) for from_, act in actions]

        probs = ValueNetwork.value_to_probs(values)
        for a,v,p in zip(actions, values, probs):
            e = Edge(upper_node=self, a=a, v=v, p=p, lambda_=self.tree.lambda_)
            self.add_edge(e)
        self.expanded = True
        # assert len(self.sub_edge) > 0, 'board:\n' + str(self.board) + '\nplayer:' + str(self.player)

    def update(self, value, q, player, dis):
        if self.parent_edge:
            pe = self.parent_edge
            pe.n += 1
            '''
            if pe.upper_node.player != player:
                value = 1 - value
            '''
            # 只使用己方的值更新
            if pe.upper_node.player == player:
                pe.n_update += 1
                value = value * self.value_decay ** dis
                # 更新价值
                pe.v = pe.v + (value - pe.v) / pe.n_update
                # 更新Q值
                # pe.q_ = pe.q_ + (q - pe.q_) / pe.n_update
            # logger.info('player:%s, value:%s, old v:%s, new v:%s', pe.upper_node.player, value, v0, pe.v)

    def add_edge(self, edge):
        self.sub_edge.append(edge)


    def __str__(self):
        return '%s, player:%s' % (self.board,self.player)

    def show(self):
        logger.debug('board is:\n%s', self.board)
        logger.debug('player: %s, level:%s', self.player, self.level)
        logger.debug('sub_edge:')
        for e in self.sub_edge:
            logger.debug(e)


class Edge:
    """
    EDGE: N, W, V, P
    W = N(win)
    V = value(action)
    P = policy(board,player)
    Q = λV + (1-λ)W/N
    U = P/(N+1)
    """
    def __init__(self, upper_node, a, v, p, lambda_):
        self.upper_node = upper_node
        self.a = a
        self.v = v
        self.v_ = v
        self.l = lambda_
        self.p = p
        self.n = 1
        self.n_update = 1
        self.w = 1
        board, player = upper_node.board.copy(), upper_node.player
        result, _ = rule.move_by_action(board, *a)
        self.down_node = Node(board=rule.flip_board(board), player=-player, tree=upper_node.tree, level=upper_node.level+1, parent_edge=self, final=result==rule.WIN)
        self.win = result == rule.WIN
        assert p != np.nan
        assert v != np.nan
        if self.win:
            self.v = 1 + 1e-15  # 1.0005

    def q(self):
        if self.win:
            return 2
        q = self.v
        u = self.p / self.n
        return q + u

    def board(self):
        return self.upper_node.board

    def __str__(self):
        return 'a:%s,n:%s,q:%s' % (self.a,self.n,self.q())


class MCTS:
    """
    蒙特卡罗树搜索
    a = argmaxQ
    """
    def __init__(self, player, init_board, first_player, policy_model, value_model,
                 expansion_gate=50, lambda_=0.5, min_search=500, max_search=10000,
                 min_search_time=15, max_search_time=300, scene=Scene.PLAY, t=100, t_decay = 0.0002):
        self.player = player
        self.expansion_gate = expansion_gate
        self.lambda_ = lambda_
        self.min_search = min_search            # 最小的搜索次数
        self.max_search = max_search            # 最大的搜索次数(超过即停止搜索)
        self.max_search_time = max_search_time  # 最大搜索时间(s)(超过即停止搜索)
        self.min_search_time = min_search_time  # 最小搜索时间(s)(同时满足最小搜索次数和最小搜索时间时，停止搜索)
        self.searching = False
        self.stop = False
        self.stop_event = Event()
        self.depth = 0
        self.n_node = 0
        self.n_search = 0
        self.scene = scene
        self.t = t                              # 训练的温度，逐渐减小，用以控制概率
        self._t = t
        self.t_decay = t_decay

        logger.info('value_model: %s', value_model)
        if isinstance(value_model, str):
            value_model = DQN(hidden_activation='relu',lr=0.001, weights_file=value_model) if 'DQN' in value_model else ValueNetwork(hidden_activation='relu',lr=0.001, weights_file=value_model)
        # self.policy = value_model
        logger.info('value_model: %s', value_model)
        self.value_model = value_model
        self.value_model_lock = threading.Lock()
        # 模拟对手走棋时用的Q值由opp_value预测，走棋的同时学习对手走棋的习惯
        self.opp_value_model = type(value_model)(hidden_activation='relu',lr=0.001)
        self.opp_value_model.copy(value_model)
        logger.info('opp_value_model: %s', self.opp_value_model)
        self.opp_value_model_lock = threading.Lock()
        self.root = Node(init_board, first_player, tree=self)
        self.predicted = set()  # 树中已经走过的走法 (board_str, from, act)
        self.root.expansion()
        self.opp_actions = {}       # 对手走过的棋局, {board:{action:n}]

    def search(self, min_search=0, max_search=0, min_search_time=0, max_search_time=0):
        logger.info('begin search...')
        self.searching = True
        if min_search == 0: min_search = self.min_search
        if max_search == 0: max_search = self.max_search
        if min_search_time == 0: min_search_time = self.min_search_time
        if max_search_time == 0: max_search_time = self.max_search_time
        start_time = time.time()
        while not self.stop and self.n_search < max_search and time.time()-start_time < max_search_time:
            if self.n_search > min_search and time.time()-start_time > min_search_time:
                break
            self.value_model.predicts.update(self.predicted)
            walked = set() # self.predicted.copy()
            self.root.search(walked)
            self.n_search += 1
            self.value_model.clear()
            self.value_model.episode += 1
            logger.debug('search %s', self.n_search)
        logger.info('search over %s', self.n_search)
        self.n_search = 0
        self.show_info()
        self.searching = False
        if self.stop:
            self.stop_event.set()

    def search_forever(self):
        self.search(min_search=2**32, max_search=2**32, min_search_time=2**32, max_search_time=2**32)

    def stop_search(self):
        if not self.searching:
            return
        self.stop_event.clear()
        self.stop = True
        # 等待搜索结束
        self.stop_event.wait()
        self.stop = False
        self.stop_event.clear()
        logger.info('search stopped...')

    def predict(self, board, player):
        logger.info('begin predict...')
        assert (self.root.board == board).all() and self.root.player == player, '%s,player:%s\nroot:\n%s,player:%s' % (board, player, self.root.board, self.root.player)
        self.search()
        e = self.root.predict()
        action = e.a
        self.predicted.add((self.root.board_str, *action))
        q = [(e.a,e.q()) for e in self.root.sub_edge]
        logger.info('predict is:%s', action)
        return action, q

    def move_down(self, board, player, action):
        assert np.all(self.root.board == board), 'root_board:\n' + str(self.root.board) + '\nboard:\n' + str(board)
        assert self.root.player == player, 'root_player:%s, player:%s' % (self.root.player, player)
        node = self.get_node(action)
        logger.debug('get_node(%s):\n%s', action, node)
        if node is None:
            logger.info('node is None, new Node()')
            board = board.copy()
            rule.move_by_action(board, *action)
            node = Node(rule.flip_board(board), -player, tree=self)
        if not node.expanded:
            node.expansion()
        self.root = node
        self.root.parent_edge = None
        self.root.level = 1
        self.n_node = 1
        self.depth = 1
        self.update_tree_info(self.root)
        logger.debug('move down to node:%s', action)
        # self.show_info()

    def update_tree_info(self, node):
        for e in node.sub_edge:
            sub_node = e.down_node
            sub_node.level = node.level + 1
            self.set_depth(sub_node.level)
            self.n_node += 1
            self.update_tree_info(sub_node)

    def get_node(self, action):
        for e in self.root.sub_edge:
            if e.a == action:
                return e.down_node

    def set_depth(self, level):
        if level > self.depth:
            self.depth = level

    def clear(self):
        self.stop = False
        self.n_search = 0

    def decay_t(self, episode):
        return self._t / (1 + self.t_decay * episode)

    def show_info(self):
        info = '\n------------- tree info --------------\n' \
               'depth:%s, n_node:%s\n' \
               'root: \n%s'
        logger_tree.info(info, self.depth, self.n_node, self.root)
        for e in self.root.sub_edge:
            logger_tree.info('edge:%s, v:%s, p:%s, N:%s, q:%s', e.a, e.v, e.p, e.n, e.q())
            logger_tree.debug(e.down_node)


class COMMAND:
    SEARCH = 'SEARCH'
    MOVE_DOWN = 'MOVE_DOWN'
    PREDICT = 'PREDICT'
    STOP = 'STOP'
    STOP_SEARCH = 'STOP_SEARCH'
    OPP_PLAY = 'OPP_PLAY'


class MCTSWorker:
    def __init__(self, player, init_board, first_player, policy_model, value_model, predict_callback, min_search=500, max_search=10000, min_search_time=15, max_search_time=120, expansion_gate=50):
        self.player = player
        self.init_board = init_board
        self.first_player = first_player
        self.policy_model = policy_model
        self.value_model = value_model
        self.min_search = min_search
        self.max_search = max_search
        self.expansion_gate = expansion_gate
        self.min_search_time = min_search_time
        self.max_search_time = max_search_time
        self.predict_callback = predict_callback
        self.command_queue = Queue()
        # self.result_queue = Queue()
        self.stopped = False
        self.opp_records = []   # 对手走棋记录
        self.records = []       # 走棋记录
        self.boards = set()     # 已经出现过的棋局: {(board,player)}
        # self.boards.add((util.board_str(init_board), first_player))
        self.started_event = Event()

    def start(self):
        Thread(target=self._start).start()
        self.started_event.wait()

    def _start(self):
        self.ts = MCTS(self.player, self.init_board, self.first_player, self.policy_model, self.value_model, lambda_=0, min_search=self.min_search, max_search=self.max_search,
                       min_search_time=self.min_search_time, max_search_time=self.max_search_time, expansion_gate=self.expansion_gate)
        self.ts.show_info()
        self.started_event.set()
        while True:
            command, board, player, action = self.command_queue.get()
            logger.info('\nget: %s, \n%s\n player:%s action:%s', command, board, player, action)
            if command == COMMAND.STOP:
                # if K.backend() == 'tensorflow':
                #     import keras.backend.tensorflow_backend as tfb
                #     tfb.clear_session()
                logger.info('MCTS WORKER ENDING...')
                break
            if command == COMMAND.PREDICT:
                self.train_circle(board, player)
                board_str = util.board_str(board)
                self.boards.add((board_str, player))
                logger.info('(board_str, player):%s', (board_str, player))
                # 走棋
                logger.info('走棋......')
                a, q = self.ts.predict(board, player)
                self.records.append((board.copy(), *a, 0.5, None))
                self.ts.move_down(self.ts.root.board, self.ts.root.player, a)
                opp_q = [(e.a, e.q()) for e in self.ts.root.sub_edge]

                if self.stopped:
                    self.predict_callback(None, None, None)
                else:
                    self.predict_callback(a, q, opp_q)
                    self.ts.search_forever()
            # elif command == COMMAND.MOVE_DOWN:
            #     # 对手走棋，向下移动树
            #     logger.info('move down...')
            #     self.ts.move_down(board, player, action)
            elif command == COMMAND.OPP_PLAY:
                self.train_circle(board, player)
                board_str = util.board_str(board)
                self.boards.add((board_str, player))
                logger.info('(board_str, player):%s', (board_str, player))
                self.records.append((board.copy(), *action, 0.5, None))
                # 将当前动作放进opp_actions里
                if board_str not in self.ts.opp_actions:
                    self.ts.opp_actions[board_str] = {}
                self.ts.opp_actions[board_str][action] = self.ts.opp_actions[board_str].get(action, 0) + 1

                # Q值最大的边
                max_e = max(self.ts.root.sub_edge, key=lambda e: e.v_)
                # 当前动作的边
                action_e = next(filter(lambda e: e.a == action, self.ts.root.sub_edge))
                # 将最大Q值与当前动作的Q值互换
                self.opp_records.append((board, *action, max_e.v_, None))
                self.opp_records.append((board, *max_e.a, action_e.v_, None))
                logger.info('maxq: %s, q: %s', max_e.v_, action_e.v_)

                # 使用调整后的动作的Q值进行训练，来更好地预测对手的走棋
                with self.ts.opp_value_model_lock:
                    self.ts.opp_value_model.train(records=self.opp_records if len(self.opp_records) < 6 else self.opp_records[-6:], epochs=3)
                self.ts.move_down(board, player, action)
            elif command == COMMAND.SEARCH:
                self.ts.search_forever()
            logger.info('records:%s', len(self.records))
        logger.info('...MCTS WORKER ENDED...')

    def train_circle(self, board, player):
        """
        检查棋局是否有环(棋局是否出现过)，如果有就将模型进行调整
        """
        logger.info('(board_str, player):%s', (util.board_str(board),player))
        if (util.board_str(board),player) in self.boards:
            finded = False
            for i in range(len(self.records) - 1, -1, -1):
                b, f, a, _, _ = self.records[i]
                if (b == board).all() and b[f] == player:
                    finded = True
                    break
            assert finded, (board, player)
            circle = self.records[i:]
            with self.ts.value_model_lock:
                self.ts.value_model.train(circle, epochs=3)
            with self.ts.opp_value_model_lock:
                self.ts.opp_value_model.train(circle, epochs=3)
            '''
            self.records = self.records[:i]
            for b, f, a, _, _ in circle:
                self.boards.remove((util.board_str(b), b[f]))
            '''
            logger.info('i:%s, 环:%s, records:%s', i, len(circle), len(self.records))
        return board

    def send(self, command, board=None, player=None, action=None):
        self.command_queue.put((command, board, player, action))

    def predict(self, board, player):
        self.send(COMMAND.PREDICT, board, player)

    def begin_search(self):
        self.send(COMMAND.SEARCH)

    def stop_search(self):
        self.started_event.wait()
        self.ts.stop_search()

    def move_down(self, action):
        raise NotImplemented

    def opp_play(self, board, player, action):
        self.send(COMMAND.OPP_PLAY, board, player, action)

    def stop(self):
        self.stopped = True
        self.stop_search()
        self.send(COMMAND.STOP)


class SimulateProcess:
    def __init__(self, record_queue, model_queue, weight_lock, weights_file, init_board='random', epsilon=1.0, epsilon_decay=2e-3, begin=0):
        self.record_queue = record_queue
        self.model_queue = model_queue
        self.weight_lock = weight_lock
        self.weights_file = weights_file
        self.init_board = init_board
        self.epsilon = epsilon
        self._epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.begin = begin
        self.episode = 0
        self.predicts = set()

    def _start(self):
        logger.info('start...')
        np.random.seed(os.getpid())
        logger.info('random:%s', [np.random.random() for _ in range(3)])
        value_model = ValueNetwork(hidden_activation='selu', output_activation='sigmoid')
        for i in range(self.begin, 2 ** 32):
            logger.info('simulate %s', i)
            self.episode = i
            self.decay_epsilon()
            board = rule.random_init_board() if self.init_board == 'random' else rule.init_board()
            player = 1
            # value_model_params = self.model_queue.get()
            with self.weight_lock:
                value_model.model.load_weights(self.weights_file)
            ts = MCTS(init_board=board.copy(), player=player, policy_model=None, value_model=value_model, max_search=50, min_search_time=0, scene=Scene.TRAIN)
            ts.decay_t(episode=i)
            records, winner = self.simulate(ts, board, player)
            if records.length() == 0:
                continue
            self.record_queue.put(records)
            if i % 100 == 0:
                records.save('records/train/alpha0/1st_%03d_' % (i//100))

    def epsilon_greedy(self, board, player, valid_action, ts):
        """
        使用epsilon-greedy策略选择下一步动作
        以epsilon以内的概率随机选择动作
        以1-epsilon的概率选择最大Q值的动作
        :return: 下一个动作: (from,to)
        """
        if np.random.random() > self.epsilon:
            # 选取Q值最大的
            action, q = ts.predict(board, player)
            return action,q
        else:
            # 随机选择
            return util.random_choice(valid_action), None

    def simulate(self, ts, board, player):
        from record import Record
        from value_network import NoActionException
        records = Record()
        while True:
            try:
                bd = board.copy()
                board_str = util.board_str(board)
                valid_action = rule.valid_actions(board, player)
                while True:
                    (from_, act), q = self.epsilon_greedy(board, player, valid_action, ts)
                    if (board_str,from_, act) not in self.predicts or len(ts.root.sub_edge) == 1:
                        break
                    ts.root.sub_edge = [e for e in ts.root.sub_edge if e.a != (from_,act)]
                    valid_action.remove((from_,act))
                assert board[from_] == player
                ts.move_down(board, player, action=(from_, act))
                if self.episode % 10 == 0:
                    logger.info('action:%s,%s', from_, act)
                    logger.info('q is %s', q)
                to_ = tuple(np.add(from_, rule.actions_move[act]))
                command, eat = rule.move(board, from_, to_)
                records.add3(bd, from_, act, len(eat), win=command == rule.WIN)
            except NoActionException:
                # 随机初始化局面后一方无路可走
                return Record(), 0
            except Exception as ex:
                logging.warning('board is:\n%s', board)
                logging.warning('player is: %s', player)
                valid = rule.valid_actions(board, player)
                logging.warning('valid is:\n%s', valid)
                logging.warning('from_:%s, act:%s', from_, act)
                ts.show_info()
                records.save('records/train/1st_')
                raise ex
            if command == rule.WIN:
                logging.info('%s WIN, step use: %s, epsilon:%s', str(player), records.length(), self.epsilon)
                return records, player
            if records.length() > 10000:
                logging.info('走子数过多: %s', records.length())
                return Record(), 0
            player = -player
            board = rule.flip_board(board)

    def decay_epsilon(self):
        self.epsilon = self._epsilon / (1 + self.epsilon_decay * (1 + self.episode))

    def start(self):
        Process(target=self._start).start()


def train():
    from multiprocessing import Queue, Lock
    import os
    record_queue = Queue()
    model_queue = Queue()
    weight_lock = Lock()
    weights_file = 'model/alpha0/weights'
    model_file = 'model/alpha0/value_network_00067h.model'
    begin = 6700
    _value_model = ValueNetwork(hidden_activation='selu', output_activation='sigmoid', model_file=model_file)
    value_model = ValueNetwork(hidden_activation='selu', output_activation='sigmoid')
    value_model.copy(_value_model)

    if os.path.exists(weights_file):
        value_model.model.load_weights(weights_file)
    else:
        value_model.model.save_weights(weights_file)

    for _ in range(3):
        SimulateProcess(record_queue, model_queue, weight_lock, weights_file, init_board='random', epsilon=1.0, epsilon_decay=2e-3, begin=begin//3).start()

    for i in range(begin+1, 2 ** 32):
        records = record_queue.get()
        logger.info('train %s, records:%s', i, len(records))
        value_model.train(records, epochs=5)
        with weight_lock:
            value_model.model.save_weights(weights_file)
        if i % 100 == 0:
            value_model.save_model('model/alpha0/value_network_%05dh.model' % (i // 100))


def main():
    board = rule.init_board()
    player = 1
    ts = MCTS(board, player)
    # ts.search()
    a,q = ts.predict()
    print(board)
    print('action:', a)
    print('q:', q)

if __name__ == '__main__':
    import logging.config
    logging.config.fileConfig('logging.conf')
    train()