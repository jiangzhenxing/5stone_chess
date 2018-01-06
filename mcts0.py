import numpy as np
import time
from threading import Thread,Event
from multiprocessing import Process,Pipe
from queue import Queue
from policy_network import PolicyNetwork
from qlearning_network import DQN
from value_network import ValueNetwork
import chess_rule as rule
import util
import logging

logger = logging.getLogger('train')
logger_tree = logging.getLogger('tree')

# policy = PolicyNetwork.load('model/model_149/convolution_policy_network_5995.model')
# worker = policy
# expansion_gate = 25
# lambda_ = 0
import sys
sys.setrecursionlimit(10000) # 最大递归深度设置为一万

class Node:
    def __init__(self, board, player, tree, level=1, parent_edge=None, expanded=False, final=False):
        self.board = board
        self.player = player
        self.tree = tree
        self.level = level
        self.parent_edge = parent_edge
        self.expanded = expanded
        self.final = final
        self.board_str = ''.join(map(str, board.flatten()))
        self.sub_edge = []
        tree.set_depth(level)
        tree.n_node += 1

    def predict(self):
        return max(self.sub_edge, key=lambda e: e.n)

    def search(self, walked):
        if self.final:
            value = 0
            player = self.player
        elif not self.expanded:
            self.expansion(walked)
            e = self.selection()
            value = e.v
            player = self.player
            logger.debug('level:%s, value:%s, player:%s', self.level, value, player)
        elif self.level > 1000:
            e = self.selection()
            value = e.v
            player = self.player
        else:
            e = self.selection()
            walked.add((self.board_str, self.player, e.a))
            value, player = e.down_node.search(walked)
        self.update(value, player)
        return value, player

    def update(self, value, player):
        if self.parent_edge:
            pe = self.parent_edge
            if pe.upper_node.player != player:
                value = 1 - value
            pe.n += 1
            # v0 = pe.v
            pe.v = pe.v + (value - pe.v) / pe.n
            # logger.info('player:%s, value:%s, old v:%s, new v:%s', pe.upper_node.player, value, v0, pe.v)

    def selection(self):
        """
        返加Q值最大的一条边
        """
        return max(self.sub_edge, key=lambda e:e.q())

    def policy(self):
        pass

    def expansion(self, walked):
        board,player = self.board, self.player
        actions = rule.valid_actions(board, player)
        actions_ = list(filter(lambda a:(self.board_str, player, a) not in walked, actions))
        if len(actions_) == 0:
            # 全部已经走过，重新选
            actions_ = actions
        values = [self.tree.value.q(board, from_, act) for from_,act in actions_]
        for action,value in zip(actions_, values):
            e = Edge(upper_node=self, a=action, v=value, p=0, lambda_=self.tree.lambda_)
            self.add_edge(e)
        values = [e.v for e in self.sub_edge]
        probs = DQN.value_to_probs(values)
        for p,e in zip(probs,self.sub_edge):
            e.p = p
        self.expanded = True
        # assert len(self.sub_edge) > 0, 'board:\n' + str(self.board) + '\nplayer:' + str(self.player)

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
        self.l = lambda_
        self.p = p
        self.n = 1
        self.w = 1
        board, player = upper_node.board.copy(), upper_node.player
        result, _ = rule.move_by_action(board, *a)
        self.down_node = Node(board=rule.flip_board(board), player=-player, tree=upper_node.tree, level=upper_node.level+1, parent_edge=self, final=result==rule.WIN)
        self.win = result == rule.WIN
        assert p != np.nan
        assert v != np.nan
        # if self.win:
        #     self.v = 1

    def q(self):
        # if self.win:
        #     return 1
        q = self.l * self.v # Q
        u = self.p / self.n
        return q + u

    def __str__(self):
        return 'a:%s,n:%s,q:%s' % (self.a,self.n,self.q())


class MCTS:
    """
    蒙特卡罗树搜索
    a = argmaxQ
    """
    def __init__(self, board, player, policy_model, value_model, expansion_gate=50, lambda_=1.0, max_search=1000, max_search_time=600):
        self.expansion_gate = expansion_gate
        self.lambda_ = lambda_
        self.max_search = max_search            # 最大的搜索次数
        self.max_search_time = max_search_time  # 最大的搜索时间(s)
        self.searching = False
        self.stop = False
        self.stop_event = Event()
        self.depth = 0
        self.n_node = 0
        self.n_search = 0
        if isinstance(value_model, str):
            value_model = DQN.load(value_model) if 'DQN' in value_model else ValueNetwork.load(value_model)
        # self.policy = value_model
        self.value = value_model
        self.root = Node(board, player, tree=self)
        self.predicted = set()  # 树中已经走过的走法 (board_str, player, action)
        self.root.expansion(set())

    def search(self, max_search=0, max_search_time=0):
        self.searching = True
        if max_search == 0: max_search = self.max_search
        if max_search_time == 0: max_search_time = self.max_search_time
        start_time = time.time()
        while not self.stop and self.n_search < max_search and time.time()-start_time < max_search_time:
            self.value.predicts.update(self.predicted)
            self.root.search(set())
            self.n_search += 1
            self.value.clear()
            self.value.episode += 1
            # logger.info('search %s', self.n_search)
        logger.info('search over %s', self.n_search)
        self.n_search = 0
        self.show_info()
        self.searching = False
        if self.stop:
            self.stop_event.set()

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
        assert (self.root.board == board).all() and self.root.player == player, '%s,player:%s\nroot:\n%s,player:%s' % (board, player, self.root.board, self.root.player)
        self.search()
        e = self.root.predict()
        action = e.a
        self.predicted.add((self.root.board_str, self.root.player, action))
        q = [(e.a,e.q()) for e in self.root.sub_edge]
        logger.info('predict is:%s', action)
        return action, q

    def move_down(self, board, player, action):
        assert np.all(self.root.board == board), 'root_board:\n' + str(
            self.root.board) + '\nboard:\n' + str(board)
        assert self.root.player == player, 'root_player:%s, player:%s' % (
        self.root.player, player)
        node = self.get_node(action)
        logger.debug('get_node(%s):\n%s', action, node)
        if node is None:
            logger.info('node is None, new Node()')
            board = board.copy()
            rule.move_by_action(board, *action)
            node = Node(rule.flip_board(board), -player, tree=self)
        if not node.expanded:
            node.expansion(set())
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


class MCTSWorker:
    def __init__(self, board, first_player, policy_model, value_model, predict_callback, max_search=1000, expansion_gate=50):
        self.predict_callback = predict_callback
        self.command_queue = Queue()
        # self.result_queue = Queue()
        self.stopped = False
        self.ts =  MCTS(board, first_player, policy_model, value_model, max_search=max_search, expansion_gate=expansion_gate)
        self.ts.show_info()

    def start(self):
        Thread(target=self._start).start()

    def _start(self):
        while True:
            command, board, player, action = self.command_queue.get()
            logger.info('\nget: %s, command==COMMAND.STOP:%s\n%s\n player:%s action:%s', command, command == COMMAND.STOP, board, player, action)
            if command == COMMAND.STOP:
                logger.info('MCTS WORKER ENDING...')
                break
            if command == COMMAND.PREDICT:
                # 走棋
                a, q = self.ts.predict(board, player)
                if self.stopped:
                    self.predict_callback(None, None)
                else:
                    self.predict_callback(a,q)
                    self.ts.move_down(self.ts.root.board, self.ts.root.player, a)
                    self.ts.search(max_search=2 ** 32, max_search_time=2 ** 32)
            elif command == COMMAND.MOVE_DOWN:
                # 对手走棋，向下移动树
                logger.info('move down...')
                self.ts.move_down(board, player, action)
            elif command == COMMAND.SEARCH:
                self.ts.search(max_search=2**32, max_search_time=2**32)
        logger.info('...MCTS WORKER ENDED...')

    def send(self, command, board=None, player=None, action=None):
        self.command_queue.put((command, board, player, action))

    def predict(self, board, player):
        self.send(COMMAND.PREDICT, board, player)

    def begin_search(self):
        self.send(COMMAND.SEARCH)

    def stop_search(self):
        self.ts.stop_search()

    def move_down(self, action):
        self.send(COMMAND.MOVE_DOWN, self.ts.root.board, self.ts.root.player, action)

    def stop(self):
        self.stopped = True
        self.send(COMMAND.STOP)
        self.stop_search()


class MCTSProcess:
    """
    MCTS进程
    该进程是非阻塞的，只是发送命令和接收结果
    具体搜索任务由另一个线程MCTSWorker完成
    """
    def __init__(self, policy_model, value_model, init_board, first_player, player_val):
        self.policy_model = policy_model
        self.value_model = value_model
        self.player_val = player_val
        conn1, conn2 = Pipe()
        self.player_end = conn1
        self.mcts_end = conn2
        if first_player == -1:
            init_board = rule.flip_board(init_board)
        self.worker = MCTSWorker(init_board, first_player, self.policy_model, self.value_model, self.predict_callback, max_search=500, expansion_gate=10)
        self.stopped = False

    def start(self):
        Process(target=self._start).start()
        self.stopped = False

    def predict_callback(self, a, q):
        self.mcts_end.send((a, q))

    def _start(self):
        logger.info('start...')

        self.worker.start()
        while True:
            board, player, action = self.mcts_end.recv()
            logger.info('\nrecv: \n%s\n player:%s action:%s', board, player, action)
            if board is None:
                self.worker.stop()
                break
            elif action is None:
                # 走棋
                self.worker.predict(board, player)
            elif action is not None:
                # 对手走棋，先停止搜索，然后沿对手走棋的方向向下移动树
                logger.info('对手走棋:%s stop search', action)
                self.worker.stop_search()
                logger.info('move down along %s', action)
                self.worker.move_down(action)
        logger.info('...MCTS PROCESS ENDED...')

    def begin_search(self):
        self.worker.begin_search()

    def predict(self, board, player):
        self.player_end.send((board, player, None))
        return self.player_end.recv()   # action, q

    def opponent_play(self, board, player, action):
        """
        对手走完棋:action,搜索树需要沿对手走棋的方向移动树的根结点
        """
        self.player_end.send((board, player, action))

    def stop(self):
        self.player_end.send((None, None, None))
        self.stopped = True


class SimulateProcess:
    def __init__(self, record_queue, model_queue, weight_lock, weights_file, epsilon=1.0, epsilon_decay=0.25):
        self.record_queue = record_queue
        self.model_queue = model_queue
        self.weight_lock = weight_lock
        self.weights_file = weights_file
        self.epsilon = epsilon
        self._epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.episode = 0
        self.predicts = set()

    def _start(self):
        logger.info('start...')
        value_model = ValueNetwork(output_activation='sigmoid')
        for i in range(2 ** 32):
            logger.info('simulate %s', i)
            board = rule.random_init_board()
            player = 1
            # value_model_params = self.model_queue.get()
            with self.weight_lock:
                value_model.model.load_weights(self.weights_file)
            ts = MCTS(board=board.copy(), player=player, policy_model=None, value_model=value_model, max_search=50)
            records, winner = self.simulate(ts, board, player)
            if records.length() == 0:
                continue
            self.record_queue.put(records)
            if i % 1000 == 0:
                records.save('records/train/alpha0/1st_')
            self.episode = i
            self.decay_epsilon()

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
        from qlearning_network import NoActionException
        records = Record()
        while True:
            try:
                bd = board.copy()
                board_str = util.board_str(board)
                valid_action = rule.valid_actions(board, player)
                while True:
                    (from_, act), q = self.epsilon_greedy(board, player, valid_action, ts)
                    if (board_str,from_, act) not in self.predicts:
                        break
                    ts.root.sub_edge = list(filter(lambda e: e.a != (from_,act), ts.root.sub_edge))
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
        self.epsilon = self._epsilon / (1 + self.epsilon_decay * np.log(1 + self.episode))

    def start(self):
        Process(target=self._start).start()


def train():
    from multiprocessing import Queue, Lock
    record_queue = Queue()
    model_queue = Queue()
    weight_lock = Lock()
    weights_file = 'model/alpha0/weights'
    value_model = ValueNetwork(output_activation='sigmoid')
    value_model.model.load_weights(weights_file)
    for _ in range(3):
        value_model.model.save_weights(weights_file)
        model_queue.put(weights_file)
        SimulateProcess(record_queue, model_queue, weight_lock, weights_file, epsilon=1.0, epsilon_decay=0.25).start()

    for i in range(2 ** 32):
        logger.info('.......... train %s ..............', i)
        records = record_queue.get()
        logger.info('records: %s', len(records))
        value_model.train(records, epochs=5)
        # model_queue.put(value_model)
        with weight_lock:
            value_model.model.save_weights(weights_file)
            model_queue.put(weights_file)
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