import numpy as np
import time
from threading import Thread,Event
from queue import Queue
from policy_network import PolicyNetwork
from qlearning_network import DQN
import chess_rule as rule
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
            logger.info('level:%s, value:%s, player:%s', self.level, value, player)
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
        values = [self.tree.policy.q(board, from_, act) for from_,act in actions_]
        for action,value in zip(actions_, values):
            e = Edge(upper_node=self, a=action, v=value, p=0, lambda_=self.tree.lambda_)
            self.add_edge(e)
        values = [e.v for e in self.sub_edge]
        probs = DQN.value_to_probs(values)
        for p,e in zip(probs,self.sub_edge):
            e.p = p
        self.expanded = True
        assert len(self.sub_edge) > 0, 'board:\n' + str(self.board) + '\nplayer:' + str(self.player)

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
    def __init__(self, board, player, policy_model, worker_model, expansion_gate=50, lambda_=1.0, max_search=1000, max_search_time=600):
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
        self.policy = DQN.load(policy_model)
        self.worker = DQN.load(worker_model)
        self.root = Node(board, player, tree=self)
        self.predicted = set()  # 树中已经走过的走法 (board_str, player, action)
        self.root.expansion(set())

    def search(self, max_search=0, max_search_time=0):
        self.searching = True
        if max_search == 0: max_search = self.max_search
        if max_search_time == 0: max_search_time = self.max_search_time
        start_time = time.time()
        while not self.stop and self.n_search < max_search and time.time()-start_time < max_search_time:
            self.worker.predicts.update(self.predicted)
            self.root.search(set())
            self.n_search += 1
            self.worker.clear()
            self.worker.episode += 1
            logger.info('search %s', self.n_search)
        logger.info('search over')
        self.n_search = 0
        self.show_info()
        self.searching = False
        if self.stop:
            self.stop_event.set()

    def stop_search(self):
        if not self.searching:
            return
        self.stop = True
        # 等待搜索结束
        self.stop_event.wait()
        self.stop = False
        self.stop_event.clear()
        logger.info('search stopped...')

    def predict(self, board, player):
        self.root = Node(board, player, tree=self)
        self.root.expansion(set())
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
        logger.info('move down to node:%s', action)
        self.show_info()

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
               'root is:\n%s'
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
    def __init__(self, board, first_player, policy_model, worker_model, max_search=1000, expansion_gate=50):
        self.command_queue = Queue()
        self.result_queue = Queue()
        self.ts =  MCTS(board, first_player, policy_model, worker_model, max_search=max_search, expansion_gate=expansion_gate)
        self.ts.show_info()
        Thread(target=self.start).start()

    def start(self):
        while True:
            command, board, player, action = self.command_queue.get()
            logger.info('\nget: %s\n%s\n player:%s action:%s', command, board, player, action)
            if command == COMMAND.STOP:
                break
            if command == COMMAND.PREDICT:
                # 走棋
                a, q = self.ts.predict(board, player)
                self.result_queue.put((a, q))
                self.ts.move_down(self.ts.root.board, self.ts.root.player, a)
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
        return self.result_queue.get()

    def begin_search(self):
        self.send(COMMAND.SEARCH)

    def stop_search(self):
        self.ts.stop_search()

    def move_down(self, action):
        self.send(COMMAND.MOVE_DOWN, self.ts.root.board, self.ts.root.player, action)

    def stop(self):
        self.send(COMMAND.STOP)


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
    main()