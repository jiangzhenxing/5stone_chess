import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense,Convolution2D,Flatten,Highway
from keras.optimizers import Adam, SGD
from keras.regularizers import l2
import chess_rule as rule
from util import add_print_time_fun, print_use_time
from record import Record
import logging

logger = logging.getLogger('train')


class NoActionException(BaseException):
    pass


class DQN:
    """
    q = 1 if win else 0
    value = r0 - r'0 + γr1 - γr'1 + ...
    value(s0) = r0 - r'0 + γvalue(s1)
    """
    def __init__(self, epsilon=1, epsilon_decay=0.15, filepath=None):
        if filepath:
            self.model = load_model(filepath)
        else:
            self.model = self.create_model()
        self.epsilon = epsilon
        self._epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.predicts = set()
        # 跟踪上一步的值，供调试
        self.q_value = None
        self.valid = None
        self.vq = None
        self.episode = 0 # 第几次训练

    @staticmethod
    def create_model():
        # 定义顺序模型
        model = Sequential()
        l = 1e-3
        # 第一个卷积层
        model.add(Convolution2D(
            filters=100,        # 卷积核/滤波器个数
            kernel_size=3,      # 卷积窗口大小
            input_shape=(5,5,10),  # 输入平面的形状
            strides=1,          # 步长
            padding='same',     # padding方式 same:保持图大小不变/valid
            activation='relu',  # 激活函数
            # kernel_regularizer=l2(l),
            # bias_regularizer=l2(l)
        ))

        def create_conv_layer(filters=50, kernel_size=3):
            return Convolution2D(filters=filters,
                                 kernel_size=kernel_size,
                                 strides=1,
                                 padding='same',
                                 activation='relu',
                                 # kernel_regularizer=l2(l),
                                 # bias_regularizer=l2(l)
                                 )
        # 第二个卷积层
        model.add(create_conv_layer())
        # 第三个卷积层
        model.add(create_conv_layer())
        # 第四个卷积层
        model.add(create_conv_layer())
        # 第五个卷积层
        model.add(create_conv_layer(filters=25, kernel_size=1))
        # 把卷积层的输出扁平化为1维
        model.add(Flatten())
        # 全连接层
        model.add(Dense(units=100,
                        activation='relu',
                        # kernel_initializer='zeros',
                        # kernel_regularizer=l2(l),
                        # bias_initializer='zeros',
                        # bias_regularizer=l2(l)
                        ))
        # 输出Q值
        model.add(Dense(units=1,
                        activation='linear',
                        kernel_initializer='zeros',
                        # kernel_regularizer=l2(l),
                        bias_initializer='zeros',
                        # bias_regularizer=l2(l)
                        ))
        # 定义优化器
        # opt = Adam(lr=1e-4)
        opt = SGD(lr=1e-3)
        # 定义优化器，loss function
        model.compile(optimizer=opt, loss='mse')
        return model

    @staticmethod
    def feature(board, from_, action):
        """
        第一视角的棋局特征
        :param board:   棋盘
        :param from_:   走哪颗子
        :param action:  动作，向哪个方向走
        :return: 当前动作的特征(5x5xN)
        """
        player = board[from_]
        to_ = tuple(np.add(from_, rule.actions_move[action]))
        # 棋盘特征:空白-己方棋子-对方棋子
        space = (board == 0).astype(np.int8).reshape((5, 5, 1))
        self = (board == player).astype(np.int8).reshape((5, 5, 1))
        opponent = (board == -player).astype(np.int8).reshape((5, 5, 1))
        # 动作特征
        from_location = np.zeros((5,5,1))
        from_location[from_] = 1
        to_location = np.zeros((5,5,1))
        to_location[to_] = 1
        # 走子后的棋盘
        board = board.copy()
        result,_ = rule.move(board, from_, to_)
        space2 = (board == 0).astype(np.int8).reshape((5, 5, 1))
        self2 = (board == player).astype(np.int8).reshape((5, 5, 1))
        opponent2 = (board == -player).astype(np.int8).reshape((5, 5, 1))
        # 走子后是否赢棋
        is_win = np.ones((5,5,1)) if result == rule.WIN else np.zeros((5,5,1))
        # 偏置
        bias = np.ones((5, 5, 1))
        return np.concatenate((space, self, opponent, from_location, to_location, space2, self2, opponent2, is_win, bias), axis=2)

    def q(self, board, from_, action):
        x = self.feature(board, from_, action)
        x = np.array([x])
        q = self.model.predict(x)[0][0]
        return q

    def maxq(self, board, player):
        q = [self.q(board,from_,action) for from_,action in rule.valid_actions(board,player)]
        return max(q)

    def value(self, board, player):
        return self.maxq(board, player)

    def epsilon_greedy(self, board, valid_action, q):
        """
        使用epsilon-greedy策略选择下一步动作
        以epsilon以内的概率随机选择动作
        以1-epsilon的概率选择最大Q值的动作
        :return: 下一个动作: (from,to)
        """
        if np.random.random() > self.epsilon:
            # 选取Q值最大的
            if q is None:
                q = [self.q(board, from_, action) for from_, action in valid_action]
            return self.pi_star(valid_action, q),q
        else:
            # 随机选择
            return self.random_choice(valid_action),q

    def pi_star(self, valid_action, q):
        """
        选择Q值最大的动作，即最优策略
        """
        maxq = np.max(q)
        idxes = np.argwhere(q == maxq)
        action = valid_action[self.random_choice(idxes)[0]]
        # logger.info('maxq:%s, idxes:%s, select:%s', maxq, idxes, action)
        return action

    def predict(self, board, player):
        valid = rule.valid_actions(board, player)
        q = [self.q(board, from_, action) for from_, action in valid]
        return self.pi_star(valid, q),(valid,q)

    def policy(self, board, player):
        valid = rule.valid_actions(board, player)
        q = None
        self.set_pre(q, valid, q)
        if len(valid) == 0:
            raise NoActionException
        board_str = ''.join(map(str, board.flatten()))
        while True:
            (from_,action),q = self.epsilon_greedy(board, valid, q)
            if (board_str,from_,action) not in self.predicts or len(valid) == 1:
                self.predicts.add((board_str,from_,action))
                self.set_pre(q, valid, None)
                if self.episode % 10 == 0:
                    logger.info('action:%s,%s', from_, action)
                    # logger.info('valid:%s', valid)
                    logger.info('q:%s', q)
                return from_,action
            else:
                # 将已经走过的位置移除，不再选择
                idx = valid.index((from_,action))
                valid.pop(idx)
                if q:
                    q.pop(idx)

    def train(self, records, batch_size=1, epochs=1, verbose=0):
        x_train = []
        y_train = []
        for bd, from_, action, reward, _ in records:
            x = self.feature(bd, from_, action)
            x_train.append(x)
            y_train.append(reward)
        x_train = np.array(x_train, copy=False)
        y_train = np.array(y_train, copy=False)
        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=verbose)

    def decay_epsilon(self):
        self.epsilon = self._epsilon / (1 + self.epsilon_decay * np.log(1 + self.episode))

    @staticmethod
    def random_choice(a):
        return a[np.random.randint(len(a))]

    @staticmethod
    def load(modelfile):
        return DQN(filepath=modelfile)

    def set_pre(self, q, valid, vq):
        self.q_value = q
        self.valid = valid
        self.vq = vq

    def copy(self, other):
        self.model.set_weights(other.model.get_weights())

    def clear(self):
        self.predicts.clear()

    def save_model(self, filepath):
        self.model.save(filepath)


def random_init_board():
    """
    随机初始化棋盘
    """
    board = np.zeros((5, 5))
    def random_init_stone(stone):
        num = np.random.randint(2, 6)  # 棋子数量 2-5
        logger.info('%s num is %s', stone, num)
        for _ in range(num):
            while True:
                idx = np.random.randint(25)
                pos = (idx // 5, idx % 5)
                if board[pos] == 0:
                    board[pos] = stone
                    break
    random_init_stone(1)
    random_init_stone(-1)
    return board

def init_board():
    board = np.zeros((5, 5))
    board[0, :] = -1
    board[4, :] = 1
    return board

# @print_use_time()
def simulate(nw0, nw1, init='fixed'):
    board = init_board() if init=='fixed' else random_init_board()
    player = 1
    records = Record()
    while True:
        nw = nw0 if player == 1 else nw1
        try:
            bd = board.copy()
            from_, action = nw.policy(board, player)
            assert board[from_] == player
            to_ = tuple(np.add(from_, rule.actions_move[action]))
            command,eat = rule.move(board, from_, to_)
            reward = len(eat)
            records.add(bd, from_, action, reward, win=command==rule.WIN)
        except NoActionException:
            return Record(),0
        except Exception as e:
            logging.info('board is:\n%s', board)
            logging.info('player is: %s', player)
            valid = rule.valid_actions(board, player)
            logging.info('valid is:\n%s', valid)
            logging.info('predict is:\n%s', nw.q_value)
            logging.info('valid action is:\n%s', nw.valid)
            logging.info('from:%s, action:%s', from_, action)
            records.save('records/train/1st_')
            raise e
        if command == rule.WIN:
            logging.info('%s WIN, step use: %s, epsilon:%s', str(player), records.length(), nw.epsilon)
            return records, player
        if records.length() > 10000:
            logging.info('走子数过多: %s', records.length())
            return Record(),0
        player = -player
        board = rule.flip_board(board)

@print_use_time()
def train_once(n0, n1, i, init='fixed'):
    logging.info('train: %d', i)
    records, winner = simulate(n0, n1, init)
    if records.length() == 0:
        return
    if i%1000==0:
        records.save('records/train/1st_')
    n1.copy(n0)
    n0.train(records, epochs=1)
    n0.clear()
    n1.clear()
    n0.episode = i
    n1.episode = i
    n0.decay_epsilon()
    n1.decay_epsilon()

def train():
    logging.info('...begin...')
    add_print_time_fun(['simulate', 'train_once'])
    n0 = DQN()
    n1 = DQN()
    n1.copy(n0)
    episode = 300000
    for i in range(episode+1):
        train_once(n0, n1, i, init='random')
        if i % 1000 == 0:
            n0.save_model('model/DQN_%04d.model' % (i // 100))
    for i in range(episode+1, episode*2 + 1, 1):
        train_once(n0, n1, i, init='fixed')
        if i % 1000 == 0:
            n0.save_model('model/DQN_%04d.model' % (i // 100))


if __name__ == '__main__':
    import logging.config
    logging.config.fileConfig('logging.conf')
    # _main()
    train()