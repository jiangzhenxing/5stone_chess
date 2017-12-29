import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense,Dropout,LocallyConnected1D,Convolution2D,MaxPooling2D,Flatten,Reshape,Highway
from keras.optimizers import Adam, SGD
from keras.regularizers import l2
import chess_rule as rule
from util import add_print_time_fun, print_use_time
from record import Record
import logging

logger = logging.getLogger('train')

class NoActionException(BaseException):
    pass


class PolicyNetwork:
    def __init__(self, filepath=None):
        if filepath:
            self.model = load_model(filepath, custom_objects={'SoftmaxLayer': SoftmaxLayer})
        else:
            self.model = self.create_model()
        self.predicts = set()
        # 跟踪上一步的值，供调试
        self.p = None
        self.valid = None
        self.vp = None
        self.episode = 0 # 第几次训练

    @staticmethod
    def load(modelfile):
        if 'convolution' in modelfile:
            return ConvolutionPolicyNetwork(filepath=modelfile)
        else:
            return RolloutPolicyNetwork(filepath=modelfile)

    @staticmethod
    def create_model():
        raise NotImplementedError

    def train(self, records, batch_size=1, epochs=10, verbose=0):
        raise NotImplementedError

    def _train(self, x_train, y_train, batch_size=1, epochs=5, verbose=0):
        # 训练模型
        # logger.info('x0 is:%s', x_train[0])
        # logger.info('y0 is:%s', y_train[0])
        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=verbose)
        # l = self.model.layers[-2]
        # logger.info('weights: %s', l.get_weights())

    def predict(self, board, player):
        raise NotImplementedError

    def _predict(self, x, board, player, valid):
        # x = self.feature(board, player)
        # valid = rule.valid_action(board, player)
        x = np.array([x])
        p = self.model.predict(x)[0]
        p = p.reshape(5,5,4)
        vp = p * valid   # 所有可能走法的概率
        board_str = ''.join(map(str, board.flatten()))
        n = 0
        try:
            logging.info('最大概率:%.2f:%s(%.2f:%s), 和:%.2f, 平均:%.2f', np.max(vp), np.argmax(vp), np.max(p), np.argmax(p), vp.sum(), vp.sum()/valid.sum())
            if np.max(vp) == 0:
                # 最大概率为0，随机选择
                logging.info('>> max prob is 0 random choise...')
                valid_index = np.argwhere(valid==1)
                assert len(valid_index) > 0, 'no valid action'
                row, col, action = valid_index[np.random.randint(len(valid_index))]
                self.predicts.add((board_str, player, ((row, col), action)))
                self.set_pre(p, valid, vp)
                return (row, col), action, vp, p
            check = True
            while True:
                if np.max(vp) == 0:
                    # 如果最大值是0，说明所有步法之前已经走过，重新取最大概率步
                    logging.info('>> max prob is 0, rechoise')
                    vp = p * valid
                    check = False
                # 选择最大概率
                mx = np.max(vp)  # 最大概率
                mx_index = np.argwhere(vp == mx)  # 所有最大概率的位置
                row, col, action = mx_index[np.random.randint(len(mx_index))]  # 从最大概率的动作中随机选择
                if check and (board_str,player,((row, col), action)) in self.predicts:
                    # 如果此步已经走过了，将其概率置为0，重新选择
                    logger.info('%s has gone', (row,col,action))
                    vp[row,col,action] = 0
                else:
                    self.predicts.add((board_str,player,((row, col), action)))
                    self.set_pre(p, valid, vp)
                    return (row,col),action, vp, p
                if n > 100:
                    logging.info('!select: %s, %s,', n, (row, col, action))
                n+=1
        except Exception as e:
            logging.info('board:')
            logging.info(board)
            logging.info('p(shape:%s) is:', p.shape)
            logging.info(p)
            logging.info('-' * 50)
            logging.info('valid action is:')
            logging.info(valid)
            logging.info('-' * 50)
            logging.info('vp is:')
            logging.info(vp)
            logging.info('-' * 50)
            raise e

    @staticmethod
    def select_by_prob(actions, probs):
        # logging.debug(actions)
        # logging.debug(probs)
        rd = np.random.rand()
        s = 0
        for i,p in enumerate(probs):
            s += p
            if s > rd:
                return actions[i]

    def policy(self, board, player):
        raise NotImplementedError

    def _policy(self, x, board, player, valid):
        x = np.array([x])
        p = self.model.predict(x)[0]
        # logger.info('p.shape: %s', p.shape)
        # for l in self.model.layers:
        #     logger.info('layer %s, output shape: %s', l, l.output_shape)
        p = p.reshape(5, 5, 4)
        vp = p * valid  # 所有可能走法的概率
        self.set_pre(p, valid, vp)
        board_str = ''.join(map(str, board.flatten()))
        try:
            max_vp = np.max(vp)
            if self.episode % 10 == 0:
                max_vp_ = max_vp
                maxp = np.max(p)
                max_vp_where = np.argwhere(vp==max_vp)[0]
                maxp_where = np.argwhere(p==maxp)[0]
                vp_sum = vp.sum()
                avgp = vp_sum / valid.sum()
            n = 0
            max_valid_prob_is0 = max_vp == 0
            if max_valid_prob_is0:
                # 最大概率为0，概率定为平均值，随机选择
                n_valid = valid.sum()
                if n_valid == 0:
                    raise NoActionException
                vp = valid / n_valid
                max_vp = 1 / n_valid
                # logging.info('>> max prob is 0 radom choise...')
                # valid_index = np.argwhere(valid == 1)
                # assert len(valid_index) > 0, 'no valid action'
                # row, col, action = valid_index[np.random.randint(len(valid_index))]
                # self.predicts.add((board_str, player, ((row, col), action)))
                # self.set_pre(p, valid, r)
                # return (row, col), action, vp
            check = True
            while True:
                if max_vp == 0:
                    # 最大概率为0，说明所有的步法之前均已走过，重新按概率选择
                    logging.info('>> max prob is 0 rechoise by prob...')
                    vp = p * valid if not max_valid_prob_is0 else valid / valid.sum()
                    check = False
                # 按概率选择
                vp = vp / vp.sum()
                idx = np.argwhere(vp > 0)
                prob = [vp[tuple(i)] for i in idx]
                row,col,action = self.select_by_prob(idx, prob)
                if check and (board_str,player,((row, col), action)) in self.predicts:
                    vp[row, col, action] = 0
                else:
                    self.predicts.add((board_str, player, ((row, col), action)))
                    if self.episode % 10 == 0:
                        logging.info('最大概率:%.4f(%s)_%.4f(%s), 和:%.4f, 平均:%.4f, select: %s,%s,%s',
                                     max_vp_, ','.join(map(str, max_vp_where)),
                                     maxp, ','.join(map(str, maxp_where)),
                                     vp_sum, avgp, row, col, action)
                    return (row, col), action, vp, p
                if n > 100:
                    logging.info('!select: %s, %s, %s, %s', n, (row, col, action), idx, prob)
                max_vp = np.max(vp)
                n +=1
        except Exception as e:
            logging.info('board:')
            logging.info(board)
            logging.info('p(shape:%s) is:', p.shape)
            logging.info(p)
            logging.info('-' * 50)
            logging.info('valid action is:')
            logging.info(valid)
            logging.info('-' * 50)
            logging.info('vp is:')
            logging.info(vp)
            logging.info('-' * 50)
            # for l in self.model.layers:
            #     logger.info('%s weights: %s\n', l, l.get_weights())
            raise e

    def probabilities(self, board, player):
        x = self.feature_1st(board, player)
        valid = rule.valid_action(board, player)
        x = np.array([x])
        p = self.model.predict(x)[0]
        p = p.reshape(5, 5, 4)
        vp = p * valid  # 所有可能走法的概率
        if vp.max() == 0:
            vp = valid / valid.sum()
        else:
            vp = vp / vp.sum()
        return vp

    @staticmethod
    def feature_1st(board, player):
        """
        第一视角的棋局特征
        :param board:   棋盘
        :param player:  当前的棋手
        :return: 当前局面的特征(5x5xN)
        """
        space = (board == 0).astype(np.int8).reshape((5, 5, 1))
        self = (board == player).astype(np.int8).reshape((5, 5, 1))
        opponent = (board == -player).astype(np.int8).reshape((5, 5, 1))
        v_locations = rule.valid_location(board, player).reshape((5, 5, 1))
        v_actions = rule.valid_action(board, player)
        bias = np.ones((5, 5, 1))
        return np.concatenate((space, self, opponent, v_locations, v_actions, bias), axis=2)

    @staticmethod
    def feature(board, player):
        """
        棋局的特征
        :param board:   棋盘
        :param player:  当前的棋手
        :return: 当前局面的特征(5x5x10)
        """
        space = (board==0).astype(np.int8).reshape((5,5,1))
        black = (board==1).astype(np.int8).reshape((5,5,1))
        white = (board==-1).astype(np.int8).reshape((5,5,1))
        who = np.ones((5,5,1)) if player == 1 else np.zeros((5,5,1))
        v_locations = rule.valid_location(board, player).reshape((5,5,1))
        v_actions = rule.valid_action(board, player)
        bias = np.ones((5,5,1))
        return np.concatenate((space, black, white, who, v_locations, v_actions, bias), axis=2)

    @staticmethod
    def target(board, from_, action, reward, vp):
        if vp[from_][action] == 1:
            y = vp
        else:
            if reward > 0:
                vp[from_][action] *= (1 + reward)
            else:
                vp[from_][action] /= (1 - reward)
            s = vp.sum()
            assert s > 0, 'sum is: %s, reward is: %s' % (s, reward)
            y = vp / s
        # if reward == 1:
        #     y = np.zeros((5, 5, 4))
        #     y[from_][action] = 1
        # elif reward == 0:
        #     if vp[from_][action] == 1:
        #         y = vp
        #     else:
        #         vp[from_][action] = 0
        #         y = vp / vp.sum()
        # else:
        #     raise ValueError('reward is: ' + str(reward))
        return y

    def set_pre(self, p, valid, vp):
        self.p = p
        self.valid = valid
        self.vp = vp

    def copy(self, other):
        self.model.set_weights(other.model.get_weights())

    def clear(self):
        self.predicts.clear()

    def save_model(self, filepath):
        self.model.save(filepath)


class RolloutPolicyNetwork(PolicyNetwork):
    @staticmethod
    def create_model():
        # 定义顺序模型
        model = Sequential()
        # 输出层
        model.add(Dense(100, activation='softmax', input_dim=5*5*9, use_bias=False, kernel_initializer='zeros', bias_initializer='zeros'))
        # 定义优化器
        # opt = Adam(lr=1e-4)
        opt = SGD(lr=1e-3)
        # 定义优化器，loss function，训练过程中计算准确率
        model.compile(optimizer=opt, loss='categorical_crossentropy')
        return model

    def train(self, records, batch_size=1, epochs=1, verbose=0):
        x_train = []
        y_train = []
        for bd, from_, action, reward, vp in records:
            player = bd[from_]
            x = self.feature_1st(bd, player)
            y = self.target(bd, from_, action, reward, vp)
            x_train.append(x.flatten())
            y_train.append(y.flatten())
        self._train(np.array(x_train, copy=False), np.array(y_train, copy=False), batch_size=batch_size, epochs=epochs, verbose=verbose)

    def predict(self, board, player):
        x = self.feature_1st(board, player).flatten()
        valid = rule.valid_action(board, player)
        return self._predict(x, board, player, valid)

    def policy(self, board, player):
        x = self.feature_1st(board, player).flatten()
        valid = rule.valid_action(board, player)
        return self._policy(x, board, player, valid)


class ConvolutionPolicyNetwork(PolicyNetwork):
    @staticmethod
    def create_model():
        # 定义顺序模型
        model = Sequential()
        l = 1e-4
        # 第一个卷积层
        model.add(Convolution2D(
            filters = 100,           # 卷积核/滤波器个数
            kernel_size = 3,        # 卷积窗口大小
            input_shape = (5,5,9), # 输入平面的形状
            strides = 1,            # 步长
            padding = 'same',       # padding方式 same:保持图大小不变/valid
            activation = 'relu',    # 激活函数
            # kernel_regularizer = l2(l),
            # bias_regularizer = l2(l)
        ))
        def create_conv_layer(filters=50, kernel_size=3,):
            return Convolution2D(filters,
                                 kernel_size,
                                 strides = 1,
                                 padding = 'same',
                                 activation = 'relu',
                                 # kernel_regularizer = l2(l),
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
        # model.add(Convolution2D(4, 1, strides=1, padding='same', use_bias=False, kernel_initializer='zeros', kernel_regularizer=l2(l), name='c2'))
        # 把卷积层的输出扁平化为1维
        model.add(Flatten())
        # 全连接层
        model.add(Dense(units=100, activation='relu'))
        # 输出层
        model.add(Dense(units=100,
                        activation='softmax',
                        kernel_initializer='zeros',
                        kernel_regularizer=l2(l),
                        bias_initializer='zeros',
                        bias_regularizer=l2(l)
                        ))

        # 定义优化器
        # opt = Adam(lr=1e-4)
        opt = SGD(lr=2e-4)
        # 定义优化器，loss function
        model.compile(optimizer=opt, loss='categorical_crossentropy')
        return model

    def set_dropout(self, rate):
        # dropout = self.model.get_layer(name='dropout')
        # dropout.rate = rate
        pass

    def get_layer(self, name):
        return self.model.get_layer(name)

    def train(self, records, batch_size=1, epochs=1, verbose=0):
        x_train = []
        y_train = []
        for bd, from_, action, reward, vp in records:
            player = bd[from_]
            x = self.feature_1st(bd, player)
            x_train.append(x)
            y = self.target(bd, from_, action, reward, vp)
            y_train.append(y.flatten())
        self.set_dropout(0.5)
        self._train(np.array(x_train, copy=False), np.array(y_train, copy=False), batch_size=batch_size, epochs=epochs, verbose=verbose)
        # logger.info('weights: %s', self.model.get_weights())

    def predict(self, board, player):
        x = self.feature_1st(board, player)
        valid = rule.valid_action(board, player)
        self.set_dropout(0)
        return self._predict(x, board, player, valid)

    def policy(self, board, player):
        x = self.feature_1st(board, player)
        valid = rule.valid_action(board, player)
        self.set_dropout(0)
        return self._policy(x, board, player, valid)

    def policy_1st(self, board, player):
        x = self.feature_1st(board, player)
        valid = rule.valid_action(board, player)
        self.set_dropout(0)
        return self._policy(x, board, valid)


from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

class SoftmaxLayer(Layer):
    def __init__(self, output_dim=(5,5,4), **kwargs):
        self.output_dim = output_dim
        super(SoftmaxLayer, self).__init__(**kwargs)

    # def build(self, input_shape):
    #     # Create a trainable weight variable for this layer.
    #     self.kernel = self.add_weight(name='kernel',
    #                                   shape=(input_shape[1], 1),
    #                                   initializer='ones',
    #                                   trainable=True)
    #     self.bias = None
    #     super(SoftmaxLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):
        flatten = K.batch_flatten(inputs)
        logger.info('len(flatten): %s', flatten)
        return K.reshape(K.softmax(flatten), (-1,*self.output_dim))

    def compute_output_shape(self, input_shape):
        return (-1,*self.output_dim)


@print_use_time()
def simulate(nw0, nw1, init='fixed'):
    board = rule.init_board() if init == 'fixed' else rule.random_init_board()
    player = 1
    records = Record()
    while True:
        nw = nw0 if player == 1 else nw1
        try:
            bd = board.copy()
            from_, action, vp, p = nw.policy(board, player)
            # print('>', from_, action)
            assert board[from_] == player
            to_ = tuple(np.add(from_, rule.actions_move[action]))
            command,eat = rule.move(board, from_, to_)
            reward = len(eat)
            records.add(bd, from_, action, reward, vp, win=command==rule.WIN)
        except NoActionException:
            return Record(),0
        except Exception as e:
            logging.info('board is:')
            logging.info(board)
            logging.info('player is: %s', player)
            valid = rule.valid_action(board, player)
            logging.info('predict is:')
            print(nw.p)
            logging.info('sum is: %s', nw.p.sum())
            logging.info('valid action is:')
            logging.info(nw.valid)
            logging.info('p * valid is:')
            logging.info(nw.vp)
            logging.info('from:%s, action:%s', from_, action)
            logging.info('prob is: %s', valid[from_][action])
            records.save('records/train/1st_')
            raise e
        # if eat:
        #     print(player, from_, to_, eat, N)
        if command == rule.WIN:
            logging.info('%s WIN, step use: %s', str(player), records.length())
            return records, player
        if records.length() > 10000:
            logging.info('走子数过多: %s', records.length())
            return Record(),0
        player = -player
        board = rule.flip_board(board)

@print_use_time()
def train(n0, n1, i, init='fixed'):
    logging.info('train: %d', i)
    records, winner = simulate(n0, n1, init)
    n0.clear()
    n1.clear()
    if records.length() == 0:
        return
    if i%500==0:
        records.save('records/train/1st_rollout_')
    n1.copy(n0)
    n0.train(records, epochs=1)
    n0.episode += 1
    n1.episode += 1

def train_rpn():
    logging.info('...begin...')
    add_print_time_fun(['simulate', 'train'])
    n0 = RolloutPolicyNetwork()
    n1 = RolloutPolicyNetwork()
    n1.copy(n0)
    episode = 300000
    for i in range(0,episode+1,1):
        train(n0, n1, i, init='random')
        if i % 100 == 0:
            n0.save_model('model/policy_network/rollout_%03d.model' % (i // 100))

def train_cpn():
    logging.info('...begin...')
    add_print_time_fun(['simulate', 'train'])
    n0 = ConvolutionPolicyNetwork()
    n1 = ConvolutionPolicyNetwork()
    n1.copy(n0)
    episode = 300000
    for i in range(0, episode+1,1):
        train(n0, n1, i, init='random')
        if i % 1000 == 0:
            n0.save_model('model/policy_network/convolution_%04dk.model' % (i // 1000))
    for i in range(episode+1, episode*2 + 1, 1):
        train(n0, n1, i, init='fixed')
        if i % 1000 == 0:
            n0.save_model('model/policy_network/convolution_%04dk.model' % (i // 1000))


if __name__ == '__main__':
    import logging.config
    logging.config.fileConfig('logging.conf')
    # _main()
    train_rpn()