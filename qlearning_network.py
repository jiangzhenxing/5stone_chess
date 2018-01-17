import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Convolution2D,Flatten
from keras.optimizers import Adam, SGD
from keras.regularizers import l2
import chess_rule as rule
from util import add_print_time_fun, print_use_time, load_model
from record import Record
from value_network import ValueNetwork, NoActionException, simulate
import logging


logger = logging.getLogger('train')


class DQN(ValueNetwork):
    """
    q = 1 if win else 0
    """
    @staticmethod
    def load_model(model_file):
        model = load_model(model_file)
        '''
        # 这里中途修改了一下输出层的正则化参数和SGD的学习率
        out = model.get_layer(index=-1)
        l = 0.01
        out.kernel_regularizer = l2(l)
        out.bias_regularizer = l2(l)
        model.optimizer = SGD(lr=1e-4, decay=1e-6)
        '''
        return model

    def create_model(self):
        # 定义顺序模型
        model = Sequential()
        l = 0.01
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
                        ))
        # 输出Q值
        if self.output_activation == 'linear':
            model.add(Dense(units=1,
                            activation='linear',
                            kernel_initializer='zeros',
                            kernel_regularizer=l2(l),
                            bias_initializer='zeros',
                            bias_regularizer=l2(l)
                            ))
        elif self.output_activation == 'sigmoid':
            model.add(Dense(units=1,
                            activation='sigmoid',
                            kernel_initializer='zeros',
                            kernel_regularizer=l2(l),
                            bias_initializer='zeros',
                            bias_regularizer=l2(l)
                            ))
        elif self.output_activation == 'selu':
            model.add(Dense(units=1,
                            activation='selu',
                            kernel_initializer='zeros',
                            kernel_regularizer=l2(l),
                            bias_initializer='zeros',
                            bias_regularizer=l2(l)
                            ))
        # 定义优化器
        # opt = Adam(lr=1e-4)
        opt = SGD(lr=2e-4, decay=1e-6)
        # loss function
        loss = 'mse' # if self.output_activation == 'linear' else 'binary_crossentropy' if self.output_activation == 'sigmoid' else None
        model.compile(optimizer=opt, loss=loss)
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

    @staticmethod
    def load(modelfile, epsilon=0.3):
        return DQN(epsilon=epsilon, filepath=modelfile)


@print_use_time()
def train_once(n0, n1, i, activation, init='fixed'):
    logging.info('train: %d', i)
    n0.episode = i
    n1.episode = i
    n0.decay_epsilon()
    n1.decay_epsilon()
    records, winner = simulate(n0, n1, activation, init)
    if records.length() == 0:
        return
    if i % 10 == 0:
        n1.copy(n0)
    n0.train(records, epochs=1)
    n0.clear()
    n1.clear()
    return records

def train():
    logging.info('...begin...')
    add_print_time_fun(['simulate', 'train_once'])
    activation = 'sigmoid'     # linear, selu, sigmoid
    n0 = DQN(output_activation=activation, filepath='model/qlearning_network/DQN_fixed_sigmoid_555_00576w.model')
    n1 = DQN(output_activation=activation, filepath='model/qlearning_network/DQN_fixed_sigmoid_555_00576w.model')
    n1.copy(n0)
    episode = 1000000
    begin = 5760000
    '''
    for i in range(begin, begin+episode+1):
        train_once(n0, n1, i, activation, init='random')
        if i % 10000 == 0:
            n0.save_model('model/qlearning_network/DQN_random_%s_%05dw.model' % (activation, i // 10000))
    '''
    for i in range(begin+1, begin + episode + 1):
        records = train_once(n0, n1, i, activation, init='fixed')
        if i % 1000 == 0:
            records.save('records/train/qlearning_network/1st_')
        if i % 10000 == 0:
            n0.save_model('model/qlearning_network/DQN_fixed_%s_555_%05dw.model' % (activation, i // 10000))


if __name__ == '__main__':
    import logging.config
    logging.config.fileConfig('logging.conf')
    # _main()
    train()