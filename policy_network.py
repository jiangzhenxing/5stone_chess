import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense,Dropout,Convolution2D,MaxPooling2D,Flatten
from keras.optimizers import Adam, SGD
import chess_rule as rule
from util import print_time_func, print_use_time
from record import Record


class PolicyNetwork:
    def __init__(self, filepath=None):
        if filepath:
            self.model = load_model(filepath)
            self.predict = self.predict0 if filepath.endswith('model0') else self.predict1
        else:
            self.model = self.create_model0()
            self.predict = self.predict0
        self.predicts = set()
        # 跟踪上一步的值，供调试
        self.p = None
        self.valid = None
        self.r = None

    @staticmethod
    def create_model0():
        # 定义顺序模型
        model = Sequential()
        # 输出层
        model.add(Dense(100, activation='softmax', input_dim=125, use_bias=False, kernel_initializer='zeros'))
        # 定义优化器
        # opt = Adam(lr=1e-4)
        opt = SGD()
        # 定义优化器，loss function，训练过程中计算准确率
        model.compile(optimizer=opt, loss='categorical_crossentropy')

        return model

    @staticmethod
    def create_model():
        # 定义顺序模型
        model = Sequential()

        # 第一个卷积层
        # input_shape 输入平面
        # filters 卷积核/滤波器个数
        # kernel_size 卷积窗口大小
        # strides 步长
        # padding padding方式 same/valid
        # activation 激活函数
        model.add(Convolution2D(
            filters = 25,
            kernel_size = 2,
            input_shape=(5,5,5),
            strides = 1,
            padding = 'same',
            activation = 'relu'
        ))

        def create_conv_layer(filters=25):
            return Convolution2D(filters, 2, strides=1, padding='same', activation='relu', kernel_initializer='zeros', bias_initializer='zeros', kernel_regularizer='l2', bias_regularizer='l2')

        # 第二个卷积层
        model.add(create_conv_layer())

        # 第三个卷积层
        model.add(create_conv_layer())

        # 第四个卷积层
        model.add(create_conv_layer())

        # 第五个卷积层
        model.add(create_conv_layer())

        # model.add(Convolution2D(4, 1, strides=1, padding='same', activation='linear', kernel_initializer='zeros', bias_initializer='zeros', kernel_regularizer='l2', bias_regularizer='l2'))

        # 把卷积层的输出扁平化为1维
        model.add(Flatten())

        # 输出层
        model.add(Dense(100, activation='softmax', kernel_regularizer='l2', bias_regularizer='l2'))

        # 定义优化器
        # opt = Adam(lr=1e-4)
        opt = SGD()

        # 定义优化器，loss function，训练过程中计算准确率
        model.compile(optimizer=opt, loss='categorical_crossentropy')

        return model

    def train(self, x_train, y_train, batch_size=1, epochs=5):
        # 训练模型
        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0)


    def predict0(self, board, player, trace=False):
        x = rule.feature(board, player).flatten()
        valid = rule.valid_action(board, player)
        return self._predict(x, board, valid, trace)

    def predict1(self, board, player, trace=False):
        x = rule.feature(board, player)
        valid = rule.valid_action(board, player)
        return self._predict(x, board, valid, trace)

    def _predict(self, x, board, valid, trace=False):
        # x = rule.feature(board, player)
        # valid = rule.valid_action(board, player)
        x = np.array([x])
        p = self.model.predict(x)[0]
        p = p.reshape(5,5,4)
        r = p * valid   # 所有可能走法的概率
        if trace:
            print('board:')
            print(board)
            print('p shape:', p.shape, 'is:')
            print(p)
            print('-' * 50)
            print('valid is:')
            print(valid)
            print('-' * 50)
            print('r is:')
            print(r)
            print('-' * 50)
        board_str = ''.join(map(str, board.flatten()))

        try:
            if np.max(r) == 0:
                # 最大概率为0，随机选择
                print('>> max prob is 0 radom choise...')
                valid_index = np.argwhere(valid==1)
                if len(valid_index)==0:
                    raise ValueError('no valid action')
                row, col, action = valid_index[np.random.randint(len(valid_index))]
                self.predicts.add((board_str, row, col, action))
                self.set_pre(p, valid, r)
                return (row, col), action

            def choose_max():
                mx = np.max(r)  # 最大概率
                mx_index = np.argwhere(r == mx)  # 所有最大概率的位置
                return mx_index[np.random.randint(len(mx_index))]  # 从最大概率的动作中随机选择

            while True:
                row, col, action = choose_max()
                if r[row,col,action] == 0:
                    # 如果最大值是0，说明所有步法之前已经走过，重新取最大概率步
                    print('>> max prob is 0, rechoise')
                    r = p * valid
                    row, col, action = choose_max()
                    self.predicts.add((board_str, row, col, action))
                    self.set_pre(p, valid, r)
                    return (row, col), action

                if (board_str,row,col,action) in self.predicts:
                    # 如果此步已经走过了，将其概率置为0，重新选择
                    r[row,col,action] = 0
                else:
                    self.predicts.add((board_str,row,col,action))
                    self.set_pre(p, valid, r)
                    return (row,col),action
        except Exception as e:
            print('board:')
            print(board)
            print('p shape:', p.shape, 'is:')
            print(p)
            print('-' * 50)
            print('valid action is:')
            print(valid)
            print('-' * 50)
            print('r is:')
            print(r)
            print('-' * 50)
            raise e

    @staticmethod
    def policy(actions, probs):
        # print(actions)
        # print(probs)
        rd = np.random.rand()
        s = 0
        for i,p in enumerate(probs):
            s += p
            if s > rd:
                return actions[i]

    def rollout(self, board, player):
        x = rule.feature(board, player).flatten()
        # x = rule.feature(board, player)
        x = np.array([x])
        p = self.model.predict(x)[0]
        p = p.reshape(5, 5, 4)
        valid = rule.valid_action(board, player)
        r = p * valid  # 所有可能走法的概率
        self.set_pre(p, valid, r)
        try:
            n = 0
            while True:
                if np.max(r) == 0:
                    # 最大概率为0，随机选择
                    print('>> max prob is 0 radom choise...')
                    valid_index = np.argwhere(valid == 1)
                    if len(valid_index) == 0:
                        raise ValueError('no valid action')
                    row, col, action = valid_index[np.random.randint(len(valid_index))]
                    return (row, col), action
                else:
                    r = r / r.sum()
                    idx = np.argwhere(r > 0)
                    prob = [r[tuple(i)] for i in idx]
                    row,col,action = self.policy(idx, prob)
                    if n > 100:
                        print(idx, prob)
                predict_str = ''.join(map(str,board.flatten())) + str(row) +str(col) + str(action)
                if predict_str not in self.predicts:
                    self.predicts.add(predict_str)
                    return (row, col), action
                else:
                    r[row,col,action] = 0
                if n > 100:
                    print('!select:', n, (row, col, action))
                n +=1
        except Exception as e:
            print('board:')
            print(board)
            print('player:', player)
            print('p shape:', p.shape, 'is:')
            print(p)
            print('-' * 50)
            print('valid action is:')
            print(valid)
            print('-' * 50)
            print('r is:')
            print(r)
            print('-' * 50)
            raise e

    def set_pre(self, p, valid, r):
        self.p = p
        self.valid = valid
        self.r = r

    def copy(self, other):
        self.model.set_weights(other.model.get_weights())

    def clear(self):
        self.predicts.clear()

    def save_model(self, filepath):
        self.model.save(filepath)

@print_use_time()
def simulate(nw0, nw1):
    board = np.zeros((5, 5))
    board[0,:] = -1
    board[4,:] = 1
    player = 1
    records = Record('records/train/')
    while True:
        nw = nw0 if player == 1 else nw1
        bd = board.copy()
        try:
            from_, action = nw.rollout(board, player)
            # print('>', from_, action)
            to_ = tuple(np.add(from_, rule.actions_move[action]))
            command,eat = rule.move(board, from_, to_)
            reward = len(eat)
            records.add(bd, from_, action, reward, win=command==rule.WIN)
        except Exception as e:
            print('board is:')
            print(board)
            valid = rule.valid_action(board, player)
            print('predict is:')
            print(nw.p)
            print('sum is:', nw.p.sum())
            print('valid action is:')
            print(nw.valid)
            print('p * valid is:')
            print(nw.r)
            print('from, action is:', from_, action)
            print('prob is:', valid[from_][action])
            records.save()
            raise e
        # if eat:
        #     print(player, from_, to_, eat, N)
        if command == rule.WIN:
            print(str(player) + ' WIN, step use:', records.length())
            return records, player
        if records.length() > 10000:
            print('走子数过多:', records.length())
            return Record(''),0
        player = -player

@print_use_time()
def train(n0, n1, i):
    print('train:', i)
    records, winner = simulate(n0, n1)
    n0.clear()
    n1.clear()

    if records.length == 0:
        return

    if i%100==0:
        records.save()

    x_train = []
    y_train = []
    for bd, from_, action, reward in records:
        player = bd[from_]
        x = rule.feature(bd, player)
        y = np.zeros((5,5,4))
        y[from_][action] = reward
        x_train.append(x.flatten())
        y_train.append(y.flatten())

    n1.copy(n0)
    n0.train(np.array(x_train, copy=False), np.array(y_train, copy=False))

def _main():
    print('...begin...')
    print_time_func.add('simulate')
    print_time_func.add('train')
    n0 = PolicyNetwork('model/policy_network2_016.model0')
    n1 = PolicyNetwork()
    n1.copy(n0)
    episode = 10000
    for i in range(1600,episode+1,1):
        train(n0, n1, i)
        if i % 100 == 0:
            n0.save_model('model/policy_network2_%03d.model0' % (i // 100))


if __name__ == '__main__':
    _main()