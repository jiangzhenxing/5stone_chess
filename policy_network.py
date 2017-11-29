import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense,Dropout,Convolution2D,MaxPooling2D,Flatten
from keras.optimizers import Adam
import chess_rule as rule
from util import print_time_func, print_use_time


class PolicyNetwork:
    def __init__(self, filepath=None):
        if filepath:
            self.model = load_model(filepath)
        else:
            self.model = self.create_model()
        self.predicts = set()
        # 跟踪上一步的值，供调试
        self.p = None
        self.valid = None
        self.r = None

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
            filters = 5,
            kernel_size = 3,
            input_shape=(5,5,5),
            strides = 1,
            padding = 'same',
            activation = 'relu'
        ))

        def create_conv_layer(filters=5):
            return Convolution2D(filters, 3, strides=1, padding='same', activation='relu', kernel_regularizer='l2', bias_regularizer='l2')

        # 第二个卷积层
        model.add(create_conv_layer())

        # 第三个卷积层
        model.add(create_conv_layer())

        # 第四个卷积层
        model.add(create_conv_layer())

        # 第五个卷积层
        model.add(create_conv_layer())

        # 把卷积层的输出扁平化为1维
        model.add(Flatten())

        # 输出层
        model.add(Dense(100, activation='softmax', kernel_regularizer='l2', bias_regularizer='l2'))

        # 定义优化器
        adam = Adam(lr=1e-4)

        # 定义优化器，loss function，训练过程中计算准确率
        model.compile(optimizer=adam, loss='categorical_crossentropy')

        return model

    def train(self, x_train,y_train,batch_size=5,epochs=10):
        # 训练模型
        self.model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs, verbose=0)

    def predict(self, board, player):
        x = rule.feature(board, player)
        valid = rule.valid_action(board, player)
        x = np.array([x])
        p = self.model.predict(x)[0].reshape(5,5,4)
        r = p * valid   # 所有可能走法的概率
        board_str = ''.join(map(str, board.flatten()))

        if np.max(r) == 0:
            # 最大概率为0，随机选择
            print('>> max prob is 0 radom choise...')
            valid_index = np.argwhere(valid==1)
            self.set_pre(p, valid, r)
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

    def set_pre(self, p, valid, r):
        self.p = p
        self.valid = valid
        self.r = r

    def copy(self, model):
        self.model.set_weights(model.get_weights())

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
    N = 0
    records = []
    while True:
        nw = nw0 if player == 1 else nw1
        bd = board.copy()
        from_, action = nw.predict(board, player)
        records.append((bd, (from_, action)))
        to_ = tuple(np.add(from_, rule.actions_move[action]))
        try:
            command,eat = rule._move(board, from_, to_)
        except IndexError as e:
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
            raise e
        # if eat:
        #     print(player, from_, to_, eat, N)
        if command == rule.WIN:
            print(str(player) + ' WIN, step use:', N)
            return records, player
        if len(records) > 8000:
            print('走子数过多:', len(records))
            return [],0
        player = -player
        N += 1

@print_use_time()
def train(n0, n1):
    records, winner = simulate(n0, n1)
    n0.clear()
    n1.clear()

    if len(records) == 0:
        return
    x_train = []
    y_train = []
    for bd, (from_, action) in records:
        player = bd[from_]
        x = rule.feature(bd, player)
        y = np.zeros((5,5,4))
        y[from_][action] = 1 if player==winner else -1
        x_train.append(x)
        y_train.append(y.flatten())
    n1.copy(n0.model)
    n0.train(np.array(x_train, copy=False), np.array(y_train, copy=False))

def _main():
    print_time_func.add('simulate')
    print_time_func.add('train')
    n0 = PolicyNetwork()
    n1 = PolicyNetwork()
    n1.copy(n0.model)
    episode = 100001
    for i in range(episode):
        train(n0, n1)
        if i > 99 and i % 100 == 0:
            n0.save_model('model/policy_network_%03d.model' % (i // 100))


if __name__ == '__main__':
    _main()