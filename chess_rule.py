import numpy as np
import pandas as pd
import logging

NOT_MOVE = 'NOT_MOVE'
INVALID_MOVE = 'INVALID_MOVE'
ACCQUIRE = 'ACCQUIRE'
WIN = 'WIN'
actions_name = ['LEFT', 'RIGHT', 'UP', 'DOWN']
actions_move = [(0,-1), (0,1), (-1,0), (1,0)]
actions = pd.Series(data=actions_move, index=actions_name)

def move(board, from_, to_):
    """
    判断from_处的棋子是否可以移动到to_位置
    会直接修改board
    :board: numpy数组，一方为1，另一方为-1，空白处为0
    :return: command,[被吃的子]
    """
    dis = np.abs(np.subtract(from_, to_)).sum()
    stone = board[from_]
    if stone == 0:
        raise ValueError(str(from_) + '处没有子')
    if dis == 0:
        # 未移动
        return NOT_MOVE, None
    if dis > 1 or board[to_] != 0:
        # 移动距离大于1或目标位置有子，不允许移动
        return INVALID_MOVE, None
    # 允许移动
    return _move(board, from_, to_)

def _move(board, from_, to_):
    logging.debug('move from %s to %s', from_, to_)
    stone = board[from_]
    board[to_] = stone
    board[from_] = 0
    eat_stone = judge_eat(board, to_)
    return WIN if judge_win(board, player=stone, eat_stone=eat_stone) else ACCQUIRE, eat_stone

def judge_eat(board, loc):
    """
    判断是否会吃子,会直接修改棋盘
    """
    result = []
    i, j = loc
    # 当前选手的棋子
    player = board[loc]
    # 判断行上是否可以吃子
    rs_row = judge_eat_line(board[i], player)
    if rs_row is not None:
        board[i, rs_row] = 0
        result.append((i, rs_row))
    # 判断列上是否可以吃子
    rs_col = judge_eat_line(board[:,j], player)
    if rs_col is not None:
        board[rs_col,j] = 0
        result.append((rs_col, j))
    return result

def judge_eat_line(line, player):
    """
    判断某条线上player是否存在吃子
    :return: 被吃子的位置
    """
    # player棋子的位置
    player_stone_index = np.argwhere(line == player).flatten()
    # 对手棋子的位置
    opponent_stone_index = np.argwhere(line == -player).flatten()
    # print('棋子位置：', player_stone_index, opponent_stone_index)

    if len(player_stone_index) == 2 and \
        player_stone_index[1] - player_stone_index[0] == 1 and \
        len(opponent_stone_index) == 1 and \
        (opponent_stone_index[0] == player_stone_index[0] - 1 or opponent_stone_index[0] == player_stone_index[1] + 1):
        # player有两颗子且是连续的
        # 对手有一颗子且与player的棋子连续
        # 可以吃对手的子
        return opponent_stone_index[0]

def judge_win(board, player, eat_stone):
    return len(eat_stone) > 0 and judge_win_1(board, player=player) or \
            judge_win_2(board, player=player)

def judge_win_1(board, player):
    """
    对方棋子数少于2则胜
    """
    if (board==-player).sum() < 2:
        return True

def judge_win_2(board, player):
    """
    对方无路可走则胜
    """
    if valid_location(board, -player).sum() == 0:
        return True

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
    v_locations = valid_location(board, player).reshape((5,5,1))
    v_actions = valid_action(board, player)
    bias = np.ones((5,5,1))
    return np.concatenate((space, black, white, who, v_locations, v_actions, bias), axis=2)

def feature_1st(board, player):
    """
    第一视角的棋局特征
    :param board:   棋盘
    :param player:  当前的棋手
    :return: 当前局面的特征(5x5xN)
    """
    space = (board==0).astype(np.int8).reshape((5,5,1))
    self = (board==player).astype(np.int8).reshape((5,5,1))
    opponent = (board==-player).astype(np.int8).reshape((5,5,1))
    v_locations = valid_location(board, player).reshape((5,5,1))
    v_actions = valid_action(board, player)
    bias = np.ones((5,5,1))
    return np.concatenate((space, self, opponent, v_locations, v_actions, bias), axis=2)

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

def flip_board(board):
    return np.fliplr(np.flipud(board))

def flip_location(location):
    i,j = location
    return 4-i,4-j

def flip_action_probs(p):
    p = np.fliplr(np.flipud(p))
    for i in range(5):
        for j in range(5):
            p[i,j] = np.concatenate((p[i,j][:2][::-1], p[i,j][2:][::-1]))
    return p

def flip_action(a):
    """
    将一个动作a进行左右/上下的颠倒
    ['LEFT', 'RIGHT', 'UP', 'DOWN']
    :param a:
    """
    return 1 - a if a < 2 else 5 - a

def valid_action(board, player):
    """
    棋子允许的动作
    """
    v_actions = np.zeros((5, 5, 4))
    for stone in np.argwhere(board == player):
        # stone: player棋子的位置
        neighbors = [np.add(stone, step) for step in actions_move]
        # print(stone, ': ', neighbors)
        # va = [1 if np.all(nb>=0) and np.all(nb<=4) and board[tuple(nb)]==0 else 0 for nb in neighbors]
        # v_actions[tuple(stone)] = va
        for i,nb in enumerate(neighbors):
            if np.all(nb >= 0) and np.all(nb <= 4) and board[tuple(nb)] == 0:
                v_actions[tuple(stone)][i] = 1
    return v_actions

def valid_action2(board, player):
    """
    棋子允许的动作
    """
    v_actions = np.zeros((4, 5, 5))
    stones = np.argwhere(board == player)
    # print('stones:', stones)
    for i,step in enumerate(actions_move):
        for stone in stones:
            nb = stone + np.array(step)
            # print('stone:', stone, 'nb:', nb)
            v_actions[i][tuple(stone)] = 1 if np.all(nb>=0) and np.all(nb<=4) and board[tuple(nb)]==0 else 0
        # print(v_actions[i])
        # print('=' * 50)
    return v_actions

def valid_action_by_location(board, player):
    """
    棋子允许的动作
    以[(from,to)]的形式表达
    """
    stones = np.argwhere(board == player)
    return [(tuple(stone),nb) for stone in stones for nb in neighbor(stone) if board[nb]==0]

def valid_actions(board, player):
    """
    棋子允许的动作
    以[(from,action)]的形式表达
    """
    stones = np.argwhere(board == player)
    def valid(location):
        i,j = location
        return (0 <= i <=4) and (0 <= j <=4) and board[i,j]==0
    return [(tuple(stone),idx) for stone in stones for idx,action in enumerate(actions_move) if valid(np.add(stone,action))]

def valid_location(board, player):
    """
    棋手允许走的位置
    """
    locations = np.zeros((5,5))
    for loc in np.argwhere(board==player):
        for nb in neighbor(loc):
            if board[nb] == 0:
                locations[nb] = 1
    return locations

def neighbor(location):
    """
    location的邻近位置
    """
    return map(lambda loc: tuple(loc), filter(lambda loc: np.all(loc>=0) and np.all(loc<=4), map(lambda action: np.add(location,action), actions_move)))


def _main():
    bd = np.zeros((5, 5))
    bd[0, :] = -1
    bd[4, 0] = 1
    bd[4, 2] = 1
    bd[3, 2] = 1
    print(bd)
    print('-' * 50)
    print(valid_actions(bd, player=1))
    # print(repr(valid_location(bd, 1)))
    # print('-' * 50)
    # # print(repr(valid_action2(bd, player=1)))
    # f = feature(bd, 1)
    # print(f.shape)
    # print(f)
    # action = np.arange(4)
    # print(flip_action(action))
    # p = np.arange(100).reshape(5,5,4) / 100
    # print(p)
    # print('-' * 50)
    # print(flip_action_probs(p))

if __name__ == '__main__':
    _main()