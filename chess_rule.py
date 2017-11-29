import numpy as np
import pandas as pd

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
    # print('move from', from_, 'to', to_)
    stone = board[from_]
    board[to_] = stone
    board[from_] = 0
    eat_stone = judge_eat(board, to_)
    is_win = judge_win(board, player=stone)
    return WIN if is_win else ACCQUIRE, eat_stone  # 判断是否吃子

def judge_eat(board, loc):
    """
    判断是否会吃子
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

def judge_win(board, player):
    """
    对方棋子数少于2则胜
    """
    if (board==-player).sum() < 2:
        return WIN

def feature(board, player):
    """
    棋局的特征
    :param board:   棋盘
    :param player:  当前的棋手
    :return:
    """
    space = (board==0).astype(np.int8)
    black = (board==1).astype(np.int8)
    white = (board==-1).astype(np.int8)
    who = np.ones((5,5)) if player == 1 else np.zeros((5,5))
    bias = np.ones((5,5))
    return np.array([space, black, white, who, bias])

def valid_action(board, player):
    """
    棋子允许的动作
    """
    v_actions = np.zeros((5, 5, 4))
    for stone in np.argwhere(board == player):
        # stone: player棋子的位置
        neighbors = [np.add(stone, step) for step in actions_move]
        # print(stone, ': ', neighbors)
        va = [1 if np.all(nb>=0) and np.all(nb<=4) and board[tuple(nb)]==0 else 0 for nb in neighbors]
        v_actions[tuple(stone)] = va
    return v_actions

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
    diffs = [(-1,0), (1,0), (0,-1), (0,1)]
    return map(lambda loc: tuple(loc), filter(lambda loc: np.all(loc>=0) and np.all(loc<=4), map(lambda diff: np.add(location,diff), diffs)))


if __name__ == '__main__':
    bd = np.zeros((5,5))
    bd[0,:] = -1
    bd[4,:] = 1
    bd[3,2] = 1
    print(bd)
    print('-' * 50)
    # print(repr(valid_location(bd, 1)))
    # print(repr(valid_action(bd, player=1)))
    print(feature(bd, 1))