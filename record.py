import time
import numpy as np
import logging
import chess_rule as rule

logger = logging.getLogger('app')

class Record:
    """
    棋谱,格式为：[board, from_, action, reward, valid_probs]
    """
    def __init__(self, gamma=0.9):
        self.gamma = gamma
        self.records = []
        self.winner = None

    def add4(self, board, from_, action, reward, vp=None, win=False):
        """
        吃子有正回报(被吃没有负回报)，输赢有额外奖励(1/-1)
        """
        if win and reward == 0:
            reward = 1
        self.records.append([board, from_, action, reward, vp])
        # player = board[from_]
        if win:
            winner = board[from_]
            self.winner = winner
            # 赢棋时计算每一步的总回报
            player_rewards = [0, 0, 0]
            for i,rc in enumerate(reversed(self.records)):
                b, f_, a, r, _ = rc
                player = int(b[f_])
                rc[3] += player_rewards[player] * self.gamma
                player_rewards[player] = rc[3]
            for record in self.records:
                b, f_, _,_,_ = record
                record[3] += 1 if b[f_] == winner else -1

    def add0(self, board, from_, action, reward, vp=None, win=False):
        """
        只有吃子有回报，没有赢了的额外奖励
        """
        if win and reward == 0:
            reward = 1
        self.records.append([board, from_, action, reward, vp])
        # player = board[from_]
        if reward > 0 and len(self.records) > 1:
            # 将对手上一步的回报-reward
            self.records[-2][3] -= reward
        if win:
            winner = board[from_]
            self.winner = winner
            # 赢棋时计算每一步的总回报
            player_rewards = [0, 0, 0]
            for i,rc in enumerate(reversed(self.records)):
                b, f_, a, r, _ = rc
                player = int(b[f_])
                rc[3] += player_rewards[player] * self.gamma
                player_rewards[player] = rc[3]

    def add1(self, board, from_, action, reward, vp=None, win=False):
        """
        吃子有回报，赢了有额外奖励
        """
        if win and reward == 0:
            reward = 1
        self.records.append([board, from_, action, reward, vp])
        # player = board[from_]
        if reward > 0 and len(self.records) > 1:
            # 将对手上一步的回报-reward
            self.records[-2][3] -= reward
            # for i, r in enumerate(filter(lambda rc: rc[0][rc[1]]==player, reversed(self.records[:-1]))):
            #     r[3] += (reward * self.gamma ** (i + 1))
            # for i, r in enumerate(filter(lambda rc: rc[0][rc[1]]!=player, reversed(self.records[:-1]))):
            #     r[3] -= (reward * self.gamma ** i)
        if win:
            winner = board[from_]
            self.winner = winner
            # 赢棋时计算每一步的总回报
            player_rewards = [0, 0, 0]
            winner = board[from_]
            for i,rc in enumerate(reversed(self.records)):
                b, f_, a, r, _ = rc
                player = int(b[f_])
                rc[3] += player_rewards[player] * self.gamma
                player_rewards[player] = rc[3]
            # 将reward缩放为最大为1(-1)
            # max_reward = max(map(lambda rec:rec[3], self.records))
            # min_reward = min(map(lambda rec:rec[3], self.records))
            # logger.info('max_reward:%s, min_reward:%s', max_reward, min_reward)
            # for i, r in enumerate(filter(lambda rec: rec[0][rec[1]]==winner, self.records)):
            #     r[3] /= max_reward
            # for i, r in enumerate(filter(lambda rec: rec[0][rec[1]]!=winner, self.records)):
            #     r[3] /= np.abs(min_reward)
            # 赢的一方额外奖励1，输的一方额外奖励-1
            for record in self.records:
                b, f_, _,_,_ = record
                record[3] += 1 if b[f_] == winner else -1

    add = add1

    def add2(self, board, from_, action, reward, vp=None, win=False):
        """
        只有赢/输的奖励(1/-1)
        """
        self.records.append([board, from_, action, reward, vp])
        if win:
            winner = board[from_]
            self.winner = winner
            for rc in self.records:
                rc[3] = 1 if rc[0][rc[1]]==winner else -1

    def add3(self, board, from_, action, reward, vp=None, win=False):
        """
        只有赢/输的奖励(1/0)
        """
        self.records.append([board, from_, action, reward, vp])
        if win:
            winner = board[from_]
            self.winner = winner
            for rc in self.records:
                rc[3] = 1 if rc[0][rc[1]]==winner else 0

    def save(self, path_pre):
        """
        保存棋谱
        """
        filepath = path_pre + str(int(time.time() * 1000)) + str(len(self.records)) + '.record'
        f = open(filepath, 'w')
        for board, from_, action, reward, vp in self.records:
            board = ''.join(map(str, board.flatten().astype(np.int8) + 1))
            from_ = ''.join(map(str, from_))
            action = str(action)
            reward = str(round(reward,4))
            f.write(','.join((board, from_, action, reward)) + '\n')
        f.close()
        return filepath

    def read(self, filepath):
        """
        读取棋谱
        :param filepath:
        """
        need_flip = '1st' in filepath
        with open(filepath) as f:
            for line in f:
                board, from_, action, reward = line.split(',')
                board = (np.array([int(i) for i in board]) - 1).reshape(5,5)
                from_ = tuple(map(int, from_))
                action = int(action)
                reward = float(reward)
                player = board[from_]
                if need_flip and player == -1:
                    board = rule.flip_board(board)
                    from_ = rule.flip_location(from_)
                    action = rule.flip_action(action)
                self.records.append([board, from_, action, reward])

    def __iter__(self):
        return iter(self.records)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, item):
        return self.records[item]

    def length(self):
        return len(self.records)

    def clear(self):
        self.records.clear()