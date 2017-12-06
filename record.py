import time
import numpy as np
import logging

logger = logging.getLogger('app')

class Record:
    """
    棋谱,格式为：[board, from_, action, reward]
    """
    def __init__(self, gamma=0.9):
        self.gamma = gamma
        self.records = []

    def add1(self, board, from_, action, reward, win=False):
        if win and reward == 0:
            reward = 1
        self.records.append([board, from_, action, reward])
        # player = board[from_]
        if reward > 0:
            # 将对手上一步的回报-reward
            self.records[-2][-1] -= reward
            # for i, r in enumerate(filter(lambda rc: rc[0][rc[1]]==player, reversed(self.records[:-1]))):
            #     r[-1] += (reward * self.gamma ** (i + 1))
            # for i, r in enumerate(filter(lambda rc: rc[0][rc[1]]!=player, reversed(self.records[:-1]))):
            #     r[-1] -= (reward * self.gamma ** i)
        if win:
            # 赢棋时计算每一步的总回报
            player_rewards = [[], [], []]
            winner = board[from_]
            for i,rc in enumerate(reversed(self.records)):
                b, f_, a, r = rc
                player = int(b[f_])
                for j,rj in player_rewards[player]:
                    rc[-1] += rj * self.gamma ** (i-j)
                if r != 0:
                    player_rewards[player].append((i,r))
            # 将reward缩放为最大为1(-1)
            max_reward = max(map(lambda rec:rec[-1], self.records))
            min_reward = min(map(lambda rec:rec[-1], self.records))
            logger.info('max_reward:%s, min_reward:%s', max_reward, min_reward)
            for i, r in enumerate(filter(lambda rec: rec[0][rec[1]]==winner, self.records)):
                r[-1] /= max_reward
            for i, r in enumerate(filter(lambda rec: rec[0][rec[1]]!=winner, self.records)):
                r[-1] /= np.abs(min_reward)

        # if win:
        #     for record in self.records:
        #         b, f, _, _ = record
        #         record[-1] += 1 if b[f] == player else -1

    def add(self, board, from_, action, reward, win=False):
        self.records.append([board, from_, action, 0])
        if win:
            winner = board[from_]
            for rc in self.records:
                rc[-1] = 1 if rc[0][rc[1]]==winner else 0

    def __iter__(self):
        return iter(self.records)

    def save(self, path_pre):
        """
        保存棋谱
        """
        filepath = path_pre + str(int(time.time() * 1000)) + str(len(self.records)) + '.record'
        f = open(filepath, 'w')
        for board, from_, action, reward in self.records:
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
        with open(filepath) as f:
            for line in f:
                board, from_, action, reward = line.split(',')
                board = (np.array([int(i) for i in board]) - 1).reshape(5,5)
                from_ = tuple(map(int, from_))
                action = int(action)
                reward = float(reward)
                self.records.append([board, from_, action, reward])

    def length(self):
        return len(self.records)

    def __len__(self):
        return len(self.records)

    def clear(self):
        self.records.clear()