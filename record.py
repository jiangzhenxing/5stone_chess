import time
import numpy as np


class Record:
    """
    棋谱,格式为：[board, from_, action, reward]
    """
    def __init__(self, path='records/', gamma=0.9):
        self.path = path
        self.gamma = gamma
        self.records = []

    def add(self, board, from_, action, reward, win=False):
        self.records.append([board, from_, action, reward])
        player = board[from_]
        if reward > 0:
            for i, r in enumerate(filter(lambda rc: rc[0][rc[1]]==player, reversed(self.records[:-1]))):
                r[-1] += (reward * self.gamma ** (i + 1))
        if win:
            for record in self.records:
                b, f, _, _ = record
                record[-1] += 1 if b[f] == player else -1

    def __iter__(self):
        return iter(self.records)

    def save(self):
        """
        保存棋谱
        """
        filepath = self.path + str(int(time.time() * 1000)) + str(len(self.records)) + '.record'
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

    def clear(self):
        self.records.clear()