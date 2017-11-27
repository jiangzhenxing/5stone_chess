HUMMAN = 'HUMMAN'
class Player:
    def __init__(self, name, signal, winner_text, clock, type_=HUMMAN):
        self.name = name
        self.signal = signal
        self.winner_text = winner_text
        self.clock = clock
        self.type_ = type_
        self.begin_time = 0
        self.total_time = 0

    def play(self, board):
        """
        返回(位置,动作)
        人类选手由界面操作，所以不执行任何动作
        """
        pass

    def is_humman(self):
        return self.type_ == HUMMAN

    def __str__(self):
        return self.name + ':' + self.type_

    def __repr__(self):
        return str(self)
