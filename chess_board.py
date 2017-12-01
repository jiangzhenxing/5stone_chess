import tkinter as tk
import numpy as np
import time
import threading
import chess_rule as rule
from tkinter import font
from tkinter import messagebox
from player import HummaPlayer, PolicyNetworkPlayer
from record import Record

BLACK = 'BLACK'
WHITE = 'WHITE'
BLACK_VALUE = 1
WHITE_VALUE = -1

class ChessBoard:
    def __init__(self):
        window = tk.Tk()
        window.title('五子棋')
        window.geometry('600x700')

        w = 100 # 棋格宽度
        r = 35  # 棋子半径
        row = 5
        col = 5
        canvas = tk.Canvas(window, bg='blue', width=w * 6, height=w * 6)
        canvas.pack()

        # 棋盘
        for i in range(1, 6):
            canvas.create_line(w, i * w, 5 * w, i * w, width=2)
            canvas.create_line(i * w, w, i * w, 5 * w, width=2)

        # 棋子
        self._stone = [[None for _ in range(col)] for _ in range(row)]

        # 棋手的名字
        self.name_white = canvas.create_text(40, 30, text=WHITE, fill='#EEE', font=font.Font(size=20))
        self.name_black = canvas.create_text(40, w*6-30, text=BLACK, fill='#111', font=font.Font(size=20))

        # 指示当前棋手信号灯
        sig_r = 5
        self.sig_white = canvas.create_oval(82-sig_r, 30-sig_r, 82 + sig_r, 30 + sig_r, fill='#66FF66', outline='#66FF66', state=tk.HIDDEN)
        self.sig_black = canvas.create_oval(82 - sig_r, w*6 - (30 - sig_r), 82 + sig_r, w*6 - (30 + sig_r), fill='#66FF66', outline='#66FF66', state=tk.HIDDEN)

        # 显示胜利者
        self.winner_white = canvas.create_text(125, 30, text='WINNER', fill='red', font=font.Font(size=20), state=tk.HIDDEN)
        self.winner_black = canvas.create_text(125, w*6 - 30, text='WINNER', fill='red', font=font.Font(size=20), state=tk.HIDDEN)

        # 计时钟
        self.clock_white = canvas.create_text(w*6-40, 30, text='00:00', fill='#EEE', font=font.Font(family='Times', size=20))
        self.clock_black = canvas.create_text(w*6-40, w * 6 - 30, text='00:00', fill='#111', font=font.Font(family='Times', size=20))

        # 开始按扭
        start_btn_text = tk.StringVar(window, value='start')
        tk.Button(window, textvariable=start_btn_text, command=self.start).place(x=100, y=620)

        # replay按扭
        tk.Button(window, text='replay', command=self.replay).place(x=200, y=620)

        # 速度调节按扭
        tk.Button(window, text='faster', command=lambda:self.change_period(0.5)).place(x=300, y=620)
        tk.Button(window, text='slower', command=lambda:self.change_period(2)).place(x=400, y=620)

        # 暂停按扭
        pause_text = tk.StringVar(value='pause')
        tk.Button(window, textvariable=pause_text, command=self.pause, width=5).place(x=500, y=620)

        # 消息显示
        lb = tk.Label(window, width=40)
        lb.place(x=85, y=650)

        self.w = w
        self.r = r
        self.row = row
        self.col = col
        self.period = 1
        self.window = window
        self.canvas = canvas
        self.pause_text = pause_text
        self.text = lb
        self.black_player = None
        self.white_player = None
        self.current_player = None
        self.current_stone = None
        self.timer = None
        self.winner = None
        self.ended = False
        self.start_btn_text = start_btn_text
        self.event = threading.Event()
        self.record = Record('records/app/')    # 记录棋谱: board:from_:to_:del_num
        self.record_path = ''
        self.init_stone()

    def start(self):
        if self.start_btn_text.get() == 'restart':
            if not messagebox.askokcancel(title='请确认', message='确定取消当前棋局并重新开局？'):
                return
        self.clear()
        self.init_stone()

        white_player = PolicyNetworkPlayer('PIG', WHITE_VALUE, self.sig_white, self.winner_white, self.clock_white, modelfile='model/policy_network2_082.model0')
        black_player = HummaPlayer('Jhon', BLACK_VALUE, self.sig_black, self.winner_black, self.clock_black)

        # white_player= HummaPlayer('Jhon', WHITE_VALUE, self.sig_white, self.winner_white, self.clock_white)
        # black_player = PolicyNetworkPlayer('PIG', BLACK_VALUE, self.sig_black, self.winner_black, self.clock_black, modelfile='model/policy_network_005.model')
        for stone in self._stone[0]:
            stone.player = white_player
        for stone in self._stone[4]:
            stone.player = black_player
        self.black_player = black_player
        self.white_player = white_player
        self.canvas.itemconfigure(self.name_white, text=white_player.name)
        self.canvas.itemconfigure(self.name_black, text=black_player.name)
        self.current_player = black_player
        self.show_signal()
        self.begin_timer()
        self.canvas.bind('<Button-1>', self.onclick)
        self.start_btn_text.set('restart')
        self.play()

    def change_period(self, scale):
        self.period *= scale
        print(self.period)

    def play(self):
        print(self.current_player, ' play...')
        if self.current_player.is_humman():
            return
        from_,to_ = self.current_player.play(self.board())
        stone = self.stone(from_)
        result = self.move_to(stone, to_)
        if result == rule.ACCQUIRE:
            self.switch_player_and_play()
        elif result == rule.WIN:
            print('GAME OVER, WINNER IS', stone.player.name)
            self.game_over(stone.player)

    def clear(self):
        for board_row in self._stone:
            for s in board_row:
                if s is not None:
                    self.del_stone(s)
        if self.timer:
            self.cancel_timer()
        self.hide_signal()
        self.hide_winner()
        self.black_player = None
        self.white_player = None
        self.current_player = None
        self.current_stone = None
        self.timer = None
        self.winner = None
        self.ended = False

    def init_stone(self):
        stone, canvas, w, r, row, col = self._stone, self.canvas, self.w, self.r, self.row, self.col
        white = [Stone((0, j), canvas.create_oval(w * (j + 1) - r, w - r, w * (j + 1) + r, w + r, fill='#EEE', outline='#EEE'), value=WHITE_VALUE) for j in range(col)]
        black = [Stone((4, j), canvas.create_oval(w * (j + 1) - r, w * row - r, w * (j + 1) + r, w * row + r, fill='#111', outline='#111'), value=BLACK_VALUE) for j in range(col)]
        stone[0] = white
        stone[row - 1] = black

    def show_signal(self):
        self.canvas.itemconfigure(self.current_player.signal, state=tk.NORMAL)

    def hide_signal(self):
        self.canvas.itemconfigure(self.sig_white, state=tk.HIDDEN)
        self.canvas.itemconfigure(self.sig_black, state=tk.HIDDEN)

    def show_winner(self):
        self.canvas.itemconfigure(self.winner.winner_text, state=tk.NORMAL)

    def hide_winner(self):
        self.canvas.itemconfigure(self.winner_white, state=tk.HIDDEN)
        self.canvas.itemconfigure(self.winner_black, state=tk.HIDDEN)

    def begin_timer(self):
        self.current_player.begin_time = int(time.time())
        def timer(player):
            use_time = int(time.time()) - player.begin_time
            player.total_time += use_time
            self.canvas.itemconfigure(player.clock, text='%02d:%02d' % (use_time // 60, use_time % 60))
            self.timer = self.canvas.after(1000, timer, player)
        self.timer = self.canvas.after(1000, timer, self.current_player)

    def cancel_timer(self):
        self.canvas.after_cancel(self.timer)
        self.canvas.itemconfigure(self.current_player.clock, text='00:00')

    def game_over(self, winner):
        # self.lb.config(text='winner is: ' + str(winner))
        self.winner = winner
        self.ended = True
        self.canvas.unbind('<Button-1>')
        self.cancel_timer()
        self.hide_signal()
        self.show_winner()
        self.start_btn_text.set('start')
        self.record_path = self.record.save()
        self.record.clear()

    def board(self):
        return np.array([[0 if s is None else s.value for s in row] for row in self._stone])

    def onmotion(self, event):
        self.text.config(text='position is (%d,%d)' % (event.x, event.y))
        self.move_to_pos(self.current_stone, event.x, event.y)

    def onclick(self, event):
        x,y = event.x, event.y
        if  not (self.w - self.r < x < self.w * 5 + self.r and self.w - self.r < y < self.w * 5 + self.r):
            self.text.config(text='click at (%d,%d)' % (event.x, event.y))
            return
        if not self.current_player.is_humman():
            return
        posx = x / self.w - 0.5
        posy = y / self.w - 0.5
        # 整数部分
        posx_i = int(posx)
        posy_i = int(posy)
        # 小数部分
        posx_d = posx - posx_i
        posy_d = posy - posy_i

        self.text.config(text='click at (%d,%d) position is %d,%d' % (event.x, event.y, posx_i, posy_i))

        if 0.25 < posx_d < 0.75 and 0.25 < posy_d < 0.75:
            loc = self.pos_to_loc(x, y)
            if self.current_stone is None:
                stone = self.stone(loc)
                if stone and stone.player == self.current_player:
                    self.move_to_pos(stone, x, y)
                    self.begin_moving(stone)
            else:
                self.end_moving(loc)

    def begin_moving(self, stone):
        self.current_stone = stone
        self.canvas.bind('<Motion>', self.onmotion)

    def end_moving(self, loc):
        stone = self.current_stone
        result = self.move_to(stone, loc)
        if result == rule.INVALID_MOVE:
            return False
        self.canvas.unbind('<Motion>')
        self.current_stone = None

        if result == rule.NOT_MOVE:
            self.move_to_loc(stone, loc)

        if result == rule.ACCQUIRE:
            self.switch_player_and_play()
        elif result == rule.WIN:
            print('GAME OVER, WINNER IS', stone.player.name)
            self.game_over(stone.player)

    def move_to(self, stone, to_loc):
        """
        把棋子移动到to_loc处，同时判断是否吃子
        :param stone:
        :param to_loc:
        :return: True:终止移动，False:继续移动
        """
        old_board = self.board()
        from_ = stone.loc
        result,del_stone_loc = rule.move(self.board(), stone.loc, to_loc)
        print(result, del_stone_loc)
        if result == rule.ACCQUIRE or result == rule.WIN:
            self.move_to_loc(stone, to_loc)
            for loc in del_stone_loc:
                self.del_stone(self.stone(loc))
            print(from_, to_loc)
            action = rule.actions_move.index(tuple(np.subtract(to_loc, from_)))
            print(action)
            self.record.add(old_board, from_, action, len(del_stone_loc), win=(result==rule.WIN))
        return result

    def replay(self):
        recordpath = 'records/train/1512130266665156.record'
        if self.record_path:
            recordpath = self.record_path
        self.clear()
        self.init_stone()
        self.event.set()
        record = Record()
        record.read(recordpath)
        record_iter = iter(record)
        length = record.length()
        record.n = 1
        def play_next_step():
            self.event.wait()
            board, from_, action, reward = next(record_iter)
            if not (board == self.board()).all():
                print(board)
                print('-' * 50)
                print(self.board())
                raise ValueError
            player = board[from_]
            to_ = tuple(np.add(from_, rule.actions_move[action]))
            result = self.move_to(self.stone(from_), to_)
            if result == rule.WIN:
                winner_text = self.winner_black if player==1 else self.winner_white
                self.canvas.itemconfigure(winner_text, state=tk.NORMAL)
            self.text.config(text=str(length) + ':' + str(record.n) + ',reward: ' + str(reward))
            record.n += 1
            threading.Timer(self.period, play_next_step).start()
            # self.window.after(self.period * 1000, play_next_step)
        threading.Timer(self.period, play_next_step).start()

    def pause(self):
        if self.pause_text.get() == 'pause':
            self.event.clear()
            self.pause_text.set('resume')
        else:
            self.event.set()
            self.pause_text.set('pause')

    def del_stone(self, stone):
        self.canvas.delete(stone.oval)
        self.stone(stone.loc, None)

    def switch_player(self):
        self.hide_signal()
        self.cancel_timer()
        self.current_player = self.white_player if self.current_player is self.black_player else self.black_player
        self.show_signal()
        self.begin_timer()

    def switch_player_and_play(self):
        self.switch_player()
        self.play()

    def move_to_loc(self, stone, loc):
        self.stone(stone.loc, None)
        self.stone(loc, stone)
        i, j = loc
        self.move_to_pos(stone, (j + 1) * self.w, (i + 1) * self.w)

    def move_to_pos(self, stone, x, y):
        self.canvas.coords(stone.oval, x - self.r, y - self.r, x + self.r, y + self.r)

    def stone(self, loc, value='NOVAL'):
        i,j = loc
        if value == 'NOVAL':
            return self._stone[i][j]
        elif value is None:
            self._stone[i][j] = None
        else:
            self._stone[i][j] = value
            value.loc = loc
            return value

    def pos_to_loc(self, x, y):
        i = int(y / self.w - 0.5)
        j = int(x / self.w - 0.5)
        if -1 < i < 5 and -1 < j < 5:
            return i,j

    @staticmethod
    def launch():
        tk.mainloop()

class Stone:
    def __init__(self, loc, oval, value, player=None):
        self.loc = loc
        self.oval = oval
        self.value = value
        self.player = player
