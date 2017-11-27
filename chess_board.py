import tkinter as tk
import numpy as np
import time
import chess_rule as rule
from tkinter import font
from tkinter import messagebox
from player import Player

BLACK = 'BLACK'
WHITE = 'WHITE'

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

        start_text = tk.StringVar(window, value='start')
        btn = tk.Button(window, textvariable=start_text, command=self.start)
        btn.pack()
        lb = tk.Label(window, width=40)
        lb.pack()

        self.w = w
        self.r = r
        self.row = row
        self.col = col
        self.window = window
        self.canvas = canvas
        self.lb = lb
        self.black_player = None
        self.white_player = None
        self.current_player = None
        self.current_stone = None
        self.timer = None
        self.winner = None
        self.ended = False
        self.start_text = start_text
        self.init_stone()

    def start(self):
        if self.start_text.get() == 'restart':
            if not messagebox.askokcancel(title='请确认', message='确定取消当前棋局并重新开局？'):
                return
        self.clear()
        self.init_stone()
        black_player = Player(BLACK, self.sig_black, self.winner_black, self.clock_black)
        white_player = Player(WHITE, self.sig_white, self.winner_white, self.clock_white)
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
        self.start_text.set('restart')
        self.play()

    def play(self):
        if self.current_player.play(self.board()) is None:
            pass

    def clear(self):
        if self.current_player:
            self.hide_signal()
        if self.timer:
            self.cancel_timer()
        for board_row in self._stone:
            for s in board_row:
                if s is not None:
                    self.del_stone(s)
        if self.winner:
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
        white = [Stone((0, j), canvas.create_oval(w * (j + 1) - r, w - r, w * (j + 1) + r, w + r, fill='#EEE', outline='#EEE'), value=-1) for j in range(col)]
        black = [Stone((4, j), canvas.create_oval(w * (j + 1) - r, w * row - r, w * (j + 1) + r, w * row + r, fill='#111', outline='#111'), value=1) for j in range(col)]
        stone[0] = white
        stone[row - 1] = black

    def show_signal(self):
        self.canvas.itemconfigure(self.current_player.signal, state=tk.NORMAL)

    def hide_signal(self):
        self.canvas.itemconfigure(self.current_player.signal, state=tk.HIDDEN)

    def show_winner(self):
        self.canvas.itemconfigure(self.winner.winner_text, state=tk.NORMAL)

    def hide_winner(self):
        self.canvas.itemconfigure(self.winner.winner_text, state=tk.HIDDEN)

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
        self.lb.config(text='winner is: ' + str(winner))
        self.winner = winner
        self.ended = True
        self.canvas.unbind('<Button-1>')
        self.cancel_timer()
        self.hide_signal()
        self.show_winner()
        self.start_text.set('start')

    def board(self):
        return np.array([[0 if s is None else s.value for s in row] for row in self._stone])

    def onmotion(self, event):
        self.lb.config(text='position is (%d,%d)' % (event.x, event.y))
        self.move_to_pos(self.current_stone, event.x, event.y)

    def onclick(self, event):
        x,y = event.x, event.y
        if  not (self.w - self.r < x < self.w * 5 + self.r and self.w - self.r < y < self.w * 5 + self.r):
            self.lb.config(text='click at (%d,%d)' % (event.x, event.y))
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

        self.lb.config(text='click at (%d,%d) position is %d,%d' % (event.x, event.y, posx_i, posy_i))

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
        if self.move_to(self.current_stone, loc):
            self.canvas.unbind('<Motion>')
            self.current_stone = None

    def move_to(self, stone, to_loc):
        """
        把棋子移动到to_loc处，同时判断是否吃子
        :param stone:
        :param to_loc:
        :return: True:终止移动，False:继续移动
        """
        result,del_stone_loc = rule.move(self.board(), stone.loc, to_loc)
        print(result, del_stone_loc)
        if result == rule.NOT_MOVE:
            self.move_to_loc(self.current_stone, to_loc)
            return True
        if result == rule.INVALID_MOVE:
            return False
        if result == rule.ACCQUIRE or result == rule.WIN:
            self.move_to_loc(self.current_stone, to_loc)
            if del_stone_loc:
                for loc in del_stone_loc:
                    self.del_stone(self.stone(loc))
            if result == rule.ACCQUIRE:
                self.switch_player()
            elif result == rule.WIN:
                print('GAME OVER, WINNER IS', stone.player.name)
                self.game_over(stone.player)
            return True

    def del_stone(self, stone):
        self.canvas.delete(stone.oval)
        self.stone(stone.loc, None)

    def switch_player(self):
        self.hide_signal()
        self.cancel_timer()
        self.current_player = self.white_player if self.current_player is self.black_player else self.black_player
        self.show_signal()
        self.begin_timer()
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
