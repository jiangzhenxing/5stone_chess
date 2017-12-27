import tkinter as tk
import numpy as np
import time
import threading
import chess_rule as rule
from tkinter import font
from tkinter import messagebox
from tkinter import filedialog
from tkinter import ttk
from player import HummaPlayer, PolicyNetworkPlayer, DQNPlayer,MCTSPlayer
from record import Record
import logging

logger = logging.getLogger('app')

BLACK = 'BLACK'
WHITE = 'WHITE'
BLACK_VALUE = 1
WHITE_VALUE = -1


class Stone:
    def __init__(self, loc, oval, value, player=None):
        self.loc = loc
        self.oval = oval
        self.value = value
        self.player = player
Stone.NONE = Stone(None,None,None)


class ChessBoard:
    def __init__(self):
        window = tk.Tk()
        window.title('五子棋')
        window.geometry('600x720')

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
        self._board = [[None for _ in range(col)] for _ in range(row)]

        # Q值
        self._qtext = [[[canvas.create_text(*(np.array([w * (j + 1), w * (i + 1)]) + np.array(a)[::-1] * 25), text='', fill='green') for a in rule.actions_move] for j in range(5)] for i in range(5)]

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

        # ----------------- 第一排按扭 -----------------------------------------
        # 对手选择
        tk.Label(window, text='请选择对手:').place(x=10, y=620)
        player_var = tk.StringVar()
        player_classes = [('White(HummaPlayer)', HummaPlayer,('White', WHITE_VALUE, self.sig_white, self.winner_white, self.clock_white), {}),
                          ('Quin(DQNPlayer)', DQNPlayer, ['Quin', WHITE_VALUE, self.sig_white, self.winner_white, self.clock_white], {'play_func':self._play, 'modelfile':'model/qlearning_network/DQN_sigmoid_00169.model'}),
                          ('Toms(MCTSPlayer)', MCTSPlayer, ['Toms', WHITE_VALUE, self.sig_white, self.winner_white, self.clock_white], {'play_func':self._play, 'policy_model':'model/qlearning_network/DQN_sigmoid_00169.model', 'worker_model':'model/qlearning_network/DQN_sigmoid_00169.model'}),]
        self.player_map = {n:(c, p, kp) for n, c, p, kp in player_classes}
        players = [n for n, *_ in player_classes]
        player_choosen = ttk.Combobox(window, width=18, textvariable=player_var, values=players, state='readonly')
        player_choosen.current(0)
        player_choosen.place(x=90, y=617)
        self.player_var = player_var

        # 先手
        first_player = tk.IntVar(value=BLACK_VALUE)
        tk.Radiobutton(window, text='黑先', variable=first_player, value=BLACK_VALUE).place(x=290, y=620)
        tk.Radiobutton(window, text='白先', variable=first_player, value=WHITE_VALUE).place(x=345, y=620)
        self.first_player = first_player

        # 开始按扭
        start_btn_text = tk.StringVar(window, value='start')
        tk.Button(window, textvariable=start_btn_text, command=self.start).place(x=415, y=617)

        # 帮助按扭
        tk.Button(window, text='help', command=None).place(x=510, y=617)

        # ---------------- 第二排按扭 ------------------------------------------
        # 选择棋谱
        tk.Label(window, text='棋谱:').place(x=10, y=658)
        record_path = tk.StringVar()
        record_entry = tk.Entry(window, textvariable=record_path, width=18)
        record_entry.place(x=45, y=655)
        tk.Button(text='select', command=self.select_record).place(x=222, y=655)

        # replay按扭
        tk.Button(window, text='replay', command=self.replay).place(x=290, y=655)

        # 速度调节按扭
        tk.Button(window, text='faster', command=lambda:self.change_period(0.5)).place(x=360, y=655)
        tk.Button(window, text='slower', command=lambda:self.change_period(2)).place(x=430, y=655)

        # 暂停按扭
        pause_text = tk.StringVar(value='pause')
        tk.Button(window, textvariable=pause_text, command=self.pause, width=5).place(x=510, y=655)

        # ---------------- 第三排按扭 ------------------------------------------
        # 消息显示
        lb = tk.Label(window, width=40)
        lb.place(x=85, y=690)

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
        self.players = [None,None,None]
        self.current_player = None
        self.current_stone = None
        self.timer = None
        self.play_timer = None
        self.replay_timer = None
        self.winner = None
        self.ended = False
        self.action_select_signal = None
        self.start_btn_text = start_btn_text
        self.record_path = record_path
        self.record_entry = record_entry
        self.event = threading.Event()
        self.record = Record()    # 记录棋谱: board:from_:to_:del_num
        self.stones = []
        self.init_stone()

    def start(self):
        if self.start_btn_text.get() == 'restart':
            if not messagebox.askokcancel(title='请确认', message='确定取消当前棋局并重新开局？'):
                return
        self.clear()
        bd = [[0, 0, 0, 0, 0],
              [0, 1,-1, 0, 0],
              [0, 0, 1, 0, 0],
              [0,-1, 0, 0, 0],
              [0, 0, 0, 0, 0],]
        # bd = -np.array(bd)
        self.init_stone()
        init_board = self.board()
        first_player = self.first_player.get()
        player_name = self.player_var.get()
        player_class, p, kp = self.player_map[player_name]
        white_player = player_class(*p, **kp)
        # white_player = PolicyNetworkPlayer('Paul', WHITE_VALUE, self.sig_white, self.winner_white, self.clock_white, play_func=self._play, modelfile='model/policy_network/convolution_6000.model')
        # white_player = DQNPlayer('Quin', WHITE_VALUE, self.sig_white, self.winner_white, self.clock_white, play_func=self._play, modelfile='model/qlearning_network/DQN_dr_3000.model')
        # white_player = MCTSPlayer('Toms', WHITE_VALUE, self.sig_white, self.winner_white, self.clock_white, play_func=self._play, modelfile='model/DQN_0090.model', )
        black_player = HummaPlayer('Jhon', BLACK_VALUE, self.sig_black, self.winner_black, self.clock_black)
        self.players[WHITE_VALUE] = white_player
        self.players[BLACK_VALUE] = black_player
        for stone in self.stones:
            stone.player = white_player if stone.value == WHITE_VALUE else black_player
        self.black_player = black_player
        self.white_player = white_player
        self.canvas.itemconfigure(self.name_white, text=white_player.name)
        self.canvas.itemconfigure(self.name_black, text=black_player.name)
        self.current_player = self.players[first_player]
        self.show_signal()
        self.begin_timer()
        self.canvas.bind('<Button-1>', self.onclick)
        self.start_btn_text.set('restart')
        self.white_player.start(init_board=init_board, first_player=first_player)
        self.black_player.start(init_board=init_board, first_player=first_player)
        self.play()

    def play(self):
        logger.info('%s play...', self.current_player)
        if self.current_player.is_humman():
            return
        board = self.board()
        self.current_player.play(board)

    def _play(self, board, player, from_, to_, p):
        logger.info('from:%s, to_:%s', from_, to_)
        logger.info('p:\n%s', p)
        valid_action = rule.valid_action(board, player)
        logger.info('valid_action:\n%s', valid_action)
        self.show_qtext(p, valid_action)
        self.show_select(from_, to_)
        stone = self.stone(from_)
        def play_later():
            result = self.move_to(stone, to_)
            if result == rule.ACCQUIRE:
                # 预测对手走棋
                bd = self.board()
                pl = -player
                op = self.current_player.predict_opponent(bd)
                if op is not None:
                    valid = rule.valid_action(bd, pl)
                    self.show_qtext(op, valid, hide=False)
                # 对手走棋
                self.switch_player_and_play()
            elif result == rule.WIN:
                logger.info('GAME OVER, WINNER IS %s', stone.player.name)
                self.game_over(stone.player)
        self.play_timer = self.window.after(int(self.period * 1000), play_later)

    def init_stone(self, board=None):
        if board is None:
            board = np.zeros((5,5))
            board[0,:] = -1
            board[4,:] = 1
        else:
            board = np.array(board)
        for i,j in np.argwhere(board == WHITE_VALUE):
            self.create_stone(i, j, WHITE_VALUE)
        for i, j in np.argwhere(board == BLACK_VALUE):
            self.create_stone(i, j, BLACK_VALUE)

    def onmotion(self, event):
        # self.show_message('position is (%d,%d)' % (event.x, event.y))
        self.move_to_pos(self.current_stone, event.x, event.y)

    def onclick(self, event):
        x,y = event.x, event.y
        if  not (self.w - self.r < x < self.w * 5 + self.r and self.w - self.r < y < self.w * 5 + self.r):
            # self.show_message('click at (%d,%d)' % (event.x, event.y))
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

        # self.show_message('click at (%d,%d) position is %d,%d' % (event.x, event.y, posx_i, posy_i))

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
        board = self.board()
        from_ = stone.loc
        result = self.move_to(stone, loc)
        if result == rule.INVALID_MOVE:
            return False
        self.canvas.unbind('<Motion>')
        self.current_stone = None

        if result == rule.NOT_MOVE:
            self.move_to_loc(stone, loc)

        if result == rule.ACCQUIRE:
            # 将走的子告知对手
            self.opponent().opponent_play(board, from_, loc)
            self.switch_player_and_play()
        elif result == rule.WIN:
            logger.info('GAME OVER, WINNER IS %s', stone.player.name)
            self.opponent().opponent_play(None, None, None)
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
        if result == rule.ACCQUIRE or result == rule.WIN:
            self.move_to_loc(stone, to_loc)
            for loc in del_stone_loc:
                self.del_stone(self.stone(loc))
            logger.info('from %s to %s, result:%s, del:%s', from_, to_loc, result, del_stone_loc)
            action = rule.actions_move.index(tuple(np.subtract(to_loc, from_)))
            logger.debug('action is: %s', action)
            self.record.add(old_board, from_, action, len(del_stone_loc), None, win=(result==rule.WIN))
        return result

    def replay(self):
        recordpath = self.record_path.get()
        if not recordpath:
            self.show_message(message='请点击"open"按扭选择棋谱位置')
            return
        self.clear()
        self.event.set()
        record = Record()
        record.read(recordpath)
        init_board = record[0][0]
        self.init_stone(init_board)
        record_iter = iter(record)
        length = len(record)
        record.n = 1
        def play_next_step():
            self.event.wait()
            try:
                board, from_, action, reward = next(record_iter)
                player = board[from_]
                to_ = tuple(np.add(from_, rule.actions_move[action]))
                assert (board == self.board()).all(), str(board) + '\n' + str(self.board())
                result = self.move_to(self.stone(from_), to_)
                if result == rule.WIN:
                    winner_text = self.winner_black if player==1 else self.winner_white
                    self.canvas.itemconfigure(winner_text, state=tk.NORMAL)
                self.show_message(str(length) + ':' + str(record.n) + ',reward: ' + str(reward))
                record.n += 1
                self.replay_timer = threading.Timer(self.period, play_next_step)
                self.replay_timer.start()
            except StopIteration:
                return
        self.replay_timer = threading.Timer(self.period, play_next_step)
        self.replay_timer.start()

    def select_record(self):
        f = filedialog.askopenfilename()
        if f:
            self.record_path.set(f)
            self.record_entry.index(len(f)-10)

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

    def opponent(self, player=None):
        if player is None:
            player = self.current_player
        return self.players[-player.stone_val]

    def stone(self, loc, value=Stone.NONE):
        i,j = loc
        if value != Stone.NONE:
            self._board[i][j] = value
            if value is not None:
                value.loc = loc
        return self._board[i][j]

    def create_stone(self, i, j, value):
        w, r = self.w, self.r
        color = '#EEE' if value == WHITE_VALUE else '#111'
        s = Stone((i, j), self.create_oval(w * (j + 1), w * (i + 1), r, fill=color, outline=color), value=value)
        self.stone((i, j), value=s)
        self.stones.append(s)
    def pos_to_loc(self, x, y):
        i = int(y / self.w - 0.5)
        j = int(x / self.w - 0.5)
        if -1 < i < 5 and -1 < j < 5:
            return i,j

    def clear(self):
        for s in self.stones:
            self.del_stone(s)
        self.stones.clear()
        if self.timer:
            self.cancel_timer()
        if self.play_timer:
            self.window.after_cancel(self.play_timer)
        if self.replay_timer:
            self.replay_timer.cancel()
        self.hide_signal()
        self.hide_winner()
        self.hide_qtext()
        self.hide_select()
        self.start_btn_text.set('start')
        self.pause_text.set('pause')
        self.show_message('')
        if self.black_player:
            self.black_player.stop()
            self.black_player = None
        if self.white_player:
            self.white_player.stop()
            self.white_player = None
        self.current_player = None
        self.current_stone = None
        self.timer = None
        self.winner = None
        self.ended = False
        self.record.clear()

    def game_over(self, winner):
        # self.lb.config(text='winner is: ' + str(winner))
        self.winner = winner
        self.ended = True
        self.canvas.unbind('<Button-1>')
        self.cancel_timer()
        self.hide_signal()
        self.show_winner()
        self.start_btn_text.set('start')
        self.record_path.set(self.record.save('records/app/' + winner.name + '_'))
        self.record.clear()
        self.white_player.stop()
        self.black_player.stop()

    def board(self):
        return np.array([[0 if s is None else s.value for s in row] for row in self._board])

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

    def change_period(self, scale):
        self.period *= scale
        logger.debug(self.period)

    def show_select(self, from_, to_):
        """
        显示选择的动作
        """
        self.hide_select()
        a = np.subtract(to_, from_)
        logger.debug('select action is: %s', tuple(a))
        i, j = from_
        x, y = np.array([self.w * (j + 1), self.w * (i + 1)]) + np.array(a)[::-1] * 25
        self.action_select_signal = self.create_oval(x, y, r=13, outline='#FFB90F',
                                                     width=2)

    def hide_select(self):
        """
        隐藏选择的动作
        """
        self.canvas.delete(self.action_select_signal)

    def create_oval(self, x, y, r, **config):
        return self.canvas.create_oval(x - r, y - r, x + r, y + r, **config)

    def show_qtext(self, qtable, valid_action, hide=True):
        """
        显示动作的Q值
        """
        if hide:
            self.hide_qtext(valid_action)
        maxq = np.max(qtable)
        avgq = qtable.sum() / valid_action.sum()
        idx = np.argwhere(valid_action == 1)
        # logger.info(idx)
        for i, j, k in idx:
            q = qtable[i, j, k]
            qtext = self.qtext(i, j, k)
            stone = self.stone((i, j))
            self.canvas.itemconfigure(qtext, text=str(round(q, 2)).replace('0.', '.'), fill='red' if q == maxq else ('#BF3EFF' if q > avgq else 'green'), state=tk.NORMAL)
            self.canvas.tag_raise(qtext, stone.oval)

    def hide_qtext(self, valid_action=None):
        """
        隐藏动作的Q值
        """
        if valid_action is None:
            valid_action = np.zeros((5, 5, 4))
        idx = np.argwhere(valid_action == 0)
        for i, j, k in idx:
            self.canvas.itemconfigure(self.qtext(i, j, k), text='', state=tk.HIDDEN)

    def qtext(self, i, j, k):
        return self._qtext[i][j][k]

    def show_message(self, message):
        self.text.config(text=message)

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

    @staticmethod
    def launch():
        tk.mainloop()

