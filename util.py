import time
import numpy as np
import logging
import os
import threading

print_time_flag = True
print_time_func = set() # update_qtable

def print_use_time(min_time=1):
    """
    生成一个方法调用计时装饰器
    :param min_time: 需要打印的最小调用时间
    """
    def timer(func):
        def call(*args, **kwargs):
            if print_time_flag and func.__name__ in print_time_func:
                start = time.time()
                result = func(*args, **kwargs)
                use_time = int((time.time() - start) * 1000)
                if use_time > min_time:
                    logging.info('%s use: %d', func.__name__, use_time)
            else:
                result = func(*args, **kwargs)
            return result
        return call
    return timer

def add_print_time_fun(func):
    for f in func:
        print_time_func.add(f)

def random_choice(a):
    """
    从a中随机选择
    """
    return a[np.random.randint(len(a))]

def select_by_prob(actions, probs):
    # logging.debug(actions)
    # logging.debug(probs)
    rd = np.random.rand()
    s = 0
    for a,p in zip(actions, probs):
        s += p
        if s > rd:
            return a

def board_str(board):
    return ''.join(map(str, board.flatten()))

def find_1st(L, func):
    """
    查找L中第一个符合条件func的元素
    """
    for e in L:
        if func(e):
            return e

def choose_max_random(a):
    """
    随机选取a中最大的一个值
    """
    a = np.array(a)
    idx = np.argwhere(a == a.max())
    return tuple(random_choice(idx))

def rand_int():
    return int(time.time() * 1000 * os.getpid() + id(threading.current_thread()))

def rand_int32():
    return rand_int() & (2 ** 32 - 1)

def softmax(a):
    a = np.array(a)
    ea = np.e ** a
    return ea / ea.sum()

LOCAL = threading.local()
def load_model(filepath):
    if not hasattr(LOCAL, 'models'):
        LOCAL.models = {}
    models = LOCAL.models

    if filepath in models:
        return models[filepath]
    else:
        from keras.models import load_model
        model = load_model(filepath)
        models[filepath] = model
        return model

def show_model(model, columns=('input_shape','output_shape','params','kernel_regularizer')):
    idx = 1
    rows = []
    for l in model.layers:
        # config = l.get_config()
        values = [idx, type(l).__name__]
        for col in columns:
            if col == 'params':
                values.append(l.count_params())
            elif hasattr(l, col):
                value = getattr(l, col)
                if not value:
                    values.append('None')
                else:
                    if 'regularizer' in col:
                        reg_config = value.get_config()
                        reg_str = ''
                        if reg_config['l1'] > 0:
                            reg_str += 'l1:%g' % reg_config['l1'] + ','
                        if reg_config['l2'] > 0:
                            reg_str += 'l2:%g' % reg_config['l2'] + ','
                        values.append(reg_str[:-1])
                    else:
                        values.append(value)
            else:
                values.append('NotExist')
        rows.append([str(v).replace(' ','') for v in values])
        idx += 1

    columns = ['#', 'Layer']+list(columns)
    widths = [[len(c) for c in columns]]
    for r in rows:
        widths.append([len(v) for v in r])
    max_widths = np.max(widths, axis=0)
    width = max_widths.sum() + (len(columns)-1) * 2

    def printline(cols, end='-'):
        for c, w in zip(cols, max_widths):
            print(c + ' ' * (w - len(c)), end='  ')
        print()
        print(end * width)

    # print title
    print('-' * width)
    printline(columns, end='=')

    # print layers
    for i,r in enumerate(rows):
        printline(r, end='-' if i < len(rows)-1 else '=')

    opt_config = model.optimizer.get_config()
    print('optimizer: lr:%g, decay:%g, momentum:%g, nesterov:%s' % (opt_config['lr'], opt_config['decay'], opt_config['momentum'], opt_config['nesterov'],))
    print('-' * width)