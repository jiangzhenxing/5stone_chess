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

