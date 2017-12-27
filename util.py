import time
import numpy as np
import logging


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


def value_to_probs(values):
    values = np.array(values)
    x = np.log(values) - np.log(1.0001 - values)
    y = np.e ** x
    return y / y.sum()
