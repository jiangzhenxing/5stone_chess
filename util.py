import time

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
                    print(func.__name__, 'use:', use_time)
            else:
                result = func(*args, **kwargs)
            return result
        return call
    return timer