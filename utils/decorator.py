import time
from functools import wraps

def timeit(func):
    @wraps(func)
    def wrapper(*args,**kwargs):
        s = time.time()
        r = func(*args,**kwargs)
        e = time.time()
        print(f"Time Costing:{e-s:.2f}")
        return r
    return wrapper


def cache4method(func):
    """类中的方法专用的缓存装饰器
    """
    @wraps(func)
    def wrapper(self,*args,**kwargs):
        keys = list(args) + list(kwargs.values())
        for key in keys:
            key = str(id(self)) + str(key)
            if key not in wrapper.cache:
                wrapper.cache[key] = func(self,*args,**kwargs)
            return wrapper.cache[key]
    wrapper.cache = {}
    return wrapper



if __name__ == "__main__":
    @timeit
    def foo():
        print(2)
    foo()