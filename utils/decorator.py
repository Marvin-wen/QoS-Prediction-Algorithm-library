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


def cache(func):
    def wrapper(*args,**kwargs):
        wraps(func)
        keys = args + tuple(kwargs.values())
        for key in keys:
            if key not in wrapper.cache:
                wrapper.cache[key] = func(*args,**kwargs)
            return wrapper.cache[key]
    wrapper.cache = {}
    return wrapper

if __name__ == "__main__":
    @timeit
    def foo():
        print(2)
    foo()