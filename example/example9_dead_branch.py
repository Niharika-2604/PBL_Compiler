def benchmark():
    if False:
        expensive_call()
    return 42

def expensive_call():
    return sum(i*i for i in range(10**6))