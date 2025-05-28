def sq(x):
    return x * x

def benchmark():
    result = 0
    for i in range(50):
        result += sq(i)
    if False:
        result -= 100
    return result