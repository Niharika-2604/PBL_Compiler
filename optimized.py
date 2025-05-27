def benchmark():
    result = 0
    i = 0
    while i < 100:
        result += i * i
        result += i * i
        i += 2
    return result
