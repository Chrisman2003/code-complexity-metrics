def add(a, b):
    return a + b

def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def main():
    x = add(2, 3)
    y = factorial(x)
    if y > 100:
        print("Large factorial")
    else:
        print("Small factorial")

if __name__ == "__main__":
    main()