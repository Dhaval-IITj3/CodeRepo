# Addition Funtion
def add(x, y):
    mysum = x + y
    return mysum


# Subtraction Function
def subtract(x, y):
    diff = x - y
    return diff


# Multiplication Function
def multiply(x, y):
    product = x * y
    return product


# Division Function
def divide(x, y):
    # Check division by 0
    if (y == 0):
        print("Can not perform division by 0")
    else:  # if y != 0, perform division
        print(f"{x} / {y} = ", x / y)
