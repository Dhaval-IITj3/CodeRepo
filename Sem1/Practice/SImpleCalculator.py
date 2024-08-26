import sys
from calculator_operations import add, subtract, multiply, divide

num_1 = int(input("Enter a number "))
num_2 = int(input("Enter another number "))

operation = input("Enter an arithematic operator ")

# Help function
def my_help():
    print('Help:')
    print('The valid operations are:')
    print(" Enter '+' = to perform addition ")
    print(" Enter '-' = to perform subtraction ")
    print(" Enter '*' = to perform multiplication ")
    print(" Enter '/' = to perform division ")
    sys.exit()
# End of help function


if operation == "h" or operation == "H":
    # Call help() function
    my_help()

if operation == "+":
    result = add(num_1, num_2)
    print(f"{num_1} + {num_2} = ", result)
elif operation == "-":
    result = subtract(num_1, num_2)
    print(f"{num_1} + {num_2} = ", result)
elif operation == "*":
    result = multiply(num_1, num_2)
    print(f"{num_1} + {num_2} = ", result)
elif operation == "/":
    divide(num_1, num_2)
else:
    print("Invalid operation")
    # Execute help function
    my_help()
