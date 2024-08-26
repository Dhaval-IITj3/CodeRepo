import sys


x = input("Enter a number")

if x.isdigit():
  x = int(x)
else:
  print("x is not a number")
  sys.exit()

if x > 0:
  print("X is a positive number")
elif x == 0:
  print("X is neither negative nor positive")
else:
  print("X is negative number")

