# first_name = input("Enter first name")
# first_lowercase = first_name.lower()
#
# if first_lowercase == "Python 3".lower():
#   print("You are using Python 3")
# elif first_lowercase == "Python 2".lower():
#   print("You are using Python 2")
#
#
# print("Bubye")



# rows = int(input("Enter number of rows: "))
#
# rows = rows // 2 + 1
#
# star = "*"
# for i in range(1,rows+1):
#     print(star * i)
#
# for i in range(rows-1,0, -1):
#     print(star * i)

name = "D"
#
# while name:
#   print(name)

while True:
  print("Hello")

i = 0
j = 0

while i < 5:
  print("value of i in outer loop", i)

  while j < 3:
    print("value of j in inner loop", j)
    j += 1

  print("In outer loop")
  i += 1

