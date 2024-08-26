# Code of developer who has probable chance of Clearing Google Interview
rows = int(input("Enter the number of rows: "))

print(*[x * "*" for x in range(1, (rows//2)+1)], sep="\n")
print(*[x * "*" for x in range((rows//2)-1, 0, -1)], sep="\n")


# Versus my code; Me who thinks, "But my code printed the intended output".
print("*")
print("**")
print("***")
print("****")
print("*****")
print("****")
print("***")
print("**")
print("*")