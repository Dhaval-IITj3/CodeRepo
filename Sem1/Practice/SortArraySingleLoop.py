# Program to sort an array in single Loop


def sort_array(arr):
    length = len(arr)

    i = 0
    while i < length-1:
        # Is current number greater than next adjacent number
        if ( arr[i] > arr[i+1]):
            # if Yes, Swap the number
            temp = arr[i]
            arr[i] = arr[i+1]
            arr[i+1] = temp

            # i++ will make i=0, thus loop begins from start
            i = -1

        i += 1

    return arr

numarr = [4,6,2,5,7]

print(sort_array(numarr))

