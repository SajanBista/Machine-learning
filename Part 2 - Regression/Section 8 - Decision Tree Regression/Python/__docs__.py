# def name(x):
#     """this function is saying it willl return name"""
#     return x

# print(name.__doc__)


# generating the list of

def is_even(numbers):
    is_even_list = []
    for number in numbers:
        if number%2 ==0:
            is_even_list.append(number)
    return is_even_list

numbers =[1,2,3,4,5,6,7,8,9]

even_numbers = is_even(numbers)
print(even_numbers)

