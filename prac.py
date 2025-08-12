# a = int(input("enter a number: "))
# b = int(input("enter a number: "))

# while b != 0:
#     temp = b
#     b = a % b
#     a = temp
# if a == 1:
#         print("co prime")
# else:
#         print("not co prime")
# start = int(input("enter a number: "))
# end = int(input("enter a number: "))
# sum = 0

# for n in range(start, end + 1):
#     if (n ** 0.5) ** 2 == n:
#         print(n,"is a perfect square")
#     else:
#         print("it is not")

# num = int(input("enter a number: "))
# strnum = str(num)
# reversed = str(strnum[::-1])

# if reversed == strnum:
#     print("palindrome")
# else:
#     print("not a palindrome")

# start = 100
# end = 999

# for n in range(start, end + 1):
#     sum = 0
#     no_of_digits = len(str(n))
#     temp = n
#     original = n
#     while temp > 0:
#         digit = temp % 10
#         sum += digit ** no_of_digits
#         temp = temp // 10
    
#     if sum == original:
#         print(original,"armstrong numbers")
    
# num = int(input("enter a number: "))
# n = num - 1
# u = n

# for n in range(num ):
#     original = num
#     no_of_digits = len(str(n))
#     while original > 0:
#         digit = original % 10

# def factorial(n):
#     if n < 0:
#         print("cannot be factorial of negative numbers")
#     result = 1
#     for i in range(1, n + 1):
#         result *= i
#     return result

# num = int(input("enter a number: "))
# strnum = str(num)
# original = num
# sum = 0


# while num > 0:
#     digit = num % 10
#     num = num // 10
#     sum += digit
# print(sum)

# if original % sum == 0:
#     print("harshad number")
# else:
#     print("it is not a harshad number")

# num = int(input("enter a number: "))
# strnum = str(num)
# original = num
# sum = 0
# s = len(str(sum))


# while num > 0:
#     digit = num % 10
#     num = num // 10
#     sum += digit
#     print(sum)
#     if sum <= 9:
#         digit = num % 10
#     num = num // 10
#     sum += digit
# print(sum)
    