# num = int(input("enter a number: "))
# if(num <= 1):
#     print("it is not prime")
# for i in range(num):
#     if(num % i == 0):
#         print("not prime")
#     else:
#         print("prime")
# word = input("enter a word")
# reversed_word = word[::-1]
# if(reversed_word == word):
#     print("it is a palindrome")
# else:
#     print("not a palindrome")
    
num = int(input("enter a number: "))
if(num <= 1):
    print("it is not prime")
for i in range(2,num):
    if num % i == 0:
    print("not prime")
    break
    is_prime = True
    