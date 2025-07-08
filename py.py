number = int(input('enter a number:'))
if number % 2 == 0:
    print(number, 'is even')
else:
    print(number, 'is odd')
fact = 1
for i in range(1,number+1):
    fact = fact *i
print(fact)
is_prime = True

if number <= 1:
    is_prime = False
else:
    for i in range(2, number):
        if number % i == 0:
            is_prime = False
            break

if is_prime:
    print("Prime")
else:
    print("Not Prime")
