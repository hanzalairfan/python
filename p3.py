n = (1,2,3,4,5)
print(max(n))
print(min(n))
print(len(n))

num = int(input("enter a number: "))
if num in n:
        print("it is in tuple")
else:
        print("it is not available")
        
lst = list(n)
print(lst)
tpl = tuple(lst)
print(tpl)

m = (1,3,5)
sum = n + m
total = 0
for i in sum:
    total += i 
    
print(total)