n1 = int(input("enter a number: "))
n2 = int(input("enter a number: "))
n3 = int(input("enter a number: "))
n4 = int(input("enter a number: "))
n5 = int(input("enter a number: "))
n6 = int(input("enter a number: "))
n7 = int(input("enter a number: "))
n8 = int(input("enter a number: "))
n9 = int(input("enter a number: "))
n10 = int(input("enter a number: "))

s = set()
s.add(n1)
s.add(n2)
s.add(n3)
s.add(n4)
s.add(n5)
s.add(n6)
s.add(n7)
s.add(n8)
s.add(n9)
s.add(n10)
print(s)

s2 = {1,3,5,7,9}
print(s2)

for i in s:
    if i in s2:
        print(i)
        
        
diff = s - s2
print(diff)
union = s.union(s2)
print(union)
if s.issubset(s2):
        print("s is subset of s2")
else:
        print("it is not")