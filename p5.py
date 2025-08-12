name= input("enter your name: ")
age = int(input("enter age: "))
marks = float(input("enter your marks: "))

dict = {name, age, marks}
print(dict)

dict2 = {
    "max": 50.00,
    "phil":25.25,
    "kevin":35.55,
    "ben":55.00,
    "ethan":47.00
    
}
print(dict2)
freq = {}
for char in dict2:
    if char in freq:
        freq[char] += 1
    else:
        freq[char] = 1
print(freq)
dict2.values()
SUM = sum(dict2.values())/(len(dict2))
print(SUM)

listofno = [1,2,3,4]
dict3 = {
    "1": 1,
    "2": 4,
    "3": 9,
    "4": 16
}
print(dict3)

combined = dict2 | dict3
print(combined)

