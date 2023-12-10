class A:
    def __init__(self,b):
        self.b = b

class B:
    def __init__(self,value):
        self.value = value

instance_b = B(10)
instance_a = A(instance_b)

instance_b.value = 42

print(instance_a.b.value)

instance_a.b.value = 67

print(instance_b.value)

def dothings(b,newval):
    b.value = newval

def ehhh(b):
    x = b
    x.value = 10

dothings(instance_b,25)

print(instance_a.b.value)

ehhh(instance_b)

print(instance_a.b.value)

print('' == None)