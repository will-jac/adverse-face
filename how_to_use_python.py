# int doSomething(int arg1, int arg2) {
#     ...
#     return(1);
# }

def do_something(arg1=1, arg2=2):
    return 1

do_something()
do_something(5)
do_something(5, 6)
do_something(arg2=6)

# for (int i = 0, i < 10, i++) {
#     printf(i);
# }

for i in range(10):
    print(i)

print(list(range(-10, 10, 2)))

a = 1
b = 2

if a == 5:
    print('a is 5')
elif b == 5:
    print('b is 5')
else:
    print('neither was 5 :(')

# public class A {
#     int val;
#     void A() {
#         this.val = 0;
#         return;
#     }
# }

class A():
    def __init__(self):
        self.val = 0
        return

b = list() 
b = []
b = [1,2,3]
b.append(4)

b = [i for i in range(1,5)] # [1,2,3,4]
b = []
for i in range(1,5):
    b.append(i)
# b = [1,2,3,4]

c = set()
# unordered, unique collection
c = set([1,2,2,3]) # {1,2,3}
c = {1,2,2,3} # {1,2,3}
c.add(4) # {1,2,3,4}

d = dict()
d = {'a':1, "b":2}
print(d['a'])

# in operator

# : operator (slicing)
import numpy as np
b = np.array([i for i in range(10)])
print(b)
print(slice(2,5))
print(b[2:5])
print(b[-1])
print(b[0:-2])


def difference(original_image, attack_image):
    # compute metric for difference
    score = 0
    return score


