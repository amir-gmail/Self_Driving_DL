from numpy import shape
import numpy as np
a = np.arange(0,9)
#a.shape = (3,3)
print(a)
b=range(0,11)
print(np.log(np.exp([1])))
print(list(b))
list1 = list(range(10))
list2 = list1
print (len(list2))
def my_f(a):
    return True

list3 = list(filter(my_f,list2))
print(len(list3))