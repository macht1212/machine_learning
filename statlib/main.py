from math_ import Agg
import random as ra

a = Agg()

data = [ra.randint(1, 10000) for _ in range(20000)]
print(data)


print(a.min(data))
print(a.max(data))
print(a.sum(data))
print(a.avg(data))