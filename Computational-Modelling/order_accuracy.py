import math
f1 = 8.056750154815475
f2 =8.056879331423856
f3 =8.056902948766579

oa = math.log((f3-f2)/(f2-f1))/math.log(2)
print(oa)

error = (f1-f2)/(2**oa-1)
print(error)