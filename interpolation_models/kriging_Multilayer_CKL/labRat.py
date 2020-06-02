# from itertools import combinations_with_replacement

# comb = combinations_with_replacement([1,2,3],3)

# for i in list(comb):
#     print(i)

import numpy as np
import itertools
import re
import string


def perm(n, seq):
    sequence = []
    for p in itertools.product(seq, repeat=n):
        #print("".join(p))
        y = ",".join(p)
        sequence.append(y)
    return sequence

perm(3,"123")

k = 3
kernels = [r"'gaussian'",r"'matern3_2'",r"'matern5_2'"]
kernel = perm(k,kernels)
thisdict = {}
for i in range(len(kernel)):
    test_string = kernel[i]
    #wordList = re.sub("[^\w]", " ",  test_string).split()
    thisdict[test_string] = 0.5

print(thisdict)




#print(wordList)

# resultDict0 = {
#     "RMSE": 0.4,
#     "theta": [0.1,0.2]
# }

# resultDict1 = {
#     "RMSE": 0.5,
#     "theta": [0.6,0.9,0.7]
# }

# thisdict =	{
#   "GMM": resultDict0,
#   "MGM": resultDict1
# }
# print(thisdict["GMM"]["RMSE"])

