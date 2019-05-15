import numpy as np
import math

def sigmoid(x: np.array):
    result = np.array(x)
    for i in range(len(result)):
        result[i] = 1 / (1 + math.exp(-result[i]))
    return result

print(sigmoid(np.array([1,2,300])))