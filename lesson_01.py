import numpy as np
from scipy import stats

class_a = np.array([55, 60, 65, 70, 75])
class_b = np.array([35, 40, 45, 50, 55])

print("Class A mean:", np.mean(class_a))
print("Class B mean:", np.mean(class_b))

print("Class A population std:", np.std(class_a))
print("Class B population std:", np.std(class_b))

print("Class A sample std:", stats.tstd(class_a))
print("Class B sample std:", stats.tstd(class_b))