
import math
# Initialization
n = int(input("Enter the number of points: "))
x = []
y = []

for i in range(n):
    xi = int(input(f"enter the value for x{i+1}: "))
    yi= int(input(f'enter the value for y{i+1}: '))
    x.append(xi)
    y.append(yi)


# Summation variables
sx = sy = sxy = sx2 = slgy = 0

# Calculations
for i in range(n):
    sx += x[i]
    slgy += (y[i])
    sxy += x[i] * (y[i])
    sx2 += x[i] ** 2

# Solve for b and a
b = ((n * sxy) - (sx * slgy)) / ((n * sx2) - (sx ** 2))
a = (slgy / n) - (b * (sx / n))
# a = math.exp(a)

print(f"Fitted curve is: y = {a}x+{b}")
