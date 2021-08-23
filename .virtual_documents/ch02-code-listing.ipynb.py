import platform
platform.python_version()


import numpy as np


data = np.array([[1, 2], [3, 4], [5, 6]])


type(data)


data


data.ndim


data.shape


data.size


data.dtype


data.nbytes


np.array([1, 2, 3], dtype=int)


np.array([1, 2, 3], dtype=float)


np.array([1, 2, 3], dtype=complex)


data = np.array([1, 2, 3], dtype=float)


data


data.dtype


data = np.array([1, 2, 3], dtype=np.int)


data.dtype


data


data = np.array([1, 2, 3], dtype=np.float)


data


data.astype(np.int)


d1 = np.array([1, 2, 3], dtype=float)


d2 = np.array([1, 2, 3], dtype=complex)


d1 + d2


(d1 + d2).dtype


np.sqrt(np.array([-1, 0, 1]))


np.sqrt(np.array([-1, 0, 1], dtype=complex))


data = np.array([1, 2, 3], dtype=complex)


data


data.real


data.imag


np.array([1, 2, 3, 4]).strides


data.ndim


data.shape


np.array([[1, 2], [3, 4]])


data.ndim


data.shape


np.zeros((2, 3))


np.ones(4)


data = np.ones(4)


data.dtype


data = np.ones(4, dtype=np.int64)


data.dtype


x1 = 5.4 * np.ones(10)


x2 = np.full(10, 5.4)


x1 = np.empty(5)


x1.fill(3.0)


x1


x2 = np.full(5, 3.0)


x2


np.arange(0.0, 10, 1)


np.linspace(0, 10, 11)


np.logspace(0, 2, 5)  # 5 data points between 10**0=1 to 10**2=100


x = np.array([-1, 0, 1])


y = np.array([-2, 0, 2])


X, Y = np.meshgrid(x, y)


X


Y


Z = (X + Y) ** 2


Z


np.empty(5, dtype=float)


def f(x):
    y = np.ones_like(x)
    # compute with x and y
    return y

x = np.array([2,3,4,4,4])
print(f(x))


np.identity(4)


np.eye(3, k=1)


np.eye(3, k=-1)


np.diag(np.arange(1, 20, 4))
# el tamaño es de acuerdo al de la matriz arange


a = np.arange(0, 11)


a


a[0]  # the first element


a[-1] # the last element


a[4]  # the fifth element, at index 4


a[1:-1]


a[1:-1:2]


a[:5]


a[-5:]


a[::-2]



f = lambda m, n: n + 10 * m
# toma dos valores m, n
f(3,6)


A = np.fromfunction(f, (3,6), dtype=float)

A


A[:, 1]  # the second column


A[1, :]  # the second row


A[:2, :]  # upper half diagonal block matrix


A[2:3, :3]  # lower left off-diagonal block matrix


A[::2, ::2]  # every second element starting from 0, 0


A[1::1, 1::2]  # every second element starting from 1, 1


B = A[1:5, 1:5]


B


B[:, :] = 0


A


C = B[1:3, 1:3].copy()


C


C[:, :] = 1  # this does not affect B since C is a copy of the view B[1:3, 1:3]


C


B


A = np.linspace(0, 1, 11)


A[np.array([0, 2, 4])]


A[[0, 2, 4]]


A > 0.5 


A[A > 0.5]


A = np.arange(10)


A


indices = [2, 4, 6]


B = A[indices]


B[0] = -1  # this does not affect A


A


A[indices] = -1


A


A = np.arange(10)
A


B = A[A > 5]


B[0] = -1  # this does not affect A


A


A[A > 5] = -1


A


import numpy as np
data = np.array([[1, 2], [3, 4]])


np.reshape(data, (1, 4))


data.reshape(4)


data = np.array([[1, 2], [3, 4]])


data


data.flatten()


data.flatten().shape


data = np.arange(0, 5)


column = data[:, np.newaxis]


column


row = data[np.newaxis, :]


row


data = np.arange(5)


data


np.vstack((data, data, data))


data = np.arange(5)


data


np.hstack((data, data, data))


data = data[:, np.newaxis]


np.hstack((data, data, data))


x = np.array([[1, 2], [3, 4]]) 


y = np.array([[5, 6], [7, 8]])


x + y


y - x


x * y


y / x


x * 2


2 ** x


y / 2


(y / 2).dtype


x = np.array([1, 2, 3, 4]).reshape(2,2)


z = np.array([1, 2, 3, 4])


x / z


z = np.array([[2.3, 4]])


z


z.shape


x / z


zz = np.concatenate([z, z], axis=0)


zz


x / zz


z = np.array([[2], [4]]) # 2  rows and 1 column


z.shape


x / z


zz = np.concatenate([z, z], axis=1)
zz


x / zz


x = np.array([[1, 3], [2, 4]])
x = x + y
x


x = np.array([[1, 3], [2, 4]])
x += y
x


x = np.linspace(-1, 1, 11)
x


y = np.sin(np.pi * x)


np.round(y, decimals=4)


np.add(np.sin(x) ** 3, np.cos(x) ** 3)


np.sin(x) ** 3 + np.cos(x) ** 3


def heaviside(x):
    return 1 if x > 0 else 0


heaviside(-1)


heaviside(1.5)


x = np.linspace(-5, 5, 11)


heaviside(x)


heaviside = np.vectorize(heaviside) # es mas conveniente usar vectorize, si la funcion es lenta usar cache =True


heaviside(x)


def heaviside(x):
    return 1.0 * (x > 0)


data = np.random.normal(size=(15,15))
data


np.mean(data)


data.mean()


data = np.random.normal(size=(5, 10, 5))
data


data.sum(axis=0).shape


data.sum(axis=(1)).shape


data.sum()


data = np.arange(1,10).reshape(3,3)
data


data.sum()


data.sum(axis=0)


data.sum(axis=1)


a = np.array([[1, 2, 3, 4],[2,4,5,2]])
a


b = np.array([4, 3, 2, 1])


a < b


np.all(a < b)


np.any(a < b)


if np.all(a < b):
    print("All elements in a are smaller than their corresponding element in b")
elif np.any(a < b):
    print("Some elements in a are smaller than their corresponding elemment in b")
else:
    print("All elements in b are smaller than their corresponding element in a")


x = np.array([-2, -1, 0, 1, 2,-2])


x > 0


1 * (x > 0)


x * (x > 0)


def pulse(x, position, height, width):
    return height * (x >= position) * (x <= (position + width))


x = np.linspace(-5, 5, 21)
x


pulse(x, position=-2, height=1, width=5)


pulse(x, position=1, height=1, width=5)


def pulse(x, position, height, width):
    return height * np.logical_and(x >= position, x <= (position + width))


x = np.linspace(-4, 4, 19)
x


np.where(x < 0, x**2, x**3)


x = np.linspace(-4,4,9)
print(x)
np.select([x < -1, x < 2, x >= 2],
          [x**2  , x**3 , x**4])


np.choose([0, 0, 0, 0, 0, 1, 1, 2, 2], 
          [x**2,    x**3,    x**4])


x[abs(x) > 2]


x[np.nonzero(abs(x) > 2)]


[np.nonzero(abs(x) > 2)]


import numpy as np
a = np.unique([1,2,3,3])
a



b = np.unique([2,3,4,4,5,6,5])
b


np.in1d(a, b) # if a is in b


1 in a


1 in b


np.all(np.in1d(a, b))


np.union1d(a, b)


np.intersect1d(a, b)


np.setdiff1d(a, b)


np.setdiff1d(b, a)


data = np.arange(9).reshape(3, 3)


data


np.transpose(data)
np.rot90(data)


data = np.random.randn(1, 2, 3, 4, 5)


data.shape


data.T.shape


A = np.arange(1, 7).reshape(2, 3)


A


B = np.arange(1, 7).reshape(3, 2)


B


np.dot(A, B)


B@A


A = np.arange(9).reshape(3, 3)


A


x = np.arange(3)


x


np.dot(A, x)


A.dot(x)


A = np.random.rand(3,3)
B = np.random.rand(3,3)


Ap = B @ A @ np.linalg.inv(B)
Ap


Ap = np.dot(B, np.dot(A, np.linalg.inv(B)))
Ap


Ap = B.dot(A.dot(np.linalg.inv(B)))
Ap


A = np.matrix(A)
B = np.matrix(B)


Ap = B * A * B.I
Ap


A = np.asmatrix(A)


B = np.asmatrix(B)
B


Ap = B * A * B.I
Ap


Ap = np.asarray(Ap)


Ap


np.inner(x, x)


np.dot(x, x)


y = x[:, np.newaxis]


y


np.dot(y.T, y)


x = np.array([1, 2, 3])


np.outer(A, B) 


# np.kron(x, x) 
np.kron(A,B)


np.kron(x[:, np.newaxis], x[np.newaxis, :])


np.kron(np.ones((2,2)), np.identity(2))


np.kron(np.identity(2), np.ones((2,2)))


x = np.array([1, 2, 3, 4])


y = np.array([5, 6, 7, 8])


np.einsum("n,n", x, y)


np.inner(x, y)


A = np.arange(9).reshape(3, 3)


B = A.T


np.inner(A,B)


np.einsum("mk,kn", A, B)


np.alltrue(np.einsum("mk,kn", A, B) == np.dot(A, B))


get_ipython().run_line_magic("reload_ext", " version_information")
get_ipython().run_line_magic("version_information", " numpy")
