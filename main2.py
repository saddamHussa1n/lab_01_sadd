import random
from typing import List

import numpy as np
import math


def bezout_recursive(a, b):
    if not b:
        return (1, 0, a)
    y, x, g = bezout_recursive(b, a % b)
    return (x, y - (a // b) * x, g)


class Vector(list):
    def sdot(self):
        return np.dot(self, self)

    def proj_coff(self, rhs):
        rhs = Vector(rhs)
        assert len(self) == len(rhs)
        return np.dot(self, rhs) / self.sdot()

    def proj(self, rhs):
        rhs = Vector(rhs)
        assert len(self) == len(rhs)
        return self.proj_coff(rhs) * self

    def __sub__(self, rhs):
        rhs = Vector(rhs)
        assert len(self) == len(rhs)
        return Vector(x - y for x, y in zip(self, rhs))

    def __mul__(self, rhs):
        return Vector(x * rhs for x in self)

    def __rmul__(self, lhs):
        return Vector(x * lhs for x in self)

    # def __repr__(self) -> str:
    #     return "[{}]".format(", ".join(str(x) for x in self))


def gramschmidt(v):
    u: List[Vector] = []
    for vi in v:
        ui = Vector(vi)
        for uj in u:
            ui = ui - uj.proj(vi)

        if any(ui):
            u.append(ui)
    return u


def lll(basis, delta):
    n = len(basis)
    basis = list(map(Vector, basis))
    ortho = gramschmidt(basis)

    def mu(i: int, j: int):
        return ortho[j].proj_coff(basis[i])

    k = 1
    while k < n:
        for j in range(k - 1, -1, -1):
            mu_kj = mu(k, j)
            if abs(mu_kj) > 0.5:
                basis[k] = basis[k] - basis[j] * round(mu_kj)
                ortho = gramschmidt(basis)

        if ortho[k].sdot() >= (delta - mu(k, k - 1) ** 2) * ortho[k - 1].sdot():
            k += 1
        else:
            basis[k], basis[k - 1] = basis[k - 1], basis[k]
            ortho = gramschmidt(basis)
            k = max(k - 1, 1)

    return [list(map(int, b)) for b in basis]


f1 = open('poly-3-9.dat')
m = int(f1.readline())
num_p = int(f1.readline())
K = 15000000

blocks = []
for line in f1:
    blocks.append(line[1:-2])
f1.close()

blocks = [x.split(' ') for x in blocks if len(x.split(' ')) == m]

i = 0
for item in blocks:
    blocks[i] = [int(i) for i in item]
    i += 1

i = 0
j = m
det = []
for i1 in range(int(len(blocks) / m)):
    lyambda = []
    step = len(blocks[i])
    for l in range(0, step):
        lyambda.append([blocks[k][l] for k in range(i, j)])
    lyambda = np.array(lyambda)
    i += step
    j += step
    det.append(int(np.linalg.det(lyambda)))
    M = [lyambda[0], lyambda[1]]
    for j1 in range(2, len(lyambda)):
        M.append(lyambda[j1] * K)
    M = np.array(M)
    M_prime = lll(M, 0.75)
    M_prime = np.array(M_prime)
    for i12 in M_prime:
        for j12 in i12:
            print(j12, end='\t')
        print('')

gcd = math.gcd(det[random.randint(0, int(len(blocks) / m)-1)], det[random.randint(0, int(len(blocks) / m)-1)])
gcd = math.gcd(gcd, det[random.randint(0, int(len(blocks) / m))])
f = open('answer.txt', 'w')
f.write('M ' + str(gcd) + '\n')

gcd2 = math.gcd(5, 1)
gcd2 = math.gcd(gcd2, gcd)
if gcd2 != 1:
    f.write('A МНОЖИТЕЛЬ НЕ НАЙДЕН' + '\n')
else:
    x, y, g = bezout_recursive(5, 1)
    # print((x, y, g))
    f.write('A ' + str((-x * (4) - y * (1)) % gcd) + '\n')
