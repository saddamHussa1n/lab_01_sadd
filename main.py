import sys

import numpy as np
from typing import List


def generate_vectors(m, n):
    if m > n:
        print('bb')
        sys.exit()
    A = np.random.randint(low=0, high=18446744073709551615, size=(m, n), dtype='uint64')  # 18446744073709551615
    rankA = np.linalg.matrix_rank(A)
    if rankA == m:
        print('Матрица А:')
        print(A)
        return A


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


def check_lll(basis, delta):
    n = len(basis)
    basis = list(map(Vector, basis))
    ortho = gramschmidt(basis)

    def mu(i: int, j: int):
        return ortho[j].proj_coff(basis[i])

    k = 1
    bbb = 0
    while k < n:
        for j in range(k - 1, -1, -1):
            mu_kj = mu(k, j)
            if abs(mu_kj) > 0.5:
                basis[k] = basis[k] - basis[j] * round(mu_kj)
                ortho = gramschmidt(basis)
        if ortho[k].sdot() >= (delta - mu(k, k - 1) ** 2) * ortho[k - 1].sdot():
            print((delta - mu(k, k - 1) ** 2) * ortho[k - 1].sdot(), '<', ortho[k].sdot())
            k += 1
        else:
            bbb = 1
            print((delta - mu(k, k - 1) ** 2) * ortho[k - 1].sdot(), '>', ortho[k].sdot())
            print('Its not LLL')
            break
    if bbb == 0:
        print('Its LLL')


# A = generate_vectors(5, 6)
# reduced_basis = lll(A, 0.75)
# reduced_basis = np.array(reduced_basis)
# print('\nМатрица LLL(δ = 0.75):')
# print(reduced_basis)
# print('Проверка матрицы А на LLL-приведенность:')
# check_lll(A, 0.75)
# print('Проверка матрицы LLL на LLL-приведенность:')
# check_lll(reduced_basis, 0.75)
# # print(lll([[2, 2, 3, 1], [7, 7, 10, 3], [11, 10, 14, 4]], 0.75))
