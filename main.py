#!/usr/bin/env python3
import math
from functools import lru_cache


def assert_divides(a, b):
    assert a % b == 0, f"Assertion failed: {a} is not divisible by {b}"
    return a // b


@lru_cache
def compute_q(m, v, l, k, t):
    if k == 0 and t == 0:
        return 1
    if k < 0 or t < 0:
        return 0
    if k == 0:
        return 0
    term1 = (2 * k - 1 + l - t) * compute_q(m, v, l, k - 1, t - 1)
    term2 = (t + 1) * (m - 1) * compute_q(m, v, l, k - 1, t + 1)
    term3 = (m * (v + k) + l * (m - 1) - (2 * k - 2 - t) - t * (m - 1)) * compute_q(m, v, l, k - 1, t)
    result = term1 + term2 + term3
    assert result >= 0, (m, v, l, k, t, result)
    return term1 + term2 + term3


@lru_cache
def compute_p_hat(m, v, l, k):
    if v < 0 or l < 0:
        return 0

    result = 0
    for i in range(k + 1):
        mul = compute_q(m, v, l, k - i, 0)
        term = ((-1) ** i) * assert_divides(math.factorial(k), math.factorial(k - i)) * mul
        result += term

    return result


@lru_cache
def compute_c(m, d, v, k, t):
    result = 0
    for i in range(max(0, k - t), (d - t + k - 1) // 2 + 1):
        v_hat = d - 2 * i - t + k - 1
        mul = compute_p_hat(m, v - d - t, t + i - k, v_hat)
        term = math.comb(d - 1, i) * math.comb(t, t + i - k) * (m ** (t + i - k)) * mul
        result += assert_divides(term, math.factorial(v_hat))
    return result


@lru_cache
def compute_a(m, d, v, k):
    if v == 0 and k == 0:
        return 1
    if v <= 0 or k < 0:
        return 0

    result = 0
    gcd_md = math.gcd(m, d)
    for l in range(1, gcd_md + 1):
        if gcd_md % l == 0:
            m_l = assert_divides(m, l)
            d_l = assert_divides(d, l)
            for t in range(max(0, k - d_l + 1), k + d_l):
                mul1 = compute_c(m_l, d_l, v, k, t)
                mul2 = compute_a(m, d, v - d_l, t)
                result += mul1 * mul2

    return result


@lru_cache
def compute_f(d, n, m):
    v = assert_divides(d * n, m)
    result = compute_a(m, d, v, 0)
    gcd_md = math.gcd(m, d)

    for l in range(1, gcd_md + 1):
        if gcd_md % l == 0:
            m_l = assert_divides(m, l)
            d_l = assert_divides(d, l)
            for k in range(min(d_l - 2, v - d_l) + 1):
                d_l_2_k = d_l - 2 - k
                term = (m_l ** k) * compute_r_hat(m_l, v - d_l - k, k, d_l_2_k) * compute_a(m, d, v - d_l, k)
                result -= assert_divides(term, math.factorial(d_l_2_k))

    return result


@lru_cache
def compute_r_hat(m, v, l, k):
    result = 0
    for i in range(k + 1):
        for j in range(k - i + 1):
            mul1 = (-1) ** (i + j)
            mul2 = assert_divides(math.factorial(k), math.factorial(k - i - j))
            mul3 = compute_q(m, v, l, k - i - j, 0)
            result += mul1 * mul2 * mul3
    return result


@lru_cache
def euler_totient(n):
    result = n
    for p in range(2, int(math.sqrt(n)) + 1):
        if n % p == 0:
            while n % p == 0:
                n //= p
            result -= result // p
    if n > 1:
        result -= result // n
    return result


@lru_cache
def compute_b_tilde(d, n):
    result = 0
    for m in range(1, d * n + 1):
        if (d * n) % m == 0:
            result += euler_totient(m) * compute_f(d, n, m)
    return assert_divides(result, d * n)


def compute_alpha_1(n, d, k):
    return math.comb(assert_divides(d * (n - 1), 2) - 1 - k, assert_divides(d, 2) - 1 - k)


def compute_alpha_2(n, d, k):
    result = 0
    half_d = assert_divides(d, 2)
    half_d_n_minus_2 = assert_divides(d * (n - 2), 2)
    for j in range(half_d):
        for s in range(max(0, k + j - half_d + 1), min(k, half_d - 1 - j) + 1):
            mul1 = math.comb(half_d - 1, j)
            mul2 = math.comb(k, s)
            mul3 = math.comb(assert_divides(d * (n - 2), 2) - k, half_d - 1 - j - s)
            mul4 = math.comb(assert_divides(d * (n - 1), 2) - 1 - (k + j - s), half_d_n_minus_2)
            result += mul1 * mul2 * mul3 * mul4
    return result


def compute_alpha_3(n, d, k):
    return math.comb(assert_divides(d * (n - 1), 2) - 1 - k, assert_divides(d, 2) - 2 - k)


def compute_h_0(n, d):
    if d % 2 == 1:
        return compute_a(2, d, assert_divides(d * n, 2), 0) if n % 2 == 0 else 0
    else:
        result = compute_a(2, d, assert_divides(d * n, 2), 0)
        for k in range(assert_divides(d, 2)):
            result -= 2 * compute_alpha_1(n, d, k) * compute_a(2, d, assert_divides(d * (n - 1), 2), k)
        for k in range(min(d - 2, assert_divides(d * (n - 2), 2)) + 1):
            result += compute_alpha_2(n, d, k) * compute_a(2, d, assert_divides(d * (n - 2), 2), k)
        for k in range(assert_divides(d, 2) - 1):
            result -= compute_alpha_3(n, d, k) * compute_a(2, d, assert_divides(d * (n - 1), 2), k)
        return result


def compute_h_1(n, d):
    if d % 2 == 0 or n % 2 == 0:
        return 0
    result = 0
    half_d_minus_1 = assert_divides(d - 1, 2)
    for k in range(half_d_minus_1 + 1):
        mul1 = math.comb(assert_divides(d * (n - 1), 2) - 1 - k, half_d_minus_1 - k)
        mul2 = compute_a(2, d, assert_divides(d * (n - 1), 2), k)
        result += mul1 * mul2
    return result


def compute_h_2(n, d):
    result = 0
    if d % 2 == 0:
        half_d = assert_divides(d, 2)
        for k in range(half_d):
            mul1 = compute_alpha_1(n, d, k)
            mul2 = compute_a(2, d, assert_divides(d * (n - 1), 2), k)
            result += mul1 * mul2
        return result
    elif n == 2:
        result = 1
    elif n % 2 == 0:
        half_d_minus_1 = assert_divides(d - 1, 2)
        half_d_n_minus_2 = assert_divides(d * (n - 2), 2)
        for k in range(min(d - 1, half_d_n_minus_2) + 1):
            for j in range(half_d_minus_1 + 1):
                for s in range(max(0, k + j - half_d_minus_1), min(k, half_d_minus_1 - j) + 1):
                    mul1 = math.comb(half_d_minus_1, j)
                    mul2 = math.comb(k, s)
                    mul3 = math.comb(half_d_n_minus_2 - k, half_d_minus_1 - j - s)
                    mul4 = math.comb(assert_divides(d * (n - 1) - 1, 2) - (k + j - s), half_d_n_minus_2)
                    mul5 = compute_a(2, d, half_d_n_minus_2, k)
                    result += mul1 * mul2 * mul3 * mul4 * mul5
    return result


def compute_b_bar(d, n):
    result = 0
    for m in range(1, d * n + 1):
        if (d * n) % m == 0:
            result += euler_totient(m) * compute_f(d, n, m)
    result = assert_divides(result, d * n)

    return assert_divides(2 * result + compute_h_0(n, d) + 2 * compute_h_1(n, d) + compute_h_2(n, d), 4)


MAXD = 9
MAXN = 30
MAXX = 1e29

for d in range(2, MAXD + 1):
    print(f'd = {d}')
    row = ['Linear', 'Chord labelled', 'Unlabelled (cyclic)', 'Unlabelled (dihedral)']
    print(f'{"n":>2}' + ' '.join(f'{row[i]:>30}' for i in range(4)))
    for n in range(2, MAXN + 1):
        row = [compute_a(1, d, d * n, 0), compute_f(d, n, 1), compute_b_tilde(d, n), compute_b_bar(d, n)]
        if any(x > MAXX for x in row):
            break
        print(f'{n:2}' + ' '.join(f'{val:30}' for val in row))
