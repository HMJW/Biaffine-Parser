import numpy as np
cimport numpy as np
from numpy cimport ndarray
cimport cython
from numpy import random
import math
import datetime
import sys

cdef extern from "math.h":
    cpdef float logf(float x);
    cpdef float expf(float x);
    cpdef double log(double x);
    cpdef double exp(double x);

ctypedef np.float64_t DTYPE_t
ctypedef np.float32_t DTYPE_t32

cdef DTYPE_t NEG_INF = -1e21
cdef DTYPE_t NEG_INF_ADD_EPS = NEG_INF + 1e20

data_type = np.float64
data_type32 = np.float32
data_type_int32 = np.int32

cdef int CMP=0
cdef int INCMP=1

def log_add_if_not_inf(DTYPE_t value, DTYPE_t a):
    if value > NEG_INF_ADD_EPS:
        return log_add_another(value, a)
    else:
        return a

def log_add_another(DTYPE_t a, DTYPE_t b):
    if (a > b):
        '''
        all math.log/exp log/exp logf/expf are ok, but the results are slightly different due to different implemantations
        logf is twice fast than math.log
        log is about 15% slower than logf
        Strictly speaking, I should use log/exp since DTYPE_t is float32 (double)
        '''
        # return a + math.log(1 + math.exp(b - a))
        # return a + logf(1 + expf(b - a))
        return a + log(1 + exp(b - a))
    else:
        # return b + math.log(1 + math.exp(a - b))
        # return b + logf(1 + expf(a - b))
        return b + log(1 + exp(a - b))


@cython.boundscheck(False)
def marginal_prob(DTYPE_t logz, np.ndarray[DTYPE_t, ndim=2] i_incmp, np.ndarray[DTYPE_t, ndim=2] o_incmp, int h, int m):
    if h == m:
        return 0.
    assert logz > NEG_INF_ADD_EPS
    cdef DTYPE_t a = i_incmp[h, m]
    cdef DTYPE_t b = o_incmp[h, m]
    cdef DTYPE_t p = 0.
    # assert a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS
    if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
        # p = math.exp(a + b - logz)
        # p = expf(a + b - logz)
        p = exp(a + b - logz)
        if not (-1e-5 <= p <= 1.0 + 1e-5):
            print("\nprob cython = ", p, " h = ", h, " m = ", m) 
    return p

@cython.boundscheck(False)
def marginal_prob_first_child_labeled(DTYPE_t logz, np.ndarray[DTYPE_t32, ndim=3] scores, np.ndarray[DTYPE_t, ndim=2] i_cmp, np.ndarray[DTYPE_t, ndim=3] o_incmp, int h, int m, int L):
    if h == m:
        return 0.
    assert logz > NEG_INF_ADD_EPS
    # C(m -> h-1) + C(h) = I(m <- h)
    # C(h) + C(h+1 <- m) = I(h <- m)
    cdef int h1 = (h+1 if m > h else h-1)
    cdef DTYPE_t a = i_cmp[m, h1]
    cdef DTYPE_t b = i_cmp[h, h]
    cdef DTYPE_t c = o_incmp[h, m, L]
    cdef DTYPE_t p = 0.
    # assert a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS
    if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS and c > NEG_INF_ADD_EPS:
        # p = math.exp(a + b - logz)
        # p = expf(a + b - logz)
        p = exp(a + b + c + scores[m, h, L+2] - logz)
        if not (-1e-5 <= p <= 1.0 + 1e-5):
            print("\nprob cython (first-child labeled) = ", p, " h = ", h, " m = ", m)
    return p


@cython.boundscheck(False)
def marginal_prob_first_child(DTYPE_t logz, np.ndarray[DTYPE_t32, ndim=3] scores, np.ndarray[DTYPE_t, ndim=2] i_cmp, np.ndarray[DTYPE_t, ndim=2] o_incmp, int h, int m):
    if h == m:
        return 0.
    assert logz > NEG_INF_ADD_EPS
    # C(m -> h-1) + C(h) = I(m <- h)
    # C(h) + C(h+1 <- m) = I(h <- m)
    cdef int h1 = (h+1 if m > h else h-1)
    cdef DTYPE_t a = i_cmp[m, h1]
    cdef DTYPE_t b = i_cmp[h, h]
    cdef DTYPE_t c = o_incmp[h, m]
    cdef DTYPE_t p = 0.
    # assert a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS
    if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS and c > NEG_INF_ADD_EPS:
        # p = math.exp(a + b - logz)
        # p = expf(a + b - logz)
        p = exp(a + b + c + scores[m, h, 0] + scores[m, h, 2] - logz)
        if not (-1e-5 <= p <= 1.0 + 1e-5):
            print("\nprob cython (first-child) = ", p, " h = ", h, " m = ", m)
    return p

@cython.boundscheck(False)
def marginal_prob_sib(DTYPE_t logz, np.ndarray[DTYPE_t, ndim=2] i_sib, np.ndarray[DTYPE_t, ndim=2] o_sib, int s, int m):
    if s == m:
        return 0.
    assert logz > NEG_INF_ADD_EPS
    cdef DTYPE_t a = i_sib[s, m]
    cdef DTYPE_t b = o_sib[s, m]
    cdef DTYPE_t p = 0.
    # assert a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS
    if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
        # p = math.exp(a + b - logz)
        # p = expf(a + b - logz)
        p = exp(a + b - logz)
        if not (-1e-5 <= p <= 1.0 + 1e-5):
            print("\nprob cython = ", p, " s = ", s, " m = ", m)
    return p

@cython.boundscheck(False)
def marginal_prob_labeled(DTYPE_t logz, np.ndarray[DTYPE_t, ndim=3] i_incmp, np.ndarray[DTYPE_t, ndim=3] o_incmp, int h, int m, int i_label):
    if h == m:
        return 0.
    assert logz > NEG_INF_ADD_EPS
    cdef DTYPE_t a = i_incmp[h, m, i_label]
    cdef DTYPE_t b = o_incmp[h, m, i_label]
    cdef DTYPE_t p = 0.
    # assert a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS
    if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
        # p = math.exp(a + b - logz)
        # p = expf(a + b - logz)
        p = exp(a + b - logz)
        if not (-1e-5 <= p <= 1.0 + 1e-5):
            print("\nprob cython = ", p, " h = ", h, " m = ", m, " l = ", i_label)
    return p

@cython.boundscheck(False)
def compute_marg_prob_w_sib(int max_len, int N, np.ndarray[DTYPE_t32, ndim=3] scores, int constrained, np.ndarray[int, ndim=2] candidate_head):
    assert max_len >= N
    cdef int N2, n, dim0, dim1, dim2, width, s, t, r, m, h, t1, sL1
    cdef DTYPE_t log_sum, a, b, prob, temp, logZ, logZ_o, eps_dynamic
    dim0, dim1, dim2 = scores.shape[0], scores.shape[1], scores.shape[2]
    assert dim0 == dim1 == max_len
    assert 3 >= dim2 >= 2
    cdef int use_first_child = (dim2 == 3)

    N2, n = N * N, N - 1
    cdef np.ndarray[DTYPE_t, ndim=2] i_cmp = np.array([NEG_INF] * N2, dtype=data_type).reshape(N, N)    # (h, m)
    cdef np.ndarray[DTYPE_t, ndim=2] i_incmp = np.array([NEG_INF] * N2, dtype=data_type).reshape(N, N)  # (h, m)
    cdef np.ndarray[DTYPE_t, ndim=2] i_sib = np.array([NEG_INF] * N2, dtype=data_type).reshape(N, N)    # (s, m)
    cdef np.ndarray[DTYPE_t, ndim=2] o_cmp = np.array([NEG_INF] * N2, dtype=data_type).reshape(N, N)
    cdef np.ndarray[DTYPE_t, ndim=2] o_incmp = np.array([NEG_INF] * N2, dtype=data_type).reshape(N, N)
    cdef np.ndarray[DTYPE_t, ndim=2] o_sib = np.array([NEG_INF] * N2, dtype=data_type).reshape(N, N)
    cdef np.ndarray[DTYPE_t32, ndim=3] marginal_probs = np.array([0.] * (max_len * max_len * dim2), dtype=data_type32).reshape(max_len, max_len, dim2)  # np.zeros((N, N), dtype=data_type)

    for s in range(N):
        i_cmp[s, s] = 0

    for width in range(1, N):
        for s in range(0, N - width):
            t = s + width
            s1 = s + 1
            tL1 = t - 1
            if not constrained or candidate_head[t, s]:  # I(s->t), I(s,t)
                log_sum = NEG_INF
                # first child: C(s,s) + C(t,s+1)
                a, b, c = i_cmp[s, s], i_cmp[t, s1], scores[t, s, 2] if use_first_child else 0.
                if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                    log_sum = log_add_if_not_inf(log_sum, a + b + c)
                if s != 0:
                    for r in range(s1, t):  #  I(s -> r) + S(r -> t)
                        a, b = i_incmp[s, r], i_sib[r, t]
                        if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                            log_sum = log_add_if_not_inf(log_sum, a + b)
                if log_sum > NEG_INF_ADD_EPS:
                    i_incmp[s, t] = log_sum + scores[t, s, 0]

            if s != 0 and (not constrained or candidate_head[s, t]):  # I(s <- t), I(t,s)
                log_sum = NEG_INF
                # first child: C(s -> t-1) + C(t)
                a, b, c = i_cmp[s, tL1], i_cmp[t, t], scores[s, t, 2] if use_first_child else 0.
                if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                    log_sum = log_add_if_not_inf(log_sum, a + b + c)
                for r in range(s1, t):  #  S(s <- r) + I(r <- t)
                    a, b = i_sib[r, s], i_incmp[t, r]
                    if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                        log_sum = log_add_if_not_inf(log_sum, a + b)
                if log_sum > NEG_INF_ADD_EPS:
                    i_incmp[t, s] = log_sum + scores[s, t, 0]

            if s != 0:  # S(s -> t) | S(s <- t) = C(s -> r) + C(r+1 <- t)
                log_sum = NEG_INF
                for r in range(s, t):
                    a, b = i_cmp[s, r], i_cmp[t, r+1]
                    if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                        log_sum = log_add_if_not_inf(log_sum, a + b)
                if log_sum > NEG_INF_ADD_EPS:
                    i_sib[s, t] = log_sum + scores[t, s, 1]
                    i_sib[t, s] = log_sum + scores[s, t, 1]

            # C(s -> t)
            if s != 0 or t == n:  # single-root
                log_sum = NEG_INF
                for r in range(s1, t+1):   # I(s -> r) + C(r -> t)
                    a, b = i_incmp[s, r], i_cmp[r, t]
                    if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                        log_sum = log_add_if_not_inf(log_sum, a + b)
                if log_sum > NEG_INF_ADD_EPS:
                    i_cmp[s, t] = log_sum

            # C(s <- t)
            if s != 0:
                log_sum = NEG_INF
                for r in range(s, t):   # C(s <- r) + I(r <- t)
                    a, b = i_cmp[r, s], i_incmp[t, r]
                    if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                        log_sum = log_add_if_not_inf(log_sum, a + b)
                if log_sum > NEG_INF_ADD_EPS:
                    i_cmp[t, s] = log_sum

    logZ = i_cmp[0, n]
    if logZ <= NEG_INF_ADD_EPS:
        print("\nlogZ = ", logZ, " constrained=", constrained)
        assert False

    o_cmp[0,n] = 0
    if not constrained or candidate_head[n, 0]:
        o_incmp[0, n] = i_cmp[n, n] + o_cmp[0, n]

    for width in range(N - 2, -1, -1):
        for s in range(0, N - width):
            t = s + width
            t1 = t + 1
            sL1 = s - 1
            # C(s -> t)
            if s != 0 or width == 0:    # single-root
                log_sum = NEG_INF
                if s != 0:
                    for r in range(0, s):
                        # I(r -> s) + C(s -> t) = C(r -> t); s is the last child of r
                        if r == 0 and t != n:  # only allow C(0, n)
                            continue
                        a, b = i_incmp[r, s], o_cmp[r, t]
                        if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                            log_sum = log_add_if_not_inf(log_sum, a + b)

                    for r in range(t1, N):
                        # C(s -> t) + C(t+1 <- r) = S(s -> r) | S(s <- r)
                        a, b, sco = i_cmp[r, t1], o_sib[s, r], scores[r, s, 1]
                        if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                            log_sum = log_add_if_not_inf(log_sum, a + b + sco)
                        a, b, sco = i_cmp[r, t1], o_sib[r, s], scores[s, r, 1]
                        if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                            log_sum = log_add_if_not_inf(log_sum, a + b + sco)

                    if t1 < N and (not constrained or candidate_head[s, t1]):
                        # C(s -> t) + C(t+1) = I(s <- t+1)  # s is the first child of t+1
                        a, b, sco, c = i_cmp[t1, t1], o_incmp[t1, s], scores[s, t1, 0], scores[s, t1, 2] if use_first_child else 0.
                        if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                            log_sum = log_add_if_not_inf(log_sum, a + b + sco + c)
                else:   # width = 0
                    # C(0) + C(1 <- r) = I(0 -> r); r is the first child of 0
                    assert t == s == 0
                    for r in range(t1, N):
                        a, b, sco, c = i_cmp[r, t1], o_incmp[s, r], scores[r, s, 0], scores[r, s, 2] if use_first_child else 0.
                        if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                            log_sum = log_add_if_not_inf(log_sum, a + b + sco + c)

                o_cmp[s, t] = log_sum

            if width == 0:
                assert s == 0
                break

            # C(s <- t)
            if s != 0:
                log_sum = NEG_INF
                for r in range(t1, N):
                    # C(s<-t) + I(t<-r) = C(s<-r)
                    a, b = i_incmp[r, t], o_cmp[r, s]
                    if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                        log_sum = log_add_if_not_inf(log_sum, a+b)

                for r in range(1, s):
                    # C(r -> s-1) + C(s <- t) = S(r -> t) | S(r <- t)
                    a, b, sco = i_cmp[r, sL1], o_sib[t, r], scores[r, t, 1]
                    if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                        log_sum = log_add_if_not_inf(log_sum, a + b + sco)
                    a, b, sco = i_cmp[r, sL1], o_sib[r, t], scores[t, r, 1]
                    if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                        log_sum = log_add_if_not_inf(log_sum, a + b + sco)

                if not constrained or candidate_head[t, sL1]:
                    # C(s-1) + C(s <- t) = I(s-1 -> t)  # t is the first child of s-1
                    a, b, sco, c = i_cmp[sL1, sL1], o_incmp[sL1, t], scores[t, sL1, 0], scores[t, sL1, 2] if use_first_child else 0.
                    if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                        log_sum = log_add_if_not_inf(log_sum, a + b + sco + c)

                o_cmp[t,s] = log_sum

            # I(s -> t)
            if not constrained or candidate_head[t, s]:
                log_sum = NEG_INF
                for r in range(t, N):
                    # I(s->t) + C(t->r) = C(s->r)
                    if s == 0 and r != N - 1:   # single-root
                        continue
                    a, b = i_cmp[t, r], o_cmp[s, r]
                    if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                        log_sum = log_add_if_not_inf(log_sum, a+b)

                if s != 0:  # single-root
                    for r in range(t1, N):
                        # I(s -> t) + S(t -> r) = I(s -> r)
                        a, b, sco = i_sib[t, r], o_incmp[s, r], scores[r, s, 0]
                        if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                            log_sum = log_add_if_not_inf(log_sum, a + b + sco)

                o_incmp[s, t] = log_sum

            # I(s <- t)
            if s != 0 and (not constrained or candidate_head[s, t]):
                log_sum = NEG_INF
                for r in range(1, s+1):
                    # C(r <- s) + I(s <- t) = C(r <- t)
                    a, b = i_cmp[s, r], o_cmp[t, r]
                    if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                        log_sum = log_add_if_not_inf(log_sum, a+b)

                for r in range(1, s):
                    # S(r <- s) + I(s <- t) = I(r <- t)
                    a, b, sco = i_sib[s, r], o_incmp[t, r], scores[r, t, 0]
                    if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                        log_sum = log_add_if_not_inf(log_sum, a + b + sco)

                o_incmp[t, s] = log_sum

            if s != 0:
                # S(s -> t)
                log_sum = NEG_INF
                for r in range(1, s):
                    # I(r -> s) + S(s -> t) = I(r -> t)
                    a, b, sco = i_incmp[r, s], o_incmp[r, t], scores[t, r, 0]
                    if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                        log_sum = log_add_if_not_inf(log_sum, a + b + sco)
                o_sib[s, t] = log_sum

                # S(s <- t)
                log_sum = NEG_INF
                for r in range(t1, N):
                    # S(s <- t) + I(t <- r) = I(s <- r)
                    a, b, sco = i_incmp[r, t], o_incmp[r, s], scores[s, r, 0]
                    if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                        log_sum = log_add_if_not_inf(log_sum, a + b + sco)
                o_sib[t, s] = log_sum


    logZ_o = o_cmp[0, 0]
    eps_dynamic = np.abs(logZ) * 1e-10
    if logZ_o <= NEG_INF_ADD_EPS or not (logZ + eps_dynamic >= logZ_o >= logZ - eps_dynamic):
        print("\nlogZ_o vs. logZ = ", logZ_o, logZ, " constrained=", constrained)

    error_occur = False
    for m in range(1, N):
        prob = 0.0
        for h in range(N):
            temp = marginal_prob(logZ, i_incmp, o_incmp, h, m)
            marginal_probs[m, h, 0] = temp
            prob += temp
            s = h
            marginal_probs[m, s, 1] = marginal_prob_sib(logZ, i_sib, o_sib, s, m) 
            if use_first_child:
                marginal_probs[m, h, 2] = marginal_prob_first_child(logZ, scores, i_cmp, o_incmp, h, m)  
        if not 1.0 + 1e-5 >= prob >= 1.0 - 1e-5:
            error_occur = True
            print("\nsum prob cython ", prob, " m: ", m, "constrained: ", constrained)
    if error_occur:
        print("\nlog_Z =", logZ)

    return logZ, marginal_probs


@cython.boundscheck(False)
def compute_marg_prob(int max_len, int N, np.ndarray[DTYPE_t32, ndim=2] scores, int constrained, np.ndarray[int, ndim=2] candidate_head):
    assert max_len >= N
    cdef int N2 = N * N
    cdef int n = N - 1
    cdef np.ndarray[DTYPE_t, ndim=2] i_cmp = np.array([NEG_INF] * N2, dtype=data_type).reshape(N, N)
    cdef np.ndarray[DTYPE_t, ndim=2] i_incmp = np.array([NEG_INF] * N2, dtype=data_type).reshape(N, N)
    cdef np.ndarray[DTYPE_t, ndim=2] o_cmp = np.array([NEG_INF] * N2, dtype=data_type).reshape(N, N)
    cdef np.ndarray[DTYPE_t, ndim=2] o_incmp = np.array([NEG_INF] * N2, dtype=data_type).reshape(N, N)
    cdef np.ndarray[DTYPE_t32, ndim=2] marginal_probs = np.array([0.] * (max_len * max_len), dtype=data_type32).reshape(max_len, max_len)  # np.zeros((N, N), dtype=data_type)

    cdef int width, s, t, r, m, h
    cdef DTYPE_t log_sum, a, b, prob, temp, logZ, logZ_o, eps_dynamic 

    for s in range(N):
        i_cmp[s,s] = 0

    for width in range(1, N):
        for s in range(0, N - width):
            t = s + width
            log_sum = NEG_INF
            # print "s: ", s,"\tt: ", t,"\twidth: ", width
            assert scores[s, t] > NEG_INF_ADD_EPS
            assert scores[t, s] > NEG_INF_ADD_EPS
            for r in range(s, t):
                a = i_cmp[s,r]
                b = i_cmp[t,r + 1]
                if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                    log_sum = log_add_if_not_inf(log_sum, a+b)

            if log_sum > NEG_INF_ADD_EPS:
                if not constrained or candidate_head[t,s]:  # I(s->t)
                    c = scores[t, s]
                    i_incmp[s,t] = log_sum + c
                if s != 0 and (not constrained or candidate_head[s,t]):  # I(t->s)
                    c = scores[s, t]
                    i_incmp[t,s] = log_sum + c

            if s != 0 or t == N - 1:  # C(s->t)
                log_sum = NEG_INF
                for r in range(s + 1, t + 1):
                    a = i_incmp[s,r]
                    b = i_cmp[r,t]
                    if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                        log_sum = log_add_if_not_inf(log_sum, a+b)
                i_cmp[s,t] = log_sum
                # print "  cmp s->t: ", i_cmp[s,t]

            if s != 0:  # C(t->s)
                log_sum = NEG_INF
                for r in range(s, t):
                    a = i_cmp[r,s]
                    b = i_incmp[t,r]
                    if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                        log_sum = log_add_if_not_inf(log_sum, a+b)
                i_cmp[t,s] = log_sum
                # print "  cmp t->s: ", i_cmp[t,s]

    logZ = i_cmp[0, n]
    if logZ <= NEG_INF_ADD_EPS:
        print("\nlogZ = ", logZ, " constrained=", constrained)
        assert False

    #def outside( constrained):
    o_cmp[0,n] = 0
    if not constrained or candidate_head[n,0]:
        o_incmp[0,n] = i_cmp[n,n] + o_cmp[0,n]

    for width in range(N - 2, -1, -1):
        for s in range(0, N - width):
            t = s + width
            # C(s->t)
            if s != 0 or width == 0:
                log_sum = NEG_INF
                for r in range(0, s):
                    if r == 0 and t != N - 1:
                        continue
                    # I(r->s) + C(s->t) = C(r->t)
                    a = i_incmp[r,s]
                    b = o_cmp[r,t]
                    if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                        log_sum = log_add_if_not_inf(log_sum, a+b)

                for r in range(t + 1, N):
                    # C(s->t) + C(r->t+1) = I(s->r)
                    if not constrained or candidate_head[r, s]:
                        a = i_cmp[r,t + 1]
                        b = o_incmp[s,r]
                        sco = scores[r, s]
                        if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                            log_sum = log_add_if_not_inf(log_sum, a+b+sco)

                    # C(s->t) + C(r->t+1) = I(r->s)
                    if not constrained or candidate_head[s, r]:
                        a = i_cmp[r,t + 1]
                        b = o_incmp[r,s]
                        sco = scores[s, r]
                        if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS: 
                            log_sum = log_add_if_not_inf(log_sum, a+b+sco)

                o_cmp[s,t] = log_sum
                # print "cmp s->t: ", o_cmp[s,t]

            if width == 0:
                assert s == 0
                break

            # C(t->s)
            if s != 0:
                log_sum = NEG_INF
                # C(t->s) + I(r->t) = C(r->s)
                for r in range(t + 1, N):
                    a = i_incmp[r,t]
                    b = o_cmp[r,s]
                    if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                        log_sum = log_add_if_not_inf(log_sum, a+b)

                for r in range(0, s):
                    # C(r->s-1) + C(t->s) = I(r->t)
                    if r == 0 and s - 1 != 0:
                        continue
                    if not constrained or candidate_head[t,r]:
                        a = i_cmp[r,s - 1]
                        b = o_incmp[r,t]
                        sco = scores[t, r]
                        if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS: 
                            log_sum = log_add_if_not_inf(log_sum, a+b+sco)

                    # C(r->s-1) + C(t->s) = I(t->r)
                    if r != 0 and (not constrained or candidate_head[r,t]):
                        a = i_cmp[r,s - 1]
                        b = o_incmp[t,r]
                        sco = scores[r, t]
                        if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                            log_sum = log_add_if_not_inf(log_sum, a+b+sco)

                o_cmp[t,s] = log_sum
                # print "cmp t->s: ", o_cmp[t,s]

            # I(s->t)
            if not constrained or candidate_head[t,s]:
                log_sum = NEG_INF
                for r in range(t, N):
                    # I(s->t) + C(t->r) = C(s->r)
                    if s == 0 and r != N - 1:
                        continue
                    a = i_cmp[t,r]
                    b = o_cmp[s,r]
                    if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                        log_sum = log_add_if_not_inf(log_sum, a+b)
                o_incmp[s,t] = log_sum

            if s != 0 and (not constrained or candidate_head[s,t]):
                log_sum = NEG_INF
                for r in range(1, s + 1):
                    # C(s->r) + I(s->t) = C(t->r)
                    a = i_cmp[s,r]
                    b = o_cmp[t,r]
                    if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                        log_sum = log_add_if_not_inf(log_sum, a+b)
                o_incmp[t,s] = log_sum

    logZ_o = o_cmp[0, 0]
    eps_dynamic = np.abs(logZ) * 1e-10
    if logZ_o <= NEG_INF_ADD_EPS or not (logZ + eps_dynamic >= logZ_o >= logZ - eps_dynamic):
        print("\nlogZ_o vs. logZ = ", logZ_o, logZ, " constrained=", constrained)

    error_occur = False
    for m in range(1, N):
        prob = 0.0
        for h in range(N):
            temp = marginal_prob(logZ, i_incmp, o_incmp, h, m)
            marginal_probs[m, h] = temp
            prob += temp
        if not 1.0 + 1e-5 >= prob >= 1.0 - 1e-5:
            error_occur = True
            print("\nsum prob cython ", prob, " m: ", m, "constrained: ", constrained)
    if error_occur:
        print("\nlog_Z =", logZ)

    return logZ, marginal_probs


@cython.boundscheck(False)
def compute_marg_prob_labeled(int max_len, int N, int L, np.ndarray[DTYPE_t32, ndim=3] scores, int constrained, np.ndarray[int, ndim=2] candidate_head, np.ndarray[int, ndim=1] gold_labels):
    assert max_len >= N
    assert L > 1
    cdef int N2 = N * N
    cdef int N2L = N2 * L
    cdef int L1 = L + 1
    cdef int n = N - 1
    cdef np.ndarray[DTYPE_t, ndim=2] i_cmp = np.array([NEG_INF] * N2, dtype=data_type).reshape(N, N)
    cdef np.ndarray[DTYPE_t, ndim=3] i_incmp = np.array([NEG_INF] * N2L, dtype=data_type).reshape(N, N, L)
    cdef np.ndarray[DTYPE_t, ndim=2] i_incmp_all_labels = np.array([NEG_INF] * N2, dtype=data_type).reshape(N, N)
    cdef np.ndarray[DTYPE_t, ndim=2] o_cmp = np.array([NEG_INF] * N2, dtype=data_type).reshape(N, N)
    cdef np.ndarray[DTYPE_t, ndim=3] o_incmp = np.array([NEG_INF] * N2L, dtype=data_type).reshape(N, N, L)
    cdef np.ndarray[DTYPE_t, ndim=2] o_incmp_all_labels = np.array([NEG_INF] * N2, dtype=data_type).reshape(N, N)  # include the arc and label scores s->t
    cdef np.ndarray[DTYPE_t32, ndim=3] marginal_probs = np.array([0.] * (max_len * max_len *L1), dtype=data_type32).reshape(max_len, max_len, L1) # np.zeros((N, N, L1), dtype=data_type)

    cdef int width, s, t, r, m, h, i_label
    cdef DTYPE_t log_sum, a, b, prob, prob_arc, temp, logZ, logZ_o, eps_dynamic

    for s in range(N):
        i_cmp[s,s] = 0

    for width in range(1, N):
        for s in range(0, N - width):
            t = s + width
            log_sum = NEG_INF
            for r in range(s, t):
                a = i_cmp[s, r]
                b = i_cmp[t, r + 1]
                if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                    log_sum = log_add_if_not_inf(log_sum, a+b)

            if log_sum > NEG_INF_ADD_EPS:
                if not constrained or candidate_head[t, s]:  # I(s->t)
                    for i_label in range(L):
                        if constrained and gold_labels[t] >= 0 and gold_labels[t] != i_label:
                            continue
                        sco = scores[t, s, L] + scores[t, s, i_label] # L means the arc score
                        i_incmp[s, t, i_label] = log_sum + sco
                        i_incmp_all_labels[s, t] = log_add_if_not_inf(i_incmp_all_labels[s, t], i_incmp[s, t, i_label])
                if s != 0 and (not constrained or candidate_head[s, t]):  # I(t->s)
                    for i_label in range(L):
                        if constrained and gold_labels[s] >= 0 and gold_labels[s] != i_label:
                            continue
                        sco = scores[s, t, L] + scores[s, t, i_label]
                        i_incmp[t, s, i_label] = log_sum + sco
                        i_incmp_all_labels[t, s] = log_add_if_not_inf(i_incmp_all_labels[t, s], i_incmp[t, s, i_label])

            if s != 0 or t == N - 1:  # C(s->t)
                log_sum = NEG_INF
                for r in range(s + 1, t + 1):
                    a = i_incmp_all_labels[s, r]
                    b = i_cmp[r, t]
                    if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                        log_sum = log_add_if_not_inf(log_sum, a+b)
                i_cmp[s, t] = log_sum

            if s != 0:  # C(t->s)
                log_sum = NEG_INF
                for r in range(s, t):
                    a = i_cmp[r, s]
                    b = i_incmp_all_labels[t, r]
                    if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                        log_sum = log_add_if_not_inf(log_sum, a+b)
                i_cmp[t, s] = log_sum

    logZ = i_cmp[0, n]
    if logZ <= NEG_INF_ADD_EPS:
        print("\nlogZ = ", logZ, " constrained=", constrained)
        assert False

    #def outside( constrained):
    o_cmp[0, n] = 0
    if not constrained or candidate_head[n,0]:
        for i_label in range(L):
            if constrained and gold_labels[n] >= 0 and gold_labels[n] != i_label:
                continue
            o_incmp[0, n, i_label] = i_cmp[n, n] + o_cmp[0, n]
            sco = scores[n, 0, L] + scores[n, 0, i_label]
            o_incmp_all_labels[0, n] = log_add_if_not_inf(o_incmp_all_labels[0, n], sco + o_incmp[0, n, i_label])

    for width in range(N - 2, -1, -1):
        for s in range(0, N - width):
            t = s + width
            # C(s->t)
            if s != 0 or width == 0:
                log_sum = NEG_INF
                for r in range(0, s):
                    if r == 0 and t != N - 1:
                        continue
                    # I(r->s) + C(s->t) = C(r->t)
                    a = i_incmp_all_labels[r, s]
                    b = o_cmp[r, t]
                    if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                        log_sum = log_add_if_not_inf(log_sum, a+b)

                for r in range(t + 1, N):
                    # C(s->t) + C(r->t+1) = I(s->r)
                    if not constrained or candidate_head[r, s]:
                        a = i_cmp[r, t + 1]
                        b = o_incmp_all_labels[s, r]
                        if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                            log_sum = log_add_if_not_inf(log_sum, a+b)

                    # C(s->t) + C(r->t+1) = I(r->s)
                    if not constrained or candidate_head[s, r]:
                        a = i_cmp[r, t + 1]
                        b = o_incmp_all_labels[r, s]
                        if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                            log_sum = log_add_if_not_inf(log_sum, a+b)

                o_cmp[s,t] = log_sum

            if width == 0:
                assert s == 0
                break

            # C(t->s)
            if s != 0:
                log_sum = NEG_INF
                # C(t->s) + I(r->t) = C(r->s)
                for r in range(t + 1, N):
                    a = i_incmp_all_labels[r, t]
                    b = o_cmp[r, s]
                    if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                        log_sum = log_add_if_not_inf(log_sum, a+b)

                for r in range(0, s):
                    # C(r->s-1) + C(t->s) = I(r->t)
                    if r == 0 and s - 1 != 0:
                        continue
                    if not constrained or candidate_head[t, r]:
                        a = i_cmp[r,s - 1]
                        b = o_incmp_all_labels[r,t]
                        if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                            log_sum = log_add_if_not_inf(log_sum, a+b)

                    # C(r->s-1) + C(t->s) = I(t->r)
                    if r != 0 and (not constrained or candidate_head[r, t]):
                        a = i_cmp[r, s - 1]
                        b = o_incmp_all_labels[t, r]
                        if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                            log_sum = log_add_if_not_inf(log_sum, a+b)

                o_cmp[t, s] = log_sum

            # I(s->t)
            if not constrained or candidate_head[t, s]:
                log_sum = NEG_INF
                for r in range(t, N):
                    # I(s->t) + C(t->r) = C(s->r)
                    if s == 0 and r != N - 1:
                        continue
                    a = i_cmp[t, r]
                    b = o_cmp[s, r]
                    if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                        log_sum = log_add_if_not_inf(log_sum, a+b)
                if log_sum > NEG_INF_ADD_EPS:
                    for i_label in range(L):
                        if constrained and gold_labels[t] >= 0 and gold_labels[t] != i_label:
                            continue
                        o_incmp[s, t, i_label] = log_sum
                        sco = scores[t, s, L] + scores[t, s, i_label]
                        o_incmp_all_labels[s, t] = log_add_if_not_inf(o_incmp_all_labels[s, t], log_sum+sco)

            if s != 0 and (not constrained or candidate_head[s, t]):
                log_sum = NEG_INF
                for r in range(1, s + 1):
                    # C(s->r) + I(s->t) = C(t->r)
                    a = i_cmp[s,r]
                    b = o_cmp[t,r]
                    if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                        log_sum = log_add_if_not_inf(log_sum, a+b)
                if log_sum > NEG_INF_ADD_EPS:
                    for i_label in range(L):
                        if constrained and gold_labels[s] >= 0 and gold_labels[s] != i_label:
                            continue
                        o_incmp[t, s, i_label] = log_sum
                        sco = scores[s, t, L] + scores[s, t, i_label]
                        o_incmp_all_labels[t, s] = log_add_if_not_inf(o_incmp_all_labels[t, s], log_sum+sco)

    logZ_o = o_cmp[0, 0]
    eps_dynamic = np.abs(logZ) * 1e-10
    if logZ_o <= NEG_INF_ADD_EPS or not (logZ + eps_dynamic >= logZ_o >= logZ - eps_dynamic):
        print("\nlogZ_o vs. logZ = ", logZ_o, logZ, " constrained=", constrained)

    error_occur = False
    for m in range(1, N):
        prob = 0.0
        for h in range(N):
            prob_arc = 0.
            for i_label in range(L):
                temp = marginal_prob_labeled(logZ, i_incmp, o_incmp, h, m, i_label) 
                marginal_probs[m, h, i_label] = temp  
                prob_arc += temp
            marginal_probs[m, h, L] = prob_arc
            prob += prob_arc
        if not 1.0 + 1e-5 >= prob >= 1.0 - 1e-5:
            error_occur = True
            print("\nsum prob cython ", prob, " m: ", m, "constrained: ", constrained)
    if error_occur:
        print("\nlog_Z =", logZ)

    return logZ, marginal_probs


@cython.boundscheck(False)
def compute_marg_prob_w_sib_labeled(int max_len, int N, int L, np.ndarray[DTYPE_t32, ndim=3] scores, int constrained, np.ndarray[int, ndim=2] candidate_head, np.ndarray[int, ndim=1] gold_labels):
    assert max_len >= N
    cdef int L1, N2, n, N2L1, dim0, dim1, dim2, width, s, t, r, m, h, t1, sL1, first_L2, sib_L1, i_label
    cdef DTYPE_t log_sum, sco, a, b, c, prob, temp, logZ, logZ_o, eps_dynamic
    dim0, dim1, dim2 = scores.shape[0], scores.shape[1], scores.shape[2]
    assert dim0 == dim1 == max_len
    assert L+3 >= dim2 >= L+1 
    cdef int use_first_child = (dim2 == L+3)    # L:arc L+1:sib L+2:first-child
    sib_L1, first_L2 = L+1, L+2
    L1, N2, n = L + 1, N * N, N - 1
    N2L1 = N2 * (L+1)
    cdef np.ndarray[DTYPE_t, ndim=2] i_cmp = np.array([NEG_INF] * N2, dtype=data_type).reshape(N, N)            # (h, m)
    cdef np.ndarray[DTYPE_t, ndim=3] i_incmp = np.array([NEG_INF] * N2L1, dtype=data_type).reshape(N, N, L1)    # (h, m)
    cdef np.ndarray[DTYPE_t, ndim=2] i_sib = np.array([NEG_INF] * N2, dtype=data_type).reshape(N, N)            # (s, m)
    cdef np.ndarray[DTYPE_t, ndim=2] o_cmp = np.array([NEG_INF] * N2, dtype=data_type).reshape(N, N)
    cdef np.ndarray[DTYPE_t, ndim=3] o_incmp = np.array([NEG_INF] * N2L1, dtype=data_type).reshape(N, N, L1)    # L: all lables, include the arc and label scores s->t
    cdef np.ndarray[DTYPE_t, ndim=2] o_sib = np.array([NEG_INF] * N2, dtype=data_type).reshape(N, N)
    cdef np.ndarray[DTYPE_t32, ndim=3] marginal_probs = np.array([0.] * (max_len * max_len * dim2), dtype=data_type32).reshape(max_len, max_len, dim2)  # np.zeros((N, N), dtype=data_type)

    for s in range(N):
        i_cmp[s, s] = 0

    for width in range(1, N):
        for s in range(0, N - width):
            t = s + width
            s1 = s + 1
            tL1 = t - 1
            if not constrained or candidate_head[t, s]:  # I(s->t), I(s,t)
                log_sum = NEG_INF
                # first child: C(s,s) + C(t,s+1)
                a, b, c = i_cmp[s, s], i_cmp[t, s1], scores[t, s, first_L2] if use_first_child else 0.
                if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                    log_sum = log_add_if_not_inf(log_sum, a + b + c)
                if s != 0:
                    for r in range(s1, t):  #  I(s -> r) + S(r -> t)
                        a, b = i_incmp[s, r, L], i_sib[r, t]
                        if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                            log_sum = log_add_if_not_inf(log_sum, a + b)
                if log_sum > NEG_INF_ADD_EPS:
                    for i_label in range(L):
                        if constrained and gold_labels[t] >= 0 and gold_labels[t] != i_label:
                            continue
                        sco = scores[t, s, L] + scores[t, s, i_label]
                        i_incmp[s, t, i_label] = log_sum + sco
                        i_incmp[s, t, L] = log_add_if_not_inf(i_incmp[s, t, L], i_incmp[s, t, i_label])

            if s != 0 and (not constrained or candidate_head[s, t]):  # I(s <- t), I(t,s)
                log_sum = NEG_INF
                # first child: C(s -> t-1) + C(t)
                a, b, c = i_cmp[s, tL1], i_cmp[t, t], scores[s, t, first_L2] if use_first_child else 0.
                if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                    log_sum = log_add_if_not_inf(log_sum, a + b + c)
                for r in range(s1, t):  #  S(s <- r) + I(r <- t)
                    a, b = i_sib[r, s], i_incmp[t, r, L]
                    if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                        log_sum = log_add_if_not_inf(log_sum, a + b)
                if log_sum > NEG_INF_ADD_EPS:
                    for i_label in range(L):
                        if constrained and gold_labels[s] >= 0 and gold_labels[s] != i_label:
                            continue
                        sco = scores[s, t, L] + scores[s, t, i_label]
                        i_incmp[t, s, i_label] = log_sum + sco
                        i_incmp[t, s, L] = log_add_if_not_inf(i_incmp[t, s, L], i_incmp[t, s, i_label])

            if s != 0:  # S(s -> t) | S(s <- t) = C(s -> r) + C(r+1 <- t)
                log_sum = NEG_INF
                for r in range(s, t):
                    a, b = i_cmp[s, r], i_cmp[t, r+1]
                    if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                        log_sum = log_add_if_not_inf(log_sum, a + b)
                if log_sum > NEG_INF_ADD_EPS:
                    i_sib[s, t] = log_sum + scores[t, s, sib_L1]
                    i_sib[t, s] = log_sum + scores[s, t, sib_L1]

            # C(s -> t)
            if s != 0 or t == n:  # single-root
                log_sum = NEG_INF
                for r in range(s1, t+1):   # I(s -> r) + C(r -> t)
                    a, b = i_incmp[s, r, L], i_cmp[r, t]
                    if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                        log_sum = log_add_if_not_inf(log_sum, a + b)
                if log_sum > NEG_INF_ADD_EPS:
                    i_cmp[s, t] = log_sum

            # C(s <- t)
            if s != 0:
                log_sum = NEG_INF
                for r in range(s, t):   # C(s <- r) + I(r <- t)
                    a, b = i_cmp[r, s], i_incmp[t, r, L]
                    if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                        log_sum = log_add_if_not_inf(log_sum, a + b)
                if log_sum > NEG_INF_ADD_EPS:
                    i_cmp[t, s] = log_sum

    logZ = i_cmp[0, n]
    if logZ <= NEG_INF_ADD_EPS:
        print("\nlogZ = ", logZ, " constrained=", constrained)
        assert False

    o_cmp[0, n] = 0
    if not constrained or candidate_head[n, 0]:
        for i_label in range(L):
            if constrained and gold_labels[n] >= 0 and gold_labels[n] != i_label:
                continue
            o_incmp[0, n, i_label] = i_cmp[n, n] + o_cmp[0, n]
            sco = scores[n, 0, L] + scores[n, 0, i_label]
            o_incmp[0, n, L] = log_add_if_not_inf(o_incmp[0, n, L], o_incmp[0, n, i_label] + sco)

    for width in range(N - 2, -1, -1):
        for s in range(0, N - width):
            t = s + width
            t1 = t + 1
            sL1 = s - 1
            # C(s -> t)
            if s != 0 or width == 0:    # single-root
                log_sum = NEG_INF
                if s != 0:
                    for r in range(0, s):
                        # I(r -> s) + C(s -> t) = C(r -> t); s is the last child of r
                        if r == 0 and t != n:  # only allow C(0, n)
                            continue
                        a, b = i_incmp[r, s, L], o_cmp[r, t]
                        if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                            log_sum = log_add_if_not_inf(log_sum, a + b)

                    for r in range(t1, N):
                        # C(s -> t) + C(t+1 <- r) = S(s -> r) | S(s <- r)
                        a, b, sco = i_cmp[r, t1], o_sib[s, r], scores[r, s, sib_L1]
                        if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                            log_sum = log_add_if_not_inf(log_sum, a + b + sco)
                        a, b, sco = i_cmp[r, t1], o_sib[r, s], scores[s, r, sib_L1]
                        if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                            log_sum = log_add_if_not_inf(log_sum, a + b + sco)

                    if t1 < N and (not constrained or candidate_head[s, t1]):
                        # C(s -> t) + C(t+1) = I(s <- t+1)  # s is the first child of t+1
                        a, b, c = i_cmp[t1, t1], o_incmp[t1, s, L], scores[s, t1, first_L2] if use_first_child else 0.
                        if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                            log_sum = log_add_if_not_inf(log_sum, a + b + c)
                else:   # width = 0
                    # C(0) + C(1 <- r) = I(0 -> r); r is the first child of 0
                    assert t == s == 0
                    for r in range(t1, N):
                        a, b, c = i_cmp[r, t1], o_incmp[s, r, L], scores[r, s, first_L2] if use_first_child else 0.
                        if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                            log_sum = log_add_if_not_inf(log_sum, a + b + c)

                o_cmp[s, t] = log_sum

            if width == 0:
                assert s == 0
                break

            # C(s <- t)
            if s != 0:
                log_sum = NEG_INF
                for r in range(t1, N):
                    # C(s<-t) + I(t<-r) = C(s<-r)
                    a, b = i_incmp[r, t, L], o_cmp[r, s]
                    if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                        log_sum = log_add_if_not_inf(log_sum, a+b)

                for r in range(1, s):
                    # C(r -> s-1) + C(s <- t) = S(r -> t) | S(r <- t)
                    a, b, sco = i_cmp[r, sL1], o_sib[t, r], scores[r, t, sib_L1]
                    if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                        log_sum = log_add_if_not_inf(log_sum, a + b + sco)
                    a, b, sco = i_cmp[r, sL1], o_sib[r, t], scores[t, r, sib_L1]
                    if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                        log_sum = log_add_if_not_inf(log_sum, a + b + sco)

                if not constrained or candidate_head[t, sL1]:
                    # C(s-1) + C(s <- t) = I(s-1 -> t)  # t is the first child of s-1
                    a, b, c = i_cmp[sL1, sL1], o_incmp[sL1, t, L], scores[t, sL1, first_L2] if use_first_child else 0.
                    if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                        log_sum = log_add_if_not_inf(log_sum, a + b + c)

                o_cmp[t,s] = log_sum

            # I(s -> t)
            if not constrained or candidate_head[t, s]:
                log_sum = NEG_INF
                for r in range(t, N):
                    # I(s->t) + C(t->r) = C(s->r)
                    if s == 0 and r != N - 1:   # single-root
                        continue
                    a, b = i_cmp[t, r], o_cmp[s, r]
                    if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                        log_sum = log_add_if_not_inf(log_sum, a + b)

                if s != 0:  # single-root
                    for r in range(t1, N):
                        # I(s -> t) + S(t -> r) = I(s -> r)
                        a, b = i_sib[t, r], o_incmp[s, r, L]
                        if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                            log_sum = log_add_if_not_inf(log_sum, a + b)

                if log_sum > NEG_INF_ADD_EPS:
                    for i_label in range(L):
                        if constrained and gold_labels[t] >= 0 and gold_labels[t] != i_label:
                            continue
                        o_incmp[s, t, i_label] = log_sum
                        sco = scores[t, s, L] + scores[t, s, i_label]
                        o_incmp[s, t, L] = log_add_if_not_inf(o_incmp[s, t, L], log_sum + sco)

            # I(s <- t)
            if s != 0 and (not constrained or candidate_head[s, t]):
                log_sum = NEG_INF
                for r in range(1, s+1):
                    # C(r <- s) + I(s <- t) = C(r <- t)
                    a, b = i_cmp[s, r], o_cmp[t, r]
                    if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                        log_sum = log_add_if_not_inf(log_sum, a + b)

                for r in range(1, s):
                    # S(r <- s) + I(s <- t) = I(r <- t)
                    a, b = i_sib[s, r], o_incmp[t, r, L]
                    if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                        log_sum = log_add_if_not_inf(log_sum, a + b)

                if log_sum > NEG_INF_ADD_EPS:
                    for i_label in range(L):
                        if constrained and gold_labels[s] >= 0 and gold_labels[s] != i_label:
                            continue
                        o_incmp[t, s, i_label] = log_sum
                        sco = scores[s, t, L] + scores[s, t, i_label]
                        o_incmp[t, s, L] = log_add_if_not_inf(o_incmp[t, s, L], log_sum + sco)


            if s != 0:
                # S(s -> t)
                log_sum = NEG_INF
                for r in range(1, s):
                    # I(r -> s) + S(s -> t) = I(r -> t)
                    a, b = i_incmp[r, s, L], o_incmp[r, t, L]
                    if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                        log_sum = log_add_if_not_inf(log_sum, a + b)
                o_sib[s, t] = log_sum

                # S(s <- t)
                log_sum = NEG_INF
                for r in range(t1, N):
                    # S(s <- t) + I(t <- r) = I(s <- r)
                    a, b = i_incmp[r, t, L], o_incmp[r, s, L]
                    if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                        log_sum = log_add_if_not_inf(log_sum, a + b)
                o_sib[t, s] = log_sum


    logZ_o = o_cmp[0, 0]
    eps_dynamic = np.abs(logZ) * 1e-10
    if logZ_o <= NEG_INF_ADD_EPS or not (logZ + eps_dynamic >= logZ_o >= logZ - eps_dynamic):
        print("\nlogZ_o vs. logZ = ", logZ_o, logZ, " constrained=", constrained)

    error_occur = False
    for m in range(1, N):
        prob = 0.0
        for h in range(N):
            prob_arc = 0.
            for i_label in range(L):
                temp = marginal_prob_labeled(logZ, i_incmp, o_incmp, h, m, i_label)
                marginal_probs[m, h, i_label] = temp
                prob_arc += temp
            marginal_probs[m, h, L] = prob_arc
            prob += prob_arc
            s = h
            marginal_probs[m, s, sib_L1] = marginal_prob_sib(logZ, i_sib, o_sib, s, m) 
            if use_first_child:
                marginal_probs[m, h, first_L2] = marginal_prob_first_child_labeled(logZ, scores, i_cmp, o_incmp, h, m, L)
        if not 1.0 + 1e-5 >= prob >= 1.0 - 1e-5:
            error_occur = True
            print("\nsum prob cython ", prob, " m: ", m, "constrained: ", constrained)
    if error_occur:
        print("\nlog_Z =", logZ)

    return logZ, marginal_probs


@cython.boundscheck(False)
def get_result_recursively(np.ndarray[int, ndim=3] cmp_info, np.ndarray[int, ndim=3] incmp_info, int cmp_or_incmp, int s, int t, np.ndarray[int, ndim=1] head_pred):
    if s == t:
        return

    if cmp_or_incmp == INCMP:
        assert head_pred[t] < 0
        head_pred[t] = s
        this_info = incmp_info[s, t] 
    else:
        this_info = cmp_info[s, t] 

    get_result_recursively(cmp_info, incmp_info, this_info[0], this_info[1], this_info[2], head_pred)
    get_result_recursively(cmp_info, incmp_info, this_info[3], this_info[4], this_info[5], head_pred)


@cython.boundscheck(False)
def viterbi(int N, np.ndarray[DTYPE_t32, ndim=2] scores, int constrained, np.ndarray[int, ndim=2] candidate_head):
    cdef np.ndarray[DTYPE_t, ndim=2] cmp_sco = np.array([NEG_INF] * (N * N), dtype=data_type).reshape(N, N)
    cdef np.ndarray[int, ndim=3] cmp_info = np.array([0] * (N * N * 6), dtype=data_type_int32).reshape(N, N, 6)  # 0: typeL begL endL 3: typeR begR endR
    cdef np.ndarray[DTYPE_t, ndim=2] incmp_sco = np.array([NEG_INF] * (N * N), dtype=data_type).reshape(N, N)
    cdef np.ndarray[int, ndim=3] incmp_info = np.array([0] * (N * N * 6), dtype=data_type_int32).reshape(N, N, 6)  
    cdef np.ndarray[int, ndim=1] head_pred = np.array([-1] * N, dtype=data_type_int32)

    cdef int width, s, t, r, m, h 
    cdef DTYPE_t log_sum, a, b, prob, prob_arc, temp, logZ, logZ_o, eps_dynamic

    for s in range(N):
        cmp_sco[s, s] = 0

    for width in range(1, N):
        for s in range(0, N - width):
            t = s + width
            log_sum = NEG_INF
            for r in range(s, t):
                a, b = cmp_sco[s, r], cmp_sco[t, r + 1]
                if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                    if not constrained or candidate_head[t, s]:  # I(s->t)
                        sco = (scores[t, s]) + a + b 
                        if sco > incmp_sco[s, t]:
                            incmp_sco[s, t] = sco
                            incmp_info[s, t] = np.array([CMP, s, r, CMP, t, r + 1], dtype=data_type_int32)
                    if s != 0 and (not constrained or candidate_head[s, t]):  # I(t->s)
                        sco = (scores[s, t]) + a + b 
                        if sco > incmp_sco[t, s]:
                            incmp_sco[t, s] = sco
                            incmp_info[t, s] = np.array([CMP, s, r, CMP, t, r + 1], dtype=data_type_int32)

            if s != 0 or t == N - 1:  # C(s->t)
                for r in range(s + 1, t + 1):
                    a, b = incmp_sco[s, r], cmp_sco[r, t]
                    if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS and a+b > cmp_sco[s, t]:
                        cmp_sco[s, t] = a + b
                        cmp_info[s, t] = np.array([INCMP, s, r, CMP, r, t], dtype=data_type_int32)

            if s != 0:  # C(t->s)
                log_sum = NEG_INF
                for r in range(s, t):
                    a, b = cmp_sco[r, s], incmp_sco[t, r]
                    if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS and a+b > cmp_sco[t, s]:
                        cmp_sco[t, s] = a + b
                        cmp_info[t, s] = np.array([CMP, r, s, INCMP, t, r], dtype=data_type_int32)

    assert cmp_sco[0, N-1] > NEG_INF_ADD_EPS
    get_result_recursively(cmp_info, incmp_info, CMP, 0, N-1, head_pred)
    return head_pred
    




