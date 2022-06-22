#import
import math
import numpy as np
import random as rd

#val gen

q = 7681
k = 3
n = 256
du = 11
dv = 3
dt = 11
eta = 4


#fct gen

def modular_reduction1(r,a):
    r1 = r%a
    if r1 > a/2:
        r1 = r1 - a
    return r1

def modular_reduction2(r,a):
    r1 = r%a
    if r1 > (a-1)/2:
        r1 = r1 - a
    return r1

def array_to_integer(tab):
    res = 0
    for i in range(256):
        res += tab[i] * 2**i
    return res

def integer_to_array(n):
    tab = np.arange(256)
    for i in range(256):
        r = n%2
        if r == 1:
            tab[i] = 1
        else:
            tab[i] = 0
        n = n//2
    return tab   


#compress/decompress

def compress(x,d,q = 7681):
    c=round(x*(2**d)/q)
    return modular_reduction1(c,2**d)

def decompress(x,d,q = 7681):
    return round(x*q/(2**d))

#enc
"""
def key_generation():
    rho = [rd.randint(0,1) for i in range(256)]
    sigma = [rd.randint(0,1) for i in range(256)]


def encryption():

def decryption(s,u,v):
    a = decompress(u,du)
    b = decompress(v,dv)
    return compress(v - np.dot(np.transpose(s),u),1)

"""
