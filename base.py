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
l = 6
w = 64
b = 1600

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

def binomial(eta):
    a = [rd.randint(0,1) for i in range(eta)]
    b = [rd.randint(0,1) for i in range(eta)]
    return sum(a-b)

def vec_bin(k,eta):
    res = np.arange(k)
    for i in range(k):
        res[i]=binomial(eta)
    return res
    
#SHAKE 128

def convert_to_array(S):
    A1 = np.arange(5*5*w)
    np.reshape(A1,(5,5,w))
    for x in range(5):
        for y in range(5):
            for z in range(w):
                A1[x,y,z]=S[w(5*y+x)+z]
    
    
def convert_to_string(A):
    res = []
    for x in range(5):
        plan = []
        for y in range(5):
            lane = []
            for z in range(w):
                val = A[y,x,z]
                lane += [val]
            plan += lane
        res += plan
    return res


def rc(t):
    if t%255 == 0:
        return 1
    R = [1,0,0,0,0,0,0,0]
    for i in range(1,t%255):
        R = [0]+R
        R[0] = (R[0]+ R[8])%2 
        R[4] = (R[4]+ R[8])%2
        R[5] = (R[5]+ R[8])%2
        R[6] = (R[6]+ R[8])%2
        R = R[0::8]
    return R[0]
    

def f1(A,i):
    A1 = np.arange(5*5*w)
    np.reshape(A1,(5,5,w))
    for x in range(5):
        for y in range(5):
            for z in range(w):
                A1[x,y,z] = A[x,y,z]
    R = [0]*w
    for j in range(l+1):
        R[(2**j)-1] = rc(j+7i)
    for z in range(w):
        A1[0,0,z] = (A[0,0,z]+R[z])%2
    return A1
    


def f2(A):
    A1 = np.arange(5*5*w)
    np.reshape(A1,(5,5,w))
    for x in range(5):
        for y in range(5):
            for z in range(w):
                A1[x,y,z] = (A[x,y,z]+((A[(x+1)%5,y,z]+1)*A[(x+2)%5,y,z]))%2
    return A1


def f3(A):
    A1 = np.arange(5*5*w)
    np.reshape(A1,(5,5,w))
    for x in range(5):
        for y in range(5):
            for z in range(w):
                A1[x,y,z] = A[(x+3y)%5,x,z]
    return A1
                



def f4(A):
    A1 = np.arange(5*5*w)
    np.reshape(A1,(5,5,w))
    for z in range(w):
        A1[0,0,z] = A[0,0,z]
    x,y = 1,0
    for i in range(24):
        for z in range(w):
            A1[x,y,z]=A[x,y,(z-((i+1)*(i+2)/2)%w)]
            x,y = y,(2x+3y)%5
    return A1
                


def f5(A):
    A1 = np.arange(5*5*w)
    np.reshape(A1,(5,5,w))
    C = np.arange(5*w)
    D = np.arange(5*w)
    np.reshape(C,(5,w))
    np.reshape(C,(5,w))
    for x in range(5):
        for z in range(w):
            C[x,z] = (A[x,0,z]+A[x,1,z]+A[x,2,z]+A[x,3,z]+A[x,4,z])%2
    for x in range(5):
        for z in range(w):
            D[x,z]=(C[(x-1)%5,z]+C[(x+1)%5,(z-1)%w])%2 
    for x in range(5):
        for y in range(5):
            for z in range(w):
                A1[x,y,z] = (A[x,y,z]+D[x,z])%2
            



def rnd(A,i):
    return f1(f2(f3(f4(f5(A)))),i)
    
    
def keccak_p_1600_24(S):
    A = convert_to_array(S)
    for i in range(12+2*l-24,12+2*l-1):
        A = rnd(A,i)
    S1 = convert_to_string(A)
    return S1
    
    

def pad(x,m):
    j = (-m-2)%x
    return [1]+[0]*j+[1]

def sponge(N,d,r = 1344):
    P = N+pad(r,len(N))
    n = len(P)/r
    c = b-r
    S = [0]*b
    for i in range(n):
        S = keccak_p_1600_24((S+P[i*r::(i+1)*r]+[0]*c)%2)
    Z = []
    while d >= len(Z):
        Z = Z+S[0::d]
        S = keccak_p_1600_24(S)
    return Z[0::d]
    

    
def Keccak_256(N,d):
    return sponge(Keccak_p[1600,24],pad,1600-256](N,d)

def SHAKE128(M,d):
    M=M+[1111]
    return Keccak_256(M+,d)

#compress/decompress

def compress(x,d,q = 7681):
    c=round(x*(2**d)/q)
    return modular_reduction1(c,2**d)

def decompress(x,d,q = 7681):
    return round(x*q/(2**d))

def compress_mat(mat,d,q = 7681):
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            mat[i][j] = compress(mat[i][j],d,q)
    return mat

def decompress_mat(mat,d,q = 7681):
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            mat[i][j] = decompress(mat[i][j],d,q)
    return mat
    

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
