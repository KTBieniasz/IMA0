#!/usr/bin/env python3

from numpy import *
from functools import lru_cache #For memoization
from scipy.linalg import inv
from mpmath import hyp2f1
#from scipy.special import hyp2f1 #Seems to be faster but numerically unstable

#### Free propagators

@lru_cache(maxsize=None)
def G0(omega,t,a):
    '''2D Lattice Green's functions, as derived by Morita in
    J. Math. Phys. 12, 1744;
    J. Phys. Soc. Jpn. 30, 957'''
    a = -sort(-abs(array(a))) #convert to array and apply symmetries
    s = 2*t
    if abs(t)<1e-6 or len(a)==0:#0D solution for t=0
        if all(a==0):
            return 1/omega
        else:
            return 0j
    elif len(a)==2:
        m = (2*s/omega)**2
        (i,j) = a
        # Basic G0 functions
        if (i==0 and j==0):
            R = (1/omega)*hyp2f1(1/2,1/2,1,m)
        elif (i==1 and j==1):
            R = (1/omega)*m/8*hyp2f1(3/2,3/2,3,m)
            #(hyp2f1(3/2,1/2,2,m)-hyp2f1(1/2,1/2,1,m))
        elif (i==1 and j==0):
            R = (omega*G0(omega,t,(0,0))-1)/2/s
        # Recursive definitions
        elif (i>1 and j==i):
            #R = binom(2*j,j)/2**(4*j+1)/omega*m**j*hyp2f1(j+1/2,j+1/2,2*j+1,m)
            R = (4*(i-1)*(2/m-1)*G0(omega,t,(i-1,i-1))
                  -(2*i-3)*G0(omega,t,(i-2,i-2)))/(2*i-1)
        elif (i>1 and j==i-1):
            R = (omega*G0(omega,t,(j,j))-s*G0(omega,t,(j,j-1)))/s
        else:
            R = (2*omega*G0(omega,t,(i-1,j))/s
                    -(G0(omega,t,(i-2,j))+G0(omega,t,(i-1,j+1))
                        +G0(omega,t,(i-1,j-1))))
        return complex(R)
    elif len(a)==1:
        i = a[0]
        m = omega/s
        xi = -m + sqrt(m**2-1)
        return xi**i/sqrt(m**2-1)/s
#### Old implementation using recurrence relations
        # if i==0:
        #     return -1j*sign(imag(sqrt((1+m)/(1-m))))/(s*sqrt(1-m**2))
        # if i==1:
        #     return (omega*G0(omega,t,(0,))-1)/s
        # if i>1:
        #     return 2*m*G0(omega,t,(i-1,))-G0(omega,t,(i-2,))
    else:
        raise NotImplementedError("Higher dimensions are not implemented here.")


@lru_cache(maxsize=None)
def Gu(z,t,U,end,start):
    """Free propagator corrected for core hole potential U."""
    dim = len(start)
    centre = (0,)*dim
    z = z - U
    diff = tuple(array(end)-array(start))
    if U==0:
        return G0(z,t,diff)
    else:
        return G0(z,t,diff)-G0(z,t,end)*G0(z,t,start)/(1./U+G0(z,t,centre))

#### MA0 continued fraction coefficients

@lru_cache(maxsize=None)
def A1(j,z,t,U,w0,M):
    """
    Continued fraction first coefficient for the zero phonon Green's function in the IMA0 approximation.
    Based on EPL 89, 37007 (2010); PRB 85, 165117 (2012).
    """
    nph = max(4,int(ceil((M/w0)**2)))
    A = array([(n, M*Gu(z-n*w0,t,U,j,j)) for n in range(nph,0,-1)])
    A0 = reduce(lambda x,y: product(y)/(1-y[1]*product(x)), A[1:])
    A1 = reduce(lambda x,y: product(y)/(1-y[1]*product(x)), A)
    while abs((A1-A0)/A1)>1e-16:
        nph *= 2
        A = array([(n, M*Gu(z-n*w0,t,U,j,j)) for n in range(nph,0,-1)])
        A0 = A1
        A1 = reduce(lambda x,y: product(y)/(1-y[1]*product(x)), A)
    return A1


@lru_cache(maxsize=None)
def An(j,z,t,U,w0,M):
    """
    Continued fraction all coefficients for all generalized Green's functions in the IMA0 approximation.
    Based on EPL 89, 37007 (2010); PRB 85, 165117 (2012).
    """
    nph = 4*int(ceil((M/w0)**2))
    A = array([(n, M*Gu(z-n*w0,t,U,j,j)) for n in range(nph,0,-1)])
    A0 = reduce(lambda x,y: product(y)/(1-y[1]*product(x)), A[1:])
    An = list(it.accumulate(A,
                            lambda x,y: product(y)/(1-y[1]*x),
                            initial=0.))
    A1 = An[-1]
    while abs((A1-A0)/A1)>1e-16:
        nph *= 2
        A = array([(n, M*Gu(z-n*w0,t,U,j,j)) for n in range(nph,0,-1)])
        An = list(it.accumulate(A,
                                lambda x,y: product(y)/(1-y[1]*x),
                                initial=0.))
        A0 = A1
        A1 = An[-1]
    return array(An[:0:-1])/sqrt(A[::-1,0]) #For Fn normalized by sqrt(n!)

#### Full interacting propagators in IMA0

@lru_cache(maxsize=None)
def GMA0(z,t,U,w0,Me,Mh,end,start,p=0):
    """
    Zero phonon Green's function for a Holstein polaron in the IMA0 approximation.
    Based on EPL 89, 37007 (2010); PRB 85, 165117 (2012).
    """
    M = sqrt(Me**2 + Mh**2)
    m = Me + Mh
    dim = len(start)
    centre = (0,)*dim
    SigMA = M*A1(centre,z,t,0.,w0,M)
    sites = neighbor([centre], p)
    vl = diag([M*A1(d,z,t,U,w0,M)-SigMA if d!=centre
                   else m*A1(d,z,t,U,w0,m)-SigMA
                   for d in sites])
    glk = array([[Gu(z-SigMA,t,U,a,b) for b in sites] for a in sites])
    gik = array([Gu(z-SigMA,t,U,end,b) for b in sites])
    Gil = gik.dot(inv(identity(len(sites))-vl.dot(glk)))
    glj = array([[Gu(z-SigMA,t,U,a,start)] for a in sites])
    gij = Gu(z-SigMA,t,U,end,start)
    Gij = gij + Gil.dot(vl).dot(glj)
    return complex(Gij)


@lru_cache(maxsize=None)
def FMA0(z,q,t,U,w0,Me,Mh=0,p=0):
    """
    All Fn generalized Green's functions for a Holstein polaron in the IMA0 approximation.
    Based on EPL 89, 37007 (2010); PRB 85, 165117 (2012).
    """
    M = sqrt(Me**2 + Mh**2)
    m = Me + Mh
    q = array(q)
    dim = len(q)
    centre = (0,)*dim
    SigMA = M*A1(centre,z,t,0.,w0,M)
    sites = neighbor([centre], p)
    ann = [An(d,z,t,U,w0,M) if d!=centre
               else An(d,z,t,U,w0,m)
               for d in sites] #for the Fn functions
    nn = min([len(a) for a in ann])
    ann = array([a[:nn] for a in ann])
    vl = array([M*An(d,z,t,U,w0,M)[0] if d!=centre
               else m*An(d,z,t,U,w0,m)[0]
               for d in sites])-SigMA
    glk = array([[Gu(z-SigMA,t,U,a,b) for b in sites] for a in sites])
    gik = array([Gu(z-SigMA,t,U,centre,b) for b in sites])
    Gil = gik.dot(inv(identity(len(sites))-vl.dot(glk)))
    gli = array([[exp(-1j*q.dot(a))*Gu(z-n*w0,t,U,a,centre)/Gu(z-n*w0,t,U,a,a)
                      for a in sites] for n in range(1,nn+1)])
    Fnil = array(list(it.accumulate(ann.T,op.mul,initial=Gil)))
    Fnil[1:] = Fnil[1:]*gli #add initial propagator and phase factor
    Fnil[0] = GMA0(z,t,U,w0,Me,Mh,centre,centre,p) #replace G_il with G_ii
    return Fnil

#### Helper functions

def delta(d):
    '''Returns an array of all NN vectors for a hypercubic d-dimensional lattice,
    ordered (x,y,...,-x,-y,...)'''
    return vstack([identity(d,int),-identity(d,int)])

def neighbor(B,n=1):
    '''Returns all the sites lying within n lattice constants from the sites B, including the sites themselves.'''
    if not B:
        return set(B)
    if n == 1:
        d = len(list(B)[0])
        D = delta(d)
        B = sorted(B)
        N = set([tuple(x+y) for x in B for y in D])
        return N|set(B)
    elif n > 1:
        N = neighbor(B,n-1)
        return neighbor(N)|N
    else:
        return set(B)
