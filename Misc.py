# -*- coding: utf-8 -*-


import pickle  # for input/output
from numpy import array


def saveData(filename, data):
    "Save an (almost) arbitrary python object to disc."
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    # done


def loadData(filename):
    "Load and return data saved to disc with the function `save_data`."
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data




" Simpson's integration rule "
def simpsonCoeff(i,N):
    if i==0 or i==N-1:
        cc=1.
    else:
        cc = 4. if i%2==1 else 2.
    return cc


"""
int( f(x) dx ) with a<=x<=b, using Simpson's rule
"""
def integral(f, a, b, h):
    f= array(f)
    N = int( (b-a)/h ) + 1
    #x = linspace(a,b,N)
    coeff = array([ simpsonCoeff( j, N ) for j in range(N) ]) * h/3.
    return sum(f*coeff)