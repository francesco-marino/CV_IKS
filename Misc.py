# -*- coding: utf-8 -*-


import pickle  # for input/output
from numpy import array
import numpy as np


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


"""
Read tabular file
"""
def read(filename):
    lists = []
    file = open(filename)
    lines = file.readlines()
    lines = [ll.strip() for ll in lines if len(ll.strip())>0]  # remove empty lines
    file.close()
    for line in lines:
        if str(line).startswith('#'):
              continue
        else:
            # parse line
            line = [ x for x in line.split() ]
            # lists 
            while len(lists)<len(line):
                lists.append( list() )
            # load data
            for j in range( len(lists) ):
                lists[j].append( line[j] )
    # convert to np.array
    for j in range(len(lists)):
        if is_float_try(lists[j][0]):
            lists[j] = np.array( lists[j], dtype=float )
    return lists


def is_float_try(stri):
    try:
        float(stri)
        return True
    except ValueError:
        return False




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