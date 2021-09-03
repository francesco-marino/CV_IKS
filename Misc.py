# -*- coding: utf-8 -*-


import pickle  # for input/output
from numpy import array
import numpy as np
from scipy import interpolate as interp
from scipy import integrate


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
Interpolation of a function f in its (discrete) domain r
"""  
def interpolate(r, f, get_der=False):
    r=np.array(r); f=np.array(f);
    assert( len(r)==len(f) )
     
    tck  = interp.splrep(r, f)
    ff = lambda x: interp.splev(x, tck )
    
    if(get_der == False):
        return ff    
    else: 
        d1 = lambda x: interp.splev(x, tck, der=1)  
        return ff, d1


"""
Compare density with a given cutoff value
"""
def getCutoff(rho, cut=1e-9):
    for rr in np.arange(0.,50.,0.1):
        if( rho(rr) < cut ):
            return rr - 0.1
        
        
"""
Computes the integral function F(x)= \int_x0 ^x f(t) dt.
Parameters
----------
x: np.array
    independent variable
f: np.array
    values of the function
"""
def integralFunction(x, f):
    assert (x.shape[0]==f.shape[0])
    integ = np.zeros_like(x)
    for j in range(x.shape[0]):
        integ[j] = integrate.simps(f[:j+1], x[:j+1])
    return integ
    
    

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


"""
Check if string can be converted to float
"""
def is_float_try(stri):
    try:
        float(stri)
        return True
    except ValueError:
        return False


"""
Check whether an array of float contains a certain value
Parameters
----------
    a: list or np.array
    floats: float
        values to be searched
"""
def floatCompare(a, floats, **kwargs):
  return np.any(np.isclose(a, floats, **kwargs))


"""
Find the intial position of a 'plateu' in an array v.
The array is scanned seeking a sequence of points where the variation
between elements is below a given treshhold.
The position j of the first element that satifies these conditions is returned.
Parameters
----------
    v: list of np.array
    n: int
        minimal length of a plateu
    tol: float
        max. diff. between two elements allowed 
    st: int
        starting position
"""
def findPlateu(v, n=3, tol=0.05, st=0):
    v = np.array(v)
    for j in range(st, v.shape[0] ):
        flag = True
        for k in range(0,n+1):
            #print(j,j+k)
            ind = np.min((v.shape[0]-1, j+k))
            if np.abs(v[j]-v[ind]) > tol:
                flag = False
                break
        if flag == True:
            return j
        
    print("No plateau with the inserted conditions has been found, returning the ten last index")
    return -10