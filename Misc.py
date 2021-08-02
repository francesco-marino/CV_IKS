# -*- coding: utf-8 -*-


import pickle  # for input/output
from numpy import array
import numpy as np
from scipy import interpolate  as interp


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
    
    
    
    
def getCutoff(rho, cut=1e-8):
        for rr in np.arange(0.,50.,0.1):
            if( rho(rr) < cut ):
                return rr - 0.1


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


# TODO add explanation
def floatCompare(a, floats, **kwargs):
  return np.any(np.isclose(a, floats, **kwargs))


