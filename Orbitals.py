# -*- coding: utf-8 -*-

" Practical representation of a level with quantum numbers n,l,j "
class Orbital(object):    
    def __init__(self, n, l, j=0):
        self.n = n
        self.l = l
        self.j = j
        # n[l]_j
        self.name = self.getName()   
        " if j=0, then work in the nl-basis "
        self.degeneracy = int(2*j+1) if j!=0 else 2*(2*l+1)
        # maybe generalize in future versions
        self.occupation = self.degeneracy
            
    def getName(self):
        ss = ""
        if self.l==0:
            ss="s"
        elif self.l==1:
            ss="p"
        elif self.l==2:
            ss="d"
        elif self.l==3:
            ss="f"
        elif self.l==4:
            ss="g"
        elif self.l==5:
            ss="h"
        elif self.l==6:
            ss="i"
        elif self.l==7:
            ss="l"          
        name = str(self.n) + ss
        if self.j!=0:
            jj = int(2*self.j)
            name += " " + str(jj) + "/2"
        return name
    
    def __str__(self):
        return self.getName()
    
    
    def sameLj(self, other):
        return (self.l==other.l and self.j==other.j)
    
    def __eq__(self,other):
        return ( self.sameLj(other) and self.n==other.n )
    
    
"  Set of orbitals; use like a list object "
class OrbitalSet(object):
    
    def __init__(self, list_orbs):
        self.list_orbs = list_orbs
        self.index = 0      # see __next__
        self.n_particles = self.countParticles()
        
        
    def __iter__(self):
        return self
    
    def __next__(self):
        try:
            result = self.list_orbs[self.index]
        except IndexError:
            raise StopIteration
        self.index += 1
        return result 
    
    def __getitem__(self,k):    
        return self.list_orbs[k]
         
    def __len__(self):
        return len(self.list_orbs)
    
    def __str__(self):
        st =""
        for oo in self.list_orbs:
            st += "\n" + str(oo)
        return st
                 
    def countParticles(self):
        s = 0
        for oo in self.list_orbs:
            s += oo.occupation
        return s

    def getLJD(self):
        L = [ self.list_orbs[l].l for l in range(len(self.list_orbs)) ]
        J = [ self.list_orbs[j].j for j in range(len(self.list_orbs)) ]
        D = [ self.list_orbs[d].degeneracy for d in range(len(self.list_orbs)) ]
        
        return (L, J, D)
    
    """
    Reset counter
    """
    def reset(self):
        self.index = 0      # see __next__
        self.n_particles = self.countParticles()
    
    

                
        
            


" Shell model basis "
# http://calistry.org/calculate/shellModel
class ShellModelBasis(OrbitalSet):
    def __init__(self):
        orbs = [  \
        Orbital(1,0,0.5),  Orbital(1,1,1.5),    Orbital(1,1,0.5),     Orbital(1,2,2.5),      Orbital(2,0,0.5), \
        Orbital(1,2,1.5),  Orbital(1,3,3.5),    Orbital(2,1,1.5),     Orbital(1,3,2.5),      Orbital(2,1,0.5), \
        Orbital(1,4,4.5),  Orbital(1,4,3.5),    Orbital(2,2,2.5),     Orbital(2,2,1.5),      Orbital(3,0,0.5), \
        Orbital(1,5,5.5),  Orbital(1,5,4.5),    Orbital(2,3,3.5),     Orbital(2,3,2.5),      Orbital(3,1,1.5), \
        Orbital(3,1,0.5),  Orbital(1,6,6.5),    Orbital(2,4,4.5),]
        # call parent constructor
        super().__init__(orbs)



class UncoupledHOBasis(OrbitalSet):
    def __init__(self):
        orbs = [  \
        Orbital(1,0),   Orbital(1,1),   Orbital(1,2),   Orbital(2,0),   Orbital(1,3),  \
        Orbital(2,1),   Orbital(1,4),   Orbital(2,2),   Orbital(3,0),   Orbital(1,5),  \
        Orbital(2,3),   Orbital(3,1),   Orbital(1,6),   Orbital(2,4),   Orbital(3,2),  \
        Orbital(4,0), ]
        # call parent constructor
        super().__init__(orbs)
        
        
        
"""
Returns the orbitals filled by n_particles in a given basis
"""   
def getOrbitalSet(n_particles, basis=ShellModelBasis() ):
    ll = []
    nn = n_particles
    for oo in basis:
        nn -= oo.occupation
        if nn>=0:
            ll.append(oo)
            if nn==0:
                break
        else:
            print ("Warning: not a closed-shell system in this basis")
            break
    return OrbitalSet(ll)
        
        

if __name__=="__main__":
    ss = getOrbitalSet(20)
    print (ss,'\n', ss.getLJD())
