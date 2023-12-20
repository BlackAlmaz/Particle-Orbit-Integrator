class Particle:
    
    def __init__(self, m, L, q, a, timeSteps):
        
        import scipy as sc
        from scipy.stats import maxwell
        import numpy as np
        import random
        from scipy import integrate
        from scipy import optimize

        self.m = m
        self.q = q
        
        vmag0 = maxwellJuttnerSample(a) # sample the velocity magnitude
        
        theta = 2 * np.pi * random.random()
        phi = np.pi * random.random()
        
        v0 = (vmag0*np.sin(theta)*np.cos(phi),vmag0*np.sin(theta)*np.sin(phi),vmag0*np.cos(theta)) # randomize the velocity components
        
        self.xp    = np.zeros(timeSteps)
        self.xp[0] = L * random.random()
        self.yp    = np.zeros(timeSteps)
        self.yp[0] = L * random.random()
        self.zp    = np.zeros(timeSteps)
        self.zp[0] = L * random.random()
        self.vx    = np.zeros(timeSteps)  
        self.vx[0] = v0[0]
        self.vy    = np.zeros(timeSteps)
        self.vy[0] = v0[1]
        self.vz    = np.zeros(timeSteps)
        self.vz[0] = v0[2]
        
        print("Position is " + str(self.xp[0]) + "," + str(self.yp[0]) + "," + str(self.zp[0]))
        print("Velocity is " + str(self.vx[0]) + "," + str(self.vy[0]) + "," + str(self.vz[0]))
                
    def setx(self, x, i):
        self.xp[i] = x
        
    def sety(self, y, i):
        self.yp[i] = y
        
    def setz(self, z, i):
        self.zp[i] = z
        
    def setvx(self, vx, i):
        self.vx[i] = vx
        
    def setvy(self, vy, i):
        self.vy[i] = vy
        
    def setvz(self, vz, i):
        self.vz[i] = vz
        
    def getvxp(self):
        return self.vx
    def getvyp(self):
        return self.vy
    def getvzp(self):
        return self.vz
        
    def getx(self, i):
        return self.xp[i]
        
    def gety(self, i):
        return self.yp[i]
        
    def getz(self, i):
        return self.zp[i]
        
    def getvx(self, i):
        return self.vx[i]
    
    def getxp(self):
        return self.xp
    
    def getyp(self):
        return self.yp
    
    def getzp(self):
        return self.zp
    
    def getvy(self, i):
        return self.vy[i]
        
    def getvz(self, i):
        return self.vz[i]
    
    def getm(self):    
        return self.m
    
    def getq(self):
        return self.q
    
    
def maxwellJuttnerSample(simDelgam): # sample the velocities
    import random
    from scipy import optimize
    from scipy import integrate
    import scipy as sc
    import numpy as np
    
    c = 0.225
            
    def maxwllJuttnerDist(vgiven): # define the Maxwell-Juttner distribution
        const = 1/(simDelgam * 8**2 * c**3 * sc.special.kn(2, 1/simDelgam))
        gamma = 1/np.sqrt(1 - vgiven**2/c**2)
        prob = vgiven * gamma * np.exp(-gamma/simDelgam)
        return prob

    def cumulativeMaxwell(v): # compute the cumulative distribution
        vmax = 0.999*c
        if(v >= vmax):
            return 1
        num = integrate.quad(lambda x : maxwllJuttnerDist(x), 0, v)[0]
        dem = integrate.quad(lambda x : maxwllJuttnerDist(x), 0, vmax)[0]
        return num/dem
    
    X = random.random()
    
    print(X)
    
    velocity = optimize.fsolve(lambda x: cumulativeMaxwell(x)-X,0.04)[0] # solve for the root
        
    print(velocity/c)

    return velocity/c