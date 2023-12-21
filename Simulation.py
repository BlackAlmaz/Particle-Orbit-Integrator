class Simulation:
    
    def __init__(self, n, dt, numGyro, sigma, delgam, mi, L, istep, c_omp, Ex, Ey, Ez, Bx, By, Bz):
        import numpy as np
        import scipy as sc
        import matplotlib
        import h5py
        import Particle
        import multiprocessing
        
        # PARAMETERS (MAIN METHOD)
        self.dt = dt
        self.Ex = Ex
        self.Ey = Ey
        self.Ez = Ez
        self.Bx = Bx
        self.By = By
        self.Bz = Bz
        self.istep = istep
        self.L = L
        self.numGyro = numGyro
        
        self.c = 0.225/dt
        c_old = 0.225
        
        self.wci = np.sqrt(sigma)*(1/mi)*(c_old/c_omp)*(1/np.sqrt(1 + 1/mi))
        
        x = 0
        
        self.wci /= dt
        
        if x == 0:
            delgam *= mi 
            m = 1
            q = -1
            self.wci *= mi
        else:
            m = mi
            q = 1
            
        self.T = 2 * np.pi/self.wci
        
        a = delgam # value for Maxwell-Boltzmann sampling
        
        tmax = int(numGyro * self.T)
        
        print(self.T)
        
        t = np.arange(0, tmax, dt)
        timeSteps = np.size(t)

        self.exint = np.full(timeSteps-1, None)
        self.eyint = np.full(timeSteps-1, None)
        self.ezint = np.full(timeSteps-1, None)
        
        self.bxint = np.full(timeSteps-1, None)
        self.byint = np.full(timeSteps-1, None)
        self.bzint = np.full(timeSteps-1, None)

        self.ptcls = np.full(n, None) 

        # Main Method
        for i in range(n): # spawn particles
            self.ptcls[i] = Particle.Particle(m,L,q, a, timeSteps)

        m = self.ptcls[0].getm()

        q = self.ptcls[0].getq()
        
        self.ptcls[0].setx(900.7555399473988, 0) # Initial seed
        self.ptcls[0].sety(350.4364737388687, 0)
        self.ptcls[0].setz(568.40211540535671, 0)


        for i in range(1, timeSteps): # Boris method loop
            for j in range(n):    
                x = self.ptcls[j].getx(i-1) # get particle positions
                y = self.ptcls[j].gety(i-1)
                z = self.ptcls[j].getz(i-1)
                vx = self.ptcls[j].getvx(i-1)
                vy = self.ptcls[j].getvy(i-1)
                vz = self.ptcls[j].getvz(i-1)

                #interpolate fields to particle positions
                
                interpolatedFields = self.interpolateFields(x,y,istep,L)

                Bxptcl = interpolatedFields[0]
                Byptcl = interpolatedFields[1]
                Bzptcl = interpolatedFields[2]
                Exptcl = interpolatedFields[3]
                Eyptcl = interpolatedFields[4]
                Ezptcl = interpolatedFields[5]

                self.exint[i-1] = Exptcl

                self.eyint[i-1] = Eyptcl

                self.ezint[i-1] = Ezptcl
                
                self.bxint[i-1] = Bxptcl

                self.byint[i-1] = Byptcl

                self.bzint[i-1] = Bzptcl
                
                borisArr = self.Boris(dt,x,y,z,vx,vy,vz,Exptcl,Eyptcl,Ezptcl,Bxptcl,Byptcl,Bzptcl,q,self.c,m)
                
                self.ptcls[j].setx(borisArr[0]%L,i);
                self.ptcls[j].sety(borisArr[2]%L,i);
                self.ptcls[j].setz(borisArr[4]%L,i);
                self.ptcls[j].setvx(borisArr[1],i);
                self.ptcls[j].setvy(borisArr[3],i);
                self.ptcls[j].setvz(borisArr[5],i);
                
    def getPtcls(self):
        return self.ptcls
    def getEint(self):
        return np.array([self.exint, self.eyint, self.ezint])
    def getBint(self):
        return np.array([self.bxint, self.byint, self.bzint])
    def getdt(self):
        return self.dt
    def getKE(self):
        import numpy as np
        KE = 0
        for ptcl in self.ptcls:
            v = np.sqrt(ptcl.getvxp()**2 + ptcl.getvyp()**2 + ptcl.getvzp()**2)
            gamma = 1/np.sqrt(1-v**2)
            KE += (gamma-1)*ptcl.getm()*self.c**2
        return KE
    def getT(self):
        return self.T
    def interpolateFields(self, x, y, istep, L):        
        import numpy as np
        
        i1 = int(x/istep)
        i2 = int(y/istep)

        boxSize = L/istep

        deltax = abs(x - istep * i1)/istep
        deltay = abs(y - istep * i2)/istep
        
        i11 = int((i1+1)%boxSize)
        i21 = int((i2+1)%boxSize)

        Bxxjk = 0.5*(self.Bx[i1][i2] + self.Bx[i1][i2-1]) + (0.5*(self.Bx[i11][i2]+self.Bx[i11][i2-1]) - 0.5*(self.Bx[i1][i2] + self.Bx[i1][i2-1]))*deltax

        Bxxj1k = 0.5*(self.Bx[i1][i21] + self.Bx[i1][i2]) + (0.5*(self.Bx[i11][i21]+self.Bx[i11][i2]) - 0.5*(self.Bx[i1][i21] + self.Bx[i1][i2]))*deltax

        Bxptcl = Bxxjk + (Bxxj1k - Bxxjk)*deltay

        Byxjk = 0.5*(self.By[i1][i2] + self.By[i1][i2-1]) + (0.5*(self.By[i11][i2]+self.By[i11][i2-1]) - 0.5*(self.By[i1][i2] + self.By[i1][i2-1]))*deltax

        Byxj1k = 0.5*(self.By[i1][i21] + self.By[i1][i2]) + (0.5*(self.By[i11][i21]+self.By[i11][i2]) - 0.5*(self.By[i1][i21] + self.By[i1][i2]))*deltax

        Byptcl = Byxjk + (Byxj1k - Byxjk)*deltay

        Bzxjk = 0.5*(self.Bz[i1][i2] + self.Bz[i1][i2-1]) + (0.5*(self.Bz[i11][i2]+self.Bz[i11][i2-1]) - 0.5*(self.Bz[i1][i2] + self.Bz[i1][i2-1]))*deltax

        Bzxj1k = 0.5*(self.Bz[i1][i21] + self.Bz[i1][i2]) + (0.5*(self.Bz[i11][i21]+self.Bz[i11][i2]) - 0.5*(self.Bz[i1][i21] + self.Bz[i1][i2]))*deltax

        Bzptcl = Bzxjk + (Bzxj1k - Bzxjk)*deltay

        Exxjk = 0.5*(self.Ex[i1][i2] + self.Ex[i1-1][i2] + (self.Ex[i11][i2] + self.Ex[i1-1][i2])*deltax) # Ex[x][j][k] interpolate over x

        Exxj1k = 0.5*(self.Ex[i1][i21] + self.Ex[i1-1][i21] + (self.Ex[i11][i21] + self.Ex[i1-1][i21])*deltax) # Ex[x][j+1][k]

        Exptcl = Exxjk + (Exxj1k - Exxjk)*deltay # Ex[x][y][k] perform interpolation over y

        Eyxjk = 0.5*(self.Ey[i1][i2] + self.Ey[i1-1][i2] + (self.Ey[i11][i2] + self.Ey[i1-1][i2])*deltax) # Ey[x][j][k] interpolate over x

        Eyxj1k = 0.5*(self.Ey[i1][i21] + self.Ey[i1-1][i21] + (self.Ey[i11][i21] + self.Ey[i1-1][i21])*deltax) # Ey[x][j+1][k]

        Eyptcl = Eyxjk + (Eyxj1k - Eyxjk)*deltay # Ey[x][y][k] perform interpolation over y

        Ezxjk = 0.5*(self.Ez[i1][i2] + self.Ez[i1-1][i2] + (self.Ez[i11][i2] + self.Ez[i1-1][i2])*deltax) # Ez[x][j][k] interpolate over x

        Ezxj1k = 0.5*(self.Ez[i1][i21] + self.Ez[i1-1][i21] + (self.Ez[i11][i21] + self.Ez[i1-1][i21])*deltax) # Ez[x][j+1][k]

        Ezptcl = Ezxjk + (Ezxj1k - Ezxjk)*deltay # Ez[x][y][k] perform interpolation over y

        return np.array([Bxptcl, Byptcl, Bzptcl, Exptcl, Eyptcl, Ezptcl])
    def plotPosition(self, i):
        from matplotlib import pyplot as plt
     
        xp = self.ptcls[i].getxp()
        yp = self.ptcls[i].getyp()
        plt.plot(xp, yp, 'o', color = 'black', markersize = 0.015, linewidth = 0.5)
        
    def computeVpar(self, i):
        import numpy as np
        Bxmean = np.array([np.mean(self.Bx[i]) for i in range(int(self.L/self.istep))])
        Bxmean2 = np.mean(Bxmean)
        Bymean = np.array([np.mean(self.By[i]) for i in range(int(self.L/self.istep))])
        Bymean2 = np.mean(Bymean)
        Bzmean = np.array([np.mean(self.Bz[i]) for i in range(int(self.L/self.istep))])
        Bzmean2 = np.mean(Bzmean)

        angle = np.arctan(Bymean2/Bxmean2)

        tmax = int(self.numGyro*self.T)

        vpar = np.full(int(tmax/self.dt), None)

        for j in range(0, int(tmax/self.dt)-1):
            v = np.array([self.ptcls[i].getvx(j),self.ptcls[i].getvy(j), self.ptcls[i].getvz(j)])
            b = np.array([Bxmean2, Bymean2, Bzmean2])
            bhat = b/np.linalg.norm(b)
            vpar[j] = np.dot(v,bhat)
            
        return vpar[:-1]
    
    def computeMagMoment(self, i):   # Magnetic Moment
        import numpy as np

        tmax = int(self.numGyro*self.T)

        mu = np.full(int(tmax/self.dt), None)

        for i in range(0, int(tmax/self.dt)-1):
            v = np.array([self.ptcls[i].getvx(j),self.ptcls[i].getvy(j), self.ptcls[i].getvz(j)])
            b = np.array(([self.Bx[i], self.By[i], self.Bz[i]]))
            bhat = b/np.linalg.norm(b)
            dot = np.dot(v,bhat)
            vperp = v - (dot*bhat)
            mu[i] = 0.5 * (ptcls[0].getm() * np.linalg.norm(vperp)**2)/np.linalg.norm(b)
            vperpa[i] = np.linalg.norm(vperp)
            ba[i] = np.linalg.norm(b)
        
        return mu
    
    def computeHits0(self, i): # Sign changes methods
        import numpy as np
        vpar = self.computeVpar(i)
        hits0 = 0
        for i in range(0,len(vpar)-1):
            if np.sign(vpar[i]) != np.sign(vpar[i+1]):
                hits0 = hits0+1
        return hits0
        
    def Boris(self, dt,x,y,z,vx,vy,vz,Ex,Ey,Ez,Bx,By,Bz,q,c,m):
        import numpy as np
        #-----------------------------------------
        #Propagates the position variables x,y,z and
        #the velocity variables vx,vy,vz from t to t+dt
        #for given E and B using Boris Method
        #-----------------------------------------
        qoverm = q/m
        dummy = 0.5 * qoverm # Bnorm contributes a factor of 1/dt

        vx *= c
        vy *= c
        vz *= c

        v = np.sqrt(vx**2 + vy**2 + vz**2)
        gamma = 1/np.sqrt(1-v**2/c**2)

        vx *= gamma/c

        vy *= gamma/c

        vz *= gamma/c

        ex0 = Ex * dummy 
        ey0 = Ey * dummy 
        ez0 = Ez * dummy

        dummy /= c

        bx = Bx * dummy 
        by = By * dummy
        bz = Bz * dummy

        #Half-Acceleration
        v0x = c*vx + ex0
        v0y = c*vy + ey0
        v0z = c*vz + ez0

        g = c / ((c**2 + v0x**2 + v0y**2 + v0z**2)**0.5)

        bx0 = bx * g
        by0 = by * g
        bz0 = bz * g

        dummy = 2 / (1 + bx0**2 + by0**2 + bz0**2)

        v1x = (v0x + v0y * bz0 - v0z * by0)*dummy
        v1y = (v0y + v0z * bx0 - v0x * bz0)*dummy
        v1z = (v0z + v0x * by0 - v0y * bx0)*dummy

        v0x = (v0x + v1y * bz0 - v1z * by0 + ex0)
        v0y = (v0y + v1z * bx0 - v1x * bz0 + ey0)
        v0z = (v0z + v1x * by0 - v1y * bx0 + ez0)

        g = c / ((c**2 + v0x**2 + v0y**2 + v0z**2)**0.5)

        v0x *= g
        v0y *= g    
        v0z *= g

        x1 = x + v0x * dt
        y1 = y + v0y * dt
        z1 = z + v0z * dt    

        v0x /= c
        v0y /= c
        v0z /= c

        return [x1,v0x,y1,v0y,z1,v0z]