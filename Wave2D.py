import numpy as np
import sympy as sp
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation

x, y, t = sp.symbols('x,y,t')

class Wave2D:

    def create_mesh(self, N, sparse=False):
        """Create 2D mesh and store in self.xij and self.yij"""
        self.xij, self.yij = np.meshgrid(np.linspace(0, 1.0, N + 1), np.linspace(0, 1.0, N + 1), indexing='ij')
        self.N = N
        self.h = 1.0/self.N
        #raise NotImplementedError

    def D2(self, N):
        """Return second order differentiation matrix"""
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (N + 1, N + 1), 'lil')
        return D
        #raise NotImplementedError

    @property
    def w(self):
        """Return the dispersion coefficient"""
        return np.pi*np.sqrt(self.mx**2 + self.my**2)
        #raise NotImplementedError

    def ue(self, mx, my):
        """Return the exact standing wave"""
        return sp.sin(mx*sp.pi*x)*sp.sin(my*sp.pi*y)*sp.cos(self.w*t)

    def initialize(self, N, mx, my):
        r"""Initialize the solution at $U^{n}$ and $U^{n-1}$
    
        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        mx, my : int
            Parameters for the standing wave
        """
        self.mx = mx
        self.my = my
        self.u = sp.lambdify((x, y, t), self.ue(mx, my))(self.xij, self.yij, 0)
        #raise NotImplementedError

    @property
    def dt(self):
        """Return the time step"""
        return self.cfl*self.h/self.c
        #raise NotImplementedError

    def l2_error(self, u, t0):
        """Return l2-error norm

        Parameters
        ----------
        u : array
            The solution mesh function
        t0 : number
            The time of the comparison
        """
        return np.sqrt(self.h*self.h*np.sum((sp.lambdify((x, y, t), self.ue(self.mx, self.my))(self.xij, self.yij, t0) - u)**2))
        #raise NotImplementedError

    def apply_bcs(self):
        self.Unp1[0] = 0
        self.Unp1[-1] = 0
        self.Unp1[:, -1] = 0
        self.Unp1[:, 0] = 0
        #raise NotImplementedError

    def __call__(self, N, Nt, cfl=0.5, c=1.0, mx=3, my=3, store_data=-1):
        """Solve the wave equation

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        Nt : int
            Number of time steps
        cfl : number
            The CFL number
        c : number
            The wave speed
        mx, my : int
            Parameters for the standing wave
        store_data : int
            Store the solution every store_data time step
            Note that if store_data is -1 then you should return the l2-error
            instead of data for plotting. This is used in `convergence_rates`.

        Returns
        -------
        If store_data > 0, then return a dictionary with key, value = timestep, solution
        If store_data == -1, then return the two-tuple (h, l2-error)
        """
        self.create_mesh(N)
        self.cfl = cfl
        self.c = c
        print(self.dt)
        self.initialize(N, mx, my)
        self.Unp1, self.Un, self.Unm1 = np.zeros((3, N+1, N+1))
        self.Unm1[:] = self.u
        D = self.D2(N)/(self.h*self.h)
        self.Un[:] = self.Unm1[:] + 0.5*(c*self.dt)**2*(D @ self.Unm1 + self.Unm1 @ D.T)
        self.Un[0] = 0
        self.Un[-1] = 0
        self.Un[:, -1] = 0
        self.Un[:, 0] = 0
        time = self.dt
        plotdata = {0: self.Unm1.copy()}
        errordata = []
        if store_data == 1:
            plotdata[1] = self.Un.copy()
            errordata[1] = self.l2_error(self.Un, time)
        for n in range(1, Nt):
            time += self.dt
            self.Unp1[:] = 2*self.Un - self.Unm1 + (c*self.dt)**2*(D @ self.Un + self.Un @ D.T)
            self.apply_bcs()
            self.Unm1[:] = self.Un
            self.Un[:] = self.Unp1
            if n % store_data == 0:
                plotdata[n] = self.Unm1.copy()
                errordata.append(self.l2_error(self.Un, time))
        if store_data == -1:
            return (self.h, errordata)
        else:
            return plotdata
        #raise NotImplementedError

    def convergence_rates(self, m=4, cfl=0.1, Nt=10, mx=3, my=3):
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretizations to use
        cfl : number
            The CFL number
        Nt : int
            The number of time steps to take
        mx, my : int
            Parameters for the standing wave

        Returns
        -------
        3-tuple of arrays. The arrays represent:
            0: the orders
            1: the l2-errors
            2: the mesh sizes
        """
        E = []
        h = []
        N0 = 8
        for m in range(m):
            dx, err = self(N0, Nt, cfl=cfl, mx=mx, my=my, store_data=-1)
            E.append(err[-1])
            h.append(dx)
            N0 *= 2
            Nt *= 2
        r = [np.log(E[i-1]/E[i])/np.log(h[i-1]/h[i]) for i in range(1, m+1, 1)]
        return r, np.array(E), np.array(h)

class Wave2D_Neumann(Wave2D):


    def D2(self, N):
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (N + 1, N + 1), 'lil')
        D[0, :2] = -2, 2
        D[-1, -2:] = 2, -2
        return D
        #raise NotImplementedError

    def ue(self, mx, my):
        return sp.cos(mx*sp.pi*x)*sp.cos(my*sp.pi*y)*sp.cos(self.w*t)
        #raise NotImplementedError

    @property
    def dt(self):
        """Return the time step"""
        return self.cfl*self.h/self.c
        #raise NotImplementedError

    def l2_error(self, u, t0):
        """Return l2-error norm

        Parameters
        ----------
        u : array
            The solution mesh function
        t0 : number
            The time of the comparison
        """
        return np.sqrt(self.h*self.h*np.sum((sp.lambdify((x, y, t), self.ue(self.mx, self.my))(self.xij, self.yij, t0) - u)**2))
        #raise NotImplementedError

    #def apply_bcs(self):
     #   raise NotImplementedError

    def __call__(self, N, Nt, cfl=0.5, c=1.0, mx=3, my=3, store_data=-1):
        """Solve the wave equation

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        Nt : int
            Number of time steps
        cfl : number
            The CFL number
        c : number
            The wave speed
        mx, my : int
            Parameters for the standing wave
        store_data : int
            Store the solution every store_data time step
            Note that if store_data is -1 then you should return the l2-error
            instead of data for plotting. This is used in `convergence_rates`.

        Returns
        -------
        If store_data > 0, then return a dictionary with key, value = timestep, solution
        If store_data == -1, then return the two-tuple (h, l2-error)
        """
        self.create_mesh(N)
        self.cfl = cfl
        self.c = c
        self.initialize(N, mx, my)
        self.Unp1, self.Un, self.Unm1 = np.zeros((3, N+1, N+1))
        self.Unm1[:] = self.u
        D = self.D2(N)/(self.h*self.h)
        self.Un[:] = self.Unm1[:] + 0.5*(c*self.dt)**2*(D @ self.Unm1 + self.Unm1 @ D.T)
        time = self.dt
        plotdata = {0: self.Unm1.copy()}
        errordata = []
        if store_data == 1:
            plotdata[1] = self.Un.copy()
            errordata.append(self.l2_error(self.Un, time))
        for n in range(1, Nt):
            time += self.dt
            self.Unp1[:] = 2*self.Un - self.Unm1 + (c*self.dt)**2*(D @ self.Un + self.Un @ D.T)
            #self.apply_bcs()
            self.Unm1[:] = self.Un
            self.Un[:] = self.Unp1
            if n % store_data == 0:
                plotdata[n] = self.Unm1.copy()
                errordata.append(self.l2_error(self.Un, time))
        if store_data == -1:
            return (self.h, errordata)
        else:
            return plotdata
        #raise NotImplementedError

    def convergence_rates(self, m=4, cfl=0.1, Nt=10, mx=3, my=3):
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretizations to use
        cfl : number
            The CFL number
        Nt : int
            The number of time steps to take
        mx, my : int
            Parameters for the standing wave

        Returns
        -------
        3-tuple of arrays. The arrays represent:
            0: the orders
            1: the l2-errors
            2: the mesh sizes
        """
        E = []
        h = []
        N0 = 8
        for m in range(m):
            dx, err = self(N0, Nt, cfl=cfl, mx=mx, my=my, store_data=-1)
            E.append(err[-1])
            h.append(dx)
            N0 *= 2
            Nt *= 2
        r = [np.log(E[i-1]/E[i])/np.log(h[i-1]/h[i]) for i in range(1, m+1, 1)]
        return r, np.array(E), np.array(h)
    
    def makeAni(self, N, Nt=10, mx=2, my=2):
        data = self(N, Nt, mx=mx, my=my, store_data=1)
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        frames = []
        for n, val in data.items():
            frame = ax.plot_wireframe(self.xij, self.yij, val, rstride=2, cstride=2);
            #frame = ax.plot_surface(xij, yij, val, vmin=-0.5*data[0].max(),
            #                        vmax=data[0].max(), cmap=cm.coolwarm,
            #                        linewidth=0, antialiased=False)
            frames.append([frame])

        ani = animation.ArtistAnimation(fig, frames, interval=400, blit=True, repeat_delay=1000)
        ani.save('wavemovie2dunstable.apng', writer='pillow', fps=5)

def test_convergence_wave2d():
    sol = Wave2D()
    r, E, h = sol.convergence_rates(mx=2, my=3)
    print(E)
    assert abs(r[-1]-2) < 1e-2

def test_convergence_wave2d_neumann():
    solN = Wave2D_Neumann()
    r, E, h = solN.convergence_rates(mx=2, my=3)
    assert abs(r[-1]-2) < 0.05

def test_exact_wave2d():
    sol = Wave2D()
    r, E, h = sol.convergence_rates(m=4, cfl=1/np.sqrt(2), mx=2, my=2)
    assert abs(E[-1]) < 1e-12
    #raise NotImplementedError

def makeAnimation():
    sol = Wave2D_Neumann()
    sol.makeAni(16, 10)

if __name__ == '__main__':
    #test_convergence_wave2d()
    #test_convergence_wave2d_neumann()
    #test_exact_wave2d()
    makeAnimation()
