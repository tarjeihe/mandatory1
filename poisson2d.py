import numpy as np
import sympy as sp
import scipy.sparse as sparse

x, y = sp.symbols('x,y')

class Poisson2D:
    r"""Solve Poisson's equation in 2D::

        \nabla^2 u(x, y) = f(x, y), in [0, L]^2

    where L is the length of the domain in both x and y directions.
    Dirichlet boundary conditions are used for the entire boundary.
    The Dirichlet values depend on the chosen manufactured solution.

    """

    def __init__(self, L, ue):
        """Initialize Poisson solver for the method of manufactured solutions

        Parameters
        ----------
        L : number
            The length of the domain in both x and y directions
        ue : Sympy function
            The analytical solution used with the method of manufactured solutions.
            ue is used to compute the right hand side function f.
        """
        self.L = L
        self.ue = ue
        self.f = sp.diff(self.ue, x, 2)+sp.diff(self.ue, y, 2)

    def create_mesh(self, N):
        """Create 2D mesh and store in self.xij and self.yij"""
        self.xij, self.yij = np.meshgrid(np.linspace(0, self.L, N + 1), np.linspace(0, self.L, N + 1), indexing='ij')
        self.N = N
        self.h = self.L/self.N
        #raise NotImplementedError

    def D2(self):
        """Return second order differentiation matrix"""
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (self.N + 1, self.N + 1))
        return D
        #raise NotImplementedError

    def laplace(self):
        """Return vectorized Laplace operator"""
        Dx = sparse.kron(1.0/(self.h*self.h)*self.D2(), sparse.eye(self.N + 1))
        Dy = sparse.kron(sparse.eye(self.N + 1), 1.0/(self.h*self.h)*self.D2())
        return Dx + Dy
        #raise NotImplementedError

    def get_boundary_indices(self):
        """Return indices of vectorized matrix that belongs to the boundary"""
        B = np.ones((self.N + 1, self.N + 1), dtype=bool)
        B[1:-1, 1:-1] = 0
        bnds = np.where(B.ravel() == 1)[0]
        return bnds
        #raise NotImplementedError

    def assemble(self):
        """Return assembled matrix A and right hand side vector b"""
        A = self.laplace()
        bnds = self.get_boundary_indices()
        A = A.tolil()
        for i in bnds:
            A[i] = 0
            A[i, i] = 1
        A = A.tocsr()
        F = sp.lambdify((x, y), self.f)(self.xij, self.yij)
        b = F.ravel()
        u_exact = sp.lambdify((x, y), self.ue)(self.xij, self.yij)
        u_raveled = u_exact.ravel()
        for i in bnds:
            b[i] = u_raveled[i]
        return A, b
        #raise NotImplementedError

    def l2_error(self, u):
        """Return l2-error norm"""
        u_exact = sp.lambdify((x, y), self.ue)(self.xij, self.yij)
        en = np.sqrt(self.h*self.h*np.sum((u_exact.ravel() - u)**2))
        return en
        #raise NotImplementedError

    def __call__(self, N):
        """Solve Poisson's equation.

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction

        Returns
        -------
        The solution as a Numpy array

        """
        self.create_mesh(N)
        A, b = self.assemble()
        self.U = sparse.linalg.spsolve(A, b.flatten()).reshape((N+1, N+1))
        return self.U

    def convergence_rates(self, m=6):
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretization levels to use

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
            u = self(N0)
            E.append(self.l2_error(u))
            h.append(self.h)
            N0 *= 2
        r = [np.log(E[i-1]/E[i])/np.log(h[i-1]/h[i]) for i in range(1, m+1, 1)]
        return r, np.array(E), np.array(h)

    def Lagrangebasis(xj, x=x):
        """Construct Lagrange basis for points in xj
    
        Parameters
        ----------
        xj : array
            Interpolation points (nodes)
        x : Sympy Symbol
    
        Returns
        -------
        Lagrange basis as a list of Sympy functions
        """
        from sympy import Mul
        n = len(xj)
        ell = []
        numert = Mul(*[x - xj[i] for i in range(n)])
        for i in range(n):
            numer = numert/(x - xj[i])
            denom = Mul(*[(xj[i] - xj[j]) for j in range(n) if i != j])
            ell.append(numer/denom)
        return ell

    def Lagrangefunction2D(u, basisx, basisy):
        N, M = u.shape
        f = 0
        for i in range(N):
            for j in range(M):
                f += basisx[i]*basisy[j]*u[i, j]
        return f

    def eval(self, xVal, yVal):
        """Return u(x, y)

        Parameters
        ----------
        x, y : numbers
            The coordinates for evaluation

        Returns
        -------
        The value of u(x, y)

        """
        indXm1 = int(np.floor(x/self.h))
        indXp1 = int(np.ceil(x/self.h))
        indYm1 = int(np.floor(y/self.h))
        indYp1 = int(np.ceil(y/self.h))

        lx = Lagrangebasis(self.xij[indXm1:indXp1, 0], x=x)
        ly = Lagrangebasis(self.yij[0, indYm1:indYp1], x=y)
        f = Lagrangefunction2D(self.U[indXm1:indXp1, indYm1:indYp1], lx, ly)
        f.subs({x: xVal, y: yVal})
        #raise NotImplementedError

def test_convergence_poisson2d():
    # This exact solution is NOT zero on the entire boundary
    ue = sp.exp(sp.cos(4*sp.pi*x)*sp.sin(2*sp.pi*y))
    sol = Poisson2D(1, ue)
    r, E, h = sol.convergence_rates()
    assert abs(r[-1]-2) < 1e-2

def test_interpolation():
    ue = sp.exp(sp.cos(4*sp.pi*x)*sp.sin(2*sp.pi*y))
    sol = Poisson2D(1, ue)
    U = sol(100)
    assert abs(sol.eval(0.52, 0.63) - ue.subs({x: 0.52, y: 0.63}).n()) < 1e-3
    assert abs(sol.eval(sol.h/2, 1-sol.h/2) - ue.subs({x: sol.h/2, y: 1-sol.h/2}).n()) < 1e-3

if __name__ == '__main__':
    test_convergence_poisson2d()
    test_interpolation()
    


