import numpy as np
from scipy.interpolate import CubicSpline


def knots_gen(a,b,n):
    assert n>0, "n must be positive"
    assert b>a, "b must be greater than a"
    return np.linspace(a,b,n+1)

def Q_R_gen(knots):
    K=len(knots)-1
    Q=np.zeros((K+1,K-1))
    R=np.zeros((K-1,K-1))
    for k in range(K-1):
        h0=knots[k+1]-knots[k]
        h1=knots[k+2]-knots[k+1]
        Q[k,k]=1/h0
        Q[k+1,k]=-1/h0 - 1/h1
        Q[k+2,k]=1/h1
        R[k,k]=(h0+h1)/3
        if k<K-2:
            R[k+1,k]=h1/6
        if k>0:
            R[k-1,k]=h0/6 
    return Q,R

def K_gen(knots):
    Q,R=Q_R_gen(knots)
    return Q @ np.linalg.solve(R, Q.T)

def K_decomp(K):
    eigenvalues, eigenvectors = np.linalg.eigh(K)
    return eigenvalues, eigenvectors

class Spline:
    def __init__(self, g, knots):
        self.g = g
        self.knots = knots
        self.spline = CubicSpline(knots, g, bc_type='natural')
        self.left_slope, self.right_slope = self.get_slopes_ext()
        self.left_value, self.right_value = self.spline(knots[0]), self.spline(knots[-1])

    def get_slopes_ext(self):
        left_slope = self.spline(self.knots[0], nu=1)
        right_slope = self.spline(self.knots[-1], nu=1)
        return left_slope, right_slope

    def spline_derivative(self):
        grid = np.linspace(self.knots[0], self.knots[-1], 100)
        derivative_values = self.spline(grid, nu=1)
        return grid, derivative_values

    def __call__(self, h): 
        h_array = np.asarray(h, dtype=float)
        scalar_input = h_array.ndim == 0
        h_array = np.atleast_1d(h_array)

        result = np.empty_like(h_array, dtype=float)

        left_mask = h_array < self.knots[0]
        right_mask = h_array > self.knots[-1]
        inside_mask = ~(left_mask | right_mask)

        result[left_mask] = (
            self.left_value
            + self.left_slope * (h_array[left_mask] - self.knots[0])
        )

        result[right_mask] = (
            self.right_value
            + self.right_slope * (h_array[right_mask] - self.knots[-1])
        )

        result[inside_mask] = self.spline(h_array[inside_mask])

        if scalar_input:
            return result[0]

        return result