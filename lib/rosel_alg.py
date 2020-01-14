### Implementation of the Algorithm described in Rosel et al. 2012.
## Written by Nick Wright
# 9/15/17

import numpy as np
from scipy import optimize
import time


def spec_unmix(m_refl,scale_factor=1,srm=None):
    '''
    Solves the system of linear equations
        m_refl = srm * areafrac
    with m_refl and srm known.
    '''
    # srm: Spectral Reflectance Matrix
    # srm =[[ ao,  am,  ai],      (620-670mn)     band 1
    #       [ ao,  am,  ai],      (841-876nm)     band 2
    #       [ ao,  am,  ai],      (459-479nm)     band 3
    #       [ 1.,  1.,  1.]])
    # srm = None
    if srm is None:
        # Original Rosel table:
        srm = np.array([[.08, .16, .95], [.08, .07, .87], [.08, .22, .95], [1, 1, 1]])
        # For VIIRS, using perovich sheba data:
        # srm = np.array([[.07, .162, .756], [.06, .073, .642], [.028, .037, .153], [1, 1, 1]])
        # elcm:
        # srm = np.array([[0.050, 0.219, 0.673], [0.026, 0.066, 0.569], [0.094, 0.410, 0.691], [1,1,1]])
        # srm = np.array([[0.050, 0.21, 0.64], [0.026, 0.05, 0.569], [0.07, 0.39, 0.691], [1, 1, 1]])
        # srm = np.array([[0.04277512, 0.05038313, 0.68143116],
        #                 [0.00563799, 0.04467386, 0.53605999],
        #                 [0.14370061, 0.11760433, 0.79361057],
        #                 [1, 1, 1]])
    else:
        srm = np.array(srm)

    # print(srm)
    # Observed reflectance
    r = np.array([m_refl[0]*scale_factor,m_refl[1]*scale_factor,m_refl[2]*scale_factor,1])

    # Weights
    w = np.array([1,1,1,1])
    w = np.sqrt(np.diag(w))

    # Appy the weights to srm matrix and reflectance vector
    srm = np.dot(w,srm)
    r = np.dot(w,r)

    # Solve the system of linear equations using non-negative least squares
    #   result is a list: [water fraction, melt fraction, ice fraction]
    areafrac,residual = optimize.nnls(srm, r)

    # print(r)
    # print(areafrac)
    # print(residual)
    # print("-"*20)

    # Convert areafrac to an int-storable format
    # Scaling factor = 1000
    areafrac = [int(areafrac[0]*1000),int(areafrac[1]*1000),int(areafrac[2]*1000)]

    return areafrac


def ml_estimation(m_refl, model):
    m_refl = m_refl.reshape(1, -1)

    results = model.predict(m_refl)

    results = results[0]

    areafrac = [int(results[0]*1000),int(results[1]*1000),int(results[2]*1000)]

    return areafrac


# Optimizes the cost function 
# Accepts a tuple of (lambda1,lambda2,lambda3) reflectances 
# and the scaling factor of the data
def optimize_cost(m_refl,scale_factor,
                srm=[[.08,.16,.95],[.08,.07,.87],[.08,.22,.95],[1,1,1]]):
    
    # srm: Spectral Reflectance Matrix
    # srm =[[ ao,  am,  ai],      (620-670mn)     band 1
    #       [ ao,  am,  ai],      (841-876nm)     band 2
    #       [ ao,  am,  ai],      (459-479nm)     band 3
    #       [ 1.,  1.,  1.]])

    A = np.array(srm)
    # r = [m_refl[0]*scale_factor,m_refl[1]*scale_factor,m_refl[2]*scale_factor,1]
    # Constants
    # Defined in Rosel 2012: g = 150, w = 0.1
    # Gamma changes the 'steepness' of the curve at 0 and 1. Higher is steeper
    # Omega changes the total weight of the sigmoid function
    gamma = 2000
    omega = 0.1

    # Input reflectance
    R = [[m_refl[0]*scale_factor],[m_refl[1]*scale_factor],[m_refl[2]*scale_factor],[1]]

    # Initial guess [[Aw],[Am],[Ai]]
    x_guess = [[.25],[.25],[.25]]

    # Python (SciPy) implementaion of the bfgs optimization algorithm. 
    x_opt = optimize.fmin_bfgs(optimization_wrapper, x_guess, args=(A,R,gamma,omega), disp=False)

    # Convert x_opt to an int-storable format
    # Scaling factor = 1000
    x_opt = [int(x_opt[0]*1000),int(x_opt[1]*1000),int(x_opt[2]*1000)]

    return x_opt


# Wraps least squares and f_cost into a single function that can be called by
# the optimization algorithm.
def optimization_wrapper(x, A, R, gamma, omega):

    cost_vector = f_cost(x, A, R, gamma, omega)

    return least_squares(cost_vector)


## Least squares 
# Performs a least squares on the output of the f_cost function.
# This gives a single value that can be optimized by the 
# optimization function. 
def least_squares(vector):

    # Make sure the input is a numpy array
    vector = np.array(vector)

    # Perform the least squares operation
    ls_cost = np.sqrt(np.sum(np.power(vector,2)))

    return ls_cost


## Cost function from Rosel 2012
# x = vector of surface fractions: [Aw, Am, Ai]T
# A = matrix of spectral reflectance values (4x3)
# R = observed spectral values [lambda1,lambda3,lambda4,1]T
## Returns a 4x1 column vector with the cost associated with each of the four
# equations in eq. 2.
def f_cost(x, A, R, gamma, omega):

    # Making sure inputs are numpy arrays, since we are doing matrix
    # algebra below. 
    x = np.array(x)
    A = np.array(A)
    R = np.array(R)

    # x = np.append()
    # Reshape the x vector into a column. This is important for the
    #  matrix algebra in the next steps. 
    x = np.reshape(x,(3,1))

    # Pad the sigmoid function with a zero to make it 4x1.
    # sigmoid = np.append( ((1 - np.tanh(x*gamma) - np.tanh((x-1)*gamma)) * omega) ,[[0]],axis=0)
    # newsig:
    sigmoid = (1 - np.tanh(x*gamma) + np.tanh((x-1)*gamma)) + 1
    sigmoid = np.append(sigmoid, [[0]], axis=0)

    # Evaluate the system of equations and add the sigmoid function
    cost = (np.dot(A,x) - R) + sigmoid

    # Increase the cost of x array not adding to 1
    x_sum = x[0]+x[1]+x[2]

    # 10 chosen arbitrarily. Theoretically there should be infinite cost with things not adding to 1, but
    # this allows some flexibility. 
    sum_weight = np.array([[1],[1],[1],[10]])
    cost = cost*sum_weight

    # print x
    # print x_sum
    # print sigmoid
    # print cost
    # print "--"*40
    
    return cost

# Testing the algorithm
def main():

    start = time.clock()

    #array([[ 0.08,  0.16,  0.95],      (620-670mn)     band 1
    #       [ 0.08,  0.07,  0.87],      (841-876nm)     band 2
    #       [ 0.08,  0.22,  0.95],      (459-479nm)     band 3
    #       [ 1.  ,  1.  ,  1.  ]])

    # Spectral reflectance matrix.
    # Defined in Rosel 2012
    # A = np.array([[.08,.16,.95],[.08,.07,.87],[.08,.22,.95],[1,1,1]])
    # Testing new array
    # A = np.array([[.0106,.4088,.6212],[.007,.1328,.6319],[.0105,.4932,.6293],[1,1,1]])
    # A = np.array([[.0542,.2968,.6856],[.0279,.0889,.5084],[.1533,.5558,.8060],[1,1,1]])
    # Based on Perovich and Grenfel:
    # A = np.array([[.07,.162,.756],[.06,.073,.642],[.07,.188,.765],[1,1,1]])
    # Based on uncorrected method 2
    # A = np.array([[0.019,0.314,0.669],[0.015,0.172,0.641],[0.023,0.466,0.715],[1,1,1]])


    # refl = [0.1032, 0.1448, 0.1576]
    # refl = [0.0384, 0.0872, 0.2776]
    # refl = [.5950,.4849,.6821]
    # refl = [.9,.9,.9]
    # refl = [0.539, 0.472, 0.557]

    refl = [.4633, .8558, .7222]
    refl = [0.368, 0.431, 0.180]

    refl = [0.553, 0.408, 0.6]

    # refl = [0.8, 0.8, 0.8]
    # m_refl = [8823 8981 8526]
    A = np.array([[0,.16,.95],[0,.07,.87],[0,.22,.95],[1,1,1]])
    x_opt = optimize_cost(refl, 1, A)
    # x_opt = optimize_cost([[0.54],[0.44],[0.2],[1]])

    print(x_opt)
    #print np.sum(x_opt)

    end = time.clock()
    elapsed = end - start
    print("{} per iteration".format(elapsed))
    print("{} hours per image".format((elapsed*35781136)/(60*60)))

    refl.append(1)

    start = time.clock()
    
    # x_opt = optimize.lsq_linear(A, refl, bounds=(0,1))
    x_opt = optimize.nnls(A, refl)
    print(x_opt)
    print(np.sum(x_opt[0]))

    end = time.clock()
    elapsed = end - start
    print("{} per iteration".format(elapsed))
    print("{} hours per image".format((elapsed*35781136)/(60*60)))


if __name__ == '__main__':

    main()