#used the program-sketch from the computional physics CP lecture from physics bachelor 5S
#name: "01 - Singular value decomposition of a drawing"

# Created by:
# *Maximilian Gschaider
# *Sebastian Grosinger
# *Florian Wallner

#################################
#--> using the SVD as reduction schematic for PCA

import numpy as np

#################################

# Rayleigh Quotient, which is used to calculate eigenvalues in the power method
def rayleigh_quotient(A,x):
    
    numerator = np.dot(np.conjugate(x), np.dot(A,x))
    denominator = np.dot(np.conjugate(x), x)
    lam = numerator/denominator
    
    return lam

def power_method(B, tol, maxit, x_0):
    """

    Parameters
    ----------
    B     ... input matrix
    tol   ... possible value 1e-5
    maxit ... maximum number of iterations
    x_0   ... starting vector, may be set to None

    Returns
    -------
    eigenvalue  ... largest eigenvalue of matrix B
    eigenvector ... corresponding normalized eigenvector
    residuum    ... list of |lam(p) - lam(p-1)|

    """
    
    # Setting x to the input value x_0 --> random vector if x_0 = None
    if x_0 == None:
        eigenvector = np.random.rand(B.shape[1])
    else:
        eigenvector = x_0
    
    # Calculating the first eigenvalue with the Rayleigh-Quotient
    eigenvalue = rayleigh_quotient(B,eigenvector)
    
    residuum = []
    # Number of iterations 
    p = 0
    
    # Implementing the power method
    while p <= maxit:
        
        # New eigenvector
        eigenvector = np.dot(B,eigenvector)
        eigenvector = eigenvector/np.linalg.norm(eigenvector)
        
        # New eigenvalue with Rayleigh-Quotient
        eigenvalue_new = rayleigh_quotient(B, eigenvector)
        
        # Calculating the difference with the eigenvalue of the step before
        r_p = abs(eigenvalue_new - eigenvalue)
        residuum.append(r_p)
        
        # Updating for the next step
        eigenvalue = eigenvalue_new
        p += 1
        
        # The alghorithm breaks if the difference is below the tolerance
        if r_p <= tol:
            return eigenvalue, eigenvector, residuum
            break
        
        if p == maxit and r_p > tol:
            print('Not enough iterations to reach desired tolerance')
            return eigenvalue, eigenvector, residuum
        
#################################

# b) Calculating the n largest eigenvalues using power method and deflation


def largest_eigvals(B, n, tol, maxit, x_0):
    """

    Parameters
    ----------
    B     ... input matrix
    tol   ... possible value 1e-5
    maxit ... maximum number of iterations
    x_0   ... starting vector, may be set to None
    n     ... number of eigenvalues that will be calculated

    Returns
    -------
    eigenvalues  ... list of the n largest eigenvalues of matrix B
    eigenvectors ... list corresponding normalized eigenvectors
    residuums    ... list of residuums for every eigenvalue

    """
    
    eigenvalues = []
    eigenvectors = []
    residuums = []
    
    for _ in range(n):
        
        # Calculating the largest eigenvalue for B
        eigval, eigvec, res = power_method(B, tol, maxit, x_0)
        
        eigenvalues.append(eigval)
        eigenvectors.append(eigvec)
        residuums.append(res)
        
        # Deflating the matrix B
        B = B - eigval*np.outer(eigvec, np.conjugate(eigvec))
        
    return eigenvalues, eigenvectors, residuums


#################################

def gram_schmidt_modified(u_list):
    """
    
    Parameters
    ----------
    u_list ... list of non-orthonormalized vectors

    Returns
    -------
    v_list ... orthonormalized vectors

    """
    
    v_list = []
    
    for u_k in u_list:
        
        # Copy is necessary because the u_k would change too!
        v_k = np.copy(u_k)
        
        for v_n in v_list:
            
            # Implemeting (8) from excercise sheet
            v_k = v_k - (np.dot(v_k, v_n)/np.dot(v_n, v_n))*v_n
        
        v_k = v_k/np.linalg.norm(v_k)
        v_list.append(v_k)
    
    return v_list
   
        

def SVD(B, A, n, tol, maxit, x_0):
    """

    Parameters
    ----------
    B     ... A_conj*A
    A     ... matrix with dimensions (height, length) containing the intensities of a color
    n     ... number of calculated eigenvalues
    tol   ... tolerance for power method
    maxit ... maximum number of iterations for power method
    x_0 : ... starting vector, may be set to None

    Returns
    -------
    A_n   ... decomposited matrix of A

    """
    
    B_eigval, B_eigvec, _ = largest_eigvals(B, n, tol, maxit, x_0)
    # Making sure that the eigenvalues of B are orthonormal
    B_eigvec_orth = gram_schmidt_modified(B_eigvec)
    # The columns of V are the orthonormal eigenvectors of B
    V = np.column_stack(B_eigvec_orth)
    
    # Calculation of Lambda --> diagonal matrix with roots of eigenvalues of B
    LAMBDA = np.identity(n, dtype = np.cdouble)
    LAMBDA_inv = np.identity(n, dtype = np.cdouble)
    
    for i in range(n):
        LAMBDA[i][i] = np.sqrt(B_eigval[i])
    
    for i in range(n):
        LAMBDA_inv[i][i] = 1/np.sqrt(B_eigval[i])
    
    # U = A * V * Lambda_inverse
    U = np.dot(A, np.dot(V, LAMBDA_inv))
    
    # Calculating the decomposition of A
    A_n = np.dot(U, np.dot(LAMBDA, np.conjugate(np.transpose(V))))

    return A_n

#################################

tol = 1e-5
maxit = 1000000
x_0 = None

    
        
    
      
      
      
      
      
      


