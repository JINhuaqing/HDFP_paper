# this file contains fn for B-spline
import numpy as np
from rpy2 import robjects as robj

_r = robj.r

def obt_bsp_basis_Rfn(x, iknots, bknots, bsp_ord, intercept=1):
    """
        Obtain the b-spline basis for given knots and degree
        args:
            x: the locs you want to evaluate
            iknots: inner knots
            bknots: boundary knots
            bsp_ord: the order of b-spline; degree = order-1
            intercept: whether including intercept or not, i.e., the first col of the basis
                       it means the intercept of the polynomial.
    """
    _r["library"]("splines")
    _r_bs = _r['bs']
    iknots_rvec = robj.FloatVector(iknots)
    bknots_rvec = robj.FloatVector(bknots)
    x_rvec = robj.FloatVector(x)
    bsis_r = _r_bs(x_rvec, 
                  knots=iknots_rvec, 
                  degree=bsp_ord-1, 
                  Boundary_knots=bknots_rvec, 
                  intercept=intercept)
    return np.matrix(bsis_r)



def obt_bsp_basis_Rfn_wrapper(x, N, bsp_ord, intercept=1):
    """
        Obtain the b-spline basis for given Num of basis and degree
        args:
            x: the locs you want to evaluate
            N: Num of basis
            bsp_ord: the order of b-spline; degree = order-1
            intercept: whether including intercept or not, i.e., the first col of the basis
                       it means the intercept of the polynomial.
    """
    aknots_raw = np.linspace(0, 1, N-(bsp_ord-2))
    iknots = aknots_raw[1:-1]
    bknots = np.array([0, 1])
    basis_mat = obt_bsp_basis_Rfn(x, iknots, bknots, bsp_ord, intercept)
    return np.array(basis_mat)

def obt_bsp_obasis_Rfn(x, N, bsp_ord):
    """
        Obtain the b-spline basis for given Num of basis and degree
        args:
            x: the locs you want to evaluate
            N: Num of basis
            bsp_ord: the order of b-spline; degree = order-1
            
    """
    _r["library"]("orthogonalsplinebasis");
    knots = np.linspace(0, 1, N-(bsp_ord-2))
    eknots = _r["expand.knots"](robj.FloatVector(knots), order=bsp_ord);
    
    #basis_obj = _r['SplineBasis'](eknots, order=bsp_ord)
    basis_obj = _r['OBasis'](eknots, order=bsp_ord)
    basis_mat = _r['evaluate'](basis_obj, robj.FloatVector(x));
    return np.array(basis_mat)
