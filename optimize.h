/*
    LA: linear algebra C++ interface library
    Copyright (C) 2021 Jiri Pittner <jiri.pittner@jh-inst.cas.cz> or <jiri@pittnerovi.com>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/


//various optimization methods

#ifndef _OPTIMIZE_H_
#define _OPTIMIZE_H_

#include <stdint.h>
#include <cstdarg>
#include "vec.h"
#include "mat.h"
#include "smat.h"


namespace LA {

//base classes with purely virtual functions for multiple inheritance use

//optimization problem - provides function and optional gradient, in multiply inherited classess will store auxiliary data for the function
template<typename T>
class OptProblem {
public:
virtual T funcgrad(const NRVec<T> &x, NRVec<T> *grad) const =0; //evaluate function and optionally also gradient at point x (if grad!=NULL)
virtual ~OptProblem() {};
};

//test case
template<typename T>
class testproblem : public OptProblem<T> {
public:
virtual T funcgrad(const NRVec<T> &x, NRVec<T> *grad) const;
virtual ~testproblem() {};
};


//base classes for optimization methods

//this is a collection of parameters, which can be augmented by method-specific additions, to be used as a base class
//methods obtain their specific parameters via dynamic_cast, missing ones are taken from the default constructor
template<typename T>
class hyperparameters {
public:
int my_maxstep; //maximum number of iterations
int my_maxfunc; //maximum number of function evaluations, <=0 means unlimited
T my_f_thr; //threshhold on function change in a single step
T my_x_thr; //threshold on x norm change in a single step
int my_verbose; //0=silent, 1=per-iteration info, 2=more detailed info
hyperparameters() : my_maxstep(200), my_maxfunc(0), my_f_thr(0), my_x_thr(0), my_verbose(1) {};
virtual ~hyperparameters(){};
};

//parameters common to methods using the gradient
template<typename T>
class gradient_hyperparameters : public hyperparameters<T> {
public:
T my_g_thr; //gradient norm convergence threshold
gradient_hyperparameters() : my_g_thr(1e-6) {};
};


//result of the minimization, converts implicitly to the function value
template<typename T>
class optresult {
public:
T f; //final function value
bool converged;
optresult(const T ff, const bool c) : f(ff), converged(c) {};
operator T() const {return f;}
};


//base class for optimization methods, virtual functions to allow a pointer to derived class
template<typename T>
class Optimizer {
public:
//par==NULL means default parameters
//if achieved!=NULL, the values actually reached are returned there: my_maxstep = no. of iterations done, my_maxfunc = no. of function evaluations done, thresholds = last values of the convergence criteria
//(|f change| and |x change| in the last step, 0 if no step was done; |gradient| for gradient methods), the other parameters as they were used
virtual optresult<T> minimize(const OptProblem<T> *o, NRVec<T> &x, const hyperparameters<T> *par=NULL, hyperparameters<T> *achieved=NULL) = 0;
virtual ~Optimizer() {}; 
};


//auxiliary routines

//line search along direction p satisfying strong Wolfe conditions (cf. Nocedal & Wright, Numerical Optimization, 2nd ed., alg. 3.5 and 3.6)
//on input x,f,g at the starting point, on successful output x,f,g at the new point; returns step length or 0 on failure (x,f,g then unchanged)
//maxiter = max. number of function evaluations, nfunc is incremented by the number of evaluations done
template<typename T>
T linesearch_wolfe(const OptProblem<T> *o, NRVec<T> &x, T &f, NRVec<T> &g, const NRVec<T> &p, T alpha1, const T c1, const T c2, const T alphamax, const int maxiter, int &nfunc, const int verbose);


//specific methods

template<typename T>
class BFGS_hyperparameters : public gradient_hyperparameters<T> {
public:
T my_c1; //sufficient decrease (Armijo) parameter of Wolfe conditions, 0<c1<c2<1
T my_c2; //curvature parameter of Wolfe conditions
int my_maxlinesearch; //max. number of function evaluations in one line search
T my_maxsteplength; //max. norm of a single step in x
T my_sy_thr; //skip the update if s.y <= sy_thr*|s|*|y| (curvature condition violated numerically)
BFGS_hyperparameters() : my_c1(1e-4), my_c2(0.9), my_maxlinesearch(30), my_maxsteplength(1e10), my_sy_thr(1e-12) {};
};

template<typename T>
class BFGS_Optimizer : public Optimizer<T> {
public:
virtual optresult<T> minimize(const OptProblem<T> *o, NRVec<T> &x, const hyperparameters<T> *par=NULL, hyperparameters<T> *achieved=NULL);
virtual ~BFGS_Optimizer() {};
};



}//namespace
#endif /* _OPTIMIZE_H_ */

