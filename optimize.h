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

template<typename T> class optstate;


//this is a collection of parameters, which can be augmented by method-specific additions, to be used as a base class
//methods take their specific parameters via setfrom(), those not provided by the caller keep the defaults from the constructor
//each derived class has to override setfrom(), converged() and setreached() if it adds parameters relevant for them
template<typename T>
class hyperparameters {
public:
int my_maxstep; //maximum number of iterations
int my_maxfunc; //maximum number of function evaluations, <=0 means unlimited
T my_f_thr; //threshhold on function change in a single step
T my_x_thr; //threshold on x norm change in a single step
int my_saddle_order; //0=minimum, k=saddle point with k negative Hessian eigenvalues (only for methods supporting it)
int my_verbose; //0=silent, 1=per-iteration info, 2=more detailed info
hyperparameters() : my_maxstep(200), my_maxfunc(0), my_f_thr(0), my_x_thr(0), my_saddle_order(0), my_verbose(1) {};
virtual void setfrom(const hyperparameters &p) {*this=p;} //copy those parameters from p which exist in both classes
virtual bool converged(const optstate<T> &s) const {return s.stepdone && s.df<my_f_thr && s.dx<my_x_thr;}
virtual void setreached(const optstate<T> &s) {my_maxstep=s.iter; my_maxfunc=s.e.nfunc; my_f_thr=s.df; my_x_thr=s.dx;} //store the values reached in the minimization
virtual ~hyperparameters(){};
};

//parameters common to methods using the gradient
template<typename T>
class gradient_hyperparameters : public hyperparameters<T> {
public:
T my_g_thr; //gradient norm convergence threshold
gradient_hyperparameters() : my_g_thr(1e-6) {};
virtual void setfrom(const hyperparameters<T> &p) {const gradient_hyperparameters *q=dynamic_cast<const gradient_hyperparameters *>(&p); if(q) *this= *q; else hyperparameters<T>::setfrom(p);}
virtual bool converged(const optstate<T> &s) const {return (s.gnorm>=0 && s.gnorm<my_g_thr) || hyperparameters<T>::converged(s);}
virtual void setreached(const optstate<T> &s) {hyperparameters<T>::setreached(s); my_g_thr=s.gnorm;}
};

//parameters of gradient methods using line search
template<typename T>
class linesearch_hyperparameters : public gradient_hyperparameters<T> {
public:
T my_c1; //sufficient decrease (Armijo) parameter of Wolfe conditions, 0<c1<c2<1
T my_c2; //curvature parameter of Wolfe conditions
int my_maxlinesearch; //max. number of function evaluations in one line search
T my_maxsteplength; //max. norm of a single step in x
linesearch_hyperparameters() : my_c1(1e-4), my_c2(0.9), my_maxlinesearch(30), my_maxsteplength(1e10) {};
virtual void setfrom(const hyperparameters<T> &p) {const linesearch_hyperparameters *q=dynamic_cast<const linesearch_hyperparameters *>(&p); if(q) *this= *q; else gradient_hyperparameters<T>::setfrom(p);}
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


//evaluation of the optimized function, counts the evaluations and keeps the best point found
template<typename T>
class optevaluator {
public:
const OptProblem<T> *o;
int maxfunc; //<=0 means unlimited
int nfunc; //no. of evaluations done
T fbest; //lowest function value found
NRVec<T> xbest; //and its point
optevaluator(const OptProblem<T> *oo, const int maxf=0) : o(oo), maxfunc(maxf), nfunc(0), fbest(0) {};
T operator()(const NRVec<T> &x, NRVec<T> *grad=NULL)
	{
	T f=o->funcgrad(x,grad);
	if(nfunc++ ==0 || f<fbest) {fbest=f; xbest=x;}
	return f;
	}
bool exhausted() const {return maxfunc>0 && nfunc>=maxfunc;}
int remaining() const {return maxfunc>0 ? maxfunc-nfunc : std::numeric_limits<int>::max();}
};


//bookkeeping of the iterations common to all methods: convergence and limit tests, verbose output, results
template<typename T>
class optstate {
public:
const char *name; //of the method for verbose output
const hyperparameters<T> &hp; //parameters used
const optevaluator<T> &e;
int iter; //no. of iterations done
T f; //current function value
T gnorm; //current gradient norm, negative if not available
T df; //|f change| in the last step
T dx; //|x change| in the last step
bool stepdone; //df and dx are valid
bool converged;
optstate(const char *nm, const hyperparameters<T> &h, const optevaluator<T> &ee) : name(nm), hp(h), e(ee), iter(0), f(0), gnorm(-1), df(0), dx(0), stepdone(false), converged(false) {};
bool stop(const T ff, const T gn= -1); //to be called at the beginning of each iteration, returns true if the iterations should finish
void step(const T dff, const T dxx) {df=std::abs(dff); dx=dxx; stepdone=true;} //record a step done
optresult<T> finish(hyperparameters<T> *achieved) const; //final report, fills achieved if not NULL
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
//alpha1 = initial trial step length
template<typename T>
T linesearch_wolfe(optevaluator<T> &e, NRVec<T> &x, T &f, NRVec<T> &g, const NRVec<T> &p, const T alpha1, const linesearch_hyperparameters<T> &lp);


//derivative-free line minimization of f(x + t*u): bracketing and Brent's method (Brent, Algorithms for Minimization without Derivatives, 1973, chap. 5)
//on input x,f at the starting point, on output x,f at the minimum found; returns t, 0 if no decrease was found (x,f then unchanged)
//t1 = initial trial step, tol and abstol = relative and absolute tolerance on t, maxiter = max. no. of function evaluations
template<typename T>
T linemin_brent(optevaluator<T> &e, NRVec<T> &x, T &f, const NRVec<T> &u, const T t1, const T tol, const T abstol, const int maxiter, const int verbose=0);


//search direction of a line-search method, used by the common driver linesearch_minimize()
template<typename T>
class linesearch_direction {
public:
virtual NRVec<T> direction(const NRVec<T> &g) =0; //search direction at the point with gradient g
virtual bool update(const NRVec<T> &s, const NRVec<T> &y, const NRVec<T> &p, const NRVec<T> &g) =0; //after the step s=alpha*p with gradient change y and new gradient g; returns false if the update was skipped
virtual void reset() =0; //forget the accumulated information
virtual bool steepest() const =0; //no information accumulated, direction() gives steepest descent
virtual bool indefinite() const {return false;} //a non-descent direction is legitimate (SR1), take steepest descent step without reset
virtual bool alpha_from_previous() const {return false;} //initial step length from the previous one (Nocedal & Wright eq. 3.60), otherwise 1 (unit step in x for steepest descent)
virtual ~linesearch_direction() {};
};

//common driver of line-search methods
template<typename T>
optresult<T> linesearch_minimize(const char *name, linesearch_direction<T> &dir, const OptProblem<T> *o, NRVec<T> &x, const linesearch_hyperparameters<T> &hp, hyperparameters<T> *achieved);


//specific methods

//parameters common to quasi-Newton methods
template<typename T>
class quasinewton_hyperparameters : public linesearch_hyperparameters<T> {
public:
T my_sy_thr; //skip the BFGS-type update if s.y <= sy_thr*|s|*|y| (curvature condition violated numerically)
quasinewton_hyperparameters() : my_sy_thr(1e-12) {};
virtual void setfrom(const hyperparameters<T> &p) {const quasinewton_hyperparameters *q=dynamic_cast<const quasinewton_hyperparameters *>(&p); if(q) *this= *q; else linesearch_hyperparameters<T>::setfrom(p);}
};


//quasi-Newton methods with line search, differing by the update of the inverse Hessian approximation
enum QN_update {QN_BFGS, QN_DFP, QN_SR1};

template<typename T>
class BFGS_hyperparameters : public quasinewton_hyperparameters<T> {
public:
QN_update my_update; //BFGS, DFP or SR1 (Nocedal & Wright eqs. 6.17, 6.15, 6.25); DFP needs an accurate line search (c2~0.1), otherwise it can stall
T my_sr1_thr; //SR1: skip the update if |(s-Hy).y| <= sr1_thr*|s-Hy|*|y| (cf. Nocedal & Wright eq. 6.26)
BFGS_hyperparameters() : my_update(QN_BFGS), my_sr1_thr(1e-8) {};
virtual void setfrom(const hyperparameters<T> &p) {const BFGS_hyperparameters *q=dynamic_cast<const BFGS_hyperparameters *>(&p); if(q) *this= *q; else quasinewton_hyperparameters<T>::setfrom(p);}
};

template<typename T>
class BFGS_Optimizer : public Optimizer<T> {
public:
virtual optresult<T> minimize(const OptProblem<T> *o, NRVec<T> &x, const hyperparameters<T> *par=NULL, hyperparameters<T> *achieved=NULL);
virtual ~BFGS_Optimizer() {};
};


//limited-memory BFGS (Nocedal & Wright chap. 7.2), for large problems
template<typename T>
class LBFGS_hyperparameters : public quasinewton_hyperparameters<T> {
public:
int my_memory; //no. of stored pairs of steps and gradient changes
LBFGS_hyperparameters() : my_memory(10) {};
virtual void setfrom(const hyperparameters<T> &p) {const LBFGS_hyperparameters *q=dynamic_cast<const LBFGS_hyperparameters *>(&p); if(q) *this= *q; else quasinewton_hyperparameters<T>::setfrom(p);}
};

template<typename T>
class LBFGS_Optimizer : public Optimizer<T> {
public:
virtual optresult<T> minimize(const OptProblem<T> *o, NRVec<T> &x, const hyperparameters<T> *par=NULL, hyperparameters<T> *achieved=NULL);
virtual ~LBFGS_Optimizer() {};
};


//nonlinear conjugate gradient methods (Nocedal & Wright chap. 5.2; Hager & Zhang, SIAM J. Optim. 16, 170 (2005))
//differing by the formula for beta in p_{k+1} = -g_{k+1} + beta p_k; PR and HS are used in the nonnegative variants PR+ and HS+
enum CG_beta {CG_FR, CG_PR, CG_HS, CG_DY, CG_HZ};

template<typename T>
class CG_hyperparameters : public linesearch_hyperparameters<T> {
public:
CG_beta my_beta; //Fletcher-Reeves, Polak-Ribiere+, Hestenes-Stiefel+, Dai-Yuan, Hager-Zhang
int my_restart; //restart with steepest descent every my_restart iterations, <=0 means the problem dimension
T my_powell_thr; //restart if |g_{k+1}.g_k| >= powell_thr*|g_{k+1}|^2 (Nocedal & Wright eq. 5.52), <=0 disables
CG_hyperparameters() : my_beta(CG_PR), my_restart(0), my_powell_thr(0.2) {this->my_c2=0.1;}; //CG needs a more accurate line search
virtual void setfrom(const hyperparameters<T> &p) {const CG_hyperparameters *q=dynamic_cast<const CG_hyperparameters *>(&p); if(q) *this= *q; else linesearch_hyperparameters<T>::setfrom(p);}
};

template<typename T>
class CG_Optimizer : public Optimizer<T> {
public:
virtual optresult<T> minimize(const OptProblem<T> *o, NRVec<T> &x, const hyperparameters<T> *par=NULL, hyperparameters<T> *achieved=NULL);
virtual ~CG_Optimizer() {};
};


//Nelder-Mead downhill simplex (Lagarias et al., SIAM J. Optim. 9, 112 (1998)), derivative-free
//convergence: f_thr is used for the spread of function values in the simplex, x_thr for its diameter, both have to be satisfied
template<typename T>
class NelderMead_hyperparameters : public hyperparameters<T> {
public:
T my_initstep; //edge length of the initial simplex x0 + initstep*e_i
bool my_adaptive; //dimension-dependent coefficients (Gao & Han, Comput. Optim. Appl. 51, 259 (2012)), otherwise the following ones
T my_alpha; //reflection
T my_gamma; //expansion
T my_rho; //contraction
T my_sigma; //shrink
NelderMead_hyperparameters() : my_initstep(0.1), my_adaptive(true), my_alpha(1), my_gamma(2), my_rho(0.5), my_sigma(0.5) {this->my_maxstep=5000; this->my_f_thr=1e-8; this->my_x_thr=1e-6;};
virtual void setfrom(const hyperparameters<T> &p) {const NelderMead_hyperparameters *q=dynamic_cast<const NelderMead_hyperparameters *>(&p); if(q) *this= *q; else hyperparameters<T>::setfrom(p);}
};

template<typename T>
class NelderMead_Optimizer : public Optimizer<T> {
public:
virtual optresult<T> minimize(const OptProblem<T> *o, NRVec<T> &x, const hyperparameters<T> *par=NULL, hyperparameters<T> *achieved=NULL);
virtual ~NelderMead_Optimizer() {};
};


//Powell's conjugate direction method (Powell, Comput. J. 7, 155 (1964)), derivative-free
//convergence: f_thr and x_thr are used for the change of f and x in one sweep over all directions, both have to be satisfied
template<typename T>
class Powell_hyperparameters : public hyperparameters<T> {
public:
T my_initstep; //initial trial step of the line minimizations
T my_linmin_tol; //relative tolerance of the line minimizations, the absolute one is 0.01*x_thr
int my_maxlinemin; //max. no. of function evaluations in one line minimization
Powell_hyperparameters() : my_initstep(0.1), my_linmin_tol(1e-4), my_maxlinemin(50) {this->my_maxstep=1000; this->my_f_thr=1e-10; this->my_x_thr=1e-6;};
virtual void setfrom(const hyperparameters<T> &p) {const Powell_hyperparameters *q=dynamic_cast<const Powell_hyperparameters *>(&p); if(q) *this= *q; else hyperparameters<T>::setfrom(p);}
};

template<typename T>
class Powell_Optimizer : public Optimizer<T> {
public:
virtual optresult<T> minimize(const OptProblem<T> *o, NRVec<T> &x, const hyperparameters<T> *par=NULL, hyperparameters<T> *achieved=NULL);
virtual ~Powell_Optimizer() {};
};



}//namespace
#endif /* _OPTIMIZE_H_ */

