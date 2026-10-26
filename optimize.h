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
#include <cmath>
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

//optional capabilities of a problem, to be multiply inherited together with OptProblem; methods find them by optcapability()
//matrices are allocated by the caller, implementations writing their elements have to call copyonwrite() first

//Hessian of f
template<typename T>
class OptHessian {
public:
virtual void hessian(const NRVec<T> &x, NRSMat<T> &h) const =0; //Hessian at x
virtual NRVec<T> hessian_times(const NRVec<T> &x, const NRVec<T> &v) const {NRSMat<T> h(x.size()); hessian(x,h); return h*v;} //Hessian-vector product, can be overridden for large problems
virtual ~OptHessian() {};
};

//metric tensor of the parameter space: Fubini-Study metric for the quantum natural gradient (Stokes et al., Quantum 4, 269 (2020)),
//Fisher information for the natural gradient in machine learning (Amari, Neural Comput. 10, 251 (1998))
template<typename T>
class OptMetric {
public:
virtual void metric(const NRVec<T> &x, NRSMat<T> &g) const =0; //metric at x
virtual ~OptMetric() {};
};

//f as a function of x_i alone is a trigonometric polynomial a_0 + sum_{k=1}^{R_i} (a_k cos(k w_i x_i) + b_k sin(k w_i x_i)), w_i = 2 pi/period_i;
//for parameterized quantum circuits with gates exp(-i x_i P/2), P a Pauli string, R_i = 1 and period_i = 2 pi
template<typename T>
class OptTrigonometric {
public:
virtual int degree(const int i) const =0; //R_i, 0 = f is not trigonometric in x_i
virtual T period(const int i) const {return 2*M_PI;}
virtual ~OptTrigonometric() {};
};


//problem wrapping another one (e.g. providing gradients); the capabilities of the wrapped problem remain accessible by optcapability()
template<typename T>
class OptProblemAdapter : public OptProblem<T> {
public:
const OptProblem<T> *p; //wrapped problem
OptProblemAdapter(const OptProblem<T> *pp) : p(pp) {};
};

//capability C (OptHessian<T>, OptMetric<T>, ...) of the problem, searched also through adapters; NULL if not available
template<class C, typename T>
const C *optcapability(const OptProblem<T> *o)
{
const C *c=dynamic_cast<const C *>(o);
if(c) return c;
const OptProblemAdapter<T> *a=dynamic_cast<const OptProblemAdapter<T> *>(o);
return a ? optcapability<C>(a->p) : NULL;
}


//gradient by the general parameter-shift rule (Wierichs et al., Quantum 6, 677 (2022); degree 1: Mitarai et al., Phys. Rev. A 98, 032309 (2018),
//Schuld et al., Phys. Rev. A 99, 032331 (2019)), exact from 2R_i evaluations per parameter; parameters of degree 0 by central differences with step h
//the wrapped problem has to provide OptTrigonometric
template<typename T>
class ParameterShift : public OptProblemAdapter<T> {
public:
const OptTrigonometric<T> *trig;
T h;
ParameterShift(const OptProblem<T> *pp, const T hh=1e-6);
virtual T funcgrad(const NRVec<T> &x, NRVec<T> *grad) const;
};

//gradient by central differences with step h, 2 evaluations per parameter
template<typename T>
class FiniteDifference : public OptProblemAdapter<T> {
public:
T h;
FiniteDifference(const OptProblem<T> *pp, const T hh=1e-6) : OptProblemAdapter<T>(pp), h(hh) {};
virtual T funcgrad(const NRVec<T> &x, NRVec<T> *grad) const;
};


//test case
template<typename T>
class testproblem : public OptProblem<T> {
public:
virtual T funcgrad(const NRVec<T> &x, NRVec<T> *grad) const;
virtual ~testproblem() {};
};

//test case with all capabilities: Ising chain with longitudinal and transverse fields in the product state of qubits RY(b_i) RX(a_i)|0>,
//x = (a_0,b_0,a_1,b_1,...), E = sum_i (h_i z_i + k_i x_i) + sum_i J_i z_i z_{i+1}, Bloch vector components z_i = cos a_i cos b_i, x_i = cos a_i sin b_i;
//trigonometric of degree 1 with period 2 pi in all parameters, Fubini-Study metric 1/4 diag(1, cos^2 a_i) for each qubit
template<typename T>
class testproblem_vqe : public OptProblem<T>, public OptHessian<T>, public OptMetric<T>, public OptTrigonometric<T> {
public:
static T hfield(const int i) {return 0.7+0.1*(i%3);}
static T kfield(const int i) {return 0.4;}
static T J(const int i) {return 1;}
virtual T funcgrad(const NRVec<T> &x, NRVec<T> *grad) const;
virtual void hessian(const NRVec<T> &x, NRSMat<T> &h) const;
virtual void metric(const NRVec<T> &x, NRSMat<T> &g) const;
virtual int degree(const int i) const {return 1;}
virtual ~testproblem_vqe() {};
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
bool stop(const T ff, const T gn= -1, const int needed=1); //to be called at the beginning of each iteration, returns true if the iterations should finish; needed = no. of function evaluations of the next iteration
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


//parameters of first-order methods with a learning rate instead of line search, suitable also for noisy (stochastic) functions and gradients
template<typename T>
class firstorder_hyperparameters : public gradient_hyperparameters<T> {
public:
T my_lr; //learning rate
T my_lr_decay; //learning rate in iteration k is lr/(1+lr_decay*k)^lr_power
T my_lr_power;
T my_weight_decay; //decoupled weight decay x -= lr*weight_decay*x (Loshchilov & Hutter, ICLR 2019)
bool my_return_best; //return the point with the lowest function value found instead of the last one; only for (nearly) deterministic f, the lowest of noisy values is biased by the noise
bool my_natural; //natural gradient (G + metric_reg I)^-1 g with the metric G of the problem (OptMetric) (Amari, Neural Comput. 10, 251 (1998); Stokes et al., Quantum 4, 269 (2020)); |g| refers then to it
T my_metric_reg; //regularization added to the diagonal of the metric
firstorder_hyperparameters() : my_lr(1e-3), my_lr_decay(0), my_lr_power(1), my_weight_decay(0), my_return_best(false), my_natural(false), my_metric_reg(1e-3) {this->my_maxstep=1000;};
virtual void setfrom(const hyperparameters<T> &p) {const firstorder_hyperparameters *q=dynamic_cast<const firstorder_hyperparameters *>(&p); if(q) *this= *q; else gradient_hyperparameters<T>::setfrom(p);}
};

//update rule of a first-order method, used by the common driver firstorder_minimize()
template<typename T>
class firstorder_step {
public:
virtual NRVec<T> step(const NRVec<T> &g, const int k) =0; //direction d of the step x -= lr*d in iteration k=1,2,... with gradient g
virtual ~firstorder_step() {};
};

//function value and gradient for firstorder_minimize(), the default is the exact evaluation, derived classes estimate them (SPSA)
template<typename T>
class gradient_estimator {
public:
virtual T operator()(optevaluator<T> &e, const NRVec<T> &x, NRVec<T> &g, const int k) {return e(x,&g);} //returns f at x and the gradient g, k=0,1,... is the iteration
virtual int cost() const {return 1;} //no. of function evaluations of one call
virtual bool exact() const {return true;} //with estimates no convergence test is done, small gradients or changes can occur by chance
virtual T calibrate(optevaluator<T> &e, const NRVec<T> &x, const firstorder_hyperparameters<T> &hp) {return hp.my_lr;} //learning rate to be used
virtual ~gradient_estimator() {};
};

//common driver of first-order methods; est==NULL means exact gradients
//the learning rate actually used is returned in achieved, if it is of firstorder_hyperparameters type
template<typename T>
optresult<T> firstorder_minimize(const char *name, firstorder_step<T> &stepper, const OptProblem<T> *o, NRVec<T> &x, const firstorder_hyperparameters<T> &hp, hyperparameters<T> *achieved, gradient_estimator<T> *est=NULL);


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


//geometry DIIS (Csaszar & Pulay, J. Mol. Struct. 114, 31 (1984)) with a quasi-Newton Hessian (Farkas & Schlegel, Phys. Chem. Chem. Phys. 4, 11 (2002))
//error vectors are the quasi-Newton steps e_i = -H g_i, the new point is sum_i c_i (x_i+e_i) minimizing |sum_i c_i e_i| with sum_i c_i = 1
//one function evaluation per iteration; if the DIIS step goes uphill, is longer than maxsteplength or does not decrease f,
//the DIIS history is restarted and a line search along the quasi-Newton step is done
template<typename T>
class GDIIS_hyperparameters : public BFGS_hyperparameters<T> {
public:
int my_diis_dim; //no. of stored points
T my_diis_eps; //regularization added to the diagonal of the DIIS matrix
GDIIS_hyperparameters() : my_diis_dim(5), my_diis_eps(0) {};
virtual void setfrom(const hyperparameters<T> &p) {const GDIIS_hyperparameters *q=dynamic_cast<const GDIIS_hyperparameters *>(&p); if(q) *this= *q; else BFGS_hyperparameters<T>::setfrom(p);}
};

template<typename T>
class GDIIS_Optimizer : public Optimizer<T> {
public:
virtual optresult<T> minimize(const OptProblem<T> *o, NRVec<T> &x, const hyperparameters<T> *par=NULL, hyperparameters<T> *achieved=NULL);
virtual ~GDIIS_Optimizer() {};
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


//Rotosolve: sweeps of exact minimizations along single parameters in which f is trigonometric (OptTrigonometric)
//(Ostaszewski et al., Quantum 5, 391 (2021); Nakanishi et al., Phys. Rev. Research 2, 043158 (2020); degree > 1: Wierichs et al., Quantum 6, 677 (2022));
//f along x_i is reconstructed from 2R_i+1 equidistant samples over its period (2R_i new evaluations) and minimized without further evaluations;
//parameters of degree 0, or all if the problem is not trigonometric, are minimized by line minimization as in Powell's method (whose parameters are used)
template<typename T>
class Rotosolve_hyperparameters : public Powell_hyperparameters<T> {
public:
bool my_reevaluate; //evaluate f at the new point instead of taking the value of the reconstruction (for noisy f or doubtful degrees)
Rotosolve_hyperparameters() : my_reevaluate(false) {};
virtual void setfrom(const hyperparameters<T> &p) {const Rotosolve_hyperparameters *q=dynamic_cast<const Rotosolve_hyperparameters *>(&p); if(q) *this= *q; else Powell_hyperparameters<T>::setfrom(p);}
};

template<typename T>
class Rotosolve_Optimizer : public Optimizer<T> {
public:
virtual optresult<T> minimize(const OptProblem<T> *o, NRVec<T> &x, const hyperparameters<T> *par=NULL, hyperparameters<T> *achieved=NULL);
virtual ~Rotosolve_Optimizer() {};
};


//stochastic gradient descent with momentum v = momentum*v + g, step d = v or d = g + momentum*v (Nesterov) (Sutskever et al., ICML 2013)
template<typename T>
class SGD_hyperparameters : public firstorder_hyperparameters<T> {
public:
T my_momentum;
bool my_nesterov;
SGD_hyperparameters() : my_momentum(0.9), my_nesterov(false) {this->my_lr=1e-2;};
virtual void setfrom(const hyperparameters<T> &p) {const SGD_hyperparameters *q=dynamic_cast<const SGD_hyperparameters *>(&p); if(q) *this= *q; else firstorder_hyperparameters<T>::setfrom(p);}
};

template<typename T>
class SGD_Optimizer : public Optimizer<T> {
public:
virtual optresult<T> minimize(const OptProblem<T> *o, NRVec<T> &x, const hyperparameters<T> *par=NULL, hyperparameters<T> *achieved=NULL);
virtual ~SGD_Optimizer() {};
};


//quantum natural gradient x -= lr (G + metric_reg I)^-1 g (Stokes et al., Quantum 4, 269 (2020)): SGD without momentum with the natural gradient;
//the problem has to provide the metric (OptMetric), the gradient can come e.g. from ParameterShift; momentum, Adam or SPSA can be combined with the natural gradient by setting my_natural
//natural steps are longer than gradient steps with the same lr (metric eigenvalues ~1/4 or smaller): lr up to ~0.1, or a larger metric_reg
template<typename T>
class QNG_hyperparameters : public SGD_hyperparameters<T> {
public:
QNG_hyperparameters() {this->my_momentum=0; this->my_natural=true; this->my_lr=0.05;};
virtual void setfrom(const hyperparameters<T> &p) {const QNG_hyperparameters *q=dynamic_cast<const QNG_hyperparameters *>(&p); if(q) *this= *q; else SGD_hyperparameters<T>::setfrom(p);}
};

template<typename T>
class QNG_Optimizer : public Optimizer<T> {
public:
virtual optresult<T> minimize(const OptProblem<T> *o, NRVec<T> &x, const hyperparameters<T> *par=NULL, hyperparameters<T> *achieved=NULL);
virtual ~QNG_Optimizer() {};
};


//Adam (Kingma & Ba, ICLR 2015), with weight_decay>0 AdamW, optionally AMSGrad (Reddi et al., ICLR 2018); beta1=0 gives RMSprop
template<typename T>
class Adam_hyperparameters : public firstorder_hyperparameters<T> {
public:
T my_beta1; //decay rate of the first moment estimate
T my_beta2; //decay rate of the second moment estimate
T my_eps; //regularization of the division by the second moment
bool my_amsgrad; //use the maximum of the past second moments
Adam_hyperparameters() : my_beta1(0.9), my_beta2(0.999), my_eps(1e-8), my_amsgrad(false) {};
virtual void setfrom(const hyperparameters<T> &p) {const Adam_hyperparameters *q=dynamic_cast<const Adam_hyperparameters *>(&p); if(q) *this= *q; else firstorder_hyperparameters<T>::setfrom(p);}
};

template<typename T>
class Adam_Optimizer : public Optimizer<T> {
public:
virtual optresult<T> minimize(const OptProblem<T> *o, NRVec<T> &x, const hyperparameters<T> *par=NULL, hyperparameters<T> *achieved=NULL);
virtual ~Adam_Optimizer() {};
};


//simultaneous perturbation stochastic approximation (Spall, IEEE Trans. Autom. Control 37, 332 (1992); IEEE Trans. Aerosp. Electron. Syst. 34, 817 (1998))
//derivative-free, 2 function evaluations per gradient estimate independently of the dimension, robust to noise (VQE)
//gain sequences: a_k = lr/(1+lr_decay*k)^lr_power, i.e. Spall's a/(k+1+A)^alpha with lr=a/(1+A)^alpha, lr_decay=1/(1+A), lr_power=alpha; c_k = c/(k+1)^gamma
//the gradient and f are estimates, therefore no convergence test is done and the iterations run until maxstep or maxfunc
//the first estimate at the starting point is always done, even if maxfunc is smaller than its 2*resamplings evaluations
template<typename T>
class SPSA_hyperparameters : public firstorder_hyperparameters<T> {
public:
T my_c; //perturbation size, of the order of the noise of f
T my_gamma; //decay exponent of the perturbation size
int my_resamplings; //no. of gradient estimates averaged in one iteration
int my_calibrate; //no. of gradient estimates at the starting point for the calibration of lr, 0 = no calibration
T my_target_step; //calibration: lr is set so that the first step has this length on average
SPSA_hyperparameters() : my_c(0.1), my_gamma(0.101), my_resamplings(1), my_calibrate(20), my_target_step(0.1) {this->my_lr_decay=0.01; this->my_lr_power=0.602; this->my_g_thr=0;};
virtual void setfrom(const hyperparameters<T> &p) {const SPSA_hyperparameters *q=dynamic_cast<const SPSA_hyperparameters *>(&p); if(q) *this= *q; else firstorder_hyperparameters<T>::setfrom(p);}
};

template<typename T>
class SPSA_Optimizer : public Optimizer<T> {
public:
virtual optresult<T> minimize(const OptProblem<T> *o, NRVec<T> &x, const hyperparameters<T> *par=NULL, hyperparameters<T> *achieved=NULL);
virtual ~SPSA_Optimizer() {};
};



}//namespace
#endif /* _OPTIMIZE_H_ */

