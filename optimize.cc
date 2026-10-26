/*
    LA: linear algebra C++ interface library
    Copyright (C) 2026 Jiri Pittner <jiri.pittner@jh-inst.cas.cz> or <jiri@pittnerovi.com>

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

#include "optimize.h"

namespace LA {


template<typename T>
T testproblem<T>::funcgrad(const NRVec<T> &x, NRVec<T> *grad) const
{
int n=x.size();
T r= n*(n-1)/2;
for(int i=0; i<n; ++i)
	{
	T x2=x[i]*x[i];
	r += (i+1)*(x2 + .1*x2*x2);
	for(int j=0; j<i; ++j) r -= cos(x[i]*x[j]);
	}
if(grad)
	{
	if(grad->size()!=x.size()) laerror("grad size mismatch");
	grad->copyonwrite(); //I forgot this initially
	for(int k=0; k<n; ++k)
		{
		(*grad)[k] = (k+1)*x[k]*(2+0.4*x[k]*x[k]);
		for(int i=0; i<n; ++i) if(i!=k) (*grad)[k] += x[i]*sin(x[i]*x[k]);
		}
	}
return r;
}


//safeguarded cubic interpolation of the minimizer of phi between a and b, given function values and derivatives at both ends
template<typename T>
static T cubic_interpolate(const T a, const T fa, const T da, const T b, const T fb, const T db)
{
T lo= a<b ? a : b;
T hi= a<b ? b : a;
T safe=0.1*(hi-lo); //keep the trial point away from the interval ends
T d1 = da + db - 3*(fa-fb)/(a-b);
T disc = d1*d1 - da*db;
T r;
if(disc<0) r=(a+b)/2;
else
	{
	T d2 = std::sqrt(disc); if(b<a) d2 = -d2;
	T den = db - da + 2*d2;
	if(den==0) r=(a+b)/2;
	else r = b - (b-a)*(db + d2 - d1)/den;
	}
if(!(r>=lo+safe && r<=hi-safe)) r=(a+b)/2; //also catches NaN
return r;
}


template<typename T>
T linesearch_wolfe(optevaluator<T> &e, NRVec<T> &x, T &f, NRVec<T> &g, const NRVec<T> &p, const T alpha1, const linesearch_hyperparameters<T> &lp)
{
const T c1=lp.my_c1;
const T c2=lp.my_c2;
const int verbose=lp.my_verbose;
const int maxiter= std::min(lp.my_maxlinesearch,e.remaining());
const int n=x.size();
const NRVec<T> x0(x);
const T f0=f;
const T d0= g*p;
if(d0>=0) {if(verbose) std::cout <<"linesearch: not a descent direction\n"; return 0;}
const T alphamax=lp.my_maxsteplength/p.norm();
T alpha=std::min(alpha1,alphamax);

//data of trial points; vectors are always freshly allocated, so that assignments (which share data) are safe
T alpha0=0, f_alpha0=f0, d_alpha0=d0; NRVec<T> x_alpha0(x0), g_alpha0(g); //previous trial point
T alo=0,flo=0,dlo=0,ahi=0,fhi=0,dhi=0; NRVec<T> xlo,glo; //bracketing interval for the zoom phase, lo is the best point
bool bracketed=false;
int it;

//bracketing phase
for(it=0; it<maxiter; ++it)
	{
	NRVec<T> xt = x0 + p*alpha;
	NRVec<T> gt(n);
	T ft=e(xt,&gt);
	T dt= gt*p;
	if(verbose>1) std::cout <<"linesearch bracket "<<it<<" alpha= "<<alpha<<" f= "<<ft<<" dphi= "<<dt<<std::endl;
	if(ft > f0 + c1*alpha*d0 || (it>0 && ft>=f_alpha0))
		{
		alo=alpha0; flo=f_alpha0; dlo=d_alpha0; xlo=x_alpha0; glo=g_alpha0;
		ahi=alpha; fhi=ft; dhi=dt;
		bracketed=true;
		break;
		}
	if(std::abs(dt) <= -c2*d0 || alpha>=alphamax) {x=xt; f=ft; g=gt; return alpha;} //strong Wolfe satisfied or cannot go further
	if(dt>=0)
		{
		alo=alpha; flo=ft; dlo=dt; xlo=xt; glo=gt;
		ahi=alpha0; fhi=f_alpha0; dhi=d_alpha0;
		bracketed=true;
		break;
		}
	alpha0=alpha; f_alpha0=ft; d_alpha0=dt; x_alpha0=xt; g_alpha0=gt;
	alpha *= 2; if(alpha>alphamax) alpha=alphamax;
	}

//zoom phase, invariant: alo satisfies sufficient decrease and has the lowest f found so far, dlo*(ahi-alo)<0
if(bracketed) for(++it; it<maxiter; ++it)
	{
	T a=cubic_interpolate(alo,flo,dlo,ahi,fhi,dhi);
	NRVec<T> xt = x0 + p*a;
	NRVec<T> gt(n);
	T ft=e(xt,&gt);
	T dt= gt*p;
	if(verbose>1) std::cout <<"linesearch zoom "<<it<<" alpha= "<<a<<" f= "<<ft<<" dphi= "<<dt<<std::endl;
	if(ft > f0 + c1*a*d0 || ft>=flo)
		{
		ahi=a; fhi=ft; dhi=dt;
		}
	else
		{
		if(std::abs(dt) <= -c2*d0) {x=xt; f=ft; g=gt; return a;}
		if(dt*(ahi-alo)>=0) {ahi=alo; fhi=flo; dhi=dlo;}
		alo=a; flo=ft; dlo=dt; xlo=xt; glo=gt;
		}
	if(std::abs(ahi-alo) <= std::numeric_limits<T>::epsilon()*std::abs(alo)) break; //interval collapsed
	}

//failure - accept the best point satisfying sufficient decrease, if any
if(!bracketed && alpha0>0) {alo=alpha0; flo=f_alpha0; xlo=x_alpha0; glo=g_alpha0;} //bracketing ran out of evaluations
if(alo>0)
	{
	if(verbose) std::cout <<"linesearch: strong Wolfe conditions not satisfied, accepting sufficient decrease step "<<alo<<std::endl;
	x=xlo; f=flo; g=glo; return alo;
	}
if(verbose) std::cout <<"linesearch failed\n";
return 0;
}


//BFGS update of the inverse Hessian approximation H, cf. Nocedal & Wright eq. 6.17
//H+ = (I - rho s y^T) H (I - rho y s^T) + rho s s^T, rho=1/(y^T s)
//expanded: H+ = H + (rho^2 y^T H y + rho) s s^T - rho (s (Hy)^T + Hy s^T) = H + s u^T + u s^T with u = (rho^2 y^T H y + rho)/2 s - rho Hy
template<typename T>
static void BFGS_update(NRSMat<T> &H, const NRVec<T> &s, const NRVec<T> &y, const T sy)
{
T rho = 1/sy;
NRVec<T> Hy = H*y;
T yHy = y*Hy;
NRVec<T> u = s*((rho*rho*yHy + rho)/2);
u.axpy(-rho,Hy);
H.rank2update((T)1,s,u);
}


template<typename T>
bool optstate<T>::stop(const T ff, const T gn)
{
f=ff; gnorm=gn;
if(hp.my_verbose)
	{
	std::cout <<name<<" iter "<<iter<<" f= "<<f;
	if(gnorm>=0) std::cout <<" |g|= "<<gnorm;
	std::cout <<std::endl;
	}
if(hp.converged(*this)) {converged=true; return true;}
if(iter>=hp.my_maxstep) {if(hp.my_verbose) std::cout <<name<<": maximum number of iterations reached\n"; return true;}
if(e.exhausted()) {if(hp.my_verbose) std::cout <<name<<": maximum number of function evaluations reached\n"; return true;}
return false;
}


template<typename T>
optresult<T> optstate<T>::finish(hyperparameters<T> *achieved) const
{
if(hp.my_verbose)
	{
	std::cout <<name<<(converged?" converged in ":" NOT converged after ")<<iter<<" iterations, "<<e.nfunc<<" function evaluations, f= "<<f;
	if(gnorm>=0) std::cout <<" |g|= "<<gnorm;
	std::cout <<std::endl;
	}
if(achieved) {achieved->setfrom(hp); achieved->setreached(*this);}
return optresult<T>(f,converged);
}


template<typename T>
optresult<T> BFGS_Optimizer<T>::minimize(const OptProblem<T> *o, NRVec<T> &x, const hyperparameters<T> *par, hyperparameters<T> *achieved)
{
BFGS_hyperparameters<T> hp;
if(par) hp.setfrom(*par);
if(hp.my_saddle_order) laerror("BFGS cannot search for saddle points");
const int verbose=hp.my_verbose;
optevaluator<T> e(o,hp.my_maxfunc);
optstate<T> st("BFGS",hp,e);

const int n=x.size();
NRVec<T> g(n);
T f=e(x,&g);
NRSMat<T> H(n);
H=1.;
bool Hinit=true; //H is the initial guess, rescale it after the first step

for(;; ++st.iter)
	{
	if(st.stop(f,g.norm())) break;

	NRVec<T> p = H*g;
	p *= (T)-1;
	if(p*g >= 0) //H lost positive definiteness numerically, restart
		{
		if(verbose) std::cout <<"BFGS: not a descent direction, resetting inverse Hessian\n";
		H=1.; Hinit=true;
		p = g*(T)-1;
		}

	NRVec<T> xold(x), gold(g);
	T fold=f;
	//on the first step (or after restart) try unit step length in x, otherwise the Newton step
	T alpha=linesearch_wolfe(e,x,f,g,p,Hinit ? 1/p.norm() : (T)1,hp);
	if(alpha==0)
		{
		if(e.exhausted()) continue; //reported by stop()
		if(Hinit) {if(verbose) std::cout <<"BFGS: line search failed along steepest descent, giving up\n"; break;}
		if(verbose) std::cout <<"BFGS: line search failed, resetting inverse Hessian\n";
		H=1.; Hinit=true;
		continue;
		}

	NRVec<T> s = x-xold;
	NRVec<T> y = g-gold;
	T sy = s*y;
	T snorm=s.norm();
	T ynorm=y.norm();
	if(verbose>1) std::cout <<"BFGS step alpha= "<<alpha<<" |s|= "<<snorm<<" s.y= "<<sy<<std::endl;
	if(sy > hp.my_sy_thr*snorm*ynorm)
		{
		if(Hinit) {H = (T)(sy/(ynorm*ynorm)); Hinit=false;} //scale initial guess, Nocedal & Wright eq. 6.20
		BFGS_update(H,s,y,sy);
		}
	else if(verbose) std::cout <<"BFGS: curvature condition violated, skipping update\n";
	st.step(f-fold,snorm);
	}

return st.finish(achieved);
}



//force instantiation
template double linesearch_wolfe(optevaluator<double> &e, NRVec<double> &x, double &f, NRVec<double> &g, const NRVec<double> &p, const double alpha1, const linesearch_hyperparameters<double> &lp);
template class optstate<double>;
template class testproblem<double>;
template class BFGS_Optimizer<double>;

}//namespace

