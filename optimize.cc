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
T linesearch_wolfe(const OptProblem<T> *o, NRVec<T> &x, T &f, NRVec<T> &g, const NRVec<T> &p, T alpha1, const T c1, const T c2, const T alphamax, const int maxiter, int &nfunc, const int verbose)
{
const int n=x.size();
const NRVec<T> x0(x);
const T f0=f;
const T d0= g*p;
if(d0>=0) {if(verbose) std::cout <<"linesearch: not a descent direction\n"; return 0;}
if(alpha1>alphamax) alpha1=alphamax;

//data of trial points; vectors are always freshly allocated, so that assignments (which share data) are safe
T alpha0=0, f_alpha0=f0, d_alpha0=d0; NRVec<T> x_alpha0(x0), g_alpha0(g); //previous trial point
T alo=0,flo=0,dlo=0,ahi=0,fhi=0,dhi=0; NRVec<T> xlo,glo; //bracketing interval for the zoom phase, lo is the best point
bool bracketed=false;
int it;

//bracketing phase
for(it=0; it<maxiter; ++it)
	{
	NRVec<T> xt = x0 + p*alpha1;
	NRVec<T> gt(n);
	T ft=o->funcgrad(xt,&gt); ++nfunc;
	T dt= gt*p;
	if(verbose>1) std::cout <<"linesearch bracket "<<it<<" alpha= "<<alpha1<<" f= "<<ft<<" dphi= "<<dt<<std::endl;
	if(ft > f0 + c1*alpha1*d0 || (it>0 && ft>=f_alpha0))
		{
		alo=alpha0; flo=f_alpha0; dlo=d_alpha0; xlo=x_alpha0; glo=g_alpha0;
		ahi=alpha1; fhi=ft; dhi=dt;
		bracketed=true;
		break;
		}
	if(std::abs(dt) <= -c2*d0 || alpha1>=alphamax) {x=xt; f=ft; g=gt; return alpha1;} //strong Wolfe satisfied or cannot go further
	if(dt>=0)
		{
		alo=alpha1; flo=ft; dlo=dt; xlo=xt; glo=gt;
		ahi=alpha0; fhi=f_alpha0; dhi=d_alpha0;
		bracketed=true;
		break;
		}
	alpha0=alpha1; f_alpha0=ft; d_alpha0=dt; x_alpha0=xt; g_alpha0=gt;
	alpha1 *= 2; if(alpha1>alphamax) alpha1=alphamax;
	}

//zoom phase, invariant: alo satisfies sufficient decrease and has the lowest f found so far, dlo*(ahi-alo)<0
if(bracketed) for(++it; it<maxiter; ++it)
	{
	T a=cubic_interpolate(alo,flo,dlo,ahi,fhi,dhi);
	NRVec<T> xt = x0 + p*a;
	NRVec<T> gt(n);
	T ft=o->funcgrad(xt,&gt); ++nfunc;
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
optresult<T> BFGS_Optimizer<T>::minimize(const OptProblem<T> *o, NRVec<T> &x, const hyperparameters<T> *par, hyperparameters<T> *achieved)
{
//collect parameters, those not provided by the caller are taken from defaults
BFGS_hyperparameters<T> hp;
if(par)
	{
	static_cast<hyperparameters<T> &>(hp) = *par;
	const gradient_hyperparameters<T> *gpar = dynamic_cast<const gradient_hyperparameters<T> *>(par);
	if(gpar) static_cast<gradient_hyperparameters<T> &>(hp) = *gpar;
	const BFGS_hyperparameters<T> *bpar = dynamic_cast<const BFGS_hyperparameters<T> *>(par);
	if(bpar) hp = *bpar;
	}
const int verbose=hp.my_verbose;

const int n=x.size();
NRVec<T> g(n);
T f=o->funcgrad(x,&g);
int nfunc=1;
NRSMat<T> H(n);
H=1.;
bool Hinit=true; //H is the initial guess, rescale it after the first step
bool converged=false;
T last_df=0, last_dx=0;

int iter;
for(iter=0; iter<hp.my_maxstep; ++iter)
	{
	T gnorm=g.norm();
	if(verbose) std::cout <<"BFGS iter "<<iter<<" f= "<<f<<" |g|= "<<gnorm<<std::endl;
	if(gnorm<hp.my_g_thr) {converged=true; break;}
	if(hp.my_maxfunc>0 && nfunc>=hp.my_maxfunc) {if(verbose) std::cout <<"BFGS: maximum number of function evaluations reached\n"; break;}

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
	T pnorm=p.norm();
	T alpha1 = Hinit ? 1/pnorm : 1; //on the first step (or after restart) try unit step length in x, otherwise the Newton step
	T alphamax = hp.my_maxsteplength/pnorm;
	int maxls=hp.my_maxlinesearch;
	if(hp.my_maxfunc>0 && hp.my_maxfunc-nfunc<maxls) maxls=hp.my_maxfunc-nfunc;
	T alpha=linesearch_wolfe(o,x,f,g,p,alpha1,hp.my_c1,hp.my_c2,alphamax,maxls,nfunc,verbose);
	if(alpha==0)
		{
		if(hp.my_maxfunc>0 && nfunc>=hp.my_maxfunc) continue; //reported at the top of the loop
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

	last_df=std::abs(f-fold); last_dx=snorm;
	if(last_df<hp.my_f_thr && last_dx<hp.my_x_thr) {converged=true; ++iter; break;}
	}

if(verbose)
	{
	if(converged) std::cout <<"BFGS converged in "<<iter<<" iterations, "<<nfunc<<" function evaluations, f= "<<f<<" |g|= "<<g.norm()<<std::endl;
	else std::cout <<"BFGS NOT converged after "<<iter<<" iterations, "<<nfunc<<" function evaluations, f= "<<f<<" |g|= "<<g.norm()<<std::endl;
	}

if(achieved)
	{
	BFGS_hyperparameters<T> ach(hp);
	ach.my_maxstep=iter;
	ach.my_maxfunc=nfunc;
	ach.my_f_thr=last_df;
	ach.my_x_thr=last_dx;
	ach.my_g_thr=g.norm();
	*achieved = static_cast<hyperparameters<T> &>(ach);
	gradient_hyperparameters<T> *gach = dynamic_cast<gradient_hyperparameters<T> *>(achieved);
	if(gach) *gach = static_cast<gradient_hyperparameters<T> &>(ach);
	BFGS_hyperparameters<T> *bach = dynamic_cast<BFGS_hyperparameters<T> *>(achieved);
	if(bach) *bach = ach;
	}

return optresult<T>(f,converged);
}



//force instantiation
template double linesearch_wolfe(const OptProblem<double> *o, NRVec<double> &x, double &f, NRVec<double> &g, const NRVec<double> &p, double alpha1, const double c1, const double c2, const double alphamax, const int maxiter, int &nfunc, const int verbose);
template class testproblem<double>;
template class BFGS_Optimizer<double>;

}//namespace

