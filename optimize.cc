/*
    LA: linear algebra C++ interface library
    Copyright (C) 2026 Jiri Pittner <jiri.pittner@jh-inst.cas.cz> or <jiri@pittnerovi.com>
    This code has been prepared with the assistance of an AI (Claude Opus 5.5, Anthropic).

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
#include "la_random.h"
#include "diis.h"
#include "nonclass.h"

namespace LA {


template<typename T>
void FrozenOptProblem<T>::setfrozen(const bitvector &frozen)
{
const int n=xfull.size();
if((int)frozen.size()!=n) laerror("size mismatch in FrozenOptProblem::setfrozen");
NRVec<int> index(n-(int)frozen.population());
for(int i=0, k=0; i<n; ++i) if(!frozen[i]) index[k++]=i;
freeindex=index;
}

template<typename T>
T FrozenOptProblem<T>::funcgrad(const NRVec<T> &x, NRVec<T> *grad) const
{
NRVec<T> full=expand(x);
if(!grad) return this->p->funcgrad(full,NULL);
NRVec<T> gfull(full.size());
T f=this->p->funcgrad(full,&gfull);
grad->subvector_from(freeindex,gfull);
return f;
}

//the capabilities are used only if the wrapped problem has them, as ensured by optcapability()
template<typename T>
void FrozenOptProblem<T>::hessian(const NRVec<T> &x, NRSMat<T> &h) const
{
NRSMat<T> hfull(xfull.size());
optcapability<OptHessian<T> >(this->p)->hessian(expand(x),hfull);
h=hfull.submatrix(freeindex);
}

template<typename T>
NRVec<T> FrozenOptProblem<T>::hessian_times(const NRVec<T> &x, const NRVec<T> &v) const
{
NRVec<T> vfull((T)0,xfull.size());
v.storesubvector_to(freeindex,vfull);
return optcapability<OptHessian<T> >(this->p)->hessian_times(expand(x),vfull).subvector(freeindex);
}

template<typename T>
void FrozenOptProblem<T>::metric(const NRVec<T> &x, NRSMat<T> &g) const
{
NRSMat<T> gfull(xfull.size());
optcapability<OptMetric<T> >(this->p)->metric(expand(x),gfull);
g=gfull.submatrix(freeindex);
}

template<typename T>
int FrozenOptProblem<T>::degree(const int i) const {return optcapability<OptTrigonometric<T> >(this->p)->degree(freeindex[i]);}

template<typename T>
T FrozenOptProblem<T>::period(const int i) const {return optcapability<OptTrigonometric<T> >(this->p)->period(freeindex[i]);}

template<typename T>
int FrozenOptProblem<T>::nconstraints() const {return optcapability<OptConstraints<T> >(this->p)->nconstraints();}

template<typename T>
void FrozenOptProblem<T>::constraints(const NRVec<T> &x, NRVec<T> &c) const {optcapability<OptConstraints<T> >(this->p)->constraints(expand(x),c);}


//Jacobian of the residuals by forward differences, columns (r(x + h_j e_j) - r(x))/h_j, h_j = h max(1,|x_j|); resid(x,r) evaluates the residuals
template<typename T, typename F>
static void forward_difference_jacobian(F resid, const NRVec<T> &x, const NRVec<T> &r, NRMat<T> &J, const T h)
{
NRVec<T> xs(x);
for(int j=0; j<x.size(); ++j)
	{
	const T hj=h*std::max((T)1,std::abs(x[j]));
	xs.copyonwrite(); //the evaluator may keep a reference to the previous point as its best one
	xs[j]=x[j]+hj;
	NRVec<T> rs(r.size());
	resid(xs,rs);
	rs-=r;
	rs*= 1/hj;
	J.columnset(rs,j);
	xs.copyonwrite();
	xs[j]=x[j];
	}
}

template<typename T>
void OptLeastSquares<T>::jacobian_fd(const NRVec<T> &x, const NRVec<T> &r, NRMat<T> &J) const
{
forward_difference_jacobian([this](const NRVec<T> &xx, NRVec<T> &rr){this->residuals(xx,rr);},x,r,J,fd_step);
}

template<typename T>
T OptLeastSquares<T>::funcgrad(const NRVec<T> &x, NRVec<T> *grad) const
{
NRVec<T> r(this->nresiduals());
this->residuals(x,r);
if(grad)
	{
	NRMat<T> J(r.size(),x.size());
	if(!this->jacobian(x,J)) jacobian_fd(x,r,J);
	*grad = r*J; //J^T r
	}
return (r*r)/2;
}


//central difference (f(x+h e_i) - f(x-h e_i))/(2h); xs is a working copy of x (unshared on entry), restored on return
//the user's funcgrad may keep a shallow copy of xs, hence copyonwrite() after each call before writing again
template<typename T>
static T central_difference(const OptProblem<T> *p, NRVec<T> &xs, const NRVec<T> &x, const int i, const T h)
{
xs[i]=x[i]+h;
T d=p->funcgrad(xs,NULL);
xs.copyonwrite();
xs[i]=x[i]-h;
d-=p->funcgrad(xs,NULL);
xs.copyonwrite();
xs[i]=x[i];
return d/(2*h);
}


template<typename T>
ParameterShift<T>::ParameterShift(const OptProblem<T> *pp, const T hh) : OptProblemAdapter<T>(pp), trig(optcapability<OptTrigonometric<T> >(pp)), h(hh)
{
if(!trig) laerror("ParameterShift needs a problem providing OptTrigonometric");
}

//derivative of a trigonometric polynomial of degree R with frequencies 1..R (in units of w):
//f'(0) = sum_{mu=1}^{2R} (-1)^{mu-1} f(s_mu)/(4R sin^2(s_mu/2)), s_mu = (2mu-1)pi/(2R) (Wierichs et al., Quantum 6, 677 (2022))
template<typename T>
T ParameterShift<T>::funcgrad(const NRVec<T> &x, NRVec<T> *grad) const
{
T f=this->p->funcgrad(x,NULL);
if(!grad) return f;
grad->copyonwrite();
NRVec<T> xs(x);
xs.copyonwrite(); //working copy with shifted elements
for(int i=0; i<x.size(); ++i)
	{
	const int R=trig->degree(i);
	if(R==0) {(*grad)[i]=central_difference(this->p,xs,x,i,h); continue;}
	const T w=2*M_PI/trig->period(i);
	T d=0;
	for(int mu=1; mu<=2*R; ++mu)
		{
		const T s=(2*mu-1)*M_PI/(2*R);
		const T sn=std::sin(s/2);
		xs.copyonwrite(); //funcgrad may have kept a shallow copy of xs
		xs[i]=x[i]+s/w;
		d+= (mu%2 ? 1 : -1)*this->p->funcgrad(xs,NULL)/(4*R*sn*sn);
		}
	xs.copyonwrite();
	xs[i]=x[i];
	(*grad)[i]=w*d;
	}
return f;
}


template<typename T>
T FiniteDifference<T>::funcgrad(const NRVec<T> &x, NRVec<T> *grad) const
{
T f=this->p->funcgrad(x,NULL);
if(!grad) return f;
grad->copyonwrite();
NRVec<T> xs(x);
xs.copyonwrite(); //working copy with shifted elements
for(int i=0; i<x.size(); ++i) (*grad)[i]=central_difference(this->p,xs,x,i,h);
return f;
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


//quasi-Newton update of the inverse Hessian approximation H, returns false if the update was skipped
//Hinit = H is still the initial guess, it is then scaled by s.y/y.y (Nocedal & Wright eq. 6.20)
//for SR1 the scaling makes (s-Hy).y=0, so its first update is skipped and the next one starts from the scaled H
//rank-1 terms are done via rank2update(a,x,x) = 2a x x^T
template<typename T>
static bool QN_update_H(NRSMat<T> &H, bool &Hinit, const NRVec<T> &s, const NRVec<T> &y, const BFGS_hyperparameters<T> &hp)
{
const T sy = s*y;
const T snorm=s.norm();
const T ynorm=y.norm();
const bool curvature = sy > hp.my_sy_thr*snorm*ynorm;
if(Hinit && curvature) {H = (T)(sy/(ynorm*ynorm)); Hinit=false;}
switch(hp.my_update)
	{
	case QN_BFGS: //H+ = (I - rho s y^T) H (I - rho y s^T) + rho s s^T = H + s u^T + u s^T, rho=1/(y^T s), u = (rho^2 y^T H y + rho)/2 s - rho Hy
		{
		if(!curvature) return false;
		T rho = 1/sy;
		NRVec<T> Hy = H*y;
		NRVec<T> u = s*((rho*rho*(y*Hy) + rho)/2);
		u.axpy(-rho,Hy);
		H.rank2update((T)1,s,u);
		}
		break;
	case QN_DFP: //H+ = H - Hy (Hy)^T/(y^T H y) + s s^T/(y^T s)
		{
		if(!curvature) return false;
		NRVec<T> Hy = H*y;
		T yHy = y*Hy;
		if(yHy<=0) return false;
		H.rank2update(1/(2*sy),s,s);
		H.rank2update(-1/(2*yHy),Hy,Hy);
		}
		break;
	case QN_SR1: //H+ = H + w w^T/(w^T y), w = s - Hy; H may become indefinite
		{
		NRVec<T> w = H*y;
		w *= (T)-1;
		w += s;
		T wy = w*y;
		if(std::abs(wy) <= hp.my_sr1_thr*w.norm()*ynorm) return false;
		H.rank2update(1/(2*wy),w,w);
		}
		break;
	default: laerror("unknown quasi-Newton update");
	}
Hinit=false;
return true;
}


template<typename T>
bool optstate<T>::stop(const T ff, const T gn, const int needed)
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
if(e.remaining()<needed) {if(hp.my_verbose) std::cout <<name<<": maximum number of function evaluations reached\n"; return true;}
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


//function along the line x0 + t*u for linemin_brent, keeps the best point and counts the evaluations
template<typename T>
class linefunction {
public:
optevaluator<T> &e;
const NRVec<T> &x0;
const NRVec<T> &u;
int maxiter; //evaluation budget
int nev; //evaluations done
T tbest;
T fbest;
linefunction(optevaluator<T> &ee, const NRVec<T> &xx0, const T f0, const NRVec<T> &uu, const int maxit) : e(ee), x0(xx0), u(uu), maxiter(std::min(maxit,ee.remaining())), nev(0), tbest(0), fbest(f0) {};
NRVec<T> point(const T t) const {NRVec<T> x(x0); x.axpy(t,u); return x;}
T operator()(const T t) {T f=e(point(t)); ++nev; if(f<fbest) {fbest=f; tbest=t;} return f;}
bool exhausted() const {return nev>=maxiter;}
};


template<typename T>
T linemin_brent(optevaluator<T> &e, NRVec<T> &x, T &f, const NRVec<T> &u, const T t1, const T tol, const T abstol, const int maxiter, const int verbose)
{
const T golden=1.618033988749895; //expansion factor of the bracketing
const T cgold=0.3819660112501051; //golden section fraction (3-sqrt(5))/2
linefunction<T> phi(e,x,f,u,maxiter);

//bracketing: a,b,c with f(b)<=f(a), f(b)<=f(c)
T a=0, fa=f;
T b=t1, fb;
T c=0, fc=0;
bool bracketed=false;
if(phi.exhausted()) return 0;
fb=phi(b);
if(fb>fa) //uphill, try the opposite direction
	{
	if(phi.exhausted()) return 0;
	c= -t1; fc=phi(c);
	if(fc>=fa) {std::swap(a,b); std::swap(fa,fb); bracketed=true;} //minimum between -t1 and t1, middle point is 0
	else {b=c; fb=fc;}
	}
if(!bracketed) //expand downhill from a through b
	{
	for(;;)
		{
		if(phi.exhausted()) break;
		c= b+golden*(b-a); fc=phi(c);
		if(fc>=fb) {bracketed=true; break;}
		a=b; fa=fb; b=c; fb=fc;
		}
	}
if(verbose>1) std::cout <<"linemin bracket "<<a<<" "<<b<<" "<<c<<" f= "<<fa<<" "<<fb<<" "<<fc<<std::endl;

//Brent's method in [lo,hi]: xm = best point, w = second best, v = previous value of w
if(bracketed)
	{
	T lo=std::min(a,c), hi=std::max(a,c);
	T xm=b, w=b, v=b, fx=fb, fw=fb, fv=fb;
	T d=0, dprev=0; //last step and the one before
	const T tolabs=std::max(abstol,std::numeric_limits<T>::epsilon()*std::abs(t1)); //nonzero to guarantee termination
	while(!phi.exhausted())
		{
		T mid=(lo+hi)/2;
		T tol1=tol*std::abs(xm)+tolabs;
		T tol2=2*tol1;
		if(std::abs(xm-mid) <= tol2-(hi-lo)/2) break; //converged
		bool goldenstep=true;
		if(std::abs(dprev)>tol1) //try parabolic interpolation through xm, w, v
			{
			T r=(xm-w)*(fx-fv);
			T q=(xm-v)*(fx-fw);
			T p=(xm-v)*q-(xm-w)*r;
			q=2*(q-r);
			if(q>0) p= -p; else q= -q;
			if(std::abs(p)<std::abs(q*dprev/2) && p>q*(lo-xm) && p<q*(hi-xm)) //acceptable: inside the interval and less than half of the step before last
				{
				dprev=d;
				d=p/q;
				T t=xm+d;
				if(t-lo<tol2 || hi-t<tol2) d= mid>xm ? tol1 : -tol1;
				goldenstep=false;
				}
			}
		if(goldenstep) {dprev= xm>=mid ? lo-xm : hi-xm; d=cgold*dprev;}
		T t= std::abs(d)>=tol1 ? xm+d : xm+(d>0 ? tol1 : -tol1);
		T ft=phi(t);
		if(ft<=fx)
			{
			if(t>=xm) lo=xm; else hi=xm;
			v=w; fv=fw; w=xm; fw=fx; xm=t; fx=ft;
			}
		else
			{
			if(t<xm) lo=t; else hi=t;
			if(ft<=fw || w==xm) {v=w; fv=fw; w=t; fw=ft;}
			else if(ft<=fv || v==xm || v==w) {v=t; fv=ft;}
			}
		}
	}

if(verbose>1) std::cout <<"linemin t= "<<phi.tbest<<" f= "<<phi.fbest<<" evaluations "<<phi.nev<<std::endl;
if(phi.tbest==0) return 0;
x=phi.point(phi.tbest);
f=phi.fbest;
return phi.tbest;
}


template<typename T>
optresult<T> linesearch_minimize(const char *name, linesearch_direction<T> &dir, const OptProblem<T> *o, NRVec<T> &x, const linesearch_hyperparameters<T> &hp, hyperparameters<T> *achieved)
{
const int verbose=hp.my_verbose;
optevaluator<T> e(o,hp.my_maxfunc);
optstate<T> st(name,hp,e);

const int n=x.size();
NRVec<T> g(n);
T f=e(x,&g);
T alpha=0, dphi=0; //of the previous step

for(;; ++st.iter)
	{
	if(st.stop(f,g.norm())) break;

	NRVec<T> p = dir.direction(g);
	bool sd=dir.steepest(); //steepest descent step
	if(p*g >= 0)
		{
		if(dir.indefinite()) {if(verbose) std::cout <<name<<": not a descent direction, steepest descent step\n";}
		else //information spoiled numerically, restart
			{
			if(verbose) std::cout <<name<<": not a descent direction, restart\n";
			dir.reset();
			}
		p = g*(T)-1;
		sd=true;
		}

	//initial step length: from the previous step, or unit step in x for steepest descent, otherwise the quasi-Newton step
	T dphi_new = p*g;
	T alpha1;
	if(dir.alpha_from_previous() && alpha>0) alpha1=alpha*dphi/dphi_new;
	else alpha1= sd ? 1/p.norm() : (T)1;
	dphi=dphi_new;

	NRVec<T> xold(x), gold(g);
	T fold=f;
	alpha=linesearch_wolfe(e,x,f,g,p,alpha1,hp);
	if(alpha==0)
		{
		if(e.exhausted()) continue; //reported by stop()
		if(sd) {if(verbose) std::cout <<name<<": line search failed along steepest descent, giving up\n"; break;}
		if(verbose) std::cout <<name<<": line search failed, restart\n";
		dir.reset();
		continue;
		}

	NRVec<T> s = x-xold;
	NRVec<T> y = g-gold;
	if(verbose>1) std::cout <<name<<" step alpha= "<<alpha<<" |s|= "<<s.norm()<<" s.y= "<<s*y<<std::endl;
	if(!dir.update(s,y,p,g) && verbose) std::cout <<name<<": update skipped\n";
	st.step(f-fold,s.norm());
	}

return st.finish(achieved);
}


//BFGS, DFP, SR1: explicit inverse Hessian approximation
template<typename T>
class QN_direction : public linesearch_direction<T> {
public:
const BFGS_hyperparameters<T> &hp;
NRSMat<T> H;
bool Hinit; //H is the initial guess, rescale it after the first step
QN_direction(const BFGS_hyperparameters<T> &h, const int n) : hp(h), H(n), Hinit(true) {H=1.;};
NRVec<T> direction(const NRVec<T> &g) {NRVec<T> p = H*g; p *= (T)-1; return p;}
bool update(const NRVec<T> &s, const NRVec<T> &y, const NRVec<T> &p, const NRVec<T> &g) {return QN_update_H(H,Hinit,s,y,hp);}
void reset() {H=1.; Hinit=true;}
bool steepest() const {return Hinit;}
bool indefinite() const {return hp.my_update==QN_SR1;}
};


template<typename T>
optresult<T> BFGS_Optimizer<T>::minimize(const OptProblem<T> *o, NRVec<T> &x, const hyperparameters<T> *par, hyperparameters<T> *achieved)
{
static const char *names[]={"BFGS","DFP","SR1"};
BFGS_hyperparameters<T> hp;
if(par) hp.setfrom(*par);
if(hp.my_saddle_order) laerror("quasi-Newton with line search cannot search for saddle points");
QN_direction<T> dir(hp,x.size());
return linesearch_minimize(names[hp.my_update],dir,o,x,hp,achieved);
}


template<typename T>
optresult<T> GDIIS_Optimizer<T>::minimize(const OptProblem<T> *o, NRVec<T> &x, const hyperparameters<T> *par, hyperparameters<T> *achieved)
{
const char *name="GDIIS";
GDIIS_hyperparameters<T> hp;
if(par) hp.setfrom(*par);
if(hp.my_saddle_order) laerror("GDIIS cannot search for saddle points");
if(hp.my_diis_dim<1) laerror("GDIIS needs diis_dim >= 1");
const int verbose=hp.my_verbose;
optevaluator<T> e(o,hp.my_maxfunc);
optstate<T> st(name,hp,e);
const int n=x.size();
QN_direction<T> dir(hp,n);
DIIS<NRVec<T>,NRVec<T> > diis(hp.my_diis_dim);
bool fresh=true; //first step after a (re)start, the quasi-Newton step from the initial Hessian guess has no scale
NRVec<T> g(n);
T f=e(x,&g);

for(;; ++st.iter)
	{
	if(st.stop(f,g.norm())) break;

	NRVec<T> p = dir.direction(g); //quasi-Newton step, the error vector of DIIS
	if(p*g >= 0)
		{
		if(verbose) std::cout <<name<<": not a descent direction, restart\n";
		dir.reset();
		diis.reset();
		fresh=true;
		p = g*(T)-1;
		}

	NRVec<T> xold(x), gold(g);
	T fold=f;
	NRVec<T> s;
	bool accepted=false;
	if(!fresh) //DIIS extrapolation, otherwise a line search is done directly
		{
		s = x+p;
		NRVec<T> err(p);
		diis.extrapolate(s,err,verbose>1,hp.my_diis_eps);
		s -= x;
		if(s*g >= 0 || s.norm() > hp.my_maxsteplength)
			{
			if(verbose) std::cout <<name<<": unreasonable DIIS step, restart\n";
			diis.reset();
			s = p;
			T snorm=s.norm();
			if(snorm > hp.my_maxsteplength) s *= hp.my_maxsteplength/snorm;
			}
		NRVec<T> xnew = x+s;
		NRVec<T> gnew(n);
		T fnew=e(xnew,&gnew);
		if(fnew<f) {x=xnew; f=fnew; g=gnew; accepted=true;}
		else
			{
			if(verbose) std::cout <<name<<": no decrease, restart and line search\n";
			diis.reset();
			}
		}
	if(!accepted) //line search along the quasi-Newton step
		{
		if(e.exhausted()) continue; //reported by stop()
		T alpha=linesearch_wolfe(e,x,f,g,p,dir.steepest() ? 1/p.norm() : (T)1,hp);
		if(alpha==0)
			{
			if(e.exhausted()) continue;
			if(dir.steepest()) {if(verbose) std::cout <<name<<": line search failed along steepest descent, giving up\n"; break;}
			if(verbose) std::cout <<name<<": line search failed, restart\n";
			dir.reset();
			fresh=true;
			continue;
			}
		s = x-xold;
		}
	fresh=false;

	NRVec<T> y = g-gold;
	if(verbose>1) std::cout <<name<<" step |s|= "<<s.norm()<<" s.y= "<<s*y<<std::endl;
	if(!dir.update(s,y,p,g) && verbose) std::cout <<name<<": update skipped\n";
	st.step(f-fold,s.norm());
	}

return st.finish(achieved);
}


//L-BFGS: last m pairs s = x_{k+1}-x_k, y = g_{k+1}-g_k in a ring buffer
template<typename T>
class LBFGS_direction : public linesearch_direction<T> {
public:
const LBFGS_hyperparameters<T> &hp;
int m; //capacity
int count; //no. of pairs stored
int newest; //index of the newest pair
NRVec<NRVec<T> > S,Y;
NRVec<T> rho; //1/(s.y)
T gamma; //initial inverse Hessian H0 = gamma I, gamma = s.y/y.y of the newest pair (Nocedal & Wright eq. 7.20)
LBFGS_direction(const LBFGS_hyperparameters<T> &h) : hp(h), m(h.my_memory), count(0), newest(-1), S(m), Y(m), rho(m), gamma(1) {};
void push(const NRVec<T> &s, const NRVec<T> &y, const T sy) {newest=(newest+1)%m; S[newest]=s; Y[newest]=y; rho[newest]=1/sy; gamma=sy/(y*y); if(count<m) ++count;}
NRVec<T> direction(const NRVec<T> &g);
bool update(const NRVec<T> &s, const NRVec<T> &y, const NRVec<T> &p, const NRVec<T> &g)
	{
	T sy=s*y;
	if(sy <= hp.my_sy_thr*s.norm()*y.norm()) return false; //curvature condition violated numerically
	push(s,y,sy);
	return true;
	}
void reset() {count=0; newest= -1; gamma=1;}
bool steepest() const {return count==0;}
};

//returns -H g by the two-loop recursion (Nocedal & Wright alg. 7.4)
template<typename T>
NRVec<T> LBFGS_direction<T>::direction(const NRVec<T> &g)
{
NRVec<T> q(g);
NRVec<T> alpha(count);
int k=newest;
for(int i=0; i<count; ++i) //newest to oldest
	{
	alpha[i] = rho[k]*(S[k]*q);
	q.axpy(-alpha[i],Y[k]);
	if(i<count-1) k=(k+m-1)%m;
	}
q *= gamma;
for(int i=count-1; i>=0; --i) //oldest to newest
	{
	T beta = rho[k]*(Y[k]*q);
	q.axpy(alpha[i]-beta,S[k]);
	k=(k+1)%m;
	}
q *= (T)-1;
return q;
}


template<typename T>
optresult<T> LBFGS_Optimizer<T>::minimize(const OptProblem<T> *o, NRVec<T> &x, const hyperparameters<T> *par, hyperparameters<T> *achieved)
{
LBFGS_hyperparameters<T> hp;
if(par) hp.setfrom(*par);
if(hp.my_saddle_order) laerror("L-BFGS cannot search for saddle points");
if(hp.my_memory<1) laerror("L-BFGS needs memory >= 1");
LBFGS_direction<T> dir(hp);
return linesearch_minimize("L-BFGS",dir,o,x,hp,achieved);
}


//nonlinear CG: p_{k+1} = -g_{k+1} + beta p_k
template<typename T>
class CG_direction : public linesearch_direction<T> {
public:
const CG_hyperparameters<T> &hp;
int restart_interval;
int count; //no. of steps since the last restart
bool restart; //next direction is steepest descent
T beta;
T gg; //g.g of the gradient of the last direction
NRVec<T> d; //last search direction
CG_direction(const CG_hyperparameters<T> &h, const int n) : hp(h), restart_interval(h.my_restart>0 ? h.my_restart : n), count(0), restart(true), beta(0), gg(0) {};
NRVec<T> direction(const NRVec<T> &g) {NRVec<T> p = g*(T)-1; if(!restart) p.axpy(beta,d); gg=g*g; return p;}
bool update(const NRVec<T> &s, const NRVec<T> &y, const NRVec<T> &p, const NRVec<T> &g);
void reset() {restart=true; count=0;}
bool steepest() const {return restart;}
bool alpha_from_previous() const {return true;}
};

template<typename T>
bool CG_direction<T>::update(const NRVec<T> &s, const NRVec<T> &y, const NRVec<T> &p, const NRVec<T> &g)
{
d=p;
T gg1=g*g;
T gy=g*y;
T dy=p*y;
restart= ++count>=restart_interval || (hp.my_powell_thr>0 && std::abs(gg1-gy) >= hp.my_powell_thr*gg1); //g_{k+1}.g_k = gg1-gy
bool curvature= dy>0;
if(!restart) switch(hp.my_beta)
	{
	case CG_FR: beta=gg1/gg; break;
	case CG_PR: beta=std::max((T)0,gy/gg); break;
	case CG_HS: if(curvature) beta=std::max((T)0,gy/dy); else restart=true; break;
	case CG_DY: if(curvature) beta=gg1/dy; else restart=true; break;
	case CG_HZ: if(curvature) beta=std::max((gy-2*(p*g)*(y*y)/dy)/dy, -1/(p.norm()*std::min((T)0.01,std::sqrt(gg)))); else restart=true; break;
	default: laerror("unknown CG beta formula");
	}
if(restart)
	{
	count=0;
	if(hp.my_verbose>1) std::cout <<"CG restart\n";
	}
return true;
}


template<typename T>
optresult<T> CG_Optimizer<T>::minimize(const OptProblem<T> *o, NRVec<T> &x, const hyperparameters<T> *par, hyperparameters<T> *achieved)
{
static const char *names[]={"CG-FR","CG-PR+","CG-HS+","CG-DY","CG-HZ"};
CG_hyperparameters<T> hp;
if(par) hp.setfrom(*par);
if(hp.my_saddle_order) laerror("CG cannot search for saddle points");
CG_direction<T> dir(hp,x.size());
return linesearch_minimize(names[hp.my_beta],dir,o,x,hp,achieved);
}



template<typename T>
bool hessian_update(NRSMat<T> &B, const NRVec<T> &s, const NRVec<T> &y, const TR_hessian type, const T thr)
{
const NRVec<T> Bs=B*s;
const T ss=s*s;
NRVec<T> xi(y);
xi-=Bs; //y - Bs, vanishes if B already satisfies the secant condition
const T xis=xi*s;
switch(type)
	{
	case TR_BFGS: //B+ = B - Bs (Bs)^T/(s.Bs) + y y^T/(y.s)
		{
		const T sy=s*y, sBs=s*Bs;
		if(sy<=thr*std::sqrt(ss*(y*y)) || sBs<=0) return false;
		B.rank2update(1/(2*sy),y,y);
		B.rank2update(-1/(2*sBs),Bs,Bs);
		}
		break;
	case TR_SR1: //B+ = B + xi xi^T/(xi.s) (Nocedal & Wright eq. 6.24)
		if(std::abs(xis)<=thr*xi.norm()*std::sqrt(ss)) return false;
		B.rank2update(1/(2*xis),xi,xi);
		break;
	case TR_PSB: //B+ = B + (xi s^T + s xi^T)/(s.s) - (xi.s) s s^T/(s.s)^2 (Powell-symmetric-Broyden)
		B.rank2update(1/ss,xi,s);
		B.rank2update(-xis/(2*ss*ss),s,s);
		break;
	case TR_BOFILL: //phi*SR1 + (1-phi)*PSB, phi = (xi.s)^2/(|xi|^2 |s|^2), the SR1 term phi xi xi^T/(xi.s) = (xi.s) xi xi^T/(|xi|^2 |s|^2) has no small denominator
		{
		const T xixi=xi*xi;
		if(xixi==0) return false;
		const T phi=xis*xis/(xixi*ss);
		B.rank2update(xis/(2*xixi*ss),xi,xi);
		B.rank2update((1-phi)/ss,xi,s);
		B.rank2update(-(1-phi)*xis/(2*ss*ss),s,s);
		}
		break;
	default: laerror("unknown Hessian update");
	}
return true;
}


template<typename T>
optresult<T> trustregion_minimize(const char *name, trustregion_step<T> &stepper, const OptProblem<T> *o, NRVec<T> &x, const trustregion_hyperparameters<T> &hp, hyperparameters<T> *achieved)
{
const int verbose=hp.my_verbose;
const bool saddle=stepper.saddle();
const OptHessian<T> *hes=optcapability<OptHessian<T> >(o);
const bool exact= hp.my_hessian==TR_EXACT || (hp.my_hessian==TR_AUTO && hes);
if(exact && !hes) laerror("exact Hessian requested, but the problem does not provide OptHessian");
TR_hessian update=hp.my_hessian; //update of B between exact Hessians or without them: Bofill for saddle points; for minima SR1 after
if(update==TR_AUTO || update==TR_EXACT) update= saddle ? TR_BOFILL : (exact ? TR_SR1 : TR_BFGS); //an exact Hessian, which may be indefinite, BFGS from the identity
if(saddle && update==TR_BFGS) laerror("the BFGS update keeps the Hessian positive definite, unsuitable for saddle points");
optevaluator<T> e(o,hp.my_maxfunc);
optstate<T> st(name,hp,e);
const int n=x.size();
NRVec<T> g(n);
T f=e(x,&g);
NRSMat<T> B(n);
bool Binit=false; //B is the unscaled initial guess
if(exact) hes->hessian(x,B);
else {B=1.; Binit=true;}
int sinceexact=0;
T delta=hp.my_delta0;

for(;; ++st.iter)
	{
	if(st.stop(f,g.norm())) break;
	NRVec<T> p=stepper.step(g,B,delta);
	const T pnorm=p.norm();
	NRVec<T> Bp=B*p;
	const T pred= g*p + 0.5*(p*Bp); //predicted change of f
	NRVec<T> xn=x+p;
	NRVec<T> gn(n);
	const T fn=e(xn,&gn);
	const T ratio= pred!=0 ? (fn-f)/pred : 0;
	if(verbose>1) std::cout <<name<<" step |p|= "<<pnorm<<" delta= "<<delta<<" f= "<<fn<<" ratio= "<<ratio<<std::endl;
	bool accept;
	if(saddle) //the step is always taken, the radius follows the quality of the model
		{
		accept=true;
		if(ratio<0.25 || ratio>1.75) delta=0.25*pnorm;
		else if(ratio>0.75 && ratio<1.25 && pnorm>=0.99*delta) delta=std::min(2*delta,hp.my_deltamax);
		}
	else if(std::abs(pred)<=100*std::numeric_limits<T>::epsilon()*std::abs(f)) //change of f below its rounding errors, judged by the gradient
		{
		accept= gn.norm()<g.norm();
		if(!accept) delta=0.25*pnorm;
		}
	else //Nocedal & Wright alg. 4.1
		{
		accept= pred<0 && ratio>hp.my_eta;
		if(ratio<0.25) delta=0.25*pnorm;
		else if(ratio>0.75 && pnorm>=0.99*delta) delta=std::min(2*delta,hp.my_deltamax);
		}
	if(!accept)
		{
		if(verbose) std::cout <<name<<": step rejected\n";
		continue;
		}
	NRVec<T> y=gn-g;
	const T fold=f;
	x=xn; f=fn; g=gn;
	if(exact && ++sinceexact>=hp.my_hessian_recompute) {hes->hessian(x,B); sinceexact=0;}
	else
		{
		if(Binit) //scaling of the initial guess (cf. Nocedal & Wright eq. 6.20)
			{
			const T sy=p*y;
			if(sy>0) B=(T)((y*y)/sy);
			Binit=false;
			}
		if(!hessian_update(B,p,y,update,hp.my_update_thr) && verbose>1) std::cout <<name<<": Hessian update skipped\n";
		}
	st.step(f-fold,pnorm);
	}

//number of negative eigenvalues of the Hessian at the result
NRSMat<T> H(n);
if(hes) hes->hessian(x,H);
else H=B;
NRVec<T> w(n);
diagonalize(H,w,(NRMat<T> *)NULL);
int index=0;
for(int i=0; i<n; ++i) if(w[i]<0) ++index;
if(verbose) std::cout <<name<<": "<<index<<" negative eigenvalues of the"<<(hes ? "" : " approximate")<<" Hessian\n";
optresult<T> r=st.finish(achieved);
if(achieved) achieved->my_saddle_order=index;
return r;
}


//-a/(lambda+mu), for the components of a trust-region step in the eigenbasis of B
template<typename T>
class shifteddivision {
public:
T mu;
shifteddivision(const T m) : mu(m) {};
T operator()(const T &a, const T &lambda) const {return -a/(lambda+mu);}
};

//exact solution of min g.p + 1/2 p.B.p, |p| <= delta in the eigenbasis of B = V Lambda V^T: p(mu) = -sum_i (v_i.g)/(lambda_i+mu) v_i
//with mu = 0 for the Newton step inside the trust region, otherwise mu > -lambda_min from |p(mu)| = delta (More & Sorensen 1983)
template<typename T>
static NRVec<T> trustregion_eigenstep(const NRVec<T> &g, const NRSMat<T> &B, const T delta)
{
const int n=g.size();
NRSMat<T> Bc(B);
NRVec<T> lam(n);
NRMat<T> V(n,n);
diagonalize(Bc,lam,&V); //ascending eigenvalues, eigenvectors in the columns of V
const NRVec<T> gt=g*V; //components of g in the eigenbasis
const T lmin=lam[0];
if(lmin>0) //Newton step if inside the trust region
	{
	NRVec<T> pt(gt);
	pt.call_on_me(shifteddivision<T>(0),lam);
	if(pt.norm()<=delta) return V*pt;
	}
const T lo=std::max((T)0,-lmin);
const T tol=1e-12*std::max(std::abs(lam[0]),std::abs(lam[n-1]));
//hard case: g has no component along the eigenvectors of lambda_min and the step from the others at mu=-lambda_min is inside the trust region
if(lmin<=0)
	{
	int nlow=0;
	while(nlow<n && lam[nlow]-lmin<=tol) ++nlow;
	NRVec<T> gtlow=gt.subvector(0,nlow-1);
	if(gtlow.norm()<=1e-12*g.norm())
		{
		NRVec<T> pt(gt);
		pt.copyonwrite();
		for(int i=0; i<nlow; ++i) pt[i]=0;
		for(int i=nlow; i<n; ++i) pt[i]= -gt[i]/(lam[i]+lo);
		T pn=pt.norm();
		if(pn<=delta)
			{
			NRVec<T> p=V*pt;
			p.axpy(std::sqrt(delta*delta-pn*pn),V.column(0));
			return p;
			}
		}
	}
//|p(mu)| decreases from infinity at mu=lo; root by Newton's method on 1/|p(mu)| - 1/delta, safeguarded by bisection
T mulo=lo, muhi=lo+g.norm()/delta; //|p(muhi)| <= delta
T mu=muhi;
NRVec<T> pt(gt);
for(int it=0; it<100; ++it)
	{
	pt=gt;
	pt.call_on_me(shifteddivision<T>(mu),lam);
	const T pn=pt.norm();
	if(std::abs(pn-delta)<=1e-10*delta) break;
	if(pn>delta) mulo=mu; else muhi=mu;
	NRVec<T> qt(pt); //|q|^2 = sum_i pt_i^2/(lambda_i+mu) = -d|p|^2/dmu /2
	qt.call_on_me(shifteddivision<T>(mu),lam);
	const T qq= -(pt*qt);
	T munew= qq>0 ? mu+(pn*pn/qq)*(pn-delta)/delta : mu;
	if(!(munew>mulo && munew<muhi)) munew=(mulo+muhi)/2;
	mu=munew;
	}
return V*pt;
}

//approximate solution of the trust-region subproblem by truncated conjugate gradients (Steihaug 1983; Nocedal & Wright alg. 7.2)
template<typename T>
static NRVec<T> trustregion_steihaug(const NRVec<T> &g, const NRSMat<T> &B, const T delta, int maxit)
{
const int n=g.size();
if(maxit<=0) maxit=n;
NRVec<T> z((T)0,n);
const T gnorm=g.norm();
if(gnorm==0) return z;
const T eps=std::min((T)0.5,std::sqrt(gnorm))*gnorm;
NRVec<T> r(g);
NRVec<T> d=g*(T)-1;
for(int j=0; j<maxit; ++j)
	{
	NRVec<T> Bd=B*d;
	const T dBd=d*Bd;
	const T rr=r*r;
	NRVec<T> z1(z);
	if(dBd>0) z1.axpy(rr/dBd,d);
	if(dBd<=0 || z1.norm()>=delta) //negative curvature or outside: to the boundary along d, |z + tau d| = delta
		{
		const T a=d*d, b=2*(z*d), c=z*z-delta*delta;
		z.axpy((-b+std::sqrt(b*b-4*a*c))/(2*a),d);
		return z;
		}
	r.axpy(rr/dBd,Bd);
	z=z1;
	if(r.norm()<eps) return z;
	d*= (r*r)/rr;
	d-=r;
	}
return z;
}

template<typename T>
class TR_newtonstep : public trustregion_step<T> {
public:
const TrustRegion_hyperparameters<T> &hp;
TR_newtonstep(const TrustRegion_hyperparameters<T> &h) : hp(h) {};
NRVec<T> step(const NRVec<T> &g, const NRSMat<T> &B, const T delta) {return hp.my_subproblem==TR_STEIHAUG ? trustregion_steihaug(g,B,delta,hp.my_maxcg) : trustregion_eigenstep(g,B,delta);}
};

template<typename T>
optresult<T> TrustRegion_Optimizer<T>::minimize(const OptProblem<T> *o, NRVec<T> &x, const hyperparameters<T> *par, hyperparameters<T> *achieved)
{
TrustRegion_hyperparameters<T> hp;
if(par) hp.setfrom(*par);
if(hp.my_saddle_order) laerror("TrustRegion_Optimizer minimizes, use PRFO_Optimizer for saddle points");
TR_newtonstep<T> stepper(hp);
return trustregion_minimize(hp.my_subproblem==TR_STEIHAUG ? "TR-Steihaug" : "TR-Newton",stepper,o,x,hp,achieved);
}


//shift of the RFO step for the modes from..to-1: largest (maximization) or lowest (minimization) eigenvalue of the bordered matrix [[Lambda, gt],[gt^T, 0]]
template<typename T>
static T rfo_shift(const NRVec<T> &lam, const NRVec<T> &gt, const int from, const int to, const bool largest)
{
const int m=to-from;
NRSMat<T> A(m+1);
A=(T)0;
for(int i=0; i<m; ++i) {A(i,i)=lam[from+i]; A(m,i)=gt[from+i];}
NRVec<T> w(m+1);
diagonalize(A,w,(NRMat<T> *)NULL);
return largest ? w[m] : w[0];
}

//P-RFO step: in the eigenbasis of B, p_i = -gt_i/(lambda_i - mu) with mu from the RFO matrices of the k maximized and of the minimized modes,
//scaled to the trust region if longer
template<typename T>
class PRFO_step : public trustregion_step<T> {
public:
int k; //no. of maximized modes
PRFO_step(const int kk) : k(kk) {};
bool saddle() const {return k>0;}
NRVec<T> step(const NRVec<T> &g, const NRSMat<T> &B, const T delta)
	{
	const int n=g.size();
	NRSMat<T> Bc(B);
	NRVec<T> lam(n);
	NRMat<T> V(n,n);
	diagonalize(Bc,lam,&V);
	const NRVec<T> gt=g*V;
	const T mumax= k>0 ? rfo_shift(lam,gt,0,k,true) : 0;
	const T mumin= k<n ? rfo_shift(lam,gt,k,n,false) : 0;
	NRVec<T> pt(n);
	for(int i=0; i<n; ++i)
		{
		const T den=lam[i]-(i<k ? mumax : mumin);
		pt[i]= den!=0 ? -gt[i]/den : 0;
		}
	NRVec<T> p=V*pt;
	const T pn=p.norm();
	if(pn>delta) p*= delta/pn;
	return p;
	}
};

template<typename T>
optresult<T> PRFO_Optimizer<T>::minimize(const OptProblem<T> *o, NRVec<T> &x, const hyperparameters<T> *par, hyperparameters<T> *achieved)
{
PRFO_hyperparameters<T> hp;
if(par) hp.setfrom(*par);
if(hp.my_saddle_order<0 || hp.my_saddle_order>x.size()) laerror("invalid saddle order in PRFO_Optimizer");
PRFO_step<T> stepper(hp.my_saddle_order);
return trustregion_minimize("P-RFO",stepper,o,x,hp,achieved);
}


//Jacobian for LM: from the problem, or by forward differences with counted evaluations; false if the evaluation budget is insufficient
template<typename T>
static bool lm_jacobian(optevaluator<T> &e, const OptResiduals<T> *res, const NRVec<T> &x, const NRVec<T> &r, NRMat<T> &J, const T h)
{
if(res->jacobian(x,J)) return true;
if(e.remaining()<x.size()) return false;
forward_difference_jacobian([&e,res](const NRVec<T> &xx, NRVec<T> &rr){e(res,xx,rr);},x,r,J,h);
return true;
}

template<typename T>
optresult<T> LM_Optimizer<T>::minimize(const OptProblem<T> *o, NRVec<T> &x, const hyperparameters<T> *par, hyperparameters<T> *achieved)
{
const char *name="Levenberg-Marquardt";
LM_hyperparameters<T> hp;
if(par) hp.setfrom(*par);
if(hp.my_saddle_order) laerror("Levenberg-Marquardt cannot search for saddle points");
const OptResiduals<T> *res=optcapability<OptResiduals<T> >(o);
if(!res) laerror("Levenberg-Marquardt needs a least-squares problem providing OptResiduals");
const int verbose=hp.my_verbose;
const int n=x.size(), m=res->nresiduals();
optevaluator<T> e(o,hp.my_maxfunc);
optstate<T> st(name,hp,e);
NRVec<T> r(m);
T f=e(res,x,r);
NRMat<T> J(m,n);
const bool fdjac= !res->jacobian(x,J); //the problem does not provide the Jacobian
bool budget= !fdjac || lm_jacobian(e,res,x,r,J,hp.my_fd_step);
NRSMat<T> A=J.transposedtimes(); //J^T J
NRVec<T> g=r*J; //J^T r, the gradient of f
NRVec<T> diag(n);
for(int i=0; i<n; ++i) diag[i]=((const NRSMat<T> &)A)(i,i);
T mu=hp.my_scaling ? hp.my_tau : hp.my_tau*diag.amax(), nu=2;

for(; budget; ++st.iter)
	{
	if(st.stop(f,g.norm(),fdjac ? n+1 : 1)) break;
	//damped step (J^T J + mu D) h = -J^T r
	NRSMat<T> M(A);
	M.copyonwrite();
	for(int i=0; i<n; ++i) M(i,i)+= mu*(hp.my_scaling ? diag[i] : (T)1);
	NRVec<T> h=g*(T)-1;
	linear_solve(M,h);
	NRVec<T> xn=x+h;
	NRVec<T> rn(m);
	const T fn=e(res,xn,rn);
	NRVec<T> Dh(h);
	if(hp.my_scaling) Dh*=diag;
	Dh*=mu;
	Dh-=g;
	const T pred=(h*Dh)/2; //predicted reduction L(0)-L(h) of the linear model
	const T rho=(f-fn)/pred;
	if(verbose>1) std::cout <<name<<" step |h|= "<<h.norm()<<" mu= "<<mu<<" f= "<<fn<<" gain ratio= "<<rho<<std::endl;
	if(rho>0)
		{
		const T fold=f;
		x=xn; r=rn; f=fn;
		NRMat<T> Jn(m,n);
		budget=lm_jacobian(e,res,x,r,Jn,hp.my_fd_step);
		J=Jn;
		A=J.transposedtimes();
		g=r*J;
		for(int i=0; i<n; ++i) diag[i]=((const NRSMat<T> &)A)(i,i);
		mu*= std::max((T)1/3,1-std::pow(2*rho-1,(T)3));
		nu=2;
		st.step(fold-f,h.norm());
		}
	else
		{
		mu*=nu; nu*=2;
		const T eps=std::numeric_limits<T>::epsilon();
		//no further improvement at machine precision (cf. Madsen, Nielsen & Tingleff eq. 3.16), tested per component to be scale invariant:
		//h_i^2 <= (10 eps)^2 (x_i^2+eps^2) for all i (with |h|<=10 eps |x| badly scaled parameters stopped too early in float)
		NRVec<T> hh(h); hh*=h;
		NRVec<T> xx(x); xx*=x; xx+=eps*eps;
		hh/=xx;
		if(hh.amax()<=100*eps*eps)
			{
			if(verbose) std::cout <<name<<": step at machine precision\n";
			st.converged=true;
			break;
			}
		}
	}
if(!budget && verbose) std::cout <<name<<": maximum number of function evaluations reached\n";
st.f=f;
return st.finish(achieved);
}


//simplex of n+1 vertices with their function values
template<typename T>
class NM_simplex {
public:
int n;
NRVec<NRVec<T> > v; //vertices
NRVec<T> f; //function values
NRVec<T> sum; //sum of the vertices
int best, second, worst; //indices of the lowest, second highest and highest function value
NM_simplex(const NRVec<T> &x0, const T h, optevaluator<T> &e);
void order();
void sumup() {sum=v[0]; for(int i=1; i<=n; ++i) sum+=v[i];}
void replace(const int i, const NRVec<T> &x, const T fx) {sum-=v[i]; sum+=x; v[i]=x; f[i]=fx;}
NRVec<T> centroid() const {NRVec<T> c(sum); c-=v[worst]; c/=(T)n; return c;} //of all vertices except the worst one
NRVec<T> point(const NRVec<T> &c, const T t) const {NRVec<T> x = c*(1+t); x.axpy(-t,v[worst]); return x;} //c + t*(c - x_worst)
T diameter() const;
void shrink(const T sigma, optevaluator<T> &e);
};

//initial simplex x0, x0 + h e_i
template<typename T>
NM_simplex<T>::NM_simplex(const NRVec<T> &x0, const T h, optevaluator<T> &e) : n(x0.size()), v(n+1), f(n+1), best(0), second(0), worst(0)
{
v[0]=x0;
f[0]=e(x0);
for(int i=1; i<=n; ++i)
	{
	v[i]=x0;
	v[i].copyonwrite();
	v[i][i-1] += h;
	f[i]=e(v[i]);
	}
sumup();
}

template<typename T>
void NM_simplex<T>::order()
{
best=worst=0;
for(int i=1; i<=n; ++i)
	{
	if(f[i]<f[best]) best=i;
	if(f[i]>=f[worst]) worst=i;
	}
second=best;
for(int i=0; i<=n; ++i) if(i!=worst && f[i]>=f[second]) second=i;
}

template<typename T>
T NM_simplex<T>::diameter() const
{
T d=0;
for(int i=0; i<=n; ++i) if(i!=best)
	{
	NRVec<T> dx = v[i]-v[best];
	d=std::max(d,dx.norm());
	}
return d;
}

//all vertices move towards the best one, as far as the evaluation budget allows
template<typename T>
void NM_simplex<T>::shrink(const T sigma, optevaluator<T> &e)
{
for(int i=0; i<=n; ++i) if(i!=best)
	{
	if(e.exhausted()) break;
	NRVec<T> x = v[best]*(1-sigma);
	x.axpy(sigma,v[i]);
	v[i]=x;
	f[i]=e(x);
	}
sumup();
}


template<typename T>
optresult<T> NelderMead_Optimizer<T>::minimize(const OptProblem<T> *o, NRVec<T> &x, const hyperparameters<T> *par, hyperparameters<T> *achieved)
{
const char *name="Nelder-Mead";
NelderMead_hyperparameters<T> hp;
if(par) hp.setfrom(*par);
if(hp.my_saddle_order) laerror("Nelder-Mead cannot search for saddle points");
const int verbose=hp.my_verbose;
const int n=x.size();
T alpha=hp.my_alpha, gamma=hp.my_gamma, rho=hp.my_rho, sigma=hp.my_sigma;
if(hp.my_adaptive) {alpha=1; gamma=1+(T)2/n; rho=0.75-(T)0.5/n; sigma=1-(T)1/n;}
optevaluator<T> e(o,hp.my_maxfunc);
optstate<T> st(name,hp,e);
NM_simplex<T> s(x,hp.my_initstep,e);
const T huge=std::numeric_limits<T>::max();

for(;; ++st.iter)
	{
	s.order();
	T spread=s.f[s.worst]-s.f[s.best];
	st.step(spread, spread<hp.my_f_thr ? s.diameter() : huge); //diameter needed only if the spread is small
	if(st.stop(s.f[s.best])) break;
	if(st.iter%(n+1)==0) s.sumup(); //avoid accumulation of rounding errors

	const char *op;
	NRVec<T> c=s.centroid();
	NRVec<T> xr=s.point(c,alpha);
	T fr=e(xr);
	if(fr<s.f[s.best]) //try expansion
		{
		if(e.exhausted()) {s.replace(s.worst,xr,fr); continue;}
		NRVec<T> xe=s.point(c,alpha*gamma);
		T fe=e(xe);
		if(fe<fr) {s.replace(s.worst,xe,fe); op="expansion";}
		else {s.replace(s.worst,xr,fr); op="reflection";}
		}
	else if(fr<s.f[s.second]) {s.replace(s.worst,xr,fr); op="reflection";}
	else //contraction, outside if the reflected point is better than the worst one
		{
		bool outside= fr<s.f[s.worst];
		if(e.exhausted()) {if(outside) s.replace(s.worst,xr,fr); continue;}
		NRVec<T> xc=s.point(c, outside ? alpha*rho : -rho);
		T fc=e(xc);
		if(outside ? fc<=fr : fc<s.f[s.worst]) {s.replace(s.worst,xc,fc); op= outside ? "outside contraction" : "inside contraction";}
		else {s.shrink(sigma,e); op="shrink";}
		}
	if(verbose>1) std::cout <<name<<" "<<op<<std::endl;
	}

s.order();
x=s.v[s.best];
st.step(s.f[s.worst]-s.f[s.best],s.diameter()); //exact values for achieved
return st.finish(achieved);
}



template<typename T>
optresult<T> Powell_Optimizer<T>::minimize(const OptProblem<T> *o, NRVec<T> &x, const hyperparameters<T> *par, hyperparameters<T> *achieved)
{
const char *name="Powell";
Powell_hyperparameters<T> hp;
if(par) hp.setfrom(*par);
if(hp.my_saddle_order) laerror("Powell cannot search for saddle points");
const int verbose=hp.my_verbose;
const T abstol=0.01*hp.my_x_thr; //line minima have to be more precise than the requested accuracy in x
const int n=x.size();
optevaluator<T> e(o,hp.my_maxfunc);
optstate<T> st(name,hp,e);
T f=e(x);

NRVec<NRVec<T> > u(n); //normalized directions
for(int i=0; i<n; ++i) {u[i]=NRVec<T>((T)0,n); u[i][i]=1;}
NRVec<T> tstep(hp.my_initstep,n); //trial steps of the line minimizations, from the last step along each direction

for(;; ++st.iter)
	{
	if(st.stop(f)) break;
	NRVec<T> x0(x);
	T f0=f;
	T delta=0; //largest decrease along a single direction
	int ibig=0;
	for(int i=0; i<n && !e.exhausted(); ++i)
		{
		T fprev=f;
		T t=linemin_brent(e,x,f,u[i],tstep[i],hp.my_linmin_tol,abstol,hp.my_maxlinemin,verbose);
		if(t!=0) tstep[i]=std::abs(t);
		if(fprev-f>delta) {delta=fprev-f; ibig=i;}
		}

	//new direction x-x0 replaces the one of the largest decrease, if Powell's test indicates it is worth
	NRVec<T> unew = x-x0;
	T unorm=unew.norm();
	if(unorm>0 && !e.exhausted())
		{
		NRVec<T> xe = x+unew; //extrapolated point
		T fe=e(xe);
		T a=f0-f-delta;
		T b=f0-fe;
		if(fe<f0 && 2*(f0-2*f+fe)*a*a < delta*b*b)
			{
			if(verbose>1) std::cout <<name<<": direction "<<ibig<<" replaced\n";
			unew /= unorm;
			u[ibig]=u[n-1]; tstep[ibig]=tstep[n-1];
			u[n-1]=unew; tstep[n-1]=unorm;
			if(!e.exhausted()) linemin_brent(e,x,f,unew,unorm,hp.my_linmin_tol,abstol,hp.my_maxlinemin,verbose);
			}
		}
	NRVec<T> dx = x-x0;
	st.step(f0-f,dx.norm());
	}

return st.finish(achieved);
}



//natural gradient (G + reg I)^-1 g from the gradient of an inner estimator and the metric G of the problem
template<typename T>
class natural_gradient_estimator : public gradient_estimator<T> {
public:
gradient_estimator<T> &inner;
const OptMetric<T> *m;
T reg;
natural_gradient_estimator(gradient_estimator<T> &in, const OptMetric<T> *mm, const T r) : inner(in), m(mm), reg(r) {};
T operator()(optevaluator<T> &e, const NRVec<T> &x, NRVec<T> &g, const int k)
	{
	T f=inner(e,x,g,k);
	NRSMat<T> G(x.size());
	m->metric(x,G);
	if(reg!=0) G+=reg;
	linear_solve(G,g);
	return f;
	}
int cost() const {return inner.cost();}
bool exact() const {return inner.exact();}
T calibrate(optevaluator<T> &e, const NRVec<T> &x, const firstorder_hyperparameters<T> &hp) {return inner.calibrate(e,x,hp);}
};


template<typename T>
optresult<T> firstorder_minimize(const char *name, firstorder_step<T> &stepper, const OptProblem<T> *o, NRVec<T> &x, const firstorder_hyperparameters<T> &hp, hyperparameters<T> *achieved, gradient_estimator<T> *est)
{
gradient_estimator<T> exact;
gradient_estimator<T> *plain = est ? est : &exact;
const OptMetric<T> *metric=NULL;
if(hp.my_natural)
	{
	metric=optcapability<OptMetric<T> >(o);
	if(!metric) laerror("natural gradient needs a problem providing the metric (OptMetric)");
	}
natural_gradient_estimator<T> natural(*plain,metric,hp.my_metric_reg);
gradient_estimator<T> &grad = hp.my_natural ? natural : *plain;
optevaluator<T> e(o,hp.my_maxfunc);
optstate<T> st(name,hp,e);
const int n=x.size();
const T lr0=grad.calibrate(e,x,hp);
if(hp.my_verbose && lr0!=hp.my_lr) std::cout <<name<<": calibrated learning rate "<<lr0<<std::endl;
NRVec<T> g(n);
T f=grad(e,x,g,0);

for(;; ++st.iter)
	{
	if(st.stop(f,grad.exact() ? g.norm() : (T)-1,grad.cost())) break;
	T lr=lr0/std::pow(1+hp.my_lr_decay*st.iter,hp.my_lr_power);
	NRVec<T> dx = stepper.step(g,st.iter+1);
	dx *= -lr;
	if(hp.my_weight_decay!=0) dx.axpy(-lr*hp.my_weight_decay,x);
	x += dx;
	T fold=f;
	NRVec<T> gnew(n); //fresh vector, the stepper may keep the previous gradient
	f=grad(e,x,gnew,st.iter+1);
	g=gnew;
	if(grad.exact()) st.step(f-fold,dx.norm()); //for estimates, small changes are no sign of convergence
	}

if(hp.my_return_best && e.fbest<f)
	{
	if(hp.my_verbose) std::cout <<name<<": returning the best point found, f= "<<e.fbest<<std::endl;
	x=e.xbest;
	st.f=e.fbest;
	}
optresult<T> r=st.finish(achieved);
firstorder_hyperparameters<T> *fach=dynamic_cast<firstorder_hyperparameters<T> *>(achieved);
if(fach) fach->my_lr=lr0;
return r;
}


//SGD with momentum
template<typename T>
class SGD_stepper : public firstorder_step<T> {
public:
const SGD_hyperparameters<T> &hp;
NRVec<T> v; //velocity
SGD_stepper(const SGD_hyperparameters<T> &h) : hp(h) {};
NRVec<T> step(const NRVec<T> &g, const int k)
	{
	if(k==1) v=g;
	else {v *= hp.my_momentum; v += g;}
	if(!hp.my_nesterov) return v;
	NRVec<T> d(g);
	d.axpy(hp.my_momentum,v);
	return d;
	}
};


template<typename T>
optresult<T> QNG_Optimizer<T>::minimize(const OptProblem<T> *o, NRVec<T> &x, const hyperparameters<T> *par, hyperparameters<T> *achieved)
{
QNG_hyperparameters<T> hp;
if(par) hp.setfrom(*par);
hp.my_natural=true; //also if the caller passed e.g. SGD_hyperparameters
if(hp.my_saddle_order) laerror("QNG cannot search for saddle points");
SGD_stepper<T> stepper(hp);
return firstorder_minimize("QNG",stepper,o,x,hp,achieved);
}


//approximate solution of min c.z subject to A z >= r and |(z_0,...,z_{nd-1})| <= delta, nd = nz or nz-1, from a feasible z by an active-set method:
//z moves along the projection of -c orthogonal to the normals of the active constraints until a further constraint becomes active,
//the trust region boundary is reached (which finishes the calculation as in COBYLA), or the multipliers show optimality;
//the projection is computed by least squares, which tolerates dependent normals at degenerate points
template<typename T>
static void lp_trustregion(const NRVec<T> &c, const NRMat<T> &A, const NRVec<T> &r, const int nd, const T delta, NRVec<T> &z)
{
const int nz=c.size();
const int m=r.size();
const T tiny=1e3*std::numeric_limits<T>::epsilon()*c.norm();
NRVec<int> active(m);
active.clear();
NRVec<int> cols(nz);
for(int j=0; j<nz; ++j) cols[j]=j;
const NRVec<T> &zc=z; //element access for reading only
const int maxit=2*(m+nz)+10;
for(int it=0; it<maxit; ++it)
	{
	int k=0;
	for(int i=0; i<m; ++i) if(active[i]) ++k;
	NRVec<int> rows(k);
	k=0;
	for(int i=0; i<m; ++i) if(active[i]) rows[k++]=i;
	NRVec<T> s = c*(T)-1; //search direction
	NRVec<T> lambda; //multipliers of the active constraints
	if(k>0)
		{
		NRMat<T> N=A.submatrix(rows,cols); //normals of the active constraints
		NRMat<T> Nt=N.transpose();
		lambda=svd_solve(Nt,c); //least squares c ~ N^T lambda
		NRVec<T> nl=lambda*N;
		s+=nl;
		}
	if(s.norm()<=tiny) //stationary on the active set, optimal if the multipliers are nonnegative
		{
		int drop= -1;
		T lmin=0;
		for(int j=0; j<k; ++j) if(lambda[j]<lmin) {lmin=lambda[j]; drop=j;}
		if(drop<0) return;
		active[rows[drop]]=0;
		continue;
		}
	//step to the first blocking constraint
	NRVec<T> Az=A*z;
	NRVec<T> As=A*s;
	T alpha=std::numeric_limits<T>::max();
	int block= -1;
	for(int i=0; i<m; ++i) if(!active[i] && As[i]<0)
		{
		T a=std::max((T)0,(Az[i]-r[i])/(-As[i]));
		if(a<alpha) {alpha=a; block=i;}
		}
	//step to the trust region boundary |zd + a sd| = delta
	T zz=z*z, ss=s*s, zs=z*s;
	if(nd<nz) {zz-=zc[nz-1]*zc[nz-1]; ss-=s[nz-1]*s[nz-1]; zs-=zc[nz-1]*s[nz-1];}
	if(ss>0)
		{
		T ab=(-zs+std::sqrt(std::max((T)0,zs*zs-ss*(zz-delta*delta))))/ss;
		if(ab<alpha) {z.axpy(ab,s); return;}
		}
	if(block<0) return;
	z.axpy(alpha,s);
	active[block]=1;
	}
}


//COBYLA trust region step: minimizes g.d subject to the linearized constraints c + A d >= 0 within |d| <= rho; if they cannot be satisfied
//in the trust region, the largest violation t is minimized first (stage 1) and then kept, c + A d >= -t (stage 2) (Powell 1994)
template<typename T>
static NRVec<T> cobyla_step(const NRVec<T> &g, const NRMat<T> &A, const NRVec<T> &c, const T rho)
{
const int n=g.size();
const int m=c.size();
NRVec<T> d((T)0,n);
if(m==0) {T gn=g.norm(); if(gn>0) d.axpy(-rho/gn,g); return d;}
T t=std::max((T)0,-c.min());
if(t>0) //stage 1 in the variables (d,t): min t subject to c_i + a_i.d + t >= 0, t >= 0
	{
	NRMat<T> A1(m+1,n+1);
	A1.clear();
	A1.storesubmatrix(0,0,A);
	for(int i=0; i<=m; ++i) A1(i,n)=1;
	NRVec<T> r1(m+1); //r1_i = -c_i, r1_m = 0
	r1.clear();
	for(int i=0; i<m; ++i) r1[i]= -c[i];
	NRVec<T> c1((T)0,n+1);
	c1[n]=1;
	NRVec<T> z((T)0,n+1);
	z[n]=t;
	lp_trustregion(c1,A1,r1,n,rho,z);
	t=z[n];
	for(int j=0; j<n; ++j) d[j]=z[j];
	}
//stage 2: min g.d subject to c_i + a_i.d >= -t
NRVec<T> r2 = c*(T)-1;
r2-=t;
lp_trustregion(g,A,r2,n,rho,d);
return d;
}


template<typename T>
static T cobyla_violation(const NRVec<T> &c) {return c.size()>0 ? std::max((T)0,-c.min()) : 0;}

//f and the constraint values at x
template<typename T>
static T cobyla_eval(optevaluator<T> &e, const OptConstraints<T> *con, const NRVec<T> &x, NRVec<T> &c)
{
T f=e(x);
if(con) con->constraints(x,c);
return f;
}


template<typename T>
optresult<T> Cobyla_Optimizer<T>::minimize(const OptProblem<T> *o, NRVec<T> &x, const hyperparameters<T> *par, hyperparameters<T> *achieved)
{
const char *name="COBYLA";
Cobyla_hyperparameters<T> hp;
if(par) hp.setfrom(*par);
if(hp.my_saddle_order) laerror("COBYLA cannot search for saddle points");
const T acc_sigma=0.25, acc_eta=2.1, geo_gamma=0.5; //simplex acceptability and geometry step (Powell 1994)
const int verbose=hp.my_verbose;
const int n=x.size();
const OptConstraints<T> *con=optcapability<OptConstraints<T> >(o);
const int m= con ? con->nconstraints() : 0;
optevaluator<T> e(o,hp.my_maxfunc);
optstate<T> st(name,hp,e);

//initial simplex x, x + rhobeg e_j with function and constraint values
T rho=hp.my_rhobeg; //resolution, lower bound of the trust region radius
T delta=rho; //trust region radius, always equal to rho in Powell's original COBYLA
const bool flexible=hp.my_flexible_radius;
T mu=0; //penalty parameter of the merit function f + mu*(largest constraint violation)
NRVec<NRVec<T> > v(n+1), cv(n+1);
NRVec<T> fv(n+1);
for(int j=0; j<=n; ++j)
	{
	v[j]=x;
	if(j>0) {v[j].copyonwrite(); v[j][j-1]+=rho;}
	cv[j]=NRVec<T>(m);
	fv[j]=cobyla_eval(e,con,v[j],cv[j]);
	}

int best=0;
T fprev=fv[0];
NRVec<int> other(n); //vertices other than the best one
bool lastsuccess=false; //the last trust region step was successful, the next one follows without checking the simplex
for(;; ++st.iter)
	{
	best=0;
	for(int j=1; j<=n; ++j) if(fv[j]+mu*cobyla_violation(cv[j]) < fv[best]+mu*cobyla_violation(cv[best])) best=j;
	if(st.iter>0) {st.step(fprev-fv[best],rho); fprev=fv[best];}
	if(st.stop(fv[best])) break;
	for(int j=0, k=0; j<=n; ++j) if(j!=best) other[k++]=j;

	//displacements d_k of the other vertices from the best one and the dual basis B = D^-T, b_k.d_l = delta_kl
	NRMat<T> D(n,n);
	for(int k=0; k<n; ++k) {NRVec<T> dk=v[other[k]]-v[best]; D.rowset(dk,k);}
	NRMat<T> B=D.inverse().transpose();
	//linear models: gradient g of f, gradients of the constraints in the rows of A, values c0 at the best vertex
	NRVec<T> df(n);
	for(int k=0; k<n; ++k) df[k]=fv[other[k]]-fv[best];
	NRVec<T> g=df*B;
	const NRVec<T> c0(cv[best]);
	NRMat<T> A;
	if(m>0)
		{
		NRMat<T> dC(n,m);
		for(int k=0; k<n; ++k) {NRVec<T> dc=cv[other[k]]-c0; dC.rowset(dc,k);}
		A=dC.transpose()*B;
		}

	//acceptability of the simplex: distances eta_k = |d_k| <= acc_eta delta and sigma_k = 1/|b_k| >= acc_sigma rho of the vertices from the opposite faces
	int worst= -1;
	T etamax=acc_eta*delta;
	for(int k=0; k<n; ++k) {T eta=D.row(k).norm(); if(eta>etamax) {etamax=eta; worst=k;}}
	if(worst<0)
		{
		T sigmin=acc_sigma*rho;
		for(int k=0; k<n; ++k) {T sigma=1/B.row(k).norm(); if(sigma<sigmin) {sigmin=sigma; worst=k;}}
		}

	if(!lastsuccess && worst>=0) //geometry step: the worst vertex moves to the best one + a step along the normal of its opposite face, with the sign preferred by the merit model
		{
		NRVec<T> dir=B.row(worst);
		//step length geo_gamma rho, or with flexible radius max(min(0.1 DIST, 0.5 delta), rho) (NEWUOA eq. (6.16)), DIST the distance of the replaced vertex
		T len= flexible ? std::max(std::min((T)0.1*D.row(worst).norm(),(T)0.5*delta),rho) : geo_gamma*rho;
		dir*= len/dir.norm();
		T mplus=g*dir, mminus= -mplus;
		if(m>0)
			{
			NRVec<T> ad=A*dir;
			NRVec<T> cp(c0), cm(c0);
			cp+=ad; cm-=ad;
			mplus+=mu*cobyla_violation(cp);
			mminus+=mu*cobyla_violation(cm);
			}
		if(mminus<mplus) dir*= (T)-1;
		if(verbose>1) std::cout <<name<<": geometry step of vertex "<<other[worst]<<std::endl;
		NRVec<T> xn=v[best]+dir;
		NRVec<T> cn(m);
		T fn=cobyla_eval(e,con,xn,cn);
		v[other[worst]]=xn; fv[other[worst]]=fn; cv[other[worst]]=cn;
		continue;
		}

	//trust region step
	lastsuccess=false;
	NRVec<T> d=cobyla_step(g,A,c0,delta);
	const T dnorm=d.norm();
	const T deltaold=delta; //trust region radius of this step
	T ratio= -1; //reduction ratio of the merit function
	const bool shortstep= dnorm<0.5*rho;
	bool failed=shortstep;
	if(shortstep && flexible) delta=std::max((T)0.1*delta,rho); //NEWUOA Box 15
	if(!failed)
		{
		const T v0=cobyla_violation(c0);
		NRVec<T> cd(c0);
		if(m>0) {NRVec<T> ad=A*d; cd+=ad;}
		T fpred= -(g*d); //predicted reductions of f and of the violation
		T vpred=v0-cobyla_violation(cd);
		if(vpred>0 && mu < -1.5*fpred/vpred) //penalty has to make the predicted reduction of the merit function positive
			{
			mu= -2*fpred/vpred;
			if(verbose>1) std::cout <<name<<": penalty parameter "<<mu<<std::endl;
			int b2=best;
			for(int j=0; j<=n; ++j) if(fv[j]+mu*cobyla_violation(cv[j]) < fv[b2]+mu*cobyla_violation(cv[b2])) b2=j;
			if(b2!=best) continue; //another vertex is best with the new penalty
			}
		T pred=fpred+mu*vpred;
		if(pred<=0) failed=true;
		else
			{
			NRVec<T> xn=v[best]+d;
			NRVec<T> cn(m);
			T fn=cobyla_eval(e,con,xn,cn);
			T phi0=fv[best]+mu*v0;
			T phin=fn+mu*cobyla_violation(cn);
			ratio=(phi0-phin)/pred;
			if(verbose>1) std::cout <<name<<" step |d|= "<<dnorm<<" delta= "<<delta<<" rho= "<<rho<<" f= "<<fn<<" violation= "<<cobyla_violation(cn)<<" ratio= "<<ratio<<std::endl;
			//vertex replaced by the new point (Powell 1994): the one giving the largest simplex volume, i.e. the largest barycentric coordinate |l_k|
			//(without improvement only if the volume increases), but preferably the most distant one beyond 1.1 rho, if the new point
			//keeps an acceptable distance |l_k| sigma_k from its opposite face; the best vertex is kept, a better new point becomes the best one
			const bool improved= phin<phi0;
			NRVec<T> l=B*d;
			int rep= -1;
			T lmax= improved ? 0 : 1;
			for(int k=0; k<n; ++k) if(std::abs(l[k])>lmax) {lmax=std::abs(l[k]); rep=k;}
			T edgmax=1.1*delta;
			for(int k=0; k<n; ++k)
				{
				T sigma=1/B.row(k).norm();
				T sigbar=std::abs(l[k])*sigma;
				if(sigbar>=acc_sigma*rho || sigbar>=sigma)
					{
					T dist;
					if(improved) {NRVec<T> dk=v[other[k]]-xn; dist=dk.norm();}
					else dist=D.row(k).norm();
					if(dist>edgmax) {edgmax=dist; rep=k;}
					}
				}
			if(rep>=0) {v[other[rep]]=xn; fv[other[rep]]=fn; cv[other[rep]]=cn;}
			}
		}
	if(flexible && !shortstep) //revision of the trust region radius by the reduction ratio (NEWUOA eq. (7.1))
		{
		T dint;
		if(ratio<=0.1) dint=0.5*dnorm;
		else if(ratio<=0.7) dint=std::max(dnorm,(T)0.5*delta);
		else dint=std::max(2*dnorm,(T)0.5*delta);
		delta= dint<=1.5*rho ? rho : dint;
		}
	if(!failed && ratio>=0.1) {lastsuccess=true; continue;}

	//failed trust region step: improve the simplex at the next iteration, or reduce rho if it is acceptable
	if(worst>=0) continue;
	//the work with the current rho continues if the step was not restricted by rho or improved the point (NEWUOA Box 10, where |d| > rho is tested;
	//here the radius of the step is compared instead, as |d| = delta = rho can be slightly above rho due to rounding)
	if(flexible && ((!shortstep && deltaold>rho) || delta>rho || ratio>0)) continue;
	if(rho<=hp.my_rhoend) {st.converged=true; break;}
	if(flexible) //NEWUOA eq. (7.6) and Box 12
		{
		T rhonew;
		if(rho<=16*hp.my_rhoend) rhonew=hp.my_rhoend;
		else if(rho<=250*hp.my_rhoend) rhonew=std::sqrt(rho*hp.my_rhoend);
		else rhonew=0.1*rho;
		delta=std::max((T)0.5*rho,rhonew);
		rho=rhonew;
		}
	else
		{
		rho*=0.5;
		if(rho<=1.5*hp.my_rhoend) rho=hp.my_rhoend;
		delta=rho;
		}
	if(verbose>1) std::cout <<name<<": rho= "<<rho<<" delta= "<<delta<<std::endl;
	}

best=0;
for(int j=1; j<=n; ++j) if(fv[j]+mu*cobyla_violation(cv[j]) < fv[best]+mu*cobyla_violation(cv[best])) best=j;
x=v[best];
st.f=fv[best];
if(verbose && m>0) std::cout <<name<<": largest constraint violation "<<cobyla_violation(cv[best])<<std::endl;
return st.finish(achieved);
}


//trigonometric polynomial a_0 + sum_{k=1}^R (a_k cos(k w t) + b_k sin(k w t)) of the single variable t = x[0]
template<typename T>
class trigpolynomial : public OptProblem<T> {
public:
NRVec<T> a, b;
T w;
trigpolynomial(const int R, const T ww) : a(R+1), b(R+1), w(ww) {};
T funcgrad(const NRVec<T> &x, NRVec<T> *grad) const
	{
	if(grad) laerror("trigpolynomial provides only the function");
	T f=a[0];
	for(int k=1; k<a.size(); ++k) f+= a[k]*std::cos(k*w*x[0]) + b[k]*std::sin(k*w*x[0]);
	return f;
	}
};

//exact minimization of f along x_i, a trigonometric polynomial of degree R with the given period: reconstruction from
//2R+1 equidistant samples t_j = j*period/(2R+1) (the one at t=0 is the current f, 2R new evaluations) by discrete Fourier transform
//and minimization of the reconstruction; returns false if the evaluation budget is insufficient (x,f then unchanged)
template<typename T>
static bool trig_minimize(optevaluator<T> &e, NRVec<T> &x, T &f, const int i, const int R, const T period, const bool reevaluate, const int verbose)
{
const int N=2*R+1;
if(e.remaining()<N-1+(reevaluate?1:0)) return false;
const T w=2*M_PI/period;
const T xi=((const NRVec<T> &)x)[i]; //const access, x may be shared with the evaluator
NRVec<T> fs(N); //samples
fs[0]=f;
NRVec<T> xs(x);
for(int j=1; j<N; ++j)
	{
	xs.copyonwrite(); //the evaluator may keep a reference to the previous sample as its best point
	xs[i]=xi+j*period/N;
	fs[j]=e(xs);
	}

trigpolynomial<T> p(R,w);
p.a[0]=fs.sum()/N;
for(int k=1; k<=R; ++k)
	{
	p.a[k]=p.b[k]=0;
	for(int j=0; j<N; ++j) {T phi=2*M_PI*k*j/N; p.a[k]+= fs[j]*std::cos(phi); p.b[k]+= fs[j]*std::sin(phi);}
	p.a[k]*= (T)2/N; p.b[k]*= (T)2/N;
	}

T t, fmin;
if(R==1) //closed form: a_1 cos(wt) + b_1 sin(wt) = A cos(wt - phi), minimum at wt = phi + pi
	{
	t=(std::atan2(p.b[1],p.a[1])+M_PI)/w;
	fmin=p.a[0]-std::sqrt(p.a[1]*p.a[1]+p.b[1]*p.b[1]);
	}
else //global minimum of the reconstruction: best of a dense grid, refined by Brent's method (no evaluations of the problem)
	{
	const int M=8*N;
	NRVec<T> tt(1);
	int jbest=0;
	T fbest=0;
	for(int j=0; j<M; ++j) {tt[0]=j*period/M; T ft=p.funcgrad(tt,NULL); if(j==0 || ft<fbest) {fbest=ft; jbest=j;}}
	optevaluator<T> pe(&p);
	tt[0]=jbest*period/M;
	NRVec<T> u((T)1,1);
	linemin_brent(pe,tt,fbest,u,period/M,(T)1e-10,(T)1e-12*period,100);
	t=tt[0];
	fmin=fbest;
	}
t-= period*std::floor(t/period+0.5); //shortest step, t in (-period/2,period/2]
x.copyonwrite();
x[i]=xi+t;
f= reevaluate ? e(x) : fmin;
if(verbose>1) std::cout <<"trigonometric minimization of parameter "<<i<<" degree "<<R<<" step "<<t<<" f= "<<f<<std::endl;
return true;
}


template<typename T>
optresult<T> Rotosolve_Optimizer<T>::minimize(const OptProblem<T> *o, NRVec<T> &x, const hyperparameters<T> *par, hyperparameters<T> *achieved)
{
const char *name="Rotosolve";
Rotosolve_hyperparameters<T> hp;
if(par) hp.setfrom(*par);
if(hp.my_saddle_order) laerror("Rotosolve cannot search for saddle points");
const int verbose=hp.my_verbose;
const T abstol=0.01*hp.my_x_thr; //as in Powell's method
const int n=x.size();
const OptTrigonometric<T> *trig=optcapability<OptTrigonometric<T> >(o);
if(!trig && verbose) std::cout <<name<<": the problem is not trigonometric, line minimizations only\n";
optevaluator<T> e(o,hp.my_maxfunc);
optstate<T> st(name,hp,e);
T f=e(x);
NRVec<T> tstep(hp.my_initstep,n); //trial steps of the line minimizations

for(;; ++st.iter)
	{
	if(st.stop(f)) break;
	NRVec<T> x0(x);
	T f0=f;
	bool budget=true;
	for(int i=0; i<n && budget; ++i)
		{
		const int R= trig ? trig->degree(i) : 0;
		if(R>0) budget=trig_minimize(e,x,f,i,R,trig->period(i),hp.my_reevaluate,verbose);
		else
			{
			NRVec<T> u((T)0,n);
			u[i]=1;
			T t=linemin_brent(e,x,f,u,tstep[i],hp.my_linmin_tol,abstol,hp.my_maxlinemin,verbose);
			if(t!=0) tstep[i]=std::abs(t);
			budget=!e.exhausted();
			}
		}
	NRVec<T> dx = x-x0;
	st.step(f0-f,dx.norm());
	if(!budget) //incomplete sweep, no convergence test
		{
		if(verbose) std::cout <<name<<": maximum number of function evaluations reached\n";
		st.f=f;
		break;
		}
	}

return st.finish(achieved);
}


template<typename T>
optresult<T> SGD_Optimizer<T>::minimize(const OptProblem<T> *o, NRVec<T> &x, const hyperparameters<T> *par, hyperparameters<T> *achieved)
{
SGD_hyperparameters<T> hp;
if(par) hp.setfrom(*par);
if(hp.my_saddle_order) laerror("SGD cannot search for saddle points");
SGD_stepper<T> stepper(hp);
return firstorder_minimize(hp.my_nesterov ? "SGD-Nesterov" : "SGD",stepper,o,x,hp,achieved);
}


//elementwise functions for Adam
template<typename T>
static T elementmax(const T &a, const T &b) {return a>b ? a : b;}

//m/(sqrt(v)/sqrtc2 + eps) with the bias correction sqrtc2 = sqrt(1-beta2^k)
template<typename T>
class adamratio {
public:
T sqrtc2, eps;
adamratio(const T s, const T e) : sqrtc2(s), eps(e) {};
T operator()(const T &m, const T &v) const {return m/(std::sqrt(v)/sqrtc2+eps);}
};


//Adam, AdamW, AMSGrad
template<typename T>
class Adam_stepper : public firstorder_step<T> {
public:
const Adam_hyperparameters<T> &hp;
NRVec<T> m; //first moment estimate
NRVec<T> v; //second moment estimate
NRVec<T> vmax; //maximum of the second moments for AMSGrad
T beta1k, beta2k; //beta1^k, beta2^k
Adam_stepper(const Adam_hyperparameters<T> &h) : hp(h), beta1k(1), beta2k(1) {};
NRVec<T> step(const NRVec<T> &g, const int k)
	{
	NRVec<T> g2(g);
	g2 *= g;
	if(k==1) {m = g*(1-hp.my_beta1); v = g2*(1-hp.my_beta2);}
	else
		{
		m *= hp.my_beta1; m.axpy(1-hp.my_beta1,g);
		v *= hp.my_beta2; v.axpy(1-hp.my_beta2,g2);
		}
	beta1k *= hp.my_beta1;
	beta2k *= hp.my_beta2;
	if(hp.my_amsgrad) {if(k==1) vmax=v; else vmax.call_on_me(elementmax<T>,v);}
	NRVec<T> d = m*(1/(1-beta1k));
	d.call_on_me(adamratio<T>(std::sqrt(1-beta2k),hp.my_eps), hp.my_amsgrad ? vmax : v);
	return d;
	}
};


template<typename T>
optresult<T> Adam_Optimizer<T>::minimize(const OptProblem<T> *o, NRVec<T> &x, const hyperparameters<T> *par, hyperparameters<T> *achieved)
{
Adam_hyperparameters<T> hp;
if(par) hp.setfrom(*par);
if(hp.my_saddle_order) laerror("Adam cannot search for saddle points");
Adam_stepper<T> stepper(hp);
return firstorder_minimize(hp.my_amsgrad ? "AMSGrad" : (hp.my_weight_decay!=0 ? "AdamW" : "Adam"),stepper,o,x,hp,achieved);
}



//SPSA gradient estimate (g_hat)_i = (f(x+c_k Delta) - f(x-c_k Delta))/(2 c_k Delta_i), Delta_i = +-1, averaged over resamplings
template<typename T>
class SPSA_estimator : public gradient_estimator<T> {
public:
const SPSA_hyperparameters<T> &hp;
SPSA_estimator(const SPSA_hyperparameters<T> &h) : hp(h) {};
T operator()(optevaluator<T> &e, const NRVec<T> &x, NRVec<T> &g, const int k);
int cost() const {return 2*hp.my_resamplings;}
bool exact() const {return false;}
T calibrate(optevaluator<T> &e, const NRVec<T> &x, const firstorder_hyperparameters<T> &h);
};

//returns the average of f(x+c_k Delta) and f(x-c_k Delta) as an estimate of f(x)
template<typename T>
T SPSA_estimator<T>::operator()(optevaluator<T> &e, const NRVec<T> &x, NRVec<T> &g, const int k)
{
const int n=x.size();
const T ck=hp.my_c/std::pow((T)(k+1),hp.my_gamma);
g.clear();
T fsum=0;
for(int r=0; r<hp.my_resamplings; ++r)
	{
	NRVec<T> delta(n); //fresh vector, element access is safe
	for(int i=0; i<n; ++i) delta[i]= randdouble()<0.5 ? -1 : 1;
	NRVec<T> xp(x), xm(x);
	xp.axpy(ck,delta);
	xm.axpy(-ck,delta);
	T fp=e(xp);
	T fm=e(xm);
	g.axpy((fp-fm)/(2*ck),delta); //1/Delta_i = Delta_i
	fsum+=fp+fm;
	}
g /= (T)hp.my_resamplings;
return fsum/(2*hp.my_resamplings);
}

//learning rate such that the first step lr*|g_hat| has on average the target length
template<typename T>
T SPSA_estimator<T>::calibrate(optevaluator<T> &e, const NRVec<T> &x, const firstorder_hyperparameters<T> &h)
{
if(hp.my_calibrate<=0) return hp.my_lr;
T gsum=0;
int ncal=0;
for(; ncal<hp.my_calibrate && e.remaining()>=2*cost(); ++ncal) //leave the budget for the first estimate
	{
	NRVec<T> g(x.size());
	(*this)(e,x,g,0);
	gsum+=g.norm();
	}
if(ncal==0 || gsum==0) return hp.my_lr;
return hp.my_target_step*ncal/gsum;
}


//plain gradient step for SPSA
template<typename T>
class plain_stepper : public firstorder_step<T> {
public:
NRVec<T> step(const NRVec<T> &g, const int k) {return g;}
};


template<typename T>
optresult<T> SPSA_Optimizer<T>::minimize(const OptProblem<T> *o, NRVec<T> &x, const hyperparameters<T> *par, hyperparameters<T> *achieved)
{
SPSA_hyperparameters<T> hp;
if(par) hp.setfrom(*par);
if(hp.my_saddle_order) laerror("SPSA cannot search for saddle points");
if(hp.my_resamplings<1) laerror("SPSA needs resamplings >= 1");
plain_stepper<T> stepper;
SPSA_estimator<T> est(hp);
return firstorder_minimize("SPSA",stepper,o,x,hp,achieved,&est);
}



//force instantiation
#define INSTANTIZE(T) \
template T linesearch_wolfe(optevaluator<T> &e, NRVec<T> &x, T &f, NRVec<T> &g, const NRVec<T> &p, const T alpha1, const linesearch_hyperparameters<T> &lp); \
template class optstate<T>; \
template class ParameterShift<T>; \
template class FiniteDifference<T>; \
template class FrozenOptProblem<T>; \
template class BFGS_Optimizer<T>; \
template class LBFGS_Optimizer<T>; \
template class GDIIS_Optimizer<T>; \
template class TrustRegion_Optimizer<T>; \
template class PRFO_Optimizer<T>; \
template class LM_Optimizer<T>; \
template class OptLeastSquares<T>; \
template bool hessian_update(NRSMat<T> &B, const NRVec<T> &s, const NRVec<T> &y, const TR_hessian type, const T thr); \
template class CG_Optimizer<T>; \
template class NelderMead_Optimizer<T>; \
template class Powell_Optimizer<T>; \
template class Rotosolve_Optimizer<T>; \
template class Cobyla_Optimizer<T>; \
template class SGD_Optimizer<T>; \
template class QNG_Optimizer<T>; \
template class Adam_Optimizer<T>; \
template class SPSA_Optimizer<T>; \
template T linemin_brent(optevaluator<T> &e, NRVec<T> &x, T &f, const NRVec<T> &u, const T t1, const T tol, const T abstol, const int maxiter, const int verbose); \
template optresult<T> linesearch_minimize(const char *name, linesearch_direction<T> &dir, const OptProblem<T> *o, NRVec<T> &x, const linesearch_hyperparameters<T> &hp, hyperparameters<T> *achieved); \
template optresult<T> firstorder_minimize(const char *name, firstorder_step<T> &stepper, const OptProblem<T> *o, NRVec<T> &x, const firstorder_hyperparameters<T> &hp, hyperparameters<T> *achieved, gradient_estimator<T> *est); \
template optresult<T> trustregion_minimize(const char *name, trustregion_step<T> &stepper, const OptProblem<T> *o, NRVec<T> &x, const trustregion_hyperparameters<T> &hp, hyperparameters<T> *achieved); \

INSTANTIZE(double)
INSTANTIZE(float)


}//namespace

