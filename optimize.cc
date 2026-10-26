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
#include "la_random.h"

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



template<typename T>
optresult<T> firstorder_minimize(const char *name, firstorder_step<T> &stepper, const OptProblem<T> *o, NRVec<T> &x, const firstorder_hyperparameters<T> &hp, hyperparameters<T> *achieved, gradient_estimator<T> *est)
{
gradient_estimator<T> exact;
gradient_estimator<T> &grad = est ? *est : exact;
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
template double linesearch_wolfe(optevaluator<double> &e, NRVec<double> &x, double &f, NRVec<double> &g, const NRVec<double> &p, const double alpha1, const linesearch_hyperparameters<double> &lp);
template class optstate<double>;
template class testproblem<double>;
template class BFGS_Optimizer<double>;
template class LBFGS_Optimizer<double>;
template class CG_Optimizer<double>;
template class NelderMead_Optimizer<double>;
template class Powell_Optimizer<double>;
template class SGD_Optimizer<double>;
template class Adam_Optimizer<double>;
template class SPSA_Optimizer<double>;
template double linemin_brent(optevaluator<double> &e, NRVec<double> &x, double &f, const NRVec<double> &u, const double t1, const double tol, const double abstol, const int maxiter, const int verbose);

}//namespace

