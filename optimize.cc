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



//force instantiation
template double linesearch_wolfe(optevaluator<double> &e, NRVec<double> &x, double &f, NRVec<double> &g, const NRVec<double> &p, const double alpha1, const linesearch_hyperparameters<double> &lp);
template class optstate<double>;
template class testproblem<double>;
template class BFGS_Optimizer<double>;
template class LBFGS_Optimizer<double>;
template class CG_Optimizer<double>;

}//namespace

