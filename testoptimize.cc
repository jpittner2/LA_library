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

//regression tests of the optimization methods and of the functions of the vector and matrix classes they rely on
//each check prints PASS or FAIL, the exit code is the number of failures

#include <complex>
#include <iomanip>
#include <fcntl.h>
#include <unistd.h>
#include "la.h"
#include "testoptimize.h"
#include "diis.h"

using namespace std;
using namespace LA;

static int npass=0, nfail=0;

static void check(const char *what, const bool ok, const double value=0)
{
if(ok) ++npass; else ++nfail;
cout <<(ok?"PASS ":"FAIL ")<<what;
if(!ok) cout <<"   (value "<<value<<")";
cout <<endl;
}


//auxiliary test problems

//extended Rosenbrock function
class rosenbrock : public OptProblem<double> {
public:
double funcgrad(const NRVec<double> &x, NRVec<double> *g) const
	{
	int n=x.size();
	double r=0;
	if(g) {g->copyonwrite(); *g=0.;}
	for(int i=0; i<n-1; ++i)
		{
		double a=1-x[i], b=x[i+1]-x[i]*x[i];
		r+=a*a+100*b*b;
		if(g) {(*g)[i]+= -2*a-400*x[i]*b; (*g)[i+1]+=200*b;}
		}
	return r;
	}
};

//coupled quadratic sum_i (i+1)/2 (x_i + x_{i+1}/2)^2, minimum 0 at 0
class coupledquadratic : public OptProblem<double> {
public:
double funcgrad(const NRVec<double> &x, NRVec<double> *g) const
	{
	int n=x.size();
	double r=0;
	if(g) {g->copyonwrite(); *g=0.;}
	for(int i=0; i<n; ++i)
		{
		double s=x[i]+0.5*x[(i+1)%n];
		r+=0.5*(i+1)*s*s;
		if(g) {(*g)[i]+=(i+1)*s; (*g)[(i+1)%n]+=0.5*(i+1)*s;}
		}
	return r;
	}
};

//quadratic with additive noise sigma on f and on the gradient components
class noisyquadratic : public OptProblem<double> {
public:
double sigma;
noisyquadratic(const double s) : sigma(s) {};
static double exact(const NRVec<double> &x) {double r=0; for(int i=0; i<x.size(); ++i) r+=0.5*(i+1)*x[i]*x[i]; return r;}
double funcgrad(const NRVec<double> &x, NRVec<double> *g) const
	{
	if(g) {g->copyonwrite(); for(int i=0; i<x.size(); ++i) (*g)[i]=(i+1)*x[i]+sigma*randdoublesigned();}
	return exact(x)+sigma*randdoublesigned();
	}
};

//trigonometric of degree 3 with period 5 in x_0, not trigonometric in x_1
class trigdegree3 : public OptProblem<double>, public OptTrigonometric<double> {
public:
static double tr(const double t) {double w=2*M_PI/5; return 0.3+cos(w*t)-0.7*sin(w*t)+0.4*cos(2*w*t)+0.2*sin(2*w*t)-0.5*cos(3*w*t)+0.9*sin(3*w*t);}
static double dtr(const double t) {double w=2*M_PI/5; return w*(-sin(w*t)-0.7*cos(w*t)-0.8*sin(2*w*t)+0.4*cos(2*w*t)+1.5*sin(3*w*t)+2.7*cos(3*w*t));}
double funcgrad(const NRVec<double> &x, NRVec<double> *g) const
	{
	double s=1.5+0.5*tanh(x[1]);
	if(g) {g->copyonwrite(); (*g)[0]=dtr(x[0])*s; double th=tanh(x[1]); (*g)[1]=tr(x[0])*0.5*(1-th*th)+2*(x[1]-0.3);}
	return tr(x[0])*s+(x[1]-0.3)*(x[1]-0.3);
	}
int degree(const int i) const {return i==0 ? 3 : 0;}
double period(const int i) const {return 5;}
};

//counts the evaluations of the wrapped problem, its capabilities remain accessible
class counting : public OptProblemAdapter<double> {
public:
mutable int count;
counting(const OptProblem<double> *pp) : OptProblemAdapter<double>(pp), count(0) {};
double funcgrad(const NRVec<double> &x, NRVec<double> *g) const {++count; return this->p->funcgrad(x,g);}
};

//constrained classics with known solutions
class quadratic_linearconstraint : public OptProblem<double>, public OptConstraints<double> { //min (x0-1)^2+(x1-2)^2, x0+x1<=1: f=2 at (0,1)
public:
double funcgrad(const NRVec<double> &x, NRVec<double> *g) const {return (x[0]-1)*(x[0]-1)+(x[1]-2)*(x[1]-2);}
int nconstraints() const {return 1;}
void constraints(const NRVec<double> &x, NRVec<double> &c) const {c.copyonwrite(); c[0]=1-x[0]-x[1];}
};
class product_disk : public OptProblem<double>, public OptConstraints<double> { //min x0 x1, x0^2+x1^2<=1: f=-1/2 (Powell 1994)
public:
double funcgrad(const NRVec<double> &x, NRVec<double> *g) const {return x[0]*x[1];}
int nconstraints() const {return 1;}
void constraints(const NRVec<double> &x, NRVec<double> &c) const {c.copyonwrite(); c[0]=1-x[0]*x[0]-x[1]*x[1];}
};
class rosensuzuki : public OptProblem<double>, public OptConstraints<double> { //f=-44 at (0,1,2,-1)
public:
double funcgrad(const NRVec<double> &x, NRVec<double> *g) const {return x[0]*x[0]+x[1]*x[1]+2*x[2]*x[2]+x[3]*x[3]-5*x[0]-5*x[1]-21*x[2]+7*x[3];}
int nconstraints() const {return 3;}
void constraints(const NRVec<double> &x, NRVec<double> &c) const
	{
	c.copyonwrite();
	c[0]=8-x[0]*x[0]-x[1]*x[1]-x[2]*x[2]-x[3]*x[3]-x[0]+x[1]-x[2]+x[3];
	c[1]=10-x[0]*x[0]-2*x[1]*x[1]-x[2]*x[2]-2*x[3]*x[3]+x[0]+x[3];
	c[2]=5-2*x[0]*x[0]-x[1]*x[1]-x[2]*x[2]-2*x[0]+x[1]+x[3];
	}
};
class sum_circle : public OptProblem<double>, public OptConstraints<double> { //min x0+x1, x0^2+x1^2=1 as two inequalities: f=-sqrt(2)
public:
double funcgrad(const NRVec<double> &x, NRVec<double> *g) const {return x[0]+x[1];}
int nconstraints() const {return 2;}
void constraints(const NRVec<double> &x, NRVec<double> &c) const {c.copyonwrite(); c[0]=x[0]*x[0]+x[1]*x[1]-1; c[1]= -c[0];}
};


//auxiliary functions

static NRVec<double> startpoint(const int n, const int seed, const double scale=1)
{
NRVec<double> x(n);
srand(seed);
x.randomize(scale);
return x;
}

static NRVec<double> rosenbrockstart(const int n)
{
NRVec<double> x(n);
for(int i=0; i<n; ++i) x[i]= i%2 ? 1. : -1.2;
return x;
}

static double fd_gradient_error(const OptProblem<double> &p, const NRVec<double> &x)
{
const int n=x.size();
NRVec<double> g(n);
p.funcgrad(x,&g);
double err=0;
for(int i=0; i<n; ++i)
	{
	NRVec<double> a(x), b(x);
	a.copyonwrite(); b.copyonwrite();
	a[i]+=1e-6; b[i]-=1e-6;
	err=max(err,abs((p.funcgrad(a,NULL)-p.funcgrad(b,NULL))/2e-6-g[i]));
	}
return err;
}

//minimizes and returns f, x is modified, achieved filled
static double run(Optimizer<double> &o, const OptProblem<double> *p, NRVec<double> &x, hyperparameters<double> &h, hyperparameters<double> *achieved=NULL, bool *converged=NULL)
{
h.my_verbose=0;
x.copyonwrite();
optresult<double> r=o.minimize(p,x,&h,achieved);
if(converged) *converged=r.converged;
return r.f;
}

//number of maxfunc limits from 1 to maxmax for which the evaluation budget is exceeded or misreported (from minfunc on, below it the initial evaluations are unavoidable)
static int maxfunc_violations(Optimizer<double> &o, const OptProblem<double> *p, const NRVec<double> &x0, hyperparameters<double> &h, const int minfunc, const int maxmax)
{
int bad=0;
for(int mf=minfunc; mf<=maxmax; ++mf)
	{
	counting cp(p);
	NRVec<double> x(x0);
	h.my_maxfunc=mf;
	hyperparameters<double> a;
	run(o,&cp,x,h,&a);
	if(cp.count>mf || a.my_maxfunc!=cp.count) ++bad;
	}
return bad;
}


//numerical Fubini-Study metric of one qubit RY(b) RX(a)|0>
static void fubinistudy(const double a, const double b, double g[2][2])
{
typedef complex<double> C;
const double h=1e-5;
C psi[2], d[2][2];
for(int k=-1; k<=1; ++k) //k=-1: the state, k=0,1: derivatives
	{
	C p[2], m[2];
	for(int s= -1; s<=1; s+=2)
		{
		double aa= a+(k==0 ? s*h : 0), bb= b+(k==1 ? s*h : 0);
		C c0=cos(aa/2), c1=C(0,-sin(aa/2));
		C *q= s>0 ? p : m;
		q[0]=cos(bb/2)*c0-sin(bb/2)*c1;
		q[1]=sin(bb/2)*c0+cos(bb/2)*c1;
		}
	if(k<0) {psi[0]=(p[0]+m[0])/2.; psi[1]=(p[1]+m[1])/2.;}
	else for(int s=0; s<2; ++s) d[k][s]=(p[s]-m[s])/(2*h);
	}
for(int i=0; i<2; ++i) for(int j=0; j<2; ++j)
	{
	C dd=conj(d[i][0])*d[j][0]+conj(d[i][1])*d[j][1];
	C di=conj(d[i][0])*psi[0]+conj(d[i][1])*psi[1];
	C dj=conj(psi[0])*d[j][0]+conj(psi[1])*d[j][1];
	g[i][j]=real(dd)-real(di*dj);
	}
}


//reference implementation of the PyTorch update formulas of Adam, AdamW, AMSGrad and SGD with momentum, for comparison
static NRVec<double> reference_firstorder(const OptProblem<double> &p, NRVec<double> x, const int steps, const double lr, const int type, const double wd)
{
//type 0 Adam/AdamW, 1 AMSGrad, 2 SGD, 3 SGD-Nesterov
const int n=x.size();
const double b1=0.9, b2=0.999, eps=1e-8, mom=0.9;
x.copyonwrite();
NRVec<double> g(n), m(0.,n), v(0.,n), vmax(0.,n), vel(0.,n);
p.funcgrad(x,&g);
for(int k=1; k<=steps; ++k)
	{
	for(int i=0; i<n; ++i)
		{
		double step;
		if(type<2)
			{
			m[i]=b1*m[i]+(1-b1)*g[i];
			v[i]=b2*v[i]+(1-b2)*g[i]*g[i];
			vmax[i]=max(vmax[i],v[i]);
			double vv= type==1 ? vmax[i] : v[i];
			step=(m[i]/(1-pow(b1,k)))/(sqrt(vv)/sqrt(1-pow(b2,k))+eps);
			}
		else
			{
			vel[i]= k==1 ? g[i] : mom*vel[i]+g[i];
			step= type==3 ? g[i]+mom*vel[i] : vel[i];
			}
		x[i]-= lr*wd*x[i]+lr*step;
		}
	NRVec<double> gn(n);
	p.funcgrad(x,&gn);
	g=gn;
	}
return x;
}


static void test_problems_and_capabilities()
{
cout <<"--- test problems, capabilities and gradient adapters\n";
testproblem<double> tp;
testproblem_vqe<double> vqe;
NRVec<double> x=startpoint(8,4,2.);
check("testproblem gradient vs finite differences",fd_gradient_error(tp,x)<1e-8,fd_gradient_error(tp,x));
check("testproblem_vqe gradient vs finite differences",fd_gradient_error(vqe,x)<1e-8,fd_gradient_error(vqe,x));
{
NRSMat<double> h(8), hfd(8);
vqe.hessian(x,h);
for(int i=0; i<8; ++i)
	{
	NRVec<double> a(x), b(x), ga(8), gb(8);
	a.copyonwrite(); b.copyonwrite();
	a[i]+=1e-6; b[i]-=1e-6;
	vqe.funcgrad(a,&ga); vqe.funcgrad(b,&gb);
	for(int j=0; j<=i; ++j) hfd(i,j)=(ga[j]-gb[j])/2e-6;
	}
NRSMat<double> d=h-hfd;
check("testproblem_vqe Hessian vs finite differences of the gradient",d.norm()<1e-8,d.norm());
NRVec<double> v=startpoint(8,5), hv=vqe.hessian_times(x,v), hv2=h*v;
hv-=hv2;
check("OptHessian::hessian_times vs hessian*v",hv.norm()<1e-14,hv.norm());
}
{
NRSMat<double> g(8);
vqe.metric(x,g);
double err=0;
for(int q=0; q<4; ++q)
	{
	double fs[2][2];
	fubinistudy(x[2*q],x[2*q+1],fs);
	for(int i=0; i<2; ++i) for(int j=0; j<2; ++j) err=max(err,abs(g(2*q+i,2*q+j)-fs[i][j]));
	}
check("testproblem_vqe metric vs numerical Fubini-Study metric of the state",err<1e-9,err);
}
{
NRVec<double> g(8), gps(8), gfd(8);
vqe.funcgrad(x,&g);
ParameterShift<double> ps(&vqe);
FiniteDifference<double> fd(&vqe);
ps.funcgrad(x,&gps); fd.funcgrad(x,&gfd);
gps-=g; gfd-=g;
check("ParameterShift gradient, degree 1",gps.norm()<1e-13,gps.norm());
check("FiniteDifference gradient",gfd.norm()<1e-8,gfd.norm());
trigdegree3 t3;
ParameterShift<double> ps3(&t3);
NRVec<double> y(2), g3(2), g3ps(2);
y[0]=1.234; y[1]=0.6;
t3.funcgrad(y,&g3); ps3.funcgrad(y,&g3ps);
g3ps-=g3;
check("ParameterShift gradient, degree 3 and period 5, and a non-trigonometric parameter",g3ps.norm()<1e-8,g3ps.norm());
FiniteDifference<double> fdps(&ps);
check("optcapability through adapters",optcapability<OptMetric<double> >(&ps)==&vqe && optcapability<OptTrigonometric<double> >(&fdps)==&vqe && optcapability<OptMetric<double> >(&tp)==NULL);
testproblem_vqe_constrained<double> vc;
counting cvc(&vc);
check("constraints found through an adapter",optcapability<OptConstraints<double> >(&cvc)==&vc);
}
}


static void test_linemin()
{
cout <<"--- derivative-free line minimization (Brent)\n";
class oned : public OptProblem<double> {public: int k; double funcgrad(const NRVec<double> &x, NRVec<double> *g) const {double t=x[0];
	switch(k) {case 0: return (t-3.7)*(t-3.7)+1; case 1: return cos(t); case 2: return exp(t)-5*t; case 3: return pow(t+2.5,4); default: return abs(t-0.3);}}};
const double tmin[]={3.7,M_PI,log(5.),-2.5,0.3};
bool ok=true;
double worst=0;
for(int k=0; k<5; ++k) for(double t1: {0.1,1.})
	{
	oned p; p.k=k;
	optevaluator<double> e(&p);
	NRVec<double> x(0.,1), u(1.,1);
	double f=e(x);
	linemin_brent(e,x,f,u,t1,1e-8,1e-10,100);
	double err=abs(x[0]-tmin[k]);
	if(err>(k==3 ? 1e-4 : 1e-6)) ok=false;
	worst=max(worst,err);
	}
check("linemin_brent finds the minima of 5 one-dimensional functions",ok,worst);
oned p; p.k=0;
bool budget=true;
for(int mb=1; mb<=10; ++mb) {optevaluator<double> e(&p); NRVec<double> x(0.,1), u(1.,1); double f=e(x); linemin_brent(e,x,f,u,0.1,1e-8,1e-10,mb); if(e.nfunc-1>mb) budget=false;}
check("linemin_brent respects the evaluation budget",budget);
}


static void test_gradient_methods()
{
cout <<"--- gradient methods\n";
testproblem<double> tp;
rosenbrock rb;
//BFGS, DFP, SR1
const QN_update upd[]={QN_BFGS,QN_DFP,QN_SR1};
const char *updname[]={"BFGS","DFP","SR1"};
for(int u=0; u<3; ++u)
	{
	BFGS_hyperparameters<double> h; h.my_update=upd[u]; h.my_g_thr=1e-6; h.my_maxstep=5000;
	if(upd[u]==QN_DFP) h.my_c2=0.1; //DFP needs an accurate line search
	BFGS_Optimizer<double> o;
	NRVec<double> x=startpoint(10,1);
	bool conv;
	double f=run(o,&tp,x,h,NULL,&conv);
	string s=string(updname[u])+" converges on testproblem n=10";
	check(s.c_str(),conv && f<1e-10,f);
	x=rosenbrockstart(10);
	f=run(o,&rb,x,h,NULL,&conv);
	s=string(updname[u])+" converges on Rosenbrock n=10";
	check(s.c_str(),conv && f<1e-10,f);
	}
{
LBFGS_hyperparameters<double> h; h.my_g_thr=1e-6; h.my_maxstep=10000;
LBFGS_Optimizer<double> o;
NRVec<double> x=rosenbrockstart(100);
bool conv;
double f=run(o,&rb,x,h,NULL,&conv);
check("L-BFGS converges on Rosenbrock n=100",conv && f<1e-10,f);
}
const CG_beta beta[]={CG_FR,CG_PR,CG_HS,CG_DY,CG_HZ};
for(int b=0; b<5; ++b)
	{
	CG_hyperparameters<double> h; h.my_beta=beta[b]; h.my_g_thr=1e-6; h.my_maxstep=20000;
	CG_Optimizer<double> o;
	NRVec<double> x=rosenbrockstart(10);
	bool conv;
	double f=run(o,&rb,x,h,NULL,&conv);
	char s[64]; sprintf(s,"CG variant %d converges on Rosenbrock n=10",b);
	check(s,conv && f<1e-10,f);
	}
{
GDIIS_hyperparameters<double> h; h.my_g_thr=1e-6; h.my_maxstep=5000;
GDIIS_Optimizer<double> o;
NRVec<double> x=rosenbrockstart(10);
bool conv;
double f=run(o,&rb,x,h,NULL,&conv);
check("GDIIS converges on Rosenbrock n=10",conv && f<1e-10,f);
//classical GDIIS acceleration with a fixed Hessian (updates switched off) on a well scaled quadratic
class scaledquad : public OptProblem<double> {public: double funcgrad(const NRVec<double> &x, NRVec<double> *g) const {int n=x.size(); double r=0; if(g) {g->copyonwrite(); *g=0.;}
	for(int i=0; i<n; ++i) {double s=x[i]+0.5*x[(i+1)%n]; r+=0.005*(i+1)*s*s; if(g) {(*g)[i]+=0.01*(i+1)*s; (*g)[(i+1)%n]+=0.005*(i+1)*s;}} return r;}};
scaledquad sq;
h.my_sy_thr=1e300; h.my_diis_dim=10; h.my_maxstep=20000;
hyperparameters<double> a;
x=startpoint(30,1);
f=run(o,&sq,x,h,&a,&conv);
check("GDIIS with a fixed Hessian accelerates (< 100 evaluations on a quadratic n=30)",conv && a.my_maxfunc<100,a.my_maxfunc);
}
{
//numerical gradients give the same BFGS run as analytic ones
testproblem_vqe<double> vqe;
ParameterShift<double> ps(&vqe);
FiniteDifference<double> fd(&vqe);
BFGS_hyperparameters<double> h; h.my_g_thr=1e-6;
BFGS_Optimizer<double> o;
NRVec<double> x1=startpoint(8,9), x2(x1), x3(x1);
hyperparameters<double> a1, a2, a3;
double f1=run(o,&vqe,x1,h,&a1), f2=run(o,&ps,x2,h,&a2), f3=run(o,&fd,x3,h,&a3);
check("BFGS with ParameterShift and FiniteDifference gradients as with analytic ones",a1.my_maxstep==a2.my_maxstep && a1.my_maxstep==a3.my_maxstep && abs(f1-f2)<1e-12 && abs(f1-f3)<1e-10,f2-f1);
}
}


static void test_derivative_free()
{
cout <<"--- derivative-free methods\n";
rosenbrock rb;
testproblem<double> tp;
{
NelderMead_hyperparameters<double> h; h.my_maxstep=100000;
NelderMead_Optimizer<double> o;
NRVec<double> x=rosenbrockstart(2);
bool conv;
double f=run(o,&rb,x,h,NULL,&conv);
check("Nelder-Mead converges on Rosenbrock n=2",conv && f<1e-10,f);
h.my_adaptive=false;
x=rosenbrockstart(2);
f=run(o,&rb,x,h,NULL,&conv);
check("Nelder-Mead with standard coefficients converges on Rosenbrock n=2",conv && f<1e-10,f);
}
{
Powell_hyperparameters<double> h;
Powell_Optimizer<double> o;
NRVec<double> x=startpoint(10,1);
bool conv;
double f=run(o,&tp,x,h,NULL,&conv);
check("Powell converges on testproblem n=10",conv && f<1e-10,f);
coupledquadratic cq;
x=startpoint(10,1);
f=run(o,&cq,x,h,NULL,&conv);
check("Powell converges on a coupled quadratic n=10",conv && f<1e-10,f);
}
{
testproblem_vqe<double> vqe;
Rotosolve_hyperparameters<double> h;
Rotosolve_Optimizer<double> o;
NRVec<double> x=startpoint(12,9);
bool conv;
double f=run(o,&vqe,x,h,NULL,&conv);
BFGS_hyperparameters<double> hb;
BFGS_Optimizer<double> ob;
NRVec<double> xb=startpoint(12,9);
double fb=run(ob,&vqe,xb,hb);
check("Rotosolve reaches the minimum of testproblem_vqe found by BFGS",conv && abs(f-fb)<1e-7,f-fb);
trigdegree3 t3;
NRVec<double> y(2); y[0]=0.7; y[1]=0.3;
h.my_maxstep=1;
run(o,&t3,y,h);
double best=1e9;
for(int j=0; j<1000000; ++j) best=min(best,trigdegree3::tr(5.*j/1000000));
check("Rotosolve degree-3 reconstruction finds the exact minimum along the parameter",abs(trigdegree3::tr(y[0])-best)<1e-10,trigdegree3::tr(y[0])-best);
}
//COBYLA, both radius modes
for(int fl=0; fl<2; ++fl)
	{
	Cobyla_hyperparameters<double> h; h.my_rhoend=1e-8; h.my_flexible_radius=fl;
	Cobyla_Optimizer<double> o;
	const char *mode= fl ? "flexible radius" : "original";
	struct {const char *name; const OptProblem<double> *p; NRVec<double> x0; double fmin;} cases[]={
		{"quadratic with a linear constraint",new quadratic_linearconstraint,NRVec<double>(0.,2),2},
		{"x0 x1 in the unit disk",new product_disk,NRVec<double>(1.,2),-0.5},
		{"Rosen-Suzuki",new rosensuzuki,NRVec<double>(0.,4),-44},
		{"Rosen-Suzuki from an infeasible point",new rosensuzuki,NRVec<double>(3.,4),-44},
		{"equality constraint x0^2+x1^2=1",new sum_circle,NRVec<double>(0.5,2),-sqrt(2.)}};
	for(auto &c: cases)
		{
		NRVec<double> x(c.x0);
		bool conv;
		double f=run(o,c.p,x,h,NULL,&conv);
		const OptConstraints<double> *con=dynamic_cast<const OptConstraints<double> *>(c.p);
		NRVec<double> cv(con->nconstraints());
		con->constraints(x,cv);
		string s=string("COBYLA (")+mode+") solves "+c.name;
		check(s.c_str(),conv && abs(f-c.fmin)<1e-7 && cv.min()> -1e-10,f-c.fmin);
		delete c.p;
		}
	testproblem_vqe_constrained<double> vc;
	NRVec<double> x=startpoint(8,9);
	bool conv;
	double f=run(o,&vc,x,h,NULL,&conv);
	NRVec<double> cv(1);
	vc.constraints(x,cv);
	string s=string("COBYLA (")+mode+") on testproblem_vqe_constrained n=8";
	check(s.c_str(),conv && abs(f+3.334983606)<1e-7 && cv[0]> -1e-10,f);
	}
}


static void test_firstorder()
{
cout <<"--- first-order and stochastic methods\n";
rosenbrock rb;
NRVec<double> x0=rosenbrockstart(5);
const int steps=300;
//Adam, AdamW, AMSGrad, SGD, Nesterov vs the reference implementation
for(int type=0; type<5; ++type)
	{
	NRVec<double> x(x0);
	double lr= type<3 ? 0.01 : 1e-4;
	int rtype= type==2 ? 1 : type==3 ? 2 : type==4 ? 3 : 0;
	double wd= type==1 ? 0.1 : 0;
	if(type<3)
		{
		Adam_hyperparameters<double> h; h.my_maxstep=steps; h.my_g_thr=0; h.my_lr=lr; h.my_weight_decay=wd; h.my_amsgrad= type==2;
		Adam_Optimizer<double> o;
		run(o,&rb,x,h);
		}
	else
		{
		SGD_hyperparameters<double> h; h.my_maxstep=steps; h.my_g_thr=0; h.my_lr=lr; h.my_nesterov= type==4;
		SGD_Optimizer<double> o;
		run(o,&rb,x,h);
		}
	NRVec<double> xr=reference_firstorder(rb,x0,steps,lr,rtype,wd);
	xr-=x;
	const char *names[]={"Adam","AdamW","AMSGrad","SGD with momentum","SGD-Nesterov"};
	string s=string(names[type])+" agrees with the reference update formulas";
	check(s.c_str(),xr.norm()<1e-10,xr.norm());
	}
check("x0 unchanged by the in-place updates of copies (copy-on-write)",x0[0]== -1.2 && x0[1]==1);
{
noisyquadratic nq(0.1);
Adam_hyperparameters<double> h; h.my_maxstep=3000; h.my_g_thr=0; h.my_lr=0.01; h.my_lr_decay=0.01;
Adam_Optimizer<double> o;
NRVec<double> x=startpoint(20,5);
run(o,&nq,x,h);
check("Adam with decaying learning rate reaches the noise floor of a noisy quadratic",noisyquadratic::exact(x)<1e-3,noisyquadratic::exact(x));
}
{
//SPSA
noisyquadratic nq(0.1);
SPSA_hyperparameters<double> h; h.my_maxfunc=4000;
SPSA_Optimizer<double> o;
NRVec<double> x=startpoint(10,5);
run(o,&nq,x,h);
check("SPSA on a noisy quadratic n=10",noisyquadratic::exact(x)<0.05,noisyquadratic::exact(x));
coupledquadratic cq;
double sum=0;
NRVec<double> xs=startpoint(6,3);
for(int r=0; r<100; ++r)
	{
	SPSA_hyperparameters<double> hc; hc.my_maxstep=1; hc.my_target_step=0.1;
	NRVec<double> y(xs);
	run(o,&cq,y,hc);
	NRVec<double> d=y-xs;
	sum+=d.norm();
	}
check("SPSA calibration gives the requested first step length (within 20%)",abs(sum/100-0.1)<0.02,sum/100);
}
{
//QNG: one step by hand and convergence
testproblem_vqe<double> vqe;
NRVec<double> x=startpoint(8,4,2.), x0q(x);
x0q.copyonwrite();
QNG_hyperparameters<double> h; h.my_maxstep=1; h.my_g_thr=0; h.my_lr=0.1; h.my_metric_reg=1e-3;
QNG_Optimizer<double> o;
run(o,&vqe,x,h);
NRVec<double> g(8);
vqe.funcgrad(x0q,&g);
NRSMat<double> G(8);
vqe.metric(x0q,G);
G+=1e-3;
linear_solve(G,g);
NRVec<double> xr(x0q);
xr.axpy(-0.1,g);
xr-=x;
check("QNG step equals x - lr (G + lambda I)^-1 g",xr.norm()<1e-14,xr.norm());
QNG_hyperparameters<double> hc; hc.my_maxstep=10000;
NRVec<double> y=startpoint(12,9);
bool conv;
double f=run(o,&vqe,y,hc,NULL,&conv);
check("QNG converges on testproblem_vqe",conv && f< -5.4,f);
}
}


static void test_frozen()
{
cout <<"--- optimization with frozen parameters (FrozenOptProblem)\n";
testproblem_vqe_constrained<double> vqe;
const int n=8;
NRVec<double> x=startpoint(n,4,2.);
bitvector frozen(n);
frozen.set(1); frozen.set(4); frozen.set(5);
FrozenOptProblem<double> fp(&vqe,x,frozen);
NRVec<double> xr=fp.reduce(x);
NRVec<double> xe=fp.expand(xr);
xe-=x;
check("reduce and expand",xr.size()==5 && xe.norm()==0);
NRVec<double> y=startpoint(5,7); //other values of the free parameters
NRVec<double> yfull=fp.expand(y);
NRVec<double> gfull(n), g(5);
double ffull=vqe.funcgrad(yfull,&gfull), f=fp.funcgrad(y,&g);
NRVec<double> gd=gfull.subvector(fp.freeindex)-g;
check("reduced function value and gradient",f==ffull && gd.norm()==0);
NRSMat<double> hfull(n), h(5), mfull(n), m(5);
vqe.hessian(yfull,hfull); fp.hessian(y,h);
vqe.metric(yfull,mfull); fp.metric(y,m);
NRSMat<double> hd=hfull.submatrix(fp.freeindex)-h, md=mfull.submatrix(fp.freeindex)-m;
check("reduced Hessian and metric",hd.norm()==0 && md.norm()==0);
NRVec<double> v=startpoint(5,8), hv=fp.hessian_times(y,v), hv2=h*v;
hv-=hv2;
check("reduced Hessian-vector product",hv.norm()<1e-14,hv.norm());
NRVec<double> cfull(1), c(1);
vqe.constraints(yfull,cfull); fp.constraints(y,c);
check("constraints and trigonometric degree of the reduced problem",fp.nconstraints()==1 && c[0]==cfull[0] && fp.degree(2)==1 && fp.period(2)==2*M_PI);
testproblem<double> tp;
FrozenOptProblem<double> ftp(&tp,startpoint(6,1),bitvector(6));
ParameterShift<double> ps(&fp);
check("capabilities of the frozen problem only if the wrapped problem has them",optcapability<OptMetric<double> >(&fp)==&fp && optcapability<OptMetric<double> >(&ftp)==NULL && optcapability<OptHessian<double> >(&ftp)==NULL && optcapability<OptMetric<double> >(&ps)==&fp);
//optimizations keep the frozen parameters
{
bitvector fr(10);
for(int i=0; i<10; i+=3) fr.set(i);
NRVec<double> x0=startpoint(10,1);
FrozenOptProblem<double> fq(&tp,x0,fr);
NRVec<double> z=fq.reduce(x0);
BFGS_hyperparameters<double> hb; hb.my_g_thr=1e-8;
BFGS_Optimizer<double> o;
bool conv;
run(o,&fq,z,hb,NULL,&conv);
NRVec<double> zf=fq.expand(z), gz(z.size());
fq.funcgrad(z,&gz);
bool kept=true;
const NRVec<double> &x0c=x0; //x0 is shared with fq.xfull, read-only access
for(int i=0; i<10; i+=3) if(zf[i]!=x0c[i]) kept=false;
check("BFGS with frozen parameters: frozen ones kept, reduced gradient zero",conv && kept && gz.norm()<1e-8,gz.norm());
}
{
NRVec<double> z=fp.reduce(x);
QNG_hyperparameters<double> hq; hq.my_maxstep=10000;
QNG_Optimizer<double> o;
bool conv;
testproblem_vqe<double> vu;
FrozenOptProblem<double> fu(&vu,x,frozen);
run(o,&fu,z,hq,NULL,&conv);
NRVec<double> gz(5);
fu.funcgrad(z,&gz);
check("QNG uses the reduced metric",conv && gz.norm()<1e-5,gz.norm());
z=fp.reduce(x);
Rotosolve_hyperparameters<double> hr;
Rotosolve_Optimizer<double> orot;
run(orot,&fu,z,hr,NULL,&conv);
fu.funcgrad(z,&gz);
check("Rotosolve uses the reduced trigonometric structure",conv && gz.norm()<1e-4,gz.norm());
z=fp.reduce(x);
Cobyla_hyperparameters<double> hc; hc.my_rhoend=1e-8;
Cobyla_Optimizer<double> oc;
run(oc,&fp,z,hc,NULL,&conv);
NRVec<double> cz(1);
fp.constraints(z,cz);
NRVec<double> zf=fp.expand(z);
const NRVec<double> &xc=x; //x is shared with fp.xfull, read-only access
check("COBYLA with frozen parameters satisfies the constraint and keeps them",conv && cz[0]> -1e-10 && zf[1]==xc[1] && zf[4]==xc[4] && zf[5]==xc[5],cz[0]);
}
}


static void test_trustregion()
{
cout <<"--- trust-region methods and saddle points\n";
{
//secant condition B+ s = y of the Hessian updates
srand(3);
const int n=6;
bool ok=true;
double worst=0;
const TR_hessian upd[]={TR_BFGS,TR_SR1,TR_PSB,TR_BOFILL};
for(int t=0; t<4; ++t)
	{
	NRSMat<double> B(n); B.randomize(0.3); B+=2.;
	NRVec<double> s(n), r(n); s.randomize(1.); r.randomize(0.3);
	NRVec<double> y=B*s; y+=r;
	if(!hessian_update(B,s,y,upd[t],1e-8)) ok=false;
	NRVec<double> Bs=B*s; Bs-=y;
	worst=max(worst,Bs.norm());
	}
check("secant condition of the BFGS, SR1, PSB and Bofill updates",ok && worst<1e-12,worst);
}
testproblem_mullerbrown<double> mb;
{
NRVec<double> x(2); x[0]= -0.3; x[1]=0.9;
NRSMat<double> h(2), hfd(2);
mb.hessian(x,h);
for(int i=0; i<2; ++i) {NRVec<double> a(x), b(x), ga(2), gb(2); a.copyonwrite(); b.copyonwrite(); a[i]+=1e-6; b[i]-=1e-6; mb.funcgrad(a,&ga); mb.funcgrad(b,&gb); for(int j=0; j<=i; ++j) hfd(i,j)=(ga[j]-gb[j])/2e-6;}
NRSMat<double> d=h-hfd;
check("Mueller-Brown gradient and Hessian vs finite differences",fd_gradient_error(mb,x)<1e-7 && d.norm()<1e-6,d.norm());
}
//stationary points of the Mueller-Brown potential (Mueller & Brown 1979)
struct {const char *name; double x0, y0, x, y, v; int k;} sp[]={
	{"minimum (-0.558,1.442)",-0.4,1.2,-0.5582,1.4417,-146.700,0},
	{"minimum (0.623,0.028)",0.5,0.1,0.6235,0.0280,-108.167,0},
	{"minimum (-0.050,0.467)",0.,0.4,-0.0500,0.4667,-80.768,0},
	{"saddle point (-0.822,0.624)",-0.7,0.6,-0.8220,0.6243,-40.665,1},
	{"saddle point (0.212,0.293)",0.15,0.35,0.2125,0.2930,-72.249,1}};
for(auto &c: sp) for(int m=0; m<3; ++m)
	{
	if(m==0 && c.k) continue;
	NRVec<double> x(2); x[0]=c.x0; x[1]=c.y0;
	hyperparameters<double> a;
	bool conv;
	double f;
	string s;
	if(m==0) {TrustRegion_hyperparameters<double> h; h.my_g_thr=1e-9; TrustRegion_Optimizer<double> o; f=run(o,&mb,x,h,&a,&conv); s="TR-Newton";}
	else {PRFO_hyperparameters<double> h; h.my_g_thr=1e-9; h.my_saddle_order=c.k; if(m==2) h.my_hessian_recompute=1000; PRFO_Optimizer<double> o; f=run(o,&mb,x,h,&a,&conv); s= m==1 ? "P-RFO" : "P-RFO with Hessian updates";}
	s+=string(" finds the ")+c.name+" with the right index";
	check(s.c_str(),conv && abs(x[0]-c.x)<1e-4 && abs(x[1]-c.y)<1e-4 && abs(f-c.v)<1e-3 && a.my_saddle_order==c.k,f-c.v);
	}
{
testproblem_vqe<double> vqe;
TrustRegion_hyperparameters<double> h;
TrustRegion_Optimizer<double> o;
NRVec<double> x=startpoint(12,9);
hyperparameters<double> a;
bool conv;
double f=run(o,&vqe,x,h,&a,&conv);
check("TR-Newton with the exact Hessian on testproblem_vqe n=12",conv && f< -5.4 && a.my_maxfunc<20 && a.my_saddle_order==0,a.my_maxfunc);
}
{
rosenbrock rb;
for(int m=0; m<2; ++m)
	{
	TrustRegion_hyperparameters<double> h; h.my_maxstep=10000; h.my_subproblem= m ? TR_STEIHAUG : TR_EIGEN;
	TrustRegion_Optimizer<double> o;
	NRVec<double> x=rosenbrockstart(20);
	bool conv;
	double f=run(o,&rb,x,h,NULL,&conv);
	check(m ? "TR with BFGS updates and Steihaug-CG on Rosenbrock n=20" : "TR with BFGS updates on Rosenbrock n=20",conv && f<1e-10,f);
	}
}
}


//fit of a NIST StRD problem from both starting points by Levenberg-Marquardt, certified residual sum of squares to 9 digits
template<class P>
static void nistfit(const char *name, const int nb)
{
for(int k=0; k<2; ++k)
	{
	P p;
	NRVec<double> b(nb);
	for(int i=0; i<nb; ++i) b[i]=P::start(k,i);
	LM_hyperparameters<double> h; h.my_g_thr=1e-9; h.my_maxstep=100000;
	LM_Optimizer<double> o;
	bool conv;
	double f=run(o,&p,b,h,NULL,&conv);
	double perr=0;
	for(int i=0; i<nb; ++i) perr=max(perr,abs(b[i]-P::certified(i))/abs(P::certified(i)));
	string s=string("Levenberg-Marquardt reproduces the certified fit of NIST ")+name+(k ? " from start 2" : " from start 1");
	check(s.c_str(),conv && abs(2*f-P::rss())<1e-9*P::rss() && perr<1e-5,2*f-P::rss());
	}
}

static void test_leastsquares()
{
cout <<"--- least squares (Levenberg-Marquardt)\n";
nistfit<testproblem_misra1a<double> >("Misra1a",2);
nistfit<testproblem_mgh09<double> >("MGH09",4);
nistfit<testproblem_thurber<double> >("Thurber (finite-difference Jacobian)",7);
testproblem_mgh09<double> p;
NRVec<double> b(4);
for(int i=0; i<4; ++i) b[i]=p.start(1,i);
NRMat<double> J(11,4), Jfd(11,4);
NRVec<double> r(11);
p.jacobian(b,J);
p.residuals(b,r);
p.jacobian_fd(b,r,Jfd);
NRMat<double> d=J-Jfd;
check("analytic and forward-difference Jacobians of MGH09",d.norm()<1e-6,d.norm());
BFGS_hyperparameters<double> h; h.my_g_thr=1e-12;
BFGS_Optimizer<double> o;
double f=run(o,&p,b,h);
check("other methods work on OptLeastSquares problems (BFGS on MGH09)",abs(2*f-p.rss())<1e-9*p.rss(),2*f-p.rss());
}


static void test_maxfunc_and_parameters()
{
cout <<"--- evaluation budget and hyperparameters\n";
testproblem<double> tp;
testproblem_vqe_constrained<double> vc;
NRVec<double> x=startpoint(6,2), xv=startpoint(8,9);
hyperparameters<double> h; h.my_f_thr=1e-10; h.my_x_thr=1e-6; h.my_maxstep=100000;
BFGS_Optimizer<double> bfgs; LBFGS_Optimizer<double> lbfgs; CG_Optimizer<double> cg; GDIIS_Optimizer<double> gdiis;
NelderMead_Optimizer<double> nm; Powell_Optimizer<double> powell; Adam_Optimizer<double> adam; Rotosolve_Optimizer<double> roto; Cobyla_Optimizer<double> cobyla;
TrustRegion_Optimizer<double> trn; PRFO_Optimizer<double> prfo; LM_Optimizer<double> lm;
//LM evaluates the residuals directly (OptResiduals), so they are counted in the problem itself
class countingthurber : public testproblem_thurber<double> {public: mutable int count; countingthurber() : count(0) {}; void residuals(const NRVec<double> &b, NRVec<double> &r) const {++count; testproblem_thurber<double>::residuals(b,r);}};
{
int bad=0;
for(int mf=1; mf<=120; ++mf)
	{
	countingthurber p;
	NRVec<double> b(7);
	for(int i=0; i<7; ++i) b[i]=p.start(1,i);
	h.my_maxfunc=mf;
	hyperparameters<double> a;
	run(lm,&p,b,h,&a);
	if(p.count>mf || a.my_maxfunc!=p.count) ++bad;
	}
check("Levenberg-Marquardt (finite-difference Jacobian) respects maxfunc and reports the evaluations",bad==0,bad);
}
struct {const char *name; Optimizer<double> *o; const OptProblem<double> *p; NRVec<double> *x; int minfunc;} cases[]={
	{"BFGS",&bfgs,&tp,&x,1},{"L-BFGS",&lbfgs,&tp,&x,1},{"CG",&cg,&tp,&x,1},{"GDIIS",&gdiis,&tp,&x,1},{"Nelder-Mead",&nm,&tp,&x,7},
	{"Powell",&powell,&tp,&x,1},{"Adam",&adam,&tp,&x,1},{"Rotosolve",&roto,&vc,&xv,1},{"COBYLA",&cobyla,&vc,&xv,9},
	{"TR-Newton",&trn,&vc,&xv,1},{"P-RFO",&prfo,&vc,&xv,1}};
for(auto &c: cases)
	{
	int bad=maxfunc_violations(*c.o,c.p,*c.x,h,c.minfunc,120);
	string s=string(c.name)+" respects maxfunc and reports the evaluations";
	check(s.c_str(),bad==0,bad);
	}
//parameters given at different levels of the hierarchy, achieved filled at its level
linesearch_hyperparameters<double> lp; lp.my_c2=0.123; lp.my_g_thr=1e-4;
BFGS_hyperparameters<double> a;
NRVec<double> y=startpoint(5,1);
run(bfgs,&tp,y,lp,&a);
check("parameters taken from the caller's level, defaults kept for the rest",a.my_c2==0.123 && a.my_sy_thr==1e-12 && a.my_g_thr<1e-4);
}


static void test_diis()
{
cout <<"--- DIIS\n";
//DIIS acceleration of a linear fixed-point iteration x = Ax + b
struct fixedpoint {static NRVec<double> run(DIIS<NRVec<double>,NRVec<double> > &d, const int seed)
	{
	const int n=5;
	srand(seed);
	NRMat<double> A(n,n); A.randomize(0.3);
	NRVec<double> b(n); b.randomize(1.);
	NRVec<double> x(0.,n);
	for(int it=0; it<6; ++it) {NRVec<double> xn=A*x; xn+=b; NRVec<double> err=xn-x; d.extrapolate(xn,err); x=xn;}
	return x;
	}};
DIIS<NRVec<double>,NRVec<double> > din(4,true), dout(4,false), dfresh(4,true);
NRVec<double> a=fixedpoint::run(din,1), b=fixedpoint::run(dout,1);
a-=b;
check("DIIS in-core and out-of-core (AuxStorage) give the same extrapolation",a.norm()==0,a.norm());
din.reset();
NRVec<double> c=fixedpoint::run(din,2), f=fixedpoint::run(dfresh,2);
c-=f;
check("DIIS after reset() as a fresh object",c.norm()==0,c.norm());
}


//a member function applied to an object sharing its data with a copy must not change the copy
template<class X> static bool unchanged(const X &orig, const X &keep) {X d=orig-keep; return d.norm()==0;}

static void test_copyonwrite()
{
cout <<"--- copy-on-write in the functions of the vector and matrix classes\n";
#define FRESHMAT NRMat<double> orig(4,5); orig.randomize(1.); NRMat<double> keep(orig); keep.copyonwrite(); NRMat<double> a(orig);
#define FRESHVEC NRVec<double> orig(6); orig.randomize(1.); NRVec<double> keep(orig); keep.copyonwrite(); NRVec<double> a(orig);
#define FRESHSMAT NRSMat<double> orig(5); orig.randomize(1.); NRSMat<double> keep(orig); keep.copyonwrite(); NRSMat<double> a(orig);
NRVec<double> r(5); r.randomize(1.);
NRVec<double> col(4); col.randomize(1.);
{FRESHMAT a.rowset(r,1); check("NRMat::rowset",unchanged(orig,keep));}
{FRESHMAT a.columnset(col,2); check("NRMat::columnset",unchanged(orig,keep));}
{FRESHMAT NRMat<double> s(2,2); s=1.; a.storesubmatrix(1,1,s); check("NRMat::storesubmatrix(row,col)",unchanged(orig,keep));}
{FRESHMAT NRMat<double> s(2,2); s=1.; NRVec<int> ri(2), ci(2); ri[0]=0; ri[1]=3; ci[0]=1; ci[1]=4; a.storesubmatrix(ri,ci,s); check("NRMat::storesubmatrix(rows,cols)",unchanged(orig,keep));}
{FRESHMAT a.randomize(1.); check("NRMat::randomize",unchanged(orig,keep));}
{NRMat<double> orig(4,4); orig.randomize(1.); NRMat<double> keep(orig); keep.copyonwrite(); NRMat<double> a(orig); NRPerm<int> p(4); p.identity(); a.axpy(2.,p,true); check("NRMat::axpy(permutation)",unchanged(orig,keep));}
{NRMat<double> orig(4,4); orig.randomize(1.); NRMat<double> keep(orig); keep.copyonwrite(); NRMat<double> a(orig); a.orthonormalize(true,NULL); check("NRMat::orthonormalize",unchanged(orig,keep));}
{
NRMat<double> m(4,4); m.randomize(1.);
NRMat<double> t(m); t.copyonwrite(); t.transposeme(3);
bool ok=true;
for(int i=0; i<4; ++i) for(int j=0; j<4; ++j) if(t(i,j)!=((i<3 && j<3) ? m(j,i) : m(i,j))) ok=false;
check("NRMat::transposeme(n) transposes the leading n x n block",ok);
}
{
FILE *f=tmpfile();
fprintf(f,"2 3\n1 2 3\n4 5 6\n");
rewind(f);
NRMat<double> m;
m.fscanf(f,"%lf");
fclose(f);
check("NRMat::fscanf reads a non-square matrix",m.nrows()==2 && m.ncols()==3 && m(0,2)==3 && m(1,0)==4 && m(1,2)==6);
}
{FRESHVEC NRVec<double> s(2); s=1.; a.storesubvector(1,s); check("NRVec::storesubvector(from)",unchanged(orig,keep));}
{FRESHVEC NRVec<double> s(2); s=1.; NRVec<int> sel(2); sel[0]=0; sel[1]=4; a.storesubvector(sel,s); check("NRVec::storesubvector(selection)",unchanged(orig,keep));}
{FRESHVEC a.randomize(1.); check("NRVec::randomize",unchanged(orig,keep));}
{FRESHVEC NRVec<double> w(6); w.randomize(1.); a.call_on_me([](const double &p, const double &q){return p>q ? p : q;},w); check("NRVec::call_on_me binary",unchanged(orig,keep));}
{FRESHSMAT a.randomize(1.); check("NRSMat::randomize",unchanged(orig,keep));}
{FRESHSMAT NRSMat<double> s(2); s=1.; a.storesubmatrix(1,s); check("NRSMat::storesubmatrix(from)",unchanged(orig,keep));}
{
FRESHSMAT
NRSMat<double> s(3); s.randomize(1.);
NRVec<int> sel(3); sel[0]=0; sel[1]=2; sel[2]=4;
a.storesubmatrix(sel,s);
bool ok=true;
for(int i=0; i<3; ++i) for(int j=0; j<3; ++j) if(a(sel[i],sel[j])!=s(i,j)) ok=false;
check("NRSMat::storesubmatrix(selection) copy-on-write and size check",unchanged(orig,keep) && ok);
}
#undef FRESHMAT
#undef FRESHVEC
#undef FRESHSMAT
}


//single precision: every method with the float defaults (opt_precision<float>) on float test problems
template<class O, class H>
static float runf(O &o, const OptProblem<float> *p, NRVec<float> &x, H &h, bool *converged)
{
h.my_verbose=0;
x.copyonwrite();
optresult<float> r=o.minimize(p,x,&h,NULL);
*converged=r.converged;
return r.f;
}

static NRVec<float> startpointf(const int n, const int seed)
{
NRVec<double> xd=startpoint(n,seed);
NRVec<float> x(n);
for(int i=0; i<n; ++i) x[i]=xd[i];
return x;
}

//the same method in double and in float from the same start, the float run with its default thresholds
template<template<class> class O, template<class> class H>
static void compare_float(const char *name, const OptProblem<double> *pd, const OptProblem<float> *pf, const int n, const int seed, const int maxstep, void (*modify)(H<double> &, H<float> &)=NULL)
{
NRVec<double> xd=startpoint(n,seed); NRVec<float> xf(n); for(int i=0; i<n; ++i) xf[i]=xd[i];
H<double> hd; H<float> hf; hd.my_maxstep=hf.my_maxstep=maxstep; if(modify) modify(hd,hf);
O<double> od; O<float> of;
bool convd, convf;
hd.my_verbose=0; xd.copyonwrite(); optresult<double> rd=od.minimize(pd,xd,&hd,NULL); convd=rd.converged;
float ff=runf(of,pf,xf,hf,&convf);
string s=string("float ")+name+" agrees with double";
check(s.c_str(),convd && convf && abs(ff-rd.f)<1e-5*max(1.,abs(rd.f)),ff-rd.f);
}
static void dfp(BFGS_hyperparameters<double> &hd, BFGS_hyperparameters<float> &hf) {hd.my_update=hf.my_update=QN_DFP; hd.my_c2=hf.my_c2=0.1;}
static void sr1(BFGS_hyperparameters<double> &hd, BFGS_hyperparameters<float> &hf) {hd.my_update=hf.my_update=QN_SR1;}

static void test_float()
{
cout <<"--- single precision (float)\n";
testproblem<float> tp; //f=0 at x=0, used for the derivative-free methods; its large constant term n(n-1)/2 cancels in f, so f is not
		      //resolved below about 3e-6 in float and gradient methods cannot reach g_thr there: they use testproblem_vqe
testproblem_vqe<double> vd; testproblem_vqe<float> vf;
compare_float<BFGS_Optimizer,BFGS_hyperparameters>("BFGS on testproblem_vqe",&vd,&vf,8,3,5000);
compare_float<BFGS_Optimizer,BFGS_hyperparameters>("DFP on testproblem_vqe",&vd,&vf,8,3,5000,&dfp);
compare_float<BFGS_Optimizer,BFGS_hyperparameters>("SR1 on testproblem_vqe",&vd,&vf,8,3,5000,&sr1);
compare_float<LBFGS_Optimizer,LBFGS_hyperparameters>("L-BFGS on testproblem_vqe",&vd,&vf,8,3,5000);
compare_float<CG_Optimizer,CG_hyperparameters>("CG on testproblem_vqe",&vd,&vf,8,3,5000);
compare_float<GDIIS_Optimizer,GDIIS_hyperparameters>("GDIIS on testproblem_vqe",&vd,&vf,8,3,5000);
{NelderMead_hyperparameters<float> h; h.my_maxstep=100000; NelderMead_Optimizer<float> o; NRVec<float> x=startpointf(4,1); bool conv; float f=runf(o,&tp,x,h,&conv); check("float Nelder-Mead converges on testproblem n=4 (default f_thr, x_thr)",conv && f<1e-5,f);}
{Powell_hyperparameters<float> h; Powell_Optimizer<float> o; NRVec<float> x=startpointf(10,1); bool conv; float f=runf(o,&tp,x,h,&conv); check("float Powell converges on testproblem n=10 (default f_thr, x_thr)",conv && f<1e-5,f);}
{Cobyla_hyperparameters<float> h; Cobyla_Optimizer<float> o; NRVec<float> x=startpointf(4,1); bool conv; float f=runf(o,&tp,x,h,&conv); check("float COBYLA converges on testproblem n=4 (default rhoend)",conv && f<1e-5,f);}
{Adam_hyperparameters<float> h; h.my_maxstep=20000; h.my_lr=1e-2; Adam_Optimizer<float> o; NRVec<float> x=startpointf(10,1); bool conv; float f=runf(o,&tp,x,h,&conv); check("float Adam converges on testproblem n=10",conv && f<1e-5,f);}
{//finite-difference gradient with the float default step
FiniteDifference<float> fd(&tp); NRVec<float> x=startpointf(10,2), g(10), gfd(10); tp.funcgrad(x,&g); fd.funcgrad(x,&gfd); gfd-=g;
check("float FiniteDifference gradient (default step 5e-3)",gfd.norm()<1e-3*g.norm(),gfd.norm()/g.norm());
FiniteDifference<double> fdd(&vd); FiniteDifference<float> fdf(&vf);
compare_float<BFGS_Optimizer,BFGS_hyperparameters>("BFGS with finite-difference gradients (default steps)",&fdd,&fdf,8,3,5000);
}
{//Hessian-based methods on the VQE-like test problem, compared with double
NRVec<double> xd=startpoint(6,3); NRVec<float> xf(6); for(int i=0;i<6;++i) xf[i]=xd[i];
TrustRegion_hyperparameters<double> hd; TrustRegion_Optimizer<double> od; double fd=run(od,&vd,xd,hd);
TrustRegion_hyperparameters<float> hf; TrustRegion_Optimizer<float> of; bool conv; float ff=runf(of,&vf,xf,hf,&conv);
check("float trust region on testproblem_vqe agrees with double",conv && abs(ff-fd)<1e-5*max(1.,abs(fd)),ff-fd);
}
{//least squares: NIST Misra1a in float needs Marquardt scaling (unscaled b1 steps fall below float resolution)
testproblem_misra1a<float> p; NRVec<float> b(2); for(int i=0;i<2;++i) b[i]=p.start(0,i);
LM_hyperparameters<float> h; h.my_maxstep=100000; h.my_scaling=true; LM_Optimizer<float> o; bool conv; float f=runf(o,&p,b,h,&conv);
float perr=0; for(int i=0;i<2;++i) perr=max(perr,(float)abs(b[i]-p.certified(i))/abs(p.certified(i)));
check("float Levenberg-Marquardt fits NIST Misra1a",conv && perr<1e-3,perr);
}
}

int main(int argc, char **argv)
{
test_problems_and_capabilities();
test_linemin();
test_gradient_methods();
test_derivative_free();
test_firstorder();
test_frozen();
test_trustregion();
test_leastsquares();
test_maxfunc_and_parameters();
test_diis();
test_copyonwrite();
test_float();
cout <<"=== "<<npass<<" passed, "<<nfail<<" failed\n";
return nfail;
}
