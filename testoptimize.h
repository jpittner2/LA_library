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

//test problems for the optimization methods, used by t.cc and testoptimize.cc

#ifndef _TESTOPTIMIZE_H_
#define _TESTOPTIMIZE_H_

#include "optimize.h"

namespace LA {

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

//constrained test case: testproblem_vqe with nonnegative total magnetization sum_i z_i >= 0
template<typename T>
class testproblem_vqe_constrained : public testproblem_vqe<T>, public OptConstraints<T> {
public:
virtual int nconstraints() const {return 1;}
virtual void constraints(const NRVec<T> &x, NRVec<T> &c) const;
virtual ~testproblem_vqe_constrained() {};
};


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


template<typename T>
void testproblem_vqe_constrained<T>::constraints(const NRVec<T> &x, NRVec<T> &c) const
{
c.copyonwrite();
c[0]=0;
for(int i=0; i<x.size()/2; ++i) c[0]+= std::cos(x[2*i])*std::cos(x[2*i+1]);
}


//Bloch vector components z = cos a cos b, x = cos a sin b of the qubit RY(b) RX(a)|0> and their derivatives
template<typename T>
class blochqubit {
public:
T z, x, za, zb, xa, xb;
blochqubit(const T a, const T b)
	{
	T ca=std::cos(a), sa=std::sin(a), cb=std::cos(b), sb=std::sin(b);
	z=ca*cb; x=ca*sb; za= -sa*cb; zb= -ca*sb; xa= -sa*sb; xb=ca*cb;
	}
};

template<typename T>
T testproblem_vqe<T>::funcgrad(const NRVec<T> &x, NRVec<T> *grad) const
{
const int nq=x.size()/2;
if(2*nq!=x.size()) laerror("testproblem_vqe needs an even number of parameters");
T r=0;
for(int i=0; i<nq; ++i)
	{
	blochqubit<T> q(x[2*i],x[2*i+1]);
	r+= hfield(i)*q.z + kfield(i)*q.x;
	if(i<nq-1) r+= J(i)*q.z*std::cos(x[2*i+2])*std::cos(x[2*i+3]);
	}
if(grad)
	{
	if(grad->size()!=x.size()) laerror("grad size mismatch");
	grad->copyonwrite();
	for(int i=0; i<nq; ++i)
		{
		blochqubit<T> q(x[2*i],x[2*i+1]);
		T eff=hfield(i); //coefficient of z_i including the couplings
		if(i>0) eff+= J(i-1)*std::cos(x[2*i-2])*std::cos(x[2*i-1]);
		if(i<nq-1) eff+= J(i)*std::cos(x[2*i+2])*std::cos(x[2*i+3]);
		(*grad)[2*i]= eff*q.za + kfield(i)*q.xa;
		(*grad)[2*i+1]= eff*q.zb + kfield(i)*q.xb;
		}
	}
return r;
}

template<typename T>
void testproblem_vqe<T>::hessian(const NRVec<T> &x, NRSMat<T> &h) const
{
const int nq=x.size()/2;
h=(T)0;
for(int i=0; i<nq; ++i)
	{
	blochqubit<T> q(x[2*i],x[2*i+1]);
	T eff=hfield(i);
	if(i>0) eff+= J(i-1)*std::cos(x[2*i-2])*std::cos(x[2*i-1]);
	if(i<nq-1) eff+= J(i)*std::cos(x[2*i+2])*std::cos(x[2*i+3]);
	//z_aa = z_bb = -z, x_aa = x_bb = -x, z_ab = sin a sin b, x_ab = -sin a cos b
	h(2*i,2*i)= -eff*q.z - kfield(i)*q.x;
	h(2*i+1,2*i+1)= h(2*i,2*i);
	h(2*i,2*i+1)= eff*std::sin(x[2*i])*std::sin(x[2*i+1]) - kfield(i)*std::sin(x[2*i])*std::cos(x[2*i+1]);
	if(i<nq-1)
		{
		blochqubit<T> q1(x[2*i+2],x[2*i+3]);
		h(2*i,2*i+2)= J(i)*q.za*q1.za;
		h(2*i,2*i+3)= J(i)*q.za*q1.zb;
		h(2*i+1,2*i+2)= J(i)*q.zb*q1.za;
		h(2*i+1,2*i+3)= J(i)*q.zb*q1.zb;
		}
	}
}

template<typename T>
void testproblem_vqe<T>::metric(const NRVec<T> &x, NRSMat<T> &g) const
{
const int nq=x.size()/2;
g=(T)0;
for(int i=0; i<nq; ++i)
	{
	T ca=std::cos(x[2*i]);
	g(2*i,2*i)=0.25;
	g(2*i+1,2*i+1)=0.25*ca*ca;
	}
}



//Mueller-Brown potential (Mueller & Brown, Theor. Chim. Acta 53, 75 (1979)), a standard 2D test surface with three minima
//(-0.558,1.442), (0.623,0.028), (-0.050,0.467) and two saddle points (-0.822,0.624), (0.212,0.293)
//V = sum_k A_k exp(a_k (x-x0_k)^2 + b_k (x-x0_k)(y-y0_k) + c_k (y-y0_k)^2)
template<typename T>
class testproblem_mullerbrown : public OptProblem<T>, public OptHessian<T> {
public:
static void terms(const NRVec<T> &x, const int k, T &v, T &qx, T &qy, T &a2, T &b, T &c2)
	{
	static const T A[]={-200,-100,-170,15}, a[]={-1,-1,-6.5,0.7}, bb[]={0,0,11,0.6}, c[]={-10,-10,-6.5,0.7}, x0[]={1,0,-0.5,-1}, y0[]={0,0.5,1.5,1};
	const T dx=x[0]-x0[k], dy=x[1]-y0[k];
	v=A[k]*std::exp(a[k]*dx*dx+bb[k]*dx*dy+c[k]*dy*dy);
	qx=2*a[k]*dx+bb[k]*dy; qy=bb[k]*dx+2*c[k]*dy;
	a2=2*a[k]; b=bb[k]; c2=2*c[k];
	}
virtual T funcgrad(const NRVec<T> &x, NRVec<T> *grad) const
	{
	T f=0;
	if(grad) {grad->copyonwrite(); *grad=0.;}
	for(int k=0; k<4; ++k)
		{
		T v,qx,qy,a2,b,c2;
		terms(x,k,v,qx,qy,a2,b,c2);
		f+=v;
		if(grad) {(*grad)[0]+=v*qx; (*grad)[1]+=v*qy;}
		}
	return f;
	}
virtual void hessian(const NRVec<T> &x, NRSMat<T> &h) const
	{
	h=(T)0;
	for(int k=0; k<4; ++k)
		{
		T v,qx,qy,a2,b,c2;
		terms(x,k,v,qx,qy,a2,b,c2);
		h(0,0)+=v*(qx*qx+a2); h(0,1)+=v*(qx*qy+b); h(1,1)+=v*(qy*qy+c2);
		}
	}
virtual ~testproblem_mullerbrown() {};
};



//NIST StRD nonlinear regression test problems with certified results (https://www.itl.nist.gov/div898/strd/nls/nls_main.shtml), residuals model - data
//start(k,i) = starting value i of "Start 1" (k=0) and "Start 2" (k=1), certified(i) = certified parameter values, rss = certified residual sum of squares

//Misra1a: y = b1*(1-exp(-b2*x)), with analytic Jacobian
template<typename T>
class testproblem_misra1a : public OptLeastSquares<T> {
public:
static const T *data(const int col) //col 0: y, col 1: x
	{
	static const T d[2][14]={{10.07,14.73,17.94,23.93,29.61,35.18,40.02,44.82,50.76,55.05,61.01,66.40,75.47,81.78},
				{77.6,114.9,141.1,190.8,239.9,289.0,332.8,378.4,434.8,477.3,536.8,593.1,689.1,760.0}};
	return d[col];
	}
static T start(const int k, const int i) {static const T s[2][2]={{500,1e-4},{250,5e-4}}; return s[k][i];}
static T certified(const int i) {static const T c[2]={2.3894212918E+02,5.5015643181E-04}; return c[i];}
static T rss() {return 1.2455138894E-01;}
virtual int nresiduals() const {return 14;}
virtual void residuals(const NRVec<T> &b, NRVec<T> &r) const
	{
	r.copyonwrite();
	for(int i=0; i<14; ++i) r[i]=b[0]*(1-std::exp(-b[1]*data(1)[i]))-data(0)[i];
	}
virtual bool jacobian(const NRVec<T> &b, NRMat<T> &J) const
	{
	J.copyonwrite();
	for(int i=0; i<14; ++i) {const T x=data(1)[i], ex=std::exp(-b[1]*x); J(i,0)=1-ex; J(i,1)=b[0]*x*ex;}
	return true;
	}
};

//MGH09 (Kowalik & Osborne): y = b1*(x^2+x*b2)/(x^2+x*b3+b4), with analytic Jacobian; "Start 1" is far from the solution
template<typename T>
class testproblem_mgh09 : public OptLeastSquares<T> {
public:
static const T *data(const int col)
	{
	static const T d[2][11]={{1.957000E-01,1.947000E-01,1.735000E-01,1.600000E-01,8.440000E-02,6.270000E-02,4.560000E-02,3.420000E-02,3.230000E-02,2.350000E-02,2.460000E-02},
				{4.000000E+00,2.000000E+00,1.000000E+00,5.000000E-01,2.500000E-01,1.670000E-01,1.250000E-01,1.000000E-01,8.330000E-02,7.140000E-02,6.250000E-02}};
	return d[col];
	}
static T start(const int k, const int i) {static const T s[2][4]={{25,39,41.5,39},{0.25,0.39,0.415,0.39}}; return s[k][i];}
static T certified(const int i) {static const T c[4]={1.9280693458E-01,1.9128232873E-01,1.2305650693E-01,1.3606233068E-01}; return c[i];}
static T rss() {return 3.0750560385E-04;}
virtual int nresiduals() const {return 11;}
virtual void residuals(const NRVec<T> &b, NRVec<T> &r) const
	{
	r.copyonwrite();
	for(int i=0; i<11; ++i) {const T x=data(1)[i]; r[i]=b[0]*(x*x+x*b[1])/(x*x+x*b[2]+b[3])-data(0)[i];}
	}
virtual bool jacobian(const NRVec<T> &b, NRMat<T> &J) const
	{
	J.copyonwrite();
	for(int i=0; i<11; ++i)
		{
		const T x=data(1)[i], num=x*x+x*b[1], den=x*x+x*b[2]+b[3];
		J(i,0)=num/den; J(i,1)=b[0]*x/den; J(i,2)= -b[0]*num*x/(den*den); J(i,3)= -b[0]*num/(den*den);
		}
	return true;
	}
};

//Thurber: y = (b1 + b2*x + b3*x^2 + b4*x^3)/(1 + b5*x + b6*x^2 + b7*x^3), without Jacobian (finite differences)
template<typename T>
class testproblem_thurber : public OptLeastSquares<T> {
public:
static const T *data(const int col)
	{
	static const T d[2][37]={{80.574,84.248,87.264,87.195,89.076,89.608,89.868,90.101,92.405,95.854,100.696,101.060,401.672,390.724,567.534,635.316,733.054,759.087,894.206,
				990.785,1090.109,1080.914,1122.643,1178.351,1260.531,1273.514,1288.339,1327.543,1353.863,1414.509,1425.208,1421.384,1442.962,1464.350,1468.705,1447.894,1457.628},
				{-3.067,-2.981,-2.921,-2.912,-2.840,-2.797,-2.702,-2.699,-2.633,-2.481,-2.363,-2.322,-1.501,-1.460,-1.274,-1.212,-1.100,-1.046,-0.915,
				-0.714,-0.566,-0.545,-0.400,-0.309,-0.109,-0.103,0.010,0.119,0.377,0.790,0.963,1.006,1.115,1.572,1.841,2.047,2.200}};
	return d[col];
	}
static T start(const int k, const int i) {static const T s[2][7]={{1000,1000,400,40,0.7,0.3,0.03},{1300,1500,500,75,1,0.4,0.05}}; return s[k][i];}
static T certified(const int i) {static const T c[7]={1.2881396800E+03,1.4910792535E+03,5.8323836877E+02,7.5416644291E+01,9.6629502864E-01,3.9797285797E-01,4.9727297349E-02}; return c[i];}
static T rss() {return 5.6427082397E+03;}
virtual int nresiduals() const {return 37;}
virtual void residuals(const NRVec<T> &b, NRVec<T> &r) const
	{
	r.copyonwrite();
	for(int i=0; i<37; ++i) {const T x=data(1)[i]; r[i]=(b[0]+x*(b[1]+x*(b[2]+x*b[3])))/(1+x*(b[4]+x*(b[5]+x*b[6])))-data(0)[i];}
	}
};


}//namespace
#endif /* _TESTOPTIMIZE_H_ */
