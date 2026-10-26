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


}//namespace
#endif /* _TESTOPTIMIZE_H_ */
