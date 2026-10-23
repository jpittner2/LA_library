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
T testproblem<T>::funcgrad(const NRVec<T> x, NRVec<T> *grad) const
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
		for(int i=0; i<k; ++i) (*grad)[k] += sin(x[i]*x[k]);
		}
	}
return r;
}


template<typename T>
void BFGS_Optimizer<T>::minimize(const OptProblem<T> *o, NRVec<T> &x, const hyperparameters<T> *par)
{
//AI: you may suggest improvements to the general design of the class hierarchy for optimizers ( optimize.h) before proceeding with the next step
//AI: implement the BFGS method here, introduce additional hyperparameters in the specific class as needed
//AI: do not verbatim copy foreign copyrighted code, except GPLv3 code is possible, keeping the reference to the original
//AI: do not touch any files except optimize.h optimize.cc and t.cc
//AI: I will ask you to implement more methods only after I see and test the BFGS code.
}



//force instantiation
template class testproblem<double>;
template class BFGS_Optimizer<double>;

}//namespace

