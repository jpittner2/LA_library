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
#include "vec.h"
#include "mat.h"
#include "smat.h"


namespace LA {

//base classes with purely virtual functions for multiple inheritance use

//optimization problem - provides function and optional gradient, in multiply inherited classess will store auxiliary data for the function
template<typename T>
class OptProblem {
public:
virtual T funcgrad(const NRVec<T> x, NRVec<T> *grad) const =0; //evaluate function and optionally also gradient at point x
virtual ~OptProblem() {};
};

//test case
template<typename T>
class testproblem : public OptProblem<T> {
public:
virtual T funcgrad(const NRVec<T> x, NRVec<T> *grad) const;
virtual ~testproblem() {};
};


//base classes for optimization methods

//this is a collection of parameters, which can be augmented by method-specific additions, to be used as a base class
template<typename T>
class hyperparameters {
public:
int my_maxstep; //maximum number of iterations
T my_f_thr; //threshhold on function change in a single step
T my_x_thr; //threshold on x norm change in a single step

virtual int maxstep() const {return my_maxstep;}
virtual T f_thr() const {return my_f_thr;}
virtual T x_thr() const {return my_x_thr;}
virtual T g_thr() const =0;
virtual ~hyperparameters(){};
};


//base class for optimization methods, virtual functions to allow a pointer to derived class
template<typename T>
class Optimizer {
public:
virtual void minimize(const OptProblem<T> *o, NRVec<T> &x, const hyperparameters<T> *par) = 0;
virtual ~Optimizer() {}; 
};


//specific methods

template<typename T>
class BFGS_hyperparameters : public hyperparameters<T> {
public:
T my_g_thr;
virtual T g_thr() const {return my_g_thr;} //gradient magnitude convergence threshold
};

template<typename T>
class BFGS_Optimizer : public Optimizer<T> {
public:
virtual void minimize(const OptProblem<T> *o, NRVec<T> &x, const hyperparameters<T> *par);
virtual ~BFGS_Optimizer() {};
};



}//namespace
#endif /* _OPTIMIZE_H_ */

