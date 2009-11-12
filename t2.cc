/*
    LA: linear algebra C++ interface library
    Copyright (C) 2008 Jiri Pittner <jiri.pittner@jh-inst.cas.cz> or <jiri@pittnerovi.com>

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


#include <time.h>
#include "la.h"
#include "sparsemat.h"
#include "matexp.h"
#include "fourindex.h"
#include "bitvector.h"

#ifdef USE_TRACEBACK
#include "traceback.h"
#endif

using namespace LA;

void test(const NRVec<double> &x)
{
NRMat<double> aa(0.,2,2);
NRMat<double> cc(aa);
cc.copyonwrite(); cc[0][0]=1.;

std::cout << aa << cc <<"\n";
std::cout <<"test x= "<<x<<"\n";
}

