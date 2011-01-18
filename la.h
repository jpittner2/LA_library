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
#ifndef _LA_H_
#define _LA_H_

//this should be the single include file for the end user
//
#ifdef USE_TRACEBACK
#include "traceback.h"
#endif
#include "la_traits.h"
#include "laerror.h"
#include "auxstorage.h"
#include "bisection.h"
#include "bitvector.h"
#include "conjgrad.h"
#include "davidson.h"
#include "diis.h"
#include "fourindex.h"
#include "gmres.h"
#include "mat.h"
#include "matexp.h"
#include "noncblas.h"
#include "nonclass.h"
#include "permutation.h"
#include "qsort.h"
#include "smat.h"
#include "sparsemat.h"
#include "sparsesmat.h"
#include "csrmat.h"
#include "vec.h"

using namespace LA;
typedef NRMat<int> NRIMat;
typedef NRMat<double> NRDMat;
typedef NRMat<complex<double> > NRCMat;
typedef NRVec<int> NRIVec;
typedef NRVec<double> NRDVec;
typedef NRVec<complex<double> > NRCVec;
#endif /* _LA_H_ */

