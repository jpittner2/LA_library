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
////////////////////////////////////////////////////////////////////////////
//traits classes

#ifndef _SPARSEMAT_TRAITS_INCL
#define _SPARSEMAT_TRAITS_INCL


template<> struct NRMat_traits<SparseMat<double> > {
typedef double elementtype;
typedef SparseMat<double> producttype;
static double norm (const SparseMat<double> &x) {return x.norm();}
static void axpy (SparseMat<double>&s, const SparseMat<double> &x, const double c) {s.axpy(c,x);}
};

#endif
