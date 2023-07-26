/*
    LA: linear algebra C++ interface library
    Copyright (C) 2023 Jiri Pittner <jiri.pittner@jh-inst.cas.cz> or <jiri@pittnerovi.com>

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


//this is a haphazard collection of graph theoretical algorithms I happened to need
//

#ifndef _GRAPH_H
#define _GRAPH_H

#include "la_traits.h"
#include "vec.h"
#include "smat.h"
#include "mat.h"

namespace LA {

//presently I am not sure what the best general class for a graph would be, so leave it purely templated algorithms
//where the class G is adjacency supporting operator() and submatrix() operations

template<typename G>
NRVec<int> neighbors(const G &adjacency, const int v);

template<typename G>
NRVec<int> nonneighbors(const G &adjacency, const int v);

template<typename G>
NRVec<int> findclique(const G &adjacency, const int v=0);

template<typename G>
NRVec<int> complement(const G &adjacency, const NRVec<int> & vertexlist);

template<typename G>
NRVec<int> cliquecover(const G &adjacency);



}//namespace
#endif 

