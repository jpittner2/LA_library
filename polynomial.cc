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

#include "polynomial.h"
#include <stdio.h>
#include <string.h>


namespace LA {



/***************************************************************************//**
 * forced instantization in the corresponding object file
 ******************************************************************************/
template class Polynomial<int>;
template class Polynomial<float>;
template class Polynomial<double>;

#define INSTANTIZE(T) \


//INSTANTIZE(float)

//INSTANTIZE(double)



}//namespace
