/*
    LA: linear algebra C++ interface library
    Copyright (C) 2008-2020 Jiri Pittner <jiri.pittner@jh-inst.cas.cz> or <jiri@pittnerovi.com>

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
#ifndef _QUATERNION_H_
#define _QUATERNION_H_

#include <stdlib.h>

template <typename T>
class Quaternion 
	{
public:
	T q[4];
	Quaternion(void) {};
	Quaternion(const T x) {q[0]=x; memset(&q[1],0,3*sizeof(T));};
	};

#endif /* _QUATERNION_H_ */

