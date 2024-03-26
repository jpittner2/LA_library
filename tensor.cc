/*
    LA: linear algebra C++ interface library
    Copyright (C) 2024 Jiri Pittner <jiri.pittner@jh-inst.cas.cz> or <jiri@pittnerovi.com>

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

#include "tensor.h"
#include "laerror.h"
#include <iostream>


namespace LA {

template<typename T>
LA_largeindex Tensor<T>::index(const SUPERINDEX &I)
{
//check index structure and ranges
#ifndef DEBUG
if(I.size()!=shape.size()) laerror("mismatch in the number of tensor index groups");
for(int i=0; i<I.size; ++i)
	{
	if(shape[i].number!=I[i].size()) {std::cerr<<"error in index group no. "<<i<<std::endl; laerror("mismatch in the size of tensor index group");}
	for(int j=0; j<shape[i].number; ++j)
		{
		if(I[i][j] <shape[i].offset || I[i][j] >= shape[i].offset+shape[i].size()) 
			{
			std::cerr<<"error in index group no. "<<i<<" index no. "<<j<<std::endl;
			laerror("tensor index out of range");
			}
		}
	}
#endif


//@@@@@@@@@
}




}//namespace
