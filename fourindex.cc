/*
    LA: linear algebra C++ interface library
    Copyright (C) 2008-2019 Jiri Pittner <jiri.pittner@jh-inst.cas.cz> or <jiri@pittnerovi.com>

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

#include "fourindex.h"
namespace LA {

template <>
void fourindex<unsigned char,double>::fscanf(FILE *ff)
{
int i,j,k,l;
double elem;
if(5!= ::fscanf(ff,"%lf %d %d %d %d",&elem,&i,&j,&k,&l)) laerror("read error in fourindex::fscanf");
while(i!=terminator&&j!=terminator&&k!=terminator&&l!=terminator)
        {
        add(i,j,k,l,elem);
        if(5!= ::fscanf(ff,"%lf %d %d %d %d",&elem,&i,&j,&k,&l)) laerror("read error in fourindex::fscanf");
        }
}

template <>
void fourindex<unsigned char,double>::fprintf(FILE *f, char *format) const
{
fourindex<unsigned char,double>::iterator it=this->begin(),end=this->end();
while(it!=end)
            {
	    ::fprintf(f,format,it->elem,it->index.indiv.i,it->index.indiv.j,it->index.indiv.k,it->index.indiv.l);
            ++it;
            }
::fprintf(f,format,0.,0,0,0,0);
}



/***************************************************************************//**
 * forced instantization in the corresponding object file
 ******************************************************************************/
template class fourindex<unsigned char,double>;
template class fourindex_ext<unsigned char,double>;
template class fourindex_dense<nosymmetry,double,unsigned char>;
}
