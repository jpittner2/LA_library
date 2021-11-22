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

//this header defines some simple algorithms independent of external libraries
//using small runtime-constant size matrices and vectors
//particularly intended to embedded computers
//it should be compilable separately from LA as well as being a part of LA

#ifndef _SIMPLE_H_
#define _SIMPLE_H_

#include <stdlib.h>
#ifndef AVOID_STDSTREAM
#include <iostream>
#endif
#include <string.h>
#include <math.h>
#include <stdio.h>

namespace LA_Simple {

//a simple gauss elimination as a template also for larger-size matrices in form of C-style arrays
#define SWAP(a,b) {T temp=(a);(a)=(b);(b)=temp;}

template<typename T, int n, int m>
T simple_gaussj(T (&a)[n][n],T (&b)[m][n]) //returns determinant, m is number of rhs to solve, inverse in a, solution in b
{
	int indxc[n],indxr[n],ipiv[n];
	int i,j,k,l,ll;
	int irow=0,icol=0;
	T det,big,dum,pivinv;

	det=1;
	for (j=0;j<n;j++) ipiv[j]=0;
	for (i=0;i<n;i++) {
		big=0.0;
		for (j=0;j<n;j++)
			if (ipiv[j] != 1)
				for (k=0;k<n;k++) {
					if (ipiv[k] == 0) {
						if (abs(a[j][k]) >= big) {
							big=abs(a[j][k]);
							irow=j;
							icol=k;
						}
					} else if (ipiv[k] > 1) {return 0;}
				}
		++(ipiv[icol]);
		if (irow != icol) {
			det = (-det);
			for (l=0;l<n;l++) SWAP(a[irow][l],a[icol][l])
			for (l=0;l<m;l++) SWAP(b[l][irow],b[l][icol])
		}
		indxr[i]=irow;
		indxc[i]=icol;
		if (a[icol][icol] == 0) {return 0;}
		pivinv=1/a[icol][icol];
		det *= a[icol][icol];
		a[icol][icol]=1.0;
		for (l=0;l<n;l++) a[icol][l] *= pivinv;
		for (l=0;l<m;l++) b[l][icol] *= pivinv;
		for (ll=0;ll<n;ll++)
			if (ll != icol) {
				dum=a[ll][icol];
				a[ll][icol]=0.0;
				for (l=0;l<n;l++) a[ll][l] -= a[icol][l]*dum;
				for (l=0;l<m;l++) b[l][ll] -= b[l][icol]*dum;
			}
	}
	for (l=n-1;l>=0;l--) {
		if (indxr[l] != indxc[l])
			for (k=0;k<n;k++)
				SWAP(a[k][indxr[l]],a[k][indxc[l]]);
	}
return det;
}

#undef SWAP


template<typename T, int n>
class simple_linfit {
public:
T fitmat[n][n];
T rhsmat[1][n];
T fitcoef[n];
int npoints;
	void clear(bool keepresults=false) {npoints=0; memset(&fitmat[0][0],0,n*n*sizeof(T)); memset(&rhsmat[0][0],0,1*n*sizeof(T)); if(!keepresults) memset(&fitcoef[0],0,n*sizeof(T));};
	simple_linfit()  {clear(false);}
	const T &operator[](int i) const {return fitcoef[i];}
	void input(const T (&funcs)[n], const T y)
		{
		++npoints;
		for(int i=0; i<n; ++i) 
			{
			for(int j=0; j<=i; ++j) 
				{
				T tmp=funcs[i]*funcs[j];
				fitmat[i][j] += tmp;
				if(i!=j) fitmat[j][i] += tmp;
				}
			rhsmat[0][i] += funcs[i]*y;
			}
		}
	T solve(bool preserve=false)
		{
		//for(int i=0; i<n; ++i)  {for(int j=0; j<n; ++j) std::cout <<fitmat[i][j]<<" "; std::cout<<std::endl;}
		//for(int j=0; j<n; ++j) std::cout <<rhsmat[0][j]<<" "; std::cout<<std::endl;
		if(npoints<n) return 0;
		if(preserve)
			{
			T fitwork[n][n];memcpy(fitwork,fitmat,n*n*sizeof(T));
			T rhswork[1][n];memcpy(rhswork,rhsmat,1*n*sizeof(T));
			T det = simple_gaussj(fitwork,rhswork);
                	memcpy(&fitcoef[0],&rhswork[0][0],n*sizeof(T));
                	return det;
			}
		T det = simple_gaussj(fitmat,rhsmat);
		memcpy(&fitcoef[0],&rhsmat[0][0],n*sizeof(T));
		clear(true);
		return det;	
		}

};

	
//stream I/O
#ifndef AVOID_STDSTREAM

template <typename T, int n>
std::ostream& operator<<(std::ostream &o, const simple_linfit<T,n> &f)
{
for(int i=0; i<n; ++i) o<<f.fitcoef[i]<<" ";
return o;
}
#endif


}//namespace
#endif /* _SIMPLE_H_ */

