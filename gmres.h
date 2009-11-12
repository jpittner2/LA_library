/*
    LA: linear algebra C++ interface library
    Copyright (C) 2008 Jiri Pittner <jiri.pittner@jh-inst.cas.cz> or <jiri@pittnerovi.com>
			based on a routine originally written by Markus Warken <markus.warken@nsn.com>

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
#ifndef _GMRES_H
#define _GMRES_H
#include "vec.h"
#include "smat.h"
#include "mat.h"
#include "sparsemat.h"
#include "nonclass.h"
#include <iomanip>
#include "auxstorage.h"

namespace LA {

//GMRES solution of a linear system

//matrix can be any class which has nrows(), ncols(), diagonalof() and gemv() available
//does not even have to be explicitly stored


/* GMRES-Algorithmus nach Schwarz, S. 552, original impl. M. Warken */
/* allows zeilen!= spalten*/
/* Matrix can be any class which provides nrows(), ncols(), gemv(), and diagonalof(), does not have to store elements explicitly */

template<class T>
void gmres_backsubstitute(const NRMat<T> &R, NRVec<T> &c, const NRVec<T> &d, const int k)
{
c.copyonwrite();
if(R(k,k)==0.) laerror("singular matrix in gmres triangular solution");
c[k] = d[k]/R(k,k);
for (int i=k-1;i>=0;i--) c[i] = (d[i]-xdot(k-i,&R(i,i+1),1,&c[i+1],1)) / R(i,i);
}


//x contains ev. initial guess and on return the solution
template<typename T, typename Matrix>
bool gmres(const Matrix &bigmat, const NRVec<T> &b, NRVec<T> &x, const bool doguess=1, const double eps=1e-7, const int MAXIT=50, const bool verbose=1, bool square=1,const bool precondition=1, int neustart=0, const int incore=1)
{
int zeilen=bigmat.nrows();
int spalten=bigmat.ncols();
if(spalten==1) laerror("gmres does not work for n==1, use conjgrad if you need this trivial case");
if(x.size()!=spalten || b.size() != zeilen) laerror("incompatible vectors and matrix sizes in GMRES");

if(zeilen!=spalten) square=0;
if(!neustart)  neustart = zeilen/10;
if (neustart < 10) neustart = 10;
x.copyonwrite();

bool flag;
double beta,beta_0;
double d_alt=0;

AuxStorage<T> *st;
NRVec<T> *v;
NRVec<T> r_k(spalten),z(spalten);
NRVec<T> cci(MAXIT+1),ssi(MAXIT+1),c(MAXIT+1),d(MAXIT+1);
NRMat<T> H(MAXIT+1,MAXIT+1);
T ci,si;
v = new NRVec<T>[incore?MAXIT+1:1];
st = incore?NULL:new AuxStorage<T>;

if(doguess) 
	{
	bigmat.gemv(0,x,'t',-1.,b); //x.gemv(0,bigmat,'t',-1.,b);
        if(precondition) bigmat.diagonalof(x,true);
	x.normalize();
	}

neustart:
for (int l=0;l<neustart;l++)  // main loop for restarts
	{
	if(square) // r_0 = b + A x_0
		{
		bigmat.gemv(0,r_k,'n',1,x); //r_k.gemv(0,bigmat,'n',1,x);
		r_k -= b;
		}
	else //r_0 = A^t b + A^t A x_0
		{
		NRVec<T> dum(zeilen);
		bigmat.gemv(0,dum,'n',1,x); //dum.gemv(0,bigmat,'n',1,x);
		bigmat.gemv(0,r_k,'t',1,dum); //r_k.gemv(0,bigmat,'t',1,dum);
		bigmat.gemv(0,z,'t',-1.,b); //z.gemv(0,bigmat,'t',-1.,b);
	 	r_k += z; 
		}

         if(precondition) bigmat.diagonalof(r_k,true);

         beta = r_k.norm();
	 if(l==0) beta_0 = beta;
	 v[0] = r_k* (1./beta);
	 if(!incore) st->put(v[0],0);

         // Iteration
	 for (int k=0;k<MAXIT;k++) 
		{
		// *iter=l*MAXIT+k;
         	//if(dowarn) if (l>0) fprintf(stderr,"gmres: restart %d\n",l);

	    	// Schritt 1
		if(!incore) st->get(v[0],k);
	        if(square)
                	{
			bigmat.gemv(0,z,'n',1,v[incore?k:0]); //z.gemv(0,bigmat,'n',1,v[incore?k:0]);
                	}
		else
			{
			NRVec<T> dum(zeilen);
			bigmat.gemv(0,dum,'n',1,v[incore?k:0]); //dum.gemv(0,bigmat,'n',1,v[incore?k:0]);
			bigmat.gemv(0,z,'t',1,dum); //z.gemv(0,bigmat,'t',1,dum);
			}
              	if(precondition) bigmat.diagonalof(z,true);

	        //Schritte 2 und 3
	        for (int i=0;i<=k;i++) 
			{
			if(!incore) st->get(v[0],i);
			H(i,k) = z*v[incore?i:0];
			z.axpy(-H(i,k),v[incore?i:0]);
	      		}

		//Schritt 4
		double tmp;
	    	H(k+1,k) = tmp= z.norm();
	    	if(tmp < 1.e-2*eps )
			{
	    		if(verbose) std::cerr <<("gmres restart performed\n");
	       		// Abbruchbedingung, konstruiere x_k
	       		for (int i=0;i<k;i++) 
				{
		  		ci = cci[i];si = ssi[i];
		  		for (int j=0;j<k;j++) 
					{
		     			T a = H(i,j);
		     			H(i,j) = ci*a+si*H(i+1,j);
		     			H(i+1,j) = -si*a+ci*H(i+1,j);
		  		}
	       				}
	       		// Loese R_k c = - d_k
			d *= -1.;
	       		gmres_backsubstitute(H,c,d,k-1);
	       		for (int i=0;i<k-1;i++) 
				{
				if(!incore) st->get(v[0],i);
				x.axpy(c[i],v[incore?i:0]);
				}
	       		flag=0; goto neustart;
	    		} // Ende Abbruch

	   	v[incore?k+1:0] = z * (1./H(k+1,k));
		if(!incore) st->put(v[0],k+1);

	        // Schritt 5  - berechne Phi_k
	    	for (int j=0;j<k+2;j++) d[j] = H(j,k);
	    	for (int i=0;i<k;i++) 
			{
                  	ci = cci[i];
			si = ssi[i];
		   	T a = d[i];
		   	d[i] = ci*a+si*d[i+1];
		   	d[i+1] = -si*a+ci*d[i+1];
	    		}
	    	//phi[k]= atan(d[k+1]/d[k]);
	    	ci=hypot(d[k],d[k+1]); 
		cci[k]=d[k]/ci; 
		ssi[k]=d[k+1]/ci;

	    	//berechne neuen d-Vektor
		d= 0.;
	    	d[0]=beta; 
	    	for (int i=0;i<=k;i++) 
			{
                  	ci = cci[i];si = ssi[i];
		 	T a = d[i];
		 	d[i] = ci*a+si*d[i+1];
		 	d[i+1] = -si*a+ci*d[i+1];
	    		}

	    	//Schritt 6: Konvergenz?
	    	if(verbose) 
			{
			std::cout << "gmres iter "<<l<<" "<<k<<" resid "
		<<std::setw(0)<<std::setiosflags(std::ios::scientific)<<std::setprecision(8)
		<<std::abs(d[k+1])<< " thr "<<eps*beta_0<< " reduction "
		<<std::setw(5)<<std::setprecision(2)<<std::resetiosflags(std::ios::scientific)
		<<(d_alt - std::abs(d[k+1]))/d_alt*100<< "\n" <<std::setprecision(12);
			std::cout.flush();
			}
		
		d_alt = abs(d[k+1]);
            	//*err= d_alt;
	    	if (d_alt < eps*beta_0) 
			{
	       		// konstruiere R_k
	       		for (int i=0;i<k;i++) 
				{
                  		ci = cci[i];
				si = ssi[i];
		    		for (int j=0;j<k;j++)
					{
			 		T a = H(i,j);
			 		H(i,j) = ci*a+si*H(i+1,j);
			 		H(i+1,j) = -si*a+ci*H(i+1,j);
		    			} 
	       			} 

	       		// Loese R_k c = - d_k 
			d *= -1.;
	       		gmres_backsubstitute(H,c,d,k-1);
	       		for(int i=0;i<k;i++) 
				{
				if(!incore) st->get(v[0],i);
				x.axpy(c[i],v[incore?i:0]);
				}
	       		flag=0; goto myreturn;
	    		}
         	} // k-Schleife

	 // zum Neustart: Konstruiere R_k 
	 for (int i=0;i<MAXIT;i++) 
		{
                ci = cci[i];si = ssi[i];
	     	for (int j=0;j<MAXIT;j++)
			{
		 	T a = H(i,j);
		 	H(i,j) = ci*a+si*H(i+1,j);
		 	H(i+1,j) = -si*a+ci*H(i+1,j);
	     		}
	 	}

	// Loese R_k c = - d_k
	d *= -1.;
	gmres_backsubstitute(H,c,d,MAXIT-1);
	for(int i=0;i<MAXIT;i++)
		{
                if(!incore) st->get(v[0],i);
                x.axpy(c[i],v[incore?i:0]);
                }

      	} // l schleife
flag=1;

myreturn:
delete[] v;
if(!incore) delete st;

return !flag;
}

}//namespace

#endif
