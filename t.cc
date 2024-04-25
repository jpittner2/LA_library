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

// g++ -D _GLIBCPP_NO_TEMPLATE_EXPORT -g testblas.cc testblas2.cc nrutil_modif.cc -L/usr/local/lib/atlas -lstrassen -lf77blas -lcblas -latlas -ltraceback -lbfd -liberty

#include <time.h>
#include "la.h"
#include "vecmat3.h"
#include "quaternion.h"
#include "permutation.h"
#include "polynomial.h"
#include "contfrac.h"
#include "simple.h"
#include "graph.h"
#include "numbers.h"
#include "miscfunc.h"


using namespace std;
using namespace LA_Vecmat3;
using namespace LA_Quaternion;
using namespace LA_Simple;
using namespace LA;



extern void test(const NRVec<double> &x);


double ad; 
void f1(const double *c)
{
ad=*c;
}

void f2(double *c)
{
*c=ad;
}


inline int randind(const int n)
{
	return int(RANDDOUBLE()*n);
}

complex<double> mycident (const complex<double>&x) {return x;}

void printme(const NRPerm<int> &p)
{
PERM_RANK_TYPE rank=p.rank();
int n=p.size();
cout<<p.rank()<<" "<<p;
NRPerm<int> qq(n,rank);
if(qq!=p)  laerror("error in rank algorithm");
for(int i=0; i<4; ++i)
	{
	NRVec_from1<int>  inv=p.inversions(i);
	NRPerm<int> q(i,inv);
	if(q!=p) laerror("error in inversions algorithm");
	}
}

void printme0(const NRPerm<int> &p)
{
cout<<p;
}

void printme1(const NRPerm<int> &p)
{
cout <<p.parity()<<' ';
cout<<p;
//cout<<p.inverse();
}


static NRMat<PERM_RANK_TYPE> Snmtable;
static int unitary_n;
static PERM_RANK_TYPE space_dim;
static NRVec<PermutationAlgebra<int,int> > allyoung;
static NRVec<NRMat<int> >allyoungmat;
static NRVec<NRMat<int> >allyoungregular;
static NRVec<int> allyoung_irrep;
int current_irrep;
int allyoung_index;

void yprintme(const YoungTableaux<int>&y)
{
cout <<y;
if(!y.is_standard()) laerror("internal error in young");
allyoung[allyoung_index] = y.young_operator();
cout <<"Young "<<allyoung_index<<" (irrep "<<current_irrep<<") = "<<allyoung[allyoung_index]<<endl;
allyoungmat[allyoung_index] = NRMat<int>(allyoung[allyoung_index],false);
allyoungregular[allyoung_index] = RegularRepresentation(allyoung[allyoung_index],Snmtable);
//cout <<"Matrix representation = "<<allyoungmat[allyoung_index];
cout <<"Regular representation = "<<allyoungregular[allyoung_index];
allyoung_irrep[allyoung_index]=current_irrep;
allyoung_index++;
}

void pprintme(const Partition<int> &p)
{
CompressedPartition<int> pc(p);
++current_irrep;
cout<<'['<<pc<<"]\n";


PERM_RANK_TYPE snd=p.Sn_irrep_dim();
cout<<"Sn IR dim "<<snd<<endl;

YoungTableaux<int> y(p);
PERM_RANK_TYPE dim=y.generate_all_standard(yprintme);

Partition<int> q=p.adjoint();
if(dim!=snd) laerror("inconsistency in standard tableaux generation");
PERM_RANK_TYPE und=p.Un_irrep_dim(unitary_n);
cout<<"U("<<unitary_n<<") IR dim "<<und<<endl;
space_dim += und*snd;
CompressedPartition<int> qc(q);
cout <<"("<<qc<<')';
cout<<" Class size "<<qc.Sn_class_size()<<endl;

cout <<"Chi= "<<Sn_character(p,q)<<endl;

/*
int nn=p.size();
YoungTableaux<int> y(p);
for(int i=1; i<=y.nrows(); ++i)
        {
        for(int j=1; j<=y[i].size(); ++j) y[i][j]= nn--;
	}
cout <<y;
YoungTableaux<int> yy(y);
yy.clear();
cout<<yy;
cout <<y;
*/
}



int main()
{
sigtraceback(SIGSEGV,1);
sigtraceback(SIGABRT,1);
sigtraceback(SIGBUS,1);
sigtraceback(SIGFPE,1);

//cout.setf(ios::scientific);
cc:cout.setf(ios::fixed);
cout.precision(10);
cin.exceptions ( ifstream::eofbit | ifstream::failbit | ifstream::badbit );

cout <<"LA version = "<<LA::version<<endl;

NRVec<double> x(1.,10);
NRVec<double> y(2.,10);
NRVec<double> z(-2.,10);


if(0) test(x);

y.axpy(3,x);

y+=z;
/*
cout <<y;
NRVec<double> a(x);

NRVec<double> b;
b|=x;

NRVec<double> c;
c=a;

y =10. *y  ;

int i;
for(i=0;i<y.size();i++) cout <<y[i] <<" ";
cout <<"\n";

cout << y*z <<"\n";

z|=x;
z[1]=5;

cout <<"zunit= "<<z.unitvector()<<"\n";
cout <<"z= "<<z<<"\n";
test(x);

x = x*5;


cout <<"x= "<<x<<"\n";
cout <<"y= "<<y<<"\n";

NRVec<double> u;
u=x+y;
cout <<"u= "<<u<<"\n";

NRMat<double> aa(0.,3,3);
aa[0][0]=aa[1][1]=aa(2,2)=2.;

NRMat<double> bb(aa);

double *p;
aa.copyonwrite(); p= &aa[2][2];
*p=3.;
bb.copyonwrite(); bb(0,2)=1.;

cout << "aa= " <<aa <<"\n";
cout << "bb= " <<bb <<"\n";
cout <<"aa trace "<<aa.trace() <<"\n";
cout << "bbt= " <<bb.transpose() <<"\n";
NRMat<double> cc=aa & bb;
cout << "aa o+ bb= " << cc <<"\n";
cout << cc.rsum() <<"\n";
cout << cc.csum() <<"\n";

NRVec<double>w(3);
w[0]=1; w[1]=2;w[2]=3;
NRVec<double> v(0.,3);
v.gemv(0.,bb,'n',1.,w);
cout << " v= " <<v <<"\n";
v.gemv(0.,bb,'t',1.,w);
cout << " v= " <<v <<"\n";

*/
/*
const int n=6000;
NRMat<double> bb(1.,n,n);
for(int i=0;i<n;i++) for(int j=0;j<n;j++) bb[i][j]=2.;
for(int i=0;i<n;i++) for(int j=0;j<i;j++) {double t; t=bb[i][j] +bb[j][j]; bb[i][j]=t;bb[j][i]=t;}
*/

/*
NRMat<double> amat,bmat,cmat;
cin >>amat;
cin >>bmat;
cmat=amat*bmat;
cout<<cmat;
cmat.copyonwrite(); cmat[0][0]=0;
NRMat<double> amat(1.,2,2);
NRMat<double> bmat(amat);
NRMat<double> dmat(amat);
//NRMat<double>  cmat; cmat=bmat*2.;
NRMat<double>  cmat(bmat*2); //more efficient
dmat.copyonwrite(); dmat[0][0]=0;

cout<<amat;
cout<<bmat;
cout<<cmat;
cout<<dmat;


NRMat<double> amat;
NRVec<double>  avec;

cin >>amat;
cin >>avec;

cout << amat*avec;
cout << avec*amat;

NRVec<double> avec(0.,10);

f1(avec);
f2(avec);

NRVec<double> uu(3);
uu[0]=1; uu[1]=2; uu[2]=3;
cout << uu << (uu|uu) <<"\n";

NRSMat<double> sa(0.,3);
sa(0,0)=1; sa(0,2)=5; sa(2,2)=10;sa(1,0)=2;sa(1,1)=3; sa(2,1)=-1;

NRSMat<double> sb(0.,3);
sb(0,0)=-2; sb(0,2)=1; sb(2,2)=2;sb(1,0)=-1;sb(1,1)=7; sb(2,1)=3;

cout << "symetr\n" <<sa << -sa <<"\n";
cout << "symetr\n" <<sb <<"\n";

cout << "sa*sb\n" << sa*sb <<"\n";
cout << "sb*sa\n" << sb*sa <<"\n";

NRMat<double> m10(10.,3,3);
 cout << "10 + sa" << m10 + sa <<"\n";
*/

/*

const int dim=256;
NRMat<double> big1(dim,dim),big2(dim,dim),big3;
for(int i=0;i<dim;i++)
	for(int j=0;j<dim;j++)
		{
		big1[i][j]=i*i+j*j*j-3*j;
		big2[i][j]=i*i/(j+1)+j*j-3*j;
		}
double t=clock()/((double) (CLOCKS_PER_SEC));
big3= big1*big2;
cout <<" big1*big2 "<<big3[0][0]<<" time "<<clock()/((double) (CLOCKS_PER_SEC))-t <<"\n";

*/

#ifndef NO_STRASSEN
if(0)
{
NRMat<double> atest, btest,ctest;
{
int cc,c1,c2,c3;
cin >>cc>>c1>>c2>>c3;

atest.s_cutoff(cc,c1,c2,c3);
}
cin>>atest;
cin>>btest;

NRMat<double> dtest(atest.nrows(),btest.ncols());
dtest.gemm(0., atest, 't', btest, 'n', 1.);
cout << dtest;

NRMat<double> etest(atest.nrows(),btest.ncols());
etest.strassen(0., atest, 't', btest, 'n', 1.);
cout << etest;
}

if(0)
{
int dim;
cin >>dim;
NRMat<double> big1(dim,dim),big2(dim,dim),big3,big4(dim,dim);
for(int i=0;i<dim;i++)
        for(int j=0;j<dim;j++)
                {
                big1[i][j]=i*i+j*j*j-3*j;
                big2[i][j]=i*i/(j+1)+j*j-3*j;
                }
double t=clock()/((double) (CLOCKS_PER_SEC));
big3= big1*big2;
cout <<" classical big1*big2 "<<big3[0][0]<<" time "<<clock()/((double) (CLOCKS_PER_SEC))-t <<"\n";

for (int c=64; c<=512;c+=64)
	{
	big4.s_cutoff(c,c,c,c);
	t=clock()/((double) (CLOCKS_PER_SEC));
	big4.strassen(0., big1, 'n', big2, 'n', 1.);
	cout <<"cutoff "<<c<<" big1*big2 "<<big4[0][0]<<" time "<<clock()/((double) (CLOCKS_PER_SEC))-t <<"\n";
	}
}
#endif

if(0)
{
NRMat<double> a(3,3),b;
NRVec<double> v(3);
for(int i=0;i<3;i++) for(int j=0;j<3;j++) { a[i][j]= i*i+j; v[i]=10-i;}
b=a;
b*= sin(1.)+1;
cout << a <<v;
a.diagmultl(v);
cout << a;
b.diagmultr(v);
cout << b;
}

if(0)
{
NRMat<double> a(3,3),b;
NRVec<double> v(10);
v[0]=2;v[1]=3;v[2]=1;v[3]=-3;v[4]=2;v[5]=-1;v[6]=3;v[7]=-2;v[8]=1;v[9]=1;
for(int i=0;i<3;i++) for(int j=0;j<3;j++) { a[i][j]= (i+j)/10.; }
cout <<a;
cout << a.norm() <<"\n";
b=a*a;
cout << b.norm() <<"\n";
cout << exp(a);
cout << exp(a.norm()) <<"\n";
cout << ipow(a,3);
cout<<ipow(a,11);
cout <<commutator(a,b);

}

if(0)
{
NRMat<double> a(3,3);
for(int i=0;i<3;i++) for(int j=0;j<3;j++) { a[i][j]= (i+j)/10.; }
NRSMat<double> b(a);
NRMat<double> c(b);
cout <<a;
cout <<b;
cout <<c;
}

if(0)
{
NRMat<double> a(3,3);
a[0][0]=1; a[0][1]=2;a[0][2]=3;
a[1][0]=4; a[1][1]=-5;a[1][2]=7;
a[2][0]=-3;a[2][1]=10;a[2][2]=2;
NRMat<double> b(2,3);
b[0][0]=1;b[0][1]=2;b[0][2]=3;
b[1][0]=2;b[1][1]=4;b[1][2]=6;
cout <<a;
cout <<b;
linear_solve(a,&b);
cout <<a;
cout <<b;
}

if(0)
{
NRMat<double> a(3,3);
for(int i=0;i<3;i++) for(int j=0;j<3;j++) { a[i][j]= (i+j)/10.; }
NRVec<double> b(3);
cout <<a;
diagonalize(a,b);
cout <<a;
cout <<b;
}

if(0)
{
NRSMat<double> a(3);
NRMat<double>v(3,3);
for(int i=0;i<3;i++) for(int j=0;j<3;j++) { a(i,j)= (i+j)/10.; }
NRVec<double> b(3);
cout <<a;
NRMat<double>c=(NRMat<double>)a; //nebo NRMat<double>c(a);
NRMat<double>d=exp(c);
diagonalize(a,b,&v);
cout <<b;
cout <<v;
cout <<d;
diagonalize(d,b);
cout <<b;
cout <<d;
}

if(0)
{
NRMat<double> a;
cin >>a ;
NRMat<double> abak=a;
NRMat<double> abak2=a;
NRMat<double> u(a.nrows(),a.nrows()),v(a.ncols(),a.ncols());
NRVec<double>s(a.ncols()<a.nrows()?a.ncols():a.nrows());
singular_decomposition(a,&u,s,&v,0);
cout <<u;
NRMat<double> sdiag(0., u.ncols(),v.nrows());
sdiag.diagonalset(s);
cout <<sdiag;
cout <<v;
cout << "Error "<<(u*sdiag*v-abak).norm()<<endl;
NRMat<double> ai=calcinverse(abak2);
cout <<"regular inverse "<<ai;
NRVec<double>ss(s);ss.copyonwrite();
for(int i=0; i<ss.size(); ++i) if(ss[i]>1e-12) ss[i]=1./ss[i]; else ss[i]=0;
sdiag.diagonalset(ss);
NRMat<double> aii = v.transpose() *sdiag*u.transpose();
cout <<"svd inverse "<<aii;
cout << "Error in inverse "<< (aii-ai).norm()<<endl;
}

if(0)
{
NRMat<complex<double> > a;
cin >>a ;
NRMat<complex<double> > abak=a;
NRMat<complex<double> > u(a.nrows(),a.nrows()),v(a.ncols(),a.ncols());
NRVec<double>s(a.ncols()<a.nrows()?a.ncols():a.nrows());
singular_decomposition(a,&u,s,&v,0);
cout <<u;
NRMat<complex<double> > sdiag(0., u.ncols(),v.nrows());
NRVec<complex<double> > ss = s;
sdiag.diagonalset(ss);
cout <<sdiag;
cout <<v;
cout << "Error "<<(u*sdiag*v-abak).norm()<<endl;
}

if(0)
{
//diagonalize a general matrix and reconstruct it back; assume real eigenvalues
//double aa[]={1,2,3,4,-5,7,-3,10,2};
//NRMat<double> a(aa,3,3);
NRMat<double> a;
cin >>a;
cout <<a ;
int n=a.nrows();
NRMat<double> u(n,n),v(n,n);
NRVec<double>wr(n),wi(n);
gdiagonalize(a,wr,wi,&u,&v,0);
cout <<u;
cout <<wr;
cout <<wi;
cout <<v;

NRVec<double>z=diagofproduct(u,v,1);
for(int i=0;i<a.nrows();++i) wr[i]/=z[i];//account for normalization of eigenvectors
u.diagmultl(wr);
v.transposeme();
cout <<v*u;

}

if(0)
{
//diagonalize a general matrix and reconstruct it back; allow complex eigenvalues
NRMat<double> a;
cin >>a;
cout <<a ; 
int n=a.nrows();
NRMat<complex<double> > u(n,n),v(n,n);
NRVec<complex<double> >w(n);
gdiagonalize(a,w,&u,&v);
cout <<u;
cout <<w;
cout <<v;

NRVec<complex<double> >z=diagofproduct(u,v,1,1);
//NRMat<complex<double> > zz=u*v.transpose(1);
cout <<z;
//cout <<zz;
for(int i=0;i<a.nrows();++i) w[i]/=z[i];//account for normalization of eigenvectors
u.diagmultl(w);
cout <<v.transpose(1)*u;

}


if(0)
{
NRMat<double> a;
cin >>a;
int n=a.nrows();
NRMat<complex<double> > u(n,n),v(n,n);
NRVec<complex<double> >w(n);
gdiagonalize(a,w,&u,&v,0,n,0,1);
cout <<u;
cout <<w;
cout <<v;

NRVec<complex<double> >z=diagofproduct(u,v,1,1);
cout <<z;
for(int i=0;i<a.nrows();++i) w[i]/=z[i];//account for normalization of eigenvectors
cout <<u*v.transpose(1); //check biorthonormality
u.diagmultl(w);
cout <<v.transpose(1)*u;

}


if(0)
{
NRMat<complex<double> > a;
cin >>a;
int n=a.nrows();
NRMat<complex<double> > u(n,n),v(n,n);
NRVec<complex<double> >w(n);
gdiagonalize(a,w,&u,&v);
//gdiagonalize(a,w,&u,&v,0,n,0,1);
cout <<u;
cout <<w;
cout <<v;

NRVec<complex<double> >z=diagofproduct(u,v,1,1);
cout <<z;
for(int i=0;i<a.nrows();++i) w[i]/=z[i];//account for normalization of eigenvectors
cout <<u*v.transpose(1); //check biorthonormality
u.diagmultl(w);
cout <<v.transpose(1)*u;

}



if(0)
{
SparseMat<double> a(4,4);
NRVec<double> v(4);
v[0]=1;v[1]=2;v[2]=3;v[3]=4;
a=1.;
a.copyonwrite();
a.add(3,0,.5);
a.add(0,2,.2);
a.add(2,1,.1);
a.add(3,3,1.);
a.add(1,1,-1.);
SparseMat<double> c(a);
c*=10.;
cout <<a;
a.simplify();
cout <<a;
cout <<c;
NRMat<double>b(c);
cout <<b;
cout << b*v;
cout <<c*v;
cout <<v*b;
cout <<v*c;
}

if(0)
{
SparseMat<double> a(4,4),b(4,4);
a=1.;
a.copyonwrite();
a.add(3,0,.5);
b.add(0,2,.2);
b.add(2,1,.1);
b.add(3,3,1.);
b.add(1,1,-1.);
SparseMat<double>c=a+b;
cout <<c;
a.join(b);
cout<<a;
cout<<b;
}

if(0)
{
SparseMat<double> a(4,4),b(4,4);
a=0.; b=2;
a.add(3,0,.5);
a.add(0,2,.2);
a.add(1,1,1);
a.add(1,0,.2);
b.add(2,1,.1);
b.add(3,3,1.);
b.add(1,1,-1.);
NRMat<double> aa(a),bb(b);
SparseMat<double>c;
NRMat<double>cc;
//cout << NRMat<double>(c);
//cout <<cc;
//cout <<"norms "<<c.norm()<<" " <<cc.norm()<<endl;
cout <<"original matrix \n"<<aa;
cout <<(cc=exp(aa));
c=exp(a);
cout <<NRMat<double>(c);
cout <<"norms2 "<<c.norm()<<" " <<cc.norm()<<endl;
}

#define sparsity (n/4)
if(0)
{
for(int n=8; n<=1024*1024;n+=n)
	{
	SparseMat<double> aa(n,n);
	cout << "\n\n\ntiming for size "<<n<<endl;
	if(n<=512) {
	NRMat<double> a(0.,n,n);
	for(int i=0; i<sparsity;i++) a(randind(n),randind(n))=RANDDOUBLE();
	double t0=clock()/((double) (CLOCKS_PER_SEC));	
	//cout <<a;
	NRMat<double> b(exp(a));
	//cout <<b;
	cout <<"dense norm "<<b.norm() <<"\n";
	cout << "test commutator " <<commutator(a,b).norm() <<endl;
	double t1=clock()/((double) (CLOCKS_PER_SEC));    
	cout << "dense time " <<n<<' '<< t1-t0 <<endl;
	aa=SparseMat<double>(a);
	}
	else
	{
	for(int i=0; i<sparsity;i++) aa.add(randind(n),randind(n),RANDDOUBLE());
	}
	//cout <<aa;
	double t2=clock()/((double) (CLOCKS_PER_SEC));        
	SparseMat<double> bb(exp(aa));
	//cout <<bb;
	cout <<"sparse norm "<<bb.norm() <<"\n";
	cout << "test commutator " <<commutator(aa,bb).norm() <<endl;
        double t3=clock()/((double) (CLOCKS_PER_SEC));
	 cout <<"sparse length "<<bb.length()<<"\n";
        cout << "sparse time "<<n<<' ' << t3-t2 <<endl;
	}
}

if(0)
{
int n;
cin>>n;
	SparseMat<double> aa(n,n);
	for(int i=0; i<sparsity;i++) aa.add(randind(n),randind(n),RANDDOUBLE());
	SparseMat<double> bb=exp(aa);
	NRVec<double> v(n);
	 for(int i=0; i<n;++i) v[i]=RANDDOUBLE();
	NRVec<double> res1=bb*v;
	NRVec<double> res2=exptimes(aa,v);
	cout <<"difference = "<<(res1-res2).norm()<<endl;
}

if(0)
{
int n,k,m;
cin >>n>>k>>m;
{
NRMat<double> a(n,k),b(k,m),c(n,m),d(n,m);
for(int i=0;i<n;++i) for(int j=0;j<k;++j) a(i,j)= RANDDOUBLE();
for(int i=0;i<k;++i) for(int j=0;j<m;++j) b(i,j)= RANDDOUBLE();
c.gemm(0., a, 'n', b, 'n', .6);
SparseMat<double> aa(a);
d.gemm(0., aa, 'n', b, 'n', .6);
cout<<c<<d;
cout  <<"test error = "<<(c-d).norm()<<endl;
}
{
NRMat<double> a(k,n),b(k,m),c(n,m),d(n,m);
for(int i=0;i<k;++i) for(int j=0;j<n;++j) a(i,j)= RANDDOUBLE();
for(int i=0;i<k;++i) for(int j=0;j<m;++j) b(i,j)= RANDDOUBLE();
c.gemm(0., a, 't', b, 'n', .7);
SparseMat<double> aa(a);
d.gemm(0., aa, 't', b, 'n', .7);
cout<<c<<d;
cout  <<"test error = "<<(c-d).norm()<<endl;
}
{
NRMat<double> a(n,k),b(m,k),c(n,m),d(n,m);
for(int i=0;i<n;++i) for(int j=0;j<k;++j) a(i,j)= RANDDOUBLE();
for(int i=0;i<m;++i) for(int j=0;j<k;++j) b(i,j)= RANDDOUBLE();
c.gemm(0., a, 'n', b, 't', .8);
SparseMat<double> aa(a);
d.gemm(0., aa, 'n', b, 't', .8);
cout<<c<<d;
cout  <<"test error = "<<(c-d).norm()<<endl;
}
{
NRMat<double> a(k,n),b(m,k),c(n,m),d(n,m);
for(int i=0;i<k;++i) for(int j=0;j<n;++j) a(i,j)= RANDDOUBLE();
for(int i=0;i<m;++i) for(int j=0;j<k;++j) b(i,j)= RANDDOUBLE();
c.gemm(0., a, 't', b, 't', .9);
SparseMat<double> aa(a);
d.gemm(0., aa, 't', b, 't', .9);
cout<<c<<d;
cout  <<"test error = "<<(c-d).norm()<<endl;
}

}


if(0)
{
SparseMat<double> a(4,4),b(4,4),d;
a=0.; b=2;
a.add(3,0,.5);
a.add(0,2,.2);
a.add(1,1,1);
a.add(1,0,.2);
b.add(2,1,.1);
b.add(3,3,1.);
b.add(1,1,-1.);
NRMat<double> aa(a),bb(b),dd;
SparseMat<double>c;
NRMat<double>cc;

c=commutator(a,b);
cc=commutator(aa,bb);

cout <<cc;
cout <<NRMat<double>(c);
cout <<"norms2 "<<c.norm()<<" " <<cc.norm()<<endl;
}

/*
NRVec<double> v(10.,10);
v+= 5.;
cout <<v;
*/
if(0)
{
int n;
cin >>n;
NRMat<double> a(n,n);
for(int i=0;i<n;++i) for(int j=0;j<=i;++j)
	{
	a(i,j)= RANDDOUBLE();
	a(j,i)= RANDDOUBLE();
	}
NRMat<double> b; b|=a;
NRVec<double> er(n),ei(n);
NRMat<double> vr(n,n),vl(n,n);
gdiagonalize(b,er,ei,&vl,&vr,1,0,1,1);
cout <<er<<ei;
cout <<"left eivec\n"<<vl <<"right eivec\n"<<vr;
cout <<"test orthogonality\n" << vl.transpose() * vr;
NRMat<double> u=exp(a*.1);
gdiagonalize(u,er,ei,&vl,&vr,1,0,1,1);
cout <<er<<ei;
cout <<"left eivec\n"<<vl <<"right eivec\n"<<vr;
cout <<"test orthogonality\n" << vl.transpose() * vr;
}

if(0)
{
int k;
cin >>k;
int n=2*k;
NRMat<double> a(n,n);
//matrix with known spectrum
for(int i=0;i<n;++i) 
        {
	for(int j=0;j<k;++j) a(i,j)=j+1.+k*k-(i==j?0.:i+1.);
	for(int j=k; j<n; ++j) a(i,j)=i-j-k*k+(i==j?i+1.:0.);
        }
NRVec<double> er(n),ei(n);
NRMat<double> vr(n,n),vl(n,n);
cout <<"input matrix\n"<<a;
gdiagonalize(a,er,ei,&vl,&vr,1,0,1);
cout <<er<<ei;
cout <<"left eivec\n"<<vl <<"right eivec\n"<<vr;
cout <<"test orthogonality\n" << vl.transpose() * vr;
}


if(0)
{
int n;
cin>>n;
NRMat<double> a(n,n);
for(int i=0;i<n;++i) for(int j=0;j<i;++j)
        {
        a(i,j)= RANDDOUBLE();
        a(j,i)= -a(i,j);
        }
cout <<"a matrix \n"<<a;
cout<<"EXP\n";
NRMat<double> b=exp0(a);
cout <<"b=exp(a) matrix\n"<<b;

cout <<"LOG\n";
NRMat<double> c=log(b); //matrixfunction(a,&mycident,1);
cout <<"c=log(exp(a))\n"<<c <<"error: "<<(c-a).norm()<<endl;
cout <<"EXP-MY\n";
NRMat<double> e=exp(c);
cout <<"e=exp(c)\n"<<e;
cout<<"error2 = "<<(e-b).norm()<<endl;
}

if(0)
{
int n;
double f;
cin>>n >>f ;
NRMat<double> a(n,n);
NRVec<double>u(n),v,w;
for(int i=0;i<n;++i) 
        {
	u[i]=f*RANDDOUBLE();
	for(int j=0;j<n;++j)
       		 a(i,j)= f*RANDDOUBLE();
        }
//cout <<"a matrix \n"<<a;
//cout<<"EXP\n";
double t=clock()/((double) (CLOCKS_PER_SEC));
NRMat<double> b=exp0(a);
cout <<"exp0 took "<<clock()/((double) (CLOCKS_PER_SEC))-t<<endl;
//cout <<"b=exp0(a) matrix\n"<<b;
t=clock()/((double) (CLOCKS_PER_SEC));
NRMat<double> c=exp(a,true);
cout <<" horner exp  took "<<clock()/((double) (CLOCKS_PER_SEC))-t<<endl;
//cout <<"exp(a)\n"<<c;
cout<<"error1 = "<<(c-b).norm()/b.norm()<<endl;
t=clock()/((double) (CLOCKS_PER_SEC));
c=exp(a,false);
cout <<" tricky exp  took "<<clock()/((double) (CLOCKS_PER_SEC))-t<<endl;
cout<<"error2 = "<<(c-b).norm()/b.norm()<<endl;

v=b*u;
t=clock()/((double) (CLOCKS_PER_SEC));
w=exptimes(a,u);
cout <<"exptimes  took "<<clock()/((double) (CLOCKS_PER_SEC))-t<<endl;
cout <<"error of exptimes = "<<(v-w).norm()/v.norm()<<endl;


}


if(0)
{
int n;
cin>>n;
NRMat<double> a(n,n);
for(int i=0;i<n;++i) for(int j=0;j<=i;++j)
        {
        a(i,j)= .1*RANDDOUBLE();
        a(j,i)= a(i,j);
        }
NRMat<double> b=exp(a);
NRMat<double> s=exp(a*.5);
NRMat<double> y(0.,n,n);
NRMat<double> z(0.,n,n);
        double r;
int i=0;
y=b;z=1.;
cout << "norm = "<<b.norm(1.)<<endl;
do
        {
	NRMat<double> tmp=z*y*-1.+3.;
	NRMat<double> ynew=y*tmp*.5;
	z=tmp*z*.5;
	y=ynew;
        cout <<"iter "<<i <<" residue "<< (r=(y-s).norm())<<endl;
        ++i;
        } while(abs(r)>1e-10);
}


if(0)
{
int n=3;
NRMat<double> a(n,n);
 a(0,0)=1.;
        a(0,1)=2.;
        a(1,0)=2.;
        a(1,1)=6.;
a(2,2)=-4;
a(0,2)=1;
cout <<a;
double d;
NRMat<double> c=calcinverse(a,&d);
cout <<a<<c;
}

if(0)
{
NRMat<double> a(3,3);
NRMat<double> b=a;
for(int i=1; i<4;i++) b=b*b;
}

if(0)
{
NRMat<double> aa,bb,cc;
cin >>aa;
cc=copytest(aa);
cout <<cc;

NRMat<complex<double> > a,b,c;
a=complexify(aa);
c=copytest(a);
cout <<c;
b=log(a);
cout <<b;
cout <<exp(b);
}

if(0)
{
NRMat<complex<double> > a,b,c;
cin>>a;
c=copytest(a);
cout <<c;
b=log(a);
cout <<b;
cout <<exp(b);
}

if(0)
{
NRMat<double> a,b,c;
cin >>a;
c=copytest(a);
cout <<c;
}

if(0)
{
NRMat<double> a;
NRMat<double> b=exp(a);
NRMat<double> c=log(b);
cout <<a;
cout <<b;
cout <<c;
cout << (b-exp(c)).norm() <<endl;
}

if(00)
{
NRMat<double> a;
cin >>a;
NRMat<double> c=log(a); //matrixfunction(a,&mycident,1);
cout <<c;
NRMat<double> b=exp(c);
cout <<"exp(log(x))\n"<<b;
cout<<(b-a).norm()<<endl;
}

if(0)
{
//check my exponential with respect to spectral decomposition one
NRSMat<double> a;
cin >>a;
NRMat<double> aa(a);
NRMat<double> b=exp(aa);
NRMat<double> c=matrixfunction(a,&exp);
cout <<a;
cout <<b;
cout <<c;
cout << (b-c).norm()/b.norm() <<endl;
}

if(0)
{
//verify BCH expansion
NRMat<double> h;
NRMat<double> t;
cin >>h;
cin >>t;
NRMat<double> r1= exp(-t) * h * exp(t);
NRMat<double> r2=BCHexpansion(h,t,30);
cout <<r1;
cout <<r2;
cout <<"error = "<<(r1-r2).norm()<<endl;
}

if(0)
{
int dim; cin>>dim;
int degree; cin >>degree;
NRMat<double> h(dim,dim);
NRMat<double> t(dim,dim);
NRVec<double> x(dim);
h.randomize(1.);
t.randomize(1.);
x.randomize(2.);
NRVec<double> y1 = exp(-t) * h * exp(t) *x;
NRVec<double> y2 = BCHtimes(h,'n',t,'n',x,degree,true);
NRVec<double> y3  = exp(t) * h * exp(-t) *x;
NRVec<double> y4 = BCHtimes(h,'n',t,'n',x,degree,false);
cout <<"BCHtimes error = "<<(y1-y2).norm()<<endl;
cout <<"BCHtimes error = "<<(y3-y4).norm()<<endl;
}

if(0)
{
int n;
cin >>n;
SparseMat<double> a(n,n);
for(int i=0;i<n;++i) for(int j=0;j<=i;++j)
        {
        a.add(i,j,RANDDOUBLE());
        }
a.setsymmetric();
NRSMat<double> aa(a); 
NRMat<double> aaa(a); 
NRVec<double> w(n);
NRMat<double> v(n,n);
//cout <<aa;
diagonalize(aa, w, &v,0);
//cout <<w;
//cout <<v;
//cout << v*aaa*v.transpose(); 
cout <<  (v*aaa*v.transpose() - diagonalmatrix(w)).norm()<<endl;
}

if(0)
{
NRMat<complex<double> > a;
cin >>a;
NRMat<complex<double> > b=exp(a);
cout <<b;
}

if(0)
{
int n;
double d;
cin >>n;
//NRMat<double> a(n,n);
NRSMat<double> a(n);
for(int i=0;i<n;++i) for(int j=0;j<=i;++j)
        {
        a(j,i)=a(i,j)=RANDDOUBLE()*(i==j?10.:1.);
        }
cout <<a;
NRMat<double> y(1,n);
for(int i=0;i<n;++i) y(0,i)=RANDDOUBLE();
cout <<y;
linear_solve(a,&y,&d);
cout << y;
cout <<"det is "<<d<<endl;
}

if(0)
{
int n;
cin >>n;
SparseMat<double> a(n,n);
int spars=n*n/3;
        for(int i=0; i<spars;i++) a.add(randind(n),randind(n),RANDDOUBLE());

NRMat<double> aa(a);
NRVec<double> v(aa[0],n*n);

cout <<a;
cout <<aa;
cout <<v;

cout <<"test "<<aa.dot(aa)<<endl;
cout <<"test "<<v*v<<endl;
cout <<"test "<<a.dot(aa)<<endl;
cout <<"test "<<a.dot(a)<<endl;

}

if(0)
{
NRMat<double> amat,bmat;
cin >>amat;
cin >>bmat;
NRVec<double> v(amat.nrows());
diagonalize(amat,v,1,1,0,&bmat,1);
cout <<amat;
cout <<v;
}


if(0)
{
int n,m;
cin>>n >>m;
NRSMat<double> a(n,n);
NRVec<double> rr(n);

for(int i=0;i<n;++i) for(int j=0;j<=i;++j)
        {
        a(i,j)= RANDDOUBLE();
	if(i==j) a(i,i)+= .5*(i-n);
        }

NRSMat<double> aa;
NRMat<double> vv(n,n);
aa=a; diagonalize(aa,rr,&vv);
NRVec<double> r(m);
NRVec<double> *eivecs = new NRVec<double>[m];
davidson(a,r,eivecs,NULL,m,1,1e-6,1,200);

cout <<"Davidson energies " <<r;
cout <<"Exact energies " ;
for(int i=0; i<m; ++i) cout <<rr[i]<<" ";
cout <<endl;

cout <<"Eigenvectors compare:\n";
for(int i=0; i<m; ++i) 
	{
	cout <<eivecs[i];
	for(int j=0; j<n;++j) cout <<vv[j][i]<<" ";
	cout <<endl;
	}

}

if(0) //davidson of a non-symmetric matrix
{
int n,m;
cin>>n >>m;
NRMat<double> a(n,n);
NRVec<double> rr(n),ii(n);

double tmp=0.;
for(int i=0;i<n;++i) for(int j=0;j<n;++j)
        {
        a(i,j)= RANDDOUBLE();
        a(j,i)= RANDDOUBLE();
	if(i==j) a(i,i)+= .5*(i-n);
	tmp+= (a(i,j)-a(j,i))*(a(i,j)-a(j,i));
        }
cout <<"norm of asymmetry "<<sqrt(tmp)<<endl;	

NRMat<double> aa=a;
NRMat<double> vv=aa;
gdiagonalize(aa, rr, ii, NULL, &vv, 1, 0, 2, 0, NULL,NULL);
NRVec<double> r(m);
NRVec<double> *eivecs = new NRVec<double>[m];
davidson(a,r,eivecs,NULL,m,1,1e-6,1,200);

cout <<"Davidson energies " <<r;
cout <<"Exact energies " ;
for(int i=0; i<m; ++i) cout <<rr[i]<<" ";
cout <<endl;

cout <<"Eigenvectors compare:\n";
for(int i=0; i<m; ++i) 
	{
	cout <<eivecs[i];
	for(int j=0; j<n;++j) cout <<vv[j][i]<<" ";
	cout <<endl;
	}

}

//davidson of large very sparse matrix (10n/n^2)

#undef sparsity
#define sparsity (n*2)
if(0)
{
int n,m;
cin >>n>>m;
	SparseMat<double> aa(n,n);
	aa.setsymmetric();
	for(int i=0; i<sparsity;i++) aa.add(randind(n),randind(n),RANDDOUBLE());
	for(int i=0; i<n; ++i) aa.add(i,i,500*RANDDOUBLE());
NRVec<double> r(m);
davidson(aa,r,(NRVec<double> *)NULL,"eivecs",m,1,1e-5,0,300,300);
cout <<r;
}

//davidson of symmetric matrix and of its unsymmetric similarity transform
#undef sparsity
#define sparsity (n*2)
#define sparsity2 (n/5)
if(0)
{
int n,m;
cin >>n>>m;
        SparseMat<double> aa(n,n);
        aa.setsymmetric();
        for(int i=0; i<sparsity;i++) aa.add(randind(n),randind(n),RANDDOUBLE());
        for(int i=0; i<n; ++i) aa.add(i,i,500*RANDDOUBLE());
NRVec<double> r(m);
NRVec<double> r2(m);
davidson(aa,r,(NRVec<double> *)NULL,"eivecs",m,1,1e-5,1,300,300);
 SparseMat<double> bb(n,n);
for(int i=0; i<sparsity2;i++) bb.add(randind(n),randind(n),RANDDOUBLE());
SparseMat<double> e1,e2,cc;
e1=exp(bb);
e2=exp(bb*-1.);
aa.setunsymmetric();
cc=e1*aa*e2;
davidson(cc,r2,(NRVec<double> *)NULL,"eivecs2",m,1,1e-5,1,300,300);
cout <<"original matrix" <<r;
cout <<"transformed matrix" <<r2;
}

//davidson of large very sparse matrix unsymmetric matrix
#undef sparsity
#define sparsity (n)
if(0)
{
int n,m;
cin >>n>>m;
        SparseMat<double> aa(n,n);
        for(int i=0; i<sparsity;i++) 
		{
		int k= randind(n);
		int l= randind(n);
		double a=RANDDOUBLE();
		double b=RANDDOUBLE()-.5;
		aa.add(k,l,a);
		aa.add(l,k,a+b/20);
		}
        for(int i=0; i<n; ++i) aa.add(i,i,500*RANDDOUBLE());
NRVec<double> r(m);
davidson(aa,r,(NRVec<double> *)NULL,"eivecs",m,1,1e-5,0,300,300);
cout <<r;
}


if(0)
{
int n,m;
cin>>n >>m;
NRMat<double> a(n,m);
NRVec<double> b(n);
NRVec<double> x(m);

for(int i=0;i<n;++i) for(int j=0;j<m;++j)
        {
        //a(j,i)= 2*i*i*i-5*j+ +9*8*7*6*5*4*3*2/(i+j+1.)+3*(i*i+2*j*j*j);
	a(i,j)= RANDDOUBLE();
	if(i==j) a(i,i)+= .5*(i-n);
        }
for(int i=0;i<n;++i) b[i] = i;

NRVec<double> bb=b;
cout <<a;
//cout <<b;
NRMat<double>aa=a;
double d;
linear_solve(aa,bb,&d);
cout <<"det = "<<d<<endl;
//cout <<bb;
gmres(a,b,x,1,1e-10,100,1,0,1,0);
//conjgrad(a,b,x,1,1e-10,200,1,0,1);

cout <<"\nsolution compare:\n";
for(int i=0; i<m; ++i) 
	{
	cout <<"iterative solver: ";
	cout <<x[i];
	cout <<" direct solver:";
	cout <<bb[i];
	cout <<endl;
	}

}


if(0)
{
int n,m;
cin>>n >>m;
SparseMat<double> aa(n,m);
NRVec<double> b(n);
NRVec<double> x(m);

//tridiagonal
        for(int i=0; i<n; ++i) aa.add(i,i,RANDDOUBLE());
	for(int i=0; i<n-1; i+=1) aa.add(i,i+1,.002*RANDDOUBLE());
	for(int i=0; i<n-1; i+=1) aa.add(i+1,i,.002*RANDDOUBLE());

for(int i=0;i<n;++i) b[i] = i+1;
gmres(aa,b,x,1,1e-20,20,1,1,1,1000,1);
//conjgrad(aa,b,x,1,1e-10,1000,1,0,1);
}

if(0)
{
NRMat<double> A(3,3);
A=1.;
double *p = (double *)A;
*p=2.;
cout <<A;
}

#if 0 //@@@make a more realistic test
{
int i;
DIIS<NRVec<double> > diis(5);
int dim;
cin>>dim;
NRVec<double> solution(dim), deviation(dim);
for(i=0; i<dim; ++i) solution[i]=i&1 ? i/2.:-i-3.;
for(i=0; i<dim; ++i) deviation[i]= (i&2 ? 1:-1) * RANDDOUBLE();
double norm=1e100;
for(int iter=1; iter<100 && norm>1e-8 ; ++iter)
	{
	NRVec<double> trial=solution;
	trial.copyonwrite();
	for(i=0; i<dim; ++i) trial[i] += deviation[i]/iter;
	cout <<"iter "<<iter<<endl;
	cout << "trial "<<trial;
	cout <<"diis " << (norm=diis.extrapolate(trial)) <<endl;
	cout << "after diis "<<trial;
	deviation=trial-solution;
	}
}
#endif

if(0)
{
NRMat<double> a,b;
cin >>a;
b=realsqrt(a);
cout <<b;
cout <<b*b;
cout <<(b*b-a).norm();
}

if(0)
{
NRSMat<double> a;
NRMat<double> b;
cin >>a>>b;
cout <<a*b;
}

if(0)
{
NRMat<double> a,b;
cin >>a >>b;
cout <<a.oplus(b);
cout <<a.otimes(b);
}


//test of general square matrix eigenvector derivatives
if(0)
{
const bool biorthonormalize=1;

NRMat<double> a;
cin >>a;
if(a.nrows()!=a.ncols()) laerror("must be square matrix");
int n=a.nrows();

NRMat<double>vl(n,n),vr(n,n);
NRVec<double> wr(n),wi(n);
NRMat<double> awork(a);
gdiagonalize(awork,wr,wi,&vl,&vr,1,0,1,biorthonormalize);
for(int i=0; i<n; ++i) if(wi[i]) laerror("try another matrix with real eigenvalues");

NRMat<double> eival(0.,n,n);
eival.diagonalset(wr);

cout <<"test biorthonormality "<< (vl.transpose() * vr - 1.).norm()<<endl;

NRMat<double> hlp;
hlp = a *vr - vr * eival;
cout  <<"test right eivectors "<<hlp.norm()<<endl;

hlp= vl.transpose() * a - eival * vl.transpose();
cout  <<"test left eivectors "<<hlp.norm()<<endl;

hlp =  vl.transpose() * a * vr - eival;
cout  <<"test eigenvalues "<<hlp.norm()<<endl;

NRMat<double> ader(n,n);
for(int i=0; i<n; ++i)
	for(int j=0; j<n; ++j)
		ader(i,j) = RANDDOUBLESIGNED();

cout <<"eigenvalues\n"<<wr<<endl;



//compute numerical derivatives of eigenvectors
double  h=1e-5;
awork = a+ ader*h;
NRMat<double> vlx(n,n),vrx(n,n);
NRVec<double> wrx(n),wix(n);
gdiagonalize(awork,wrx,wix,&vlx,&vrx,1,0,1,biorthonormalize);
for(int i=0; i<n; ++i) if(wix[i]) laerror("try another matrix with real eigenvalues");

awork = a - ader*h;
NRMat<double> vly(n,n),vry(n,n);
NRVec<double> wry(n),wiy(n);
gdiagonalize(awork,wry,wiy,&vly,&vry,1,0,1,biorthonormalize);
for(int i=0; i<n; ++i) if(wiy[i]) laerror("try another matrix with real eigenvalues");

NRMat<double> vld,vrd;
NRVec<double> wrd;
vld = (vlx-vly) * (.5/h);
vrd = (vrx-vry) * (.5/h);
wrd = (wrx-wry) * (.5/h);

NRMat<double> vlg,vrg;
NRVec<double> wrg(n);

//compute analytic derivative
NRMat<double> tmp(n,n);
tmp.gemm(0.,vl,'t', ader * vr,'n',1.);
hlp |= tmp;
cout <<" C~+ VH C = \n"<<tmp<<endl;

tmp.diagonalof(wrg);
for(int i=0; i<n; ++i)
        for(int j=0; j<n; ++j)
		if(i!=j) tmp(i,j) /= (wr[j] - wr[i]); else  tmp(i,j) = 0.;
cout <<" old X matrix (tmp) \n"<<tmp<<endl;

NRMat<double> Y = tmp;
NRMat<double> S = vr.transpose() * vr;
cout <<"test S\n"<<S;
NRMat<double> tmp2 = S * tmp;
cout <<"test tmp2\n"<<tmp2;
Y.copyonwrite();
for(int i=0; i<n; ++i) Y(i,i) -= tmp2(i,i);

cout <<"Y matrix \n"<< Y;

NRMat<double> vri = calcinverse(vr);

NRMat<double> numX = vri * vrd;
cout <<" numerical X matrix \n"<< numX;
cout <<" numerical X matrix test = "<< (vr * numX - vrd).norm()<<endl;

vrg = vr * Y;

vlg = - (Y*vri).transpose();

//and compare
cout <<"eigenvalue numerical derivative\n"<<wrd<<endl;
cout <<"eigenvalue analytic derivative\n"<<wrg<<endl;
cout <<"eigenvalue derivative error = "<<(wrd-wrg).norm()<<endl;

//and for right eigenvectors
cout <<"right eigenvector  numerical derivative\n"<<vrd<<endl;
cout <<"right eigenvector  analytic derivative\n"<<vrg<<endl;
cout <<"right eigenvector derivative error = "<<(vrd-vrg).norm()<<endl;

//and for left eigenvectors
cout <<"left eigenvector  numerical derivative\n"<<vld<<endl;
cout <<"left eigenvector  analytic derivative\n"<<vlg<<endl;
cout <<"left eigenvector derivative error = "<<(vld-vlg).norm()<<endl;


}

//@@@@@@@make this derivative check in complex version


if(0)
{
try { laerror("test catch exception"); }
catch(LAerror x)
	{
	cout <<"caught exception: "<<x <<endl;
	}
laerror("test exception 2");
}

if(0)
{
NRVec<double> v(3);
v[0]=1; v[1]=2; v[2]=3;
NRVec<complex<double> > vv = v;
NRVec<double>u(v);
vv += u;
cout <<vv;
}

if(0)
{
complex<double> scale; cin >> scale;

NRMat<complex<double> > h; cin >>h; 
NRVec<complex<double> > x(h.nrows());
NRVec<complex<double> > y,z;

x.randomize(1.);

y=exptimes(h*scale,x,false,1.);
z=exptimes(h,x,false,scale);

cout <<x;
cout <<y;
cout <<z;
cout <<"Error "<<(y-z).norm()<<endl;

}

if(0)
{
int nocc,nvirt;
cin >>nocc>>nvirt;
int n=nocc+nvirt;
NRMat<double> t(nocc,nvirt);
t.randomize(0.5);
NRSMat<double> A(n);
A=1.;
for(int i=0; i<nocc;++i) for(int a=nocc; a<n; ++a) A(i,a) = t(i,a-nocc);
NRMat<double> B(n,n);
NRVec<double> E(n);
cout <<A;
NRSMat<double> Awork=A;
diagonalize(Awork,E,&B);
cout <<B<<E;
NRMat<double> U = A*B;
for(int p=0; p<n; ++p) E[p] = 1./E[p];
U.diagmultr(E);
cout << "Unitary\n"<<U;
cout <<"Error = "<<(U*U.transpose() -1.).norm()<<endl;
}

if(0)
{
double t;
NRMat<complex<double> > ham;
cin >>ham;
NRMat<double> hamreal = realpart(ham);
t=clock()/((double) (CLOCKS_PER_SEC));
NRMat<complex<double> >hamexp1 = exp(ham*complex<double>(0,1));
cout <<"dense exp time "<<clock()/((double) (CLOCKS_PER_SEC))-t <<"\n";
SparseMat<complex<double> > h(ham);
h *= complex<double>(0,1);
h.simplify();
cout <<"norms of input (should be same) "<<ham.norm()<<" "<<h.norm()<<endl;
cout <<"length of hamiltonian "<<h.length()<<endl;
t=clock()/((double) (CLOCKS_PER_SEC));
SparseMat<complex<double> > hexp = exp(h);
cout <<"sparse exp time "<<clock()/((double) (CLOCKS_PER_SEC))-t <<"\n";
cout <<"length of exp of ihamiltonian "<<hexp.length()<<endl;
NRMat<complex<double> >hamexp2(hexp);
cout <<"norm of results "<<hamexp1.norm() <<" "<<hamexp2.norm()<<endl;
cout <<"error = "<<(hamexp1-hamexp2).norm()<<endl;
NRMat<double> s,c;
t=clock()/((double) (CLOCKS_PER_SEC));
sincos(s,c,hamreal);
cout <<"dense sincos time "<<clock()/((double) (CLOCKS_PER_SEC))-t <<"\n";
NRMat<complex<double> >hamexp3 = complexmatrix(c,s);
cout <<"sincos error = "<<(hamexp1-hamexp3).norm()<<endl;
SparseMat<double> hreal(hamreal);
SparseMat<double>  si,co;
t=clock()/((double) (CLOCKS_PER_SEC));
sincos(si,co,hreal);
cout <<"sparse sincos time "<<clock()/((double) (CLOCKS_PER_SEC))-t <<"\n";
cout <<"length of sin,cos "<<si.length()<<" "<<co.length()<<endl;
NRMat<complex<double> >hamexp4 = complexmatrix(NRMat<double>(co),NRMat<double>(si));
cout <<"sincos error 2 = "<<(hamexp1-hamexp4).norm()<<endl;

NRVec<complex<double> > rhs(ham.ncols());
rhs.randomize(1.);
t=clock()/((double) (CLOCKS_PER_SEC));
NRVec<complex<double> > r1 = exptimes(ham*complex<double>(0,1),rhs);
cout <<"dense exptimes "<<clock()/((double) (CLOCKS_PER_SEC))-t <<"\n";
t=clock()/((double) (CLOCKS_PER_SEC));
NRVec<complex<double> > r2 = exptimes(h,rhs);
cout <<"sparse exptimes "<<clock()/((double) (CLOCKS_PER_SEC))-t <<"\n";
cout <<"exptimes error = "<<(r1-r2).norm()<<endl;
NRVec<complex<double> > siv,cov;
t=clock()/((double) (CLOCKS_PER_SEC));
//sincostimes(ham,siv,cov,rhs);
sincostimes(hamreal,siv,cov,rhs);
cout <<"dense sincostimes "<<clock()/((double) (CLOCKS_PER_SEC))-t <<"\n";
cout <<"sincostimes errors = "<<(siv-s*rhs).norm()<<" "<<(cov-c*rhs).norm()<<endl;
NRVec<complex<double> > r3 = cov + siv*complex<double>(0,1);
cout <<"sincostimes error  = "<<(r1-r3).norm()<<endl;
/*
 * real sparse matrix times complex vector not implemented
NRVec<complex<double> > siv2,cov2;
t=clock()/((double) (CLOCKS_PER_SEC));
sincostimes(hreal,siv2,cov2,rhs);
cout <<"sparse sincostimes "<<clock()/((double) (CLOCKS_PER_SEC))-t <<"\n";
NRVec<complex<double> > r4 = cov2 + siv2*complex<double>(0,1);
cout <<"sincostimes error 2 = "<<(r1-r4).norm()<<endl;
*/
}

if(0)
{
double t;
SparseMat<complex<double> > h;
cin >> h;
h *= complex<double>(0,1);
h.simplify();
cout <<"length of hamiltonian "<<h.length()<<endl;
t=clock()/((double) (CLOCKS_PER_SEC));
SparseMat<complex<double> > hexp = exp(h);
cout <<"sparse exp time "<<clock()/((double) (CLOCKS_PER_SEC))-t <<"\n";
cout <<"length of exp hamiltonian "<<hexp.length()<<endl;
NRMat<complex<double> > he(hexp);
NRMat<complex<double> > hec(he);
NRMat<complex<double> > het(he);
hec.conjugateme();
het.transposeme();
//cout <<he;
cout <<"unitarity error "<<(hec*he).norm(1)<<endl;
cout <<"symmetry error "<<(het-he).norm()<<endl;
}

if(0)
{
int n;
cin >>n;
NRSMat<double> hd(n);
hd.randomize(1);
SparseSMat<double> h(hd);
NRMat<double> rd = hd*hd;
SparseSMat<double> r = h*h;
NRSMat<double>rx(r);
NRMat<double> r2(rx);
cout <<"Error = "<<(r2-rd).norm()<<endl;
}


if(0)
{
SparseSMat<double> h0;
cin>>h0;
cout <<"matrix read\n"; cout.flush();
SparseSMat<double> h1 = h0; //.submatrix(0,2047,0,2047);
SparseSMat<complex<double> > h = imagmatrix(h1);
double t=clock()/((double) (CLOCKS_PER_SEC));
SparseSMat<complex<double> > r = h*h;
cout <<"SparseSMat mult time "<<clock()/((double) (CLOCKS_PER_SEC))-t <<"\n";
cout.flush();
t=clock()/((double) (CLOCKS_PER_SEC));
r = exp(h);
cout <<"SparseSMat exp time "<<clock()/((double) (CLOCKS_PER_SEC))-t <<"\n";
cout.flush();
if(h.nrows()<=1024)
{
NRSMat<complex<double> > h3(h);
NRMat<complex<double> > h2(h3);
NRMat<complex<double> >r2 = exp(h2);
cout <<"error = "<<(r2-NRMat<complex<double> >(NRSMat<complex<double> >(r))).norm()<<endl;
cout <<"errorx = "<<(r2-NRSMat<complex<double> >(r)).norm()<<endl;
}
}


if(0)
{
int n;
cin >>n;
NRMat<double> a(n,n);
a.randomize(1);
for(int i=0; i<n; ++i) for(int j=0; j<i; ++j) {a(i,j)=0.;}
cout <<a;
NRMat<double> bb=a.transpose()*a;
NRMat<double> cc(bb);
NRMat<double> b(bb);
NRMat<double> c(cc);
cholesky(b,0);
cout <<b;
cout << "Error = "<<(b*b.transpose()-bb).norm()<<endl;
cholesky(c,1);
cout <<c;
cout << "Error = "<<(c.transpose()*c-cc).norm()<<endl;
}



if(0)
{
int n;
cin >>n;
NRMat<complex<double> > a(n,n);
a.randomize(1);
for(int i=0; i<n; ++i) for(int j=0; j<i; ++j) {a(i,j)=0.;}
for(int i=0; i<n; ++i) {a(i,i).imag(0.); if(a(i,i).real()<0) a(i,i).real(-a(i,i).real());}
cout <<a;
NRMat<complex<double> > bb=a.transpose(true)*a;
NRMat<complex<double> > cc(bb);
NRMat<complex<double> > b(bb);
NRMat<complex<double> > c(cc);
cholesky(b,0);
cout <<b;
cout << "Error = "<<(b*b.transpose(true)-bb).norm()<<endl;
cholesky(c,1);
cout <<c;
cout << "Error = "<<(c.transpose(true)*c-cc).norm()<<endl;

}

if(0)
{
int n;
cin >>n;
NRMat<double> bb(0.,n,n);
int nn=0;
for(int i=0; i<n; ++i) for(int j=i; j<n; ++j) {if(RANDDOUBLE()>0.995 || i==j) {++nn; bb(i,j)=bb(j,i)=RANDDOUBLE()*(i==j?10.:.5/(i+j)*(RANDDOUBLE()>.5?1:-1));}}
bb=bb*bb.transpose();
//cout <<bb;
nn=0;
for(int i=0; i<n; ++i) for(int j=i; j<n; ++j) if(abs(bb(i,j))>1e-16 || abs(bb(j,i))>1e-16 ) ++nn;
cout <<"Original filling = "<<nn*2./n/(n+1)<<endl;
NRMat<double> b(bb);
cholesky(b,0);
//cout <<b;
cout << "Error = "<<(b*b.transpose()-bb).norm()<<endl;
nn=0;
for(int i=0; i<n; ++i) for(int j=i; j<n; ++j) if(abs(b(i,j))>1e-16 || abs(b(j,i))>1e-16 ) ++nn;
cout <<"Cholesky factor filling = "<<nn*2./n/(n+1)<<endl;
}


if(0)
{
int n;
cin >>n;
NRMat<double> a(n,n);
a.randomize(1);
for(int i=0; i<n; ++i) for(int j=0; j<i; ++j) {a(i,j)=0.;}
cout <<a;
NRMat<double> bb=a.transpose()*a;
SparseSMat<double> cc(bb);
NRMat<double> b(bb);
cholesky(b,0);
cout <<b;
SparseSMat<double> c;
c= cc.cholesky();
cout <<c;
}


if(0)
{
int n;
cin >>n;
NRMat<complex<double> > a(n,n);
a.randomize(1);
for(int i=0; i<n; ++i) for(int j=0; j<i; ++j) {a(i,j)=0.;}
for(int i=0; i<n; ++i) {a(i,i).imag(0.); if(a(i,i).real()<0) a(i,i).real(-10.*a(i,i).real()); else a(i,i).real(10.*a(i,i).real());}
if(n<100)cout <<a;
NRMat<complex<double> > bb=a.transpose(true)*a;
SparseSMat<complex<double> > cc(bb);
if(n<100)cout <<"Input matrix\n"<<bb;
NRMat<complex<double> > b(bb);
//cholesky(b,0);
//if(n<100)cout <<"Dense Cholesky result\n"<<b;
SparseSMat<complex<double> > c;
c= cc.cholesky();
NRMat<complex<double> > cx(c);
if(n<100)cout <<"Sparse pivoted Cholesky result \n"<<cx;
if(n<100)cout <<"result of reconstruction\n"<<cx.transpose(true)*cx<<endl;
cout <<"Error = "<<(cx.transpose(true)*cx -bb).norm()<<endl;
}


if(0)
{
int n;
cin >>n;
SparseSMat<double> bh(n,n);
for(int i=0; i<=n/400; ++i) for(int j=i; j<n; ++j) {if(RANDDOUBLE()>0.995 || i==j) 
	{bh.add(i,j,RANDDOUBLE()*(i==j?10.:(RANDDOUBLE()>.5?1:-1)),false);}}
if(n<1000) cout <<"Random matrix\n"<<bh;
SparseSMat<double> bb(n,n);
bb.gemm(0.,bh,'c',bh,'n',1.);
if(n<1000) cout <<"Input matrix\n"<<bb;
cout <<"Original filling = "<<bb.simplify()<<endl;
SparseSMat<double> b = bb.cholesky();
if(n<1000) cout <<"Cholesky result\n"<<b;
SparseSMat<double> br(n,n);
br.gemm(0.,b,'c',b,'n',1.);
if(n<1000) cout <<"Result of reconstruction\n"<<br;
if(n<1000) cout <<"Difference\n"<<br-bb;
cout << "Error = "<<(br-bb).norm()<<endl;
cout <<"Cholesky factor filling = "<<b.simplify()<<endl;
}


if(0)
{
int n;
cin >>n;
NRMat<double> a(n,n),b(n,n);
a.randomize(1.);
b.randomize(1.);
NRMat<double>c;
double t=clock()/((double) (CLOCKS_PER_SEC));
int rep=1+10000000000LL/n/n/n;
cout <<"Repetitions " <<rep<<endl;
for(int i=0; i<rep; ++i) c=a*b;
cout <<"CPU time (ms) "<<(clock()/((double) (CLOCKS_PER_SEC))-t)*1000./rep <<"\n";
NRMat<double>cgpu;
a.moveto(gpu1);
b.moveto(gpu1);
 t=clock()/((double) (CLOCKS_PER_SEC));
for(int i=0; i<rep; ++i) cgpu=a*b;
cout <<"GPU time (ms) "<<(clock()/((double) (CLOCKS_PER_SEC))-t)*1000./rep <<"\n";
cgpu.moveto(cpu);
cout << "Error = "<<(c-cgpu).norm()<<endl;
}

if(0)
{
int n;
cin >>n;
NRMat<double> a(n,n);
a.randomize(1.);
NRVec<double> v(n);
v.randomize(1.);
NRSMat<double> as(n);
as.randomize(1.);
NRVec<double>w = a*v;
NRVec<double>ws = as*v;
NRMat<double>c(n,n);
c=exp(a);
a.moveto(gpu1);
v.moveto(gpu1);
as.moveto(gpu1);
NRVec<double>wgpu = a*v;
NRVec<double>wsgpu = as*v;
w.moveto(gpu1);
ws.moveto(gpu1);
cout << "Error gemv = "<<(w-wgpu).norm()<<endl;
cout << "Error symv = "<<(ws-wsgpu).norm()<<endl;
NRMat<double>cgpu;
cgpu=exp(a);
c.moveto(gpu1);
cout << "Error = "<<(c-cgpu).norm()<<endl;
}

/*
if(0)
{
CSRMat<double> h0;
cin>>h0;
cout <<"matrix read\n"; cout.flush();
CSRMat<double> h1 = h0; 
CSRMat<complex<double> > h = imagmatrix(h1);
double t=clock()/((double) (CLOCKS_PER_SEC));
CSRMat<complex<double> > r = h*h;
cout <<"CSRMat mult time "<<clock()/((double) (CLOCKS_PER_SEC))-t <<"\n";
cout.flush();
t=clock()/((double) (CLOCKS_PER_SEC));
r = exp(h);
cout <<"CSRMat exp time "<<clock()/((double) (CLOCKS_PER_SEC))-t <<"\n";
cout.flush();
if(h.nrows()<=1024)
{
NRMat<complex<double> > h2(h);
NRMat<complex<double> >r2 = exp(h2);
cout <<"error = "<<(r2-NRMat<complex<double> >(r)).norm()<<endl;
}
}
*/


if(0)
{
NRMat<complex<double> > m;
ifstream f("libormat");
f >> m;
f.close();

NRVec<double> eivals(m.nrows());
NRMat<complex<double> > m_aux = m;
NRMat<complex<double> > m_test = m;

cout << "hermiticity error = " <<(m_aux.transpose(true) - m_aux).norm()<<endl; 
diagonalize(m,eivals);
cout << "eivals "<<eivals;
cout <<"eivecs "<<m<<endl;
NRVec<complex<double> > eivalsc(eivals);

NRMat<complex<double> > m4 =  m.transpose(true);

cout <<"unitarity error "<< (m4*m).norm(true)<<endl;

NRMat<complex<double> > m5(m.nrows(),m.nrows());
for(int i=0; i<m.nrows(); ++i)  for(int j=0; j<m.nrows(); ++j)
	m5(i,j) = complex<double>(m(j,i).real(), - m(j,i).imag());

cout << "conjugatetest "<<(m4-m5).norm()<<endl;

NRMat<complex<double> > m1= m_aux * m;
NRMat<complex<double> > m1x = m; m1x.diagmultr(eivalsc);
cout << "test m1 m1x "<<(m1-m1x).norm()<<endl;
NRMat<complex<double> > m2=  m.transpose(true) * m1;
//NRMat<complex<double> > m2b=  m * m_aux * m.transpose(true);
cout <<"check "<<m2<<endl;
//cout <<"checkb "<<m2b<<endl;
double err=0.;
for(int i=0; i<m.nrows(); ++i) for(int j=0; j<m.nrows(); ++j)
	if(i!=j) err += abs(m2(i,j));
cout <<"offdiagonality error = "<<err<<endl;

err=0; 
for(int i=0; i<m.nrows(); ++i) err += abs(m2(i,i) - eivals[i]);
cout <<"eigenvalue error = "<<err<<endl;


//test as general matrix
NRVec<complex<double> > ww(m.nrows());
NRMat<complex<double> > vl(m.nrows(),m.nrows()), vr(m.nrows(),m.nrows());
gdiagonalize(m_test,ww,&vl,&vr);
cout << "eivals "<<ww<<endl;
cout << "eivecs "<<vl<<vr<<endl;
NRMat<complex<double> > m3= vl.transpose(true)* m_aux *vr;
cout <<"check2 "<<m3;
 err=0.;
for(int i=0; i<m.nrows(); ++i) for(int j=0; j<m.nrows(); ++j)
        if(i!=j) err += abs(m3(i,j));
cout <<"offdiagonality error 2 = "<<err<<endl;

err=0;
for(int i=0; i<m.nrows(); ++i) err += abs(m3(i,i) - ww[i]);
cout <<"eigenvalue error = "<<err<<endl;



}

if(0)
{
NRMat<double> a;
cin >>a ;
double det=determinant_destroy(a);
cout << "det= "<<det<<endl;
}

if(0)
{
bitvector v(3);
v.assign(0,0);
v.assign(1,1);
v.assign(2,0);
cin >>v;
cout <<v;
}

if(0)
{
Quaternion<double> v,r,tmp,q,q1;
cin >>q;
q.normalize();
q1= q;
q1.conjugateme();
cout <<"test "<<q*q1<<endl;
cin >>v;
r= q*v*q1;
cout <<"rotated "<<r<<endl;
}

if(0)
{
Quaternion<double> q(1.);
Quaternion<double> p,r(q);
p=r;
q+=p;
q-=r;
r=p+q;
r+=Quaternion<double>(0,8,8,8);
r+= 2.;
r/=4.;
cout<<r<<endl;
Quaternion<double> s(-1,2,1,2);
Quaternion<double> t=r*s.conjugate();
cout <<t<<" "<<t[0]<<" "<<t.norm()<<endl;
Quaternion<double>tt = t.inverse();
cout <<tt<<" "<<t*tt<<endl;
r=s/t;
r.normalize();
cout<<r<<endl;
Mat3<double> rotmat;
quat2rotmat(r,rotmat);
cout << rotmat[0][1]<<endl;
r.normalize(NULL,true);
NRMat<double> rotmat2(3,3),rotmat3(3,3);
Quaternion<NRMat<double> > rotmatder;
rotmatder[0].resize(3,3);
rotmatder[1].resize(3,3);
rotmatder[2].resize(3,3);
rotmatder[3].resize(3,3);
normquat2rotmatder(r,rotmatder);
cout << rotmatder;
normquat2rotmat(r,rotmat2);
normquat2rotmat(-r,rotmat3);
cout << rotmat2<<endl;
cout << rotmat3<<endl;
Quaternion<double> rr;
rotmat2normquat(rotmat2,rr);
cout <<"orig "<<r<<endl;
cout <<"reconstruct "<<rr<<endl;
cout <<"diff " <<r-rr<<endl;
Vec3<double> eul;
r.normalize(NULL,true);
r.normquat2eulerzyx(eul);
cout<< "euler "<<eul[0]<<" " <<eul[1]<<" "<<eul[2]<<endl;
Quaternion<double> rback;
rback.eulerzyx2normquat(eul);
cout <<"q from euler back "<<endl<<r<<endl<<rback<<endl;
Mat3<double> rm;
euler2rotmat((double *)eul,rm,"xyz",false,false,false);
cout <<rm;
Vec3<double> eulback;
rotmat2euler((double *)eulback,rm,"xyz",false,false,false);
cout <<"euler back from rm error "<<eulback-eul<<endl;
Vec3<double> axis={0,1/sqrt(2),1/sqrt(2)};
Quaternion<double> rrr;
rrr.axis2normquat(axis,0.5);
NRVec<double> axis2(3);
double angle;
rrr.normquat2axis(&axis2[0],angle);
cout <<"back angle "<<angle<<" "<<axis2<<endl;
Vec3<double> vec={1,2,3};
Quaternion<double> qvec(vec);
Quaternion<double>  rvec = qvec.rotateby(rrr);
Quaternion<double>  rvec2 = qvec.rotateby(rrr.conjugate());
cout <<rvec<<endl;
cout <<rvec2<<endl;
Quaternion<double> rrvec = rvec.rotateby(rrr.conjugate());
cout <<rrvec<<endl;
Vec3<double>  rotvec;
rrr.rotate(rotvec,vec);
Quaternion<double> qq={1.5,2,-2,-1.123};
cout << " test "<<qq*qq<<endl;
cout <<"exp " <<exp(qq)<< " "<< log(exp(qq))-qq<<endl;
cout <<"log " <<log(qq)<<" "<< exp(log(qq))-qq<<endl;
cout <<"pow " <<pow(qq,0.5)<<" " <<pow(qq,0.5)*pow(qq,0.5)-qq<<endl;
Quaternion<double> qq2={-.5,1,2,3.23};
Mat3<double> m(1,2,3,4,5,6,7,8,-9);
NRMat<double> mm(m,3,3);
Vec3<double> v(10,20,30);
NRVec<double> vv(v,3);
cout << m*v<<mm*vv<<endl;
cout << v*m<<vv*mm<<endl;
Mat3<double> m2 = m.inverse();
cout<<m2<<m*m2<<m2*m<<endl;
NRMat<double> mm2(3,3); mm2=1.;
multiply_by_inverse(mm2,mm);
cout <<mm<<endl;
qq.normalize(NULL,true);
Vec3<double> xeul,yeul;
qq.normquat2euler((double *)xeul,"zyx");
qq.normquat2eulerzyx((double *)yeul);
cout <<xeul<< " "<<yeul<<endl;
Quaternion<double> xqq;
xqq.euler2normquat((double *)xeul,"zyx");
cout <<"normquat2euler test "<<endl<<qq<<endl<<xqq<<endl<<qq-xqq<<endl;
}


if(0)
{
NRVec<double> a,b,c;
cin >>a>>b;
c=a+b;
cout<<c;
}

if(0)
{
NRPerm<int> p;
cin >>p;
int n=p.size();
NRVec_from1<double> v(n);
int i;
for(i=1; i<=n; ++i) v[i]=10.*i;
cout <<v.permuted(p);
}

if(0)
{
int n;
CyclePerm<int> c;
cin>>c >>n;
c.simplify();
cout<<c<<endl;
NRPerm<int> p(c);
cout <<p;
CyclePerm<int> cc(p);
cout <<cc<<endl;
NRPerm<int> pp(cc);
cout <<(p == pp)<<endl;
cout <<(c == cc)<<endl;
CyclePerm<int> cc1,cc2;
cc1=cc.pow_simple(n);
cc2=cc.pow(n);
cout<<cc<<endl;
cout<<cc2<<endl;
cout <<(cc1 == cc2)<<endl;
}

if(0)
{
int seed;
int f=open("/dev/random",O_RDONLY);
if(sizeof(int)!=read(f,&seed,sizeof(int))) laerror("cannot read /dev/random");
close(f);
srand(seed);
int n;
cin >>n;
NRPerm<int> p(n);
p.randomize();
cout <<p;
CyclePerm<int> cc(p);
cout <<cc<<endl;
NRPerm<int> pp(cc,n);
cout <<pp;
if(pp!=p) laerror("inconsistency");
NRVec<double> v(n);
for(int i=0; i<n; ++i) v[i]=10.*(i+1);
NRVec<double> vv(v);
v.permuteme(cc);
cout <<v;
NRVec<double>  vvv= vv.permuted(pp);
cout<<vvv;
cout<<"error "<<(v-vvv).norm()<<endl;
}

if(0)
{
int seed;
int f=open("/dev/random",O_RDONLY);
if(sizeof(int)!=read(f,&seed,sizeof(int))) laerror("cannot read /dev/random");
close(f);
srand(seed);
int n;
cin >>n;
NRVec<double> v(n);
v.randomize(1.);
NRVec<double> vv(v);
NRPerm<int> p(n);
vv.sort(0,p);
NRVec<double> vvv=v.permuted(p,true);
NRVec<double> v4=vv.permuted(p,false);
cout<<v<<vv;
cout<<vvv<<v4<<p;
cout <<"error "<<(vv-vvv).norm() <<" "<<(v-v4).norm()<<endl;
}

if(0)
{
int n;
cin >>n;
NRPerm<int> p(n);
p.identity();
do{
cout <<p;
}while(p.next());
}

if(0)
{
int n;
cin >>n;
NRPerm<int> p(n);
int tot=p.generate_all_lex(printme);
cout <<"generated "<<tot<<endl;
}

if(0)
{
int n;
cin >>n;
NRPerm<int> p(n);
int seed;
int f=open("/dev/random",O_RDONLY);
if(sizeof(int)!=read(f,&seed,sizeof(int))) laerror("cannot read /dev/random");
close(f);
srand(seed);
for(int i=1; i<=n; ++i) p[i]=1+RANDINT32()%(n-1);
cout <<"Initial = "<<p<<endl;
int tot=p.generate_all_multi(printme0);
cout <<"generated "<<tot<<endl;
}

if(0)
{
int n; cin >>n;
NRPerm<int> p(n); 
NRVec_from1<int> classes; cin>>classes;
if(classes.size()!=p.size()) laerror("sizes do not match");
PermutationAlgebra<int,int> tot = p.list_restricted(classes,-2);
tot.copyonwrite();
cout <<"generated " <<tot.size()<<"\n"<<tot<<endl;
PermutationAlgebra<int,int> all= p.list_all();
all.sort();
cout <<"ALL\n"<<all;
}


if(0)
{
int n;;
cin >>n >>unitary_n;
Sn_characters<int> Sn(n);
cout <<Sn;
if(!Sn.is_valid()) laerror("internal error in Sn character calculation");

Snmtable = Multable(n);
cout <<"Multiplication table = "<<Snmtable<<endl;

cout <<"allyoung.resize "<<Sn.sumirrepdims()<<endl;
allyoung.resize(Sn.sumirrepdims());
allyoungmat.resize(Sn.sumirrepdims());
allyoungregular.resize(Sn.sumirrepdims());
allyoung_irrep.resize(Sn.sumirrepdims());
allyoung_index=0;


Partition<int> p(n);
space_dim=0;
int tot=p.generate_all(pprintme,0);
cout <<"partitions generated "<<tot<<endl;
if(tot!=partitions(n)) laerror("internal error in partition generation or enumerations");
if(space_dim!=longpow(unitary_n,n)) {cout<<space_dim<<" "<<ipow(unitary_n,n)<<endl;laerror("integer overflow or internal error in space dimensions");}

for(int i=0; i<allyoung.size(); ++i)
	{
	for(int j=0; j<allyoung.size(); ++j)
		{
		//	cout <<"Young "<<i<<" "<<allyoung[i]<<endl;
		//	cout <<"Young "<<j<<" "<<allyoung[j]<<endl;
		PermutationAlgebra<int,int> r=allyoung[i]*allyoung[j];
		NRMat<int> rm(r,false,n);
		NRMat<int> rm2 = allyoungmat[i]*allyoungmat[j];
		NRMat<int> rreg=RegularRepresentation(r,Snmtable);
		NRMat<int> rreg2=allyoungregular[i]*allyoungregular[j]; 
		cout <<"Product of Young "<<i<<" and "<<j<<" = "<<r<<endl;
		//cout<<"matrix "<<rm<<endl;
		if(i!=j && !r.is_zero()) cout <<"NONORTHOGONAL Young operators found "<<i<< " "<<j<<" (irreps "<<allyoung_irrep[i]<<" "<<allyoung_irrep[j]<<")\n";
		if(rreg!=rreg2) 
			{
			cout <<"Representation of product = "<<rreg;
			cout <<"Product of representations = "<<rreg2;
			laerror("internal error in multiplication of permutationalgebra");
			}
		if(rm!=rm2) laerror("internal error in matrix representation of permutationalgebra");
		if(allyoung_irrep[i]!=allyoung_irrep[j] && !r.is_zero()) laerror("internal error in PermutationAlgebra");
		}
	}
}



if(0)
{
int n;
cin >>n ;
Sn_characters<int> Sn(n);
cout <<Sn;
if(!Sn.is_valid()) laerror("internal error in Sn character calculation");
Polynomial<int> p(1);
p[0]=p[1]=1;
CycleIndex<int> c(Sn);
PERM_RANK_TYPE d;
Polynomial<int> pp=c.substitute(p,&d);
for(int i=0; i<p.degree(); ++i) if(d!=pp[i]) laerror("error in cycle index");
}

if(0)
{
int n,m;
double x;
cin >>n >>m >>x;
if(n<m) {int t=n; n=m; m=t;}
Polynomial<double> p(n),q(m);
p.randomize(1.);
q.randomize(1.);
NRVec<double> qq(q);
Polynomial<double> qqq(qq);
Polynomial<double> pp(n); pp.randomize(1.); 
p*=10.;
Polynomial<double> a,b;
p.polydiv(q,a,b);
Polynomial<double> r=p*q;
Polynomial<double> z=value(p,q); //p(q(x))
Polynomial<double> y=value(q,p); 
cout <<p;
cout <<q;
cout <<q.companion();
cout<<Sylvester(p,q);
cout <<a;
cout <<b;
cout <<r;
cout <<z;
cout << "polydiv test "<<(a*q+b -p).norm()<<endl;
cout << value(p,x)*value(q,x) -value(r,x)<<endl;
cout << value(p,value(q,x)) -value(z,x)<<endl;
cout << value(q,value(p,x)) -value(y,x)<<endl;
NRMat<double> u(5,5);
u.randomize(1.);
cout << (value(p,u)*value(q,u) -value(r,u)).norm()<<endl;

}

if(0)
{
int n;
cin >>n ;
NRVec<double> r(n);
//r.randomize(1.);
//wilkinson's ill-conditionel polynomial
for(int i=0; i<n;++i) r[i]=i+1;
double x0=r[0]*0.8+r[1]*0.2;
r.sort(0);
Polynomial<double> p=polyfromroots(r);
cout <<p;
cout <<r;
NRVec<double> rr= p.realroots(1e-10);
rr.resize(n,true);
cout <<rr;
cout <<"root error = "<<(r-rr).norm()<<endl;
double x=p.newton(x0);
double xdif=1e10;
for(int i=0; i<n; ++i) if(abs(x-r[i])<xdif) xdif=abs(x-r[i]);
cout<<"test newton "<<xdif<<endl;
}

if(0)
{
int n;
cin >>n ;
NRVec<double> x(n+1),y(n+1);
x.randomize(1.);
y.randomize(1.);
Polynomial<double> p=lagrange_interpolation(x,y);
cout <<x<<y<<p;
NRVec<double> yy=values(p,x);
cout<<"interpolation error= "<<(y-yy).norm()<<endl;
Polynomial<double>q=p.integral(2);
Polynomial<double>pp=q.derivative(2);
cout<<"test deriv. "<<(pp-p).norm()<<endl;
}

if(0)
{
int n;
cin >>n;
NRVec<double> rr(n);
rr.randomize(1.);
rr.sort(0);
if(rr.size()>2) rr[1]=rr[0];//make a degenerate root
NRVec<double> pr(2*n);
NRVec<double> qr(2*n);
pr.randomize(1.);
qr.randomize(1.);
for(int i=0; i<n; ++i) {pr[i]=qr[i]=rr[i];}
Polynomial<double> p=polyfromroots(pr);
Polynomial<double> q=polyfromroots(qr);
Polynomial<double> g=poly_gcd(p,q,1e-8);
cout <<"GCD ="<<g;
Polynomial<double> gg=svd_gcd(p,q,1e-13);
cout <<"SVDGCD ="<<gg;
NRVec<double> rrr=g.realroots(1e-5);
NRVec<double> rrrr=gg.realroots(1e-5);
cout <<rr<<rrr<<rrrr;
cout <<"test gcd "<<(rr-rrr).norm()<<endl;
cout <<"test svdgcd "<<(rr-rrrr).norm()<<endl;
}

if(0)
{
int n;
cin >>n;
Polynomial<double> p;
p.binomial(n);
Polynomial<double> q(1);
q[0]=q[1]=1.;
Polynomial<double> qq=q.pow(n);
cout <<"test binom "<<(p-qq).norm()<<endl;
}

if(0)
{
int n;
cin >>n ;
Polynomial<double> p=hermite_polynomial<double>(n);
cout <<"Hermite = "<<p<<std::endl;
cout <<"non-zero = "<< ((n&1)?p.odd_powers():p.even_powers())<<std::endl;
cout <<"value = "<<value(p,1.23456789)<<" "<<odd_value(p,1.23456789)+even_value(p,1.23456789)<<endl;
}

if(0)
{
NRVec<int> v({1,2,3,4});
cout <<v;
NRVec_from1<int> vv({1,2});
cout <<vv;
v.resize(0);
cout<<v;
NRMat<int> m({{1,2,3},{4,5,6}});
cout<<m;
NRMat_from1<int> mm({{1,2,3},{4,5,6}});
cout<<mm;
Vec3<double> x({1,2,3});
cout<<x<<endl;
Mat3<double> y({{1,2,3},{4,5,6},{7,8,9}});
cout <<y<<endl;
Quaternion<double> q({1,2,3,4});
cout<<q<<endl;
NRPerm<int> p({1,4,2,3});
cout <<p<<endl;
CyclePerm<int> c({NRVec_from1({1,2}),NRVec_from1({3,4,5})});
cout <<c;
Polynomial<double> pp({1,2,3,4,5});
cout<<pp;
}

if(0)
{
//prepare random symmetric mat3
int seed;
int f=open("/dev/random",O_RDONLY);
if(sizeof(int)!=read(f,&seed,sizeof(int))) laerror("cannot read /dev/random");
close(f);
srand(seed);

double scale;
cin >> scale;

NRMat<double> tmp(3,3);
tmp.randomize(scale);
Mat3<double> mm(tmp);
mm.symmetrize();
mm+= 1.;
NRMat<double> m(&mm[0][0],3,3);
cout <<m<<"3 3\n"<<mm<<endl;

NRVec<double> w(3);
diagonalize(m,w);
cout << w<<m;

Vec3<double> ww;
Mat3<double> vv;
mm.eivec_sym(ww,vv);
cout <<"3\n"<<ww<<"\n3 3\n"<<vv<<endl;

NRVec<double> www(&ww[0],3);
NRMat<double> vvv(&vv[0][0],3,3);
cout<<"eival error = "<<(w-www).norm()<<endl;
cout<<"eivec error = "<<(m.diffabs(vvv)).norm()<<endl; //just ignore signs due to arb. phases (not full check)
}

if(0)
{
//prepare random  mat3
int seed;
int f=open("/dev/random",O_RDONLY);
if(sizeof(int)!=read(f,&seed,sizeof(int))) laerror("cannot read /dev/random");
close(f);
srand(seed);

NRMat<double> tmp(3,3);
tmp.randomize(2.);
Mat3<double> mm(tmp);
NRMat<double> m(&mm[0][0],3,3);
cout <<m<<"3 3\n"<<mm<<endl;

double rr[2][3]={{1,2,3},{4,5,6}};
NRMat r(rr);
cout<<r;

double d;
linear_solve(m,&r,&d);

Mat3<double> mmi=mm.inverse();
double dd=simple_gaussj(mm.elements(),rr);
cout <<"det="<<dd<<endl;
cout <<"error of inverse = "<<(mmi-mm).norm()<<endl;
cout <<"3 3\n"<<mm<<NRMat<double>(rr);

cout<<"linear solve det="<<d<<endl;
cout <<r;
cout <<"det error="<<d-dd<<endl;
cout <<"solution error="<<(r-rr).norm()<<endl;
}

if(0)
{
//prepare random  mat3
int seed;
int f=open("/dev/random",O_RDONLY);
if(sizeof(int)!=read(f,&seed,sizeof(int))) laerror("cannot read /dev/random");
close(f);
srand(seed);

simple_linfit<double,3> fit;
double funcs[3];
for(int i=0; i<100; ++i) 
	{
	double x = 10*RANDDOUBLESIGNED();
	funcs[0]=1;
	funcs[1]=x;
	funcs[2]=x*x;
	double y = 2*funcs[2] -3*funcs[1] + funcs[0];
	//cout <<"test "<<x<<" "<<y<<endl;
	fit.input(funcs,y);
	}
double det = fit.solve();
cout <<"det= "<<det<<" fit "<<fit<<endl;
}

if(0)
{
NRMat<double> m;
cin >>m;
int n=m.nrows();
int seed;
int f=open("/dev/random",O_RDONLY);
if(sizeof(int)!=read(f,&seed,sizeof(int))) laerror("cannot read /dev/random");
close(f);
srand(seed);
NRPerm<int> p(n);
p.randomize();
cout <<p;
NRMat<double> mm=m.permuted_rows(p);
cout <<mm;
}

if(0)
{
Rational<int> p,q;
cin>>p>>q;
cout <<p+q<<endl;
cout <<p-q<<endl;
cout <<p*q<<endl;
cout <<p/q<<endl;
}

if(0)
{
double x;
cin >>x;
ContFrac<long long> xx(x,20,1000000);
cout<<xx;
double y=xx.value();
cout <<y<<endl;
cout << "CF roundoff error = "<<x-y<<endl;
}

if(0)
{
int p,q;
cin>>p>>q;
ContFrac<int> xx(p,q);
double z=p; z/=q;
ContFrac<int> zz(z,20,100000);
ContFrac<int> yy=xx.reciprocal();
cout<<xx;
cout<<yy;
cout<<zz;
int pp,qq,rr,ss;
xx.convergent(&pp,&qq);
yy.convergent(&rr,&ss);
cout << pp<<" "<<qq<<endl;
cout << rr<<" "<<ss<<endl;
if(p!=pp ||q!=qq||qq!=rr||pp!=ss) cout<<"ContFrac error\n";
double zzz=zz.value();
cout <<z<<" "<<zzz<<endl;
}

if(0)
{
Rational<int> r({11,101});
ContFrac<int> x(r);
ContFrac<int> y= x+Rational<int>({2,3});
cout<<Rational<int>(y)<<endl;
ContFrac<int> z= x*Rational<int>({2,3});
cout<<Rational<int>(z)<<endl;
}

if(0)
{
ContFrac<int> x(11,101);
ContFrac<int> v(3,7);
ContFrac<int> y= x+v;
cout<<Rational<int>(y)<<endl;
ContFrac<int> z= x*v;
cout<<Rational<int>(z)<<endl;
BiHomographic<int> h({{{12,4},{3,1}},{{0,1},{-1,0}}});
ContFrac<int> zz=h.value(x,v);
cout<<Rational<int>(zz)<<endl;
cout<<(Rational<int>(x)+3)*(Rational<int>(v)+4)/(Rational<int>(x)-Rational<int>(v))<<endl;
}

if(0)
{
double x;
cin >>x;
ContFrac<int> xx(x,15,100000);
cout <<xx<<endl;
ContFrac<int> xm= -xx;
cout<< xm<<endl;
cout<<xx+xm<<endl;
}

if(0)
{
Rational<int> x;
cin >>x;
ContFrac<int> xx(x);
cout<<xx;
ContFrac<int> xm= -xx;
cout<< xm;
ContFrac<int> yy(-x);
cout<<yy;
ContFrac<int> ym= -yy;
cout<< ym;
cout <<"ZEROs\n"<<xx+xm<<" "<<yy+ym<<" "<<xx+yy<<" "<<xm+ym<<endl;
}

if(0)
{
double x;
cin >>x;
ContFrac<int> xx(x,25,100000);
ContFrac<int> xx1(x+1e-4,25,100000);
ContFrac<int> xx2(x-1e-4,25,100000);
xx.canonicalize();
xx1.canonicalize();
xx2.canonicalize();
cout<<"small "<<xx2<<endl;
cout<<"middle "<<xx<<endl;
cout<<"big "<<xx1<<endl;
cout << "TEST "<<(xx<xx1) <<" "<<(xx>xx2) <<endl;
}

if(0)
{
int seed;
int f=open("/dev/random",O_RDONLY);
if(sizeof(int)!=read(f,&seed,sizeof(int))) laerror("cannot read /dev/random");
close(f);
srand(seed);

double t=RANDINT32();
t/=(1L<<30);
t-=1;
cout <<"theta = "<<t<<endl;
NRMat<complex<double> > sz({{1,0},{0,-1}});
NRMat<complex<double> > szsz=sz.otimes(sz);
NRMat<complex<double> > c1not0({
			{1,0,0,0},
			{0,1,0,0},
			{0,0,0,1},
			{0,0,1,0},
			});
NRMat<complex<double> > itszsz= szsz * complex<double>(0,t);
NRMat<complex<double> > expitszsz = exp(itszsz);
cout <<expitszsz;
NRMat<complex<double> > id({{1,0},{0,1}});
NRMat<complex<double> > rz(2,2); rz.clear();
double c=cos(t);
double s=sin(t);
rz(0,0) = complex<double>(c,s);
rz(1,1) = complex<double>(c,-s);
NRMat<complex<double> >  idrz = id.otimes(rz);
NRMat<complex<double> > test = c1not0 * idrz *c1not0;
cout <<test;
cout <<"Error = "<<(expitszsz-test).norm()<<endl;
}

if(0)
{
NRVec<double> x({1,2,3});
NRVec<double> y({4,5,6});
//cout <<x.concat(y);
x.append(10.);
x.prepend(11.);
cout <<x;
x.concatme(y);
cout <<x;
}

if(0)
{
double x;
cin>>x;
cout <<mantissa(x,20)<<endl;
}

if(0)
{
double  t,t2;
int s;
cin>>t>>s;
bitvector m = mantissa(t,64,s);
cout <<m<<std::endl;
bitvector_decimal(t2,m,-s);
cout <<t2<<endl;
}

if(0)
{
int n;
cin >>n;
NRMat<std::complex<double> > a(n,n);
a.randomize(1);
NRMat<std::complex<double> > ah=a.transpose(true);
NRMat<std::complex<double> > b=a-ah;
//general non-symmetric unitary matrix
NRMat<std::complex<double> > u=exp(b);
NRMat<std::complex<double> > bsym(b),banti(b);
bsym.copyonwrite();
banti.copyonwrite();
for(int i=0; i<n; ++i) for(int j=0; j<n; ++j)
	{
	bsym(i,j).real(0);
	banti(i,j).imag(0);
	}
NRMat<std::complex<double> > usym=exp(bsym);
NRMat<std::complex<double> > uanti=exp(banti);
cout <<b<<u;
cout<<bsym<<usym;
//usym is unitary symmetric complex
cout<<banti<<uanti;
//uanti is unitary real
NRMat<std::complex<double> > uh=u.transpose(true);
NRMat<std::complex<double> > usymh=usym.transpose(true);
NRMat<std::complex<double> > uantih=uanti.transpose(true);
cout <<"error of unitarity of u*uh = "<<(u*uh).norm(1.)<<endl;
cout <<"error of unitarity of usym*usymh = "<<(usym*usymh).norm(1.)<<endl;
cout <<"error of unitarity of uanti*uantih = "<<(uanti*uantih).norm(1.)<<endl;
cout <<"expected nonzero norm of commutator of sym and anti parts = "<<commutator(bsym,banti).norm()<<endl;
cout <<"expected nonzero norm of difference between u and usym*uanti = "<<(u-usym*uanti).norm()<<endl;
NRSMat<double> hs(n);
hs.randomize(1);
NRMat<double> h(hs);
NRMat<std::complex<double> > ih=imagmatrix(h);
NRMat<std::complex<double> > c=commutator(bsym,ih);
//cout <<"commutator of two sym parts = "<<c<<endl; //is antisymmetric
NRMat<std::complex<double> > uc=exp(c);
NRMat<std::complex<double> > usym2=exp(ih);
cout <<"error of unitarity of uc = "<<(uc*uc.transpose(true)).norm(1.)<<endl;
cout <<"error of unitarity of usym2 = "<<(usym2*usym2.transpose(true)).norm(1.)<<endl;
NRMat<std::complex<double> > up = exp(bsym+ih);
cout <<"error of unitarity of up = "<<(up*up.transpose(true)).norm(1.)<<endl;
cout <<"expected nonzero norm of difference between up and usym*usym2 = "<<(up-usym*usym2).norm()<<endl;
NRMat<std::complex<double> > ubch1 = exp(bsym+ih+c*0.5);
cout <<"error of unitarity of ubch1 = "<<(ubch1*ubch1.transpose(true)).norm(1.)<<endl;
cout <<"expected nonzero norm of difference between ubch1 and usym*usym2 = "<<(ubch1-usym*usym2).norm()<<endl;
}

if(0)
{
int n;
cin >>n;
NRSMat<double> hs(n);
hs.randomize(1);
NRMat<double> h(hs);
//since H is real, the unitary U will be symmetric and its complex conjugation thus equal to hermitian conjugation
NRMat<std::complex<double> > ih=imagmatrix(h);
NRMat<std::complex<double> > u=expi(h);
NRMat<std::complex<double> > uh = u.transpose(true);
NRMat<std::complex<double> > uc = u.conjugate();
NRMat<std::complex<double> > ut = u.transpose(false);
cout <<"error of symmetry of exp(iH) = "<<(u-ut).norm()<<endl;
NRMat<std::complex<double> > uct = uc.transpose(false);
cout <<"error of uct - uh = "<<(uct-uh).norm()<<endl;
//cout <<u << uc<<uh;
cout <<"nonunitarity of u*uh = "<<(u*uh).norm(1.)<<endl;
cout <<"nonunitarity of u*uc = "<<(u*uc).norm(1.)<<endl;
NRMat<std::complex<double> > uu=exp(ih);
NRMat<std::complex<double> > uuh = uu.transpose(true);
cout <<"nonunitarity of uu = "<<(uu*uuh).norm(1.)<<endl;
cout <<"error of complex exponential = "<<(u-uu).norm()<<endl;
NRMat<double> d(h);
NRVec<double> e(n);
diagonalize(d,e);

//now try similarity transformed H
NRMat<double> a(n,n);
a.randomize(1);
NRMat<double> ainv = calcinverse(a);
NRMat<double> atr = a.transpose();
NRMat<double> ainvtr = ainv.transpose();
cout <<"error of inverse = "<<(a*ainv).norm(1.)<<endl;
NRMat<double> hbar = ainv * h * a;
NRVec<double> er(n),ei(n);
NRMat<double> hwork(hbar);
gdiagonalize(hwork,er,ei);
er.sort();
cout <<"errors of isospectral eigenvalues = "<<(er-e).norm()<<" "<<ei.norm()<<endl;
NRMat<std::complex<double> > uubar=expi(hbar);
NRMat<std::complex<double> > ubar = NRMat<std::complex<double> >(ainv) * u * NRMat<std::complex<double> >(a);
cout <<"error of similarity transformed exponential = "<<(ubar-uubar).norm()<<endl;
NRMat<std::complex<double> > ubarh = ubar.transpose(true);
NRMat<std::complex<double> > ubarh2 =  NRMat<std::complex<double> >(atr) * u.transpose(true) * NRMat<std::complex<double> >(ainvtr);
cout <<"error of ubar hermit conjugation = "<<(ubarh-ubarh2).norm()<<endl;
NRMat<std::complex<double> > uubarh = uubar.transpose(true);
//cout <<"H = "<<h;
//cout <<"Hbar = "<<hbar;
//cout <<"U = "<<u;
//cout <<"Ubar = "<<ubar;
NRMat<std::complex<double> > ubartest2=ubar*ubarh;
cout <<"expected nonunitarity of ubar = "<<ubartest2.norm(1.)<<endl;
NRMat<std::complex<double> > uubartest=uubarh*uubar;
cout <<"expected nonunitarity of uubar = "<<uubartest.norm(1.)<<endl;
NRMat<std::complex<double> > uubartest2 = NRMat<std::complex<double> >(atr) * u.transpose(true) * NRMat<std::complex<double> >(ainvtr * ainv) * u * NRMat<std::complex<double> >(a);
cout <<"expected nonunitarity of explicit uubar = "<<uubartest2.norm(1.)<<endl;
//pseudounitarity
NRMat<std::complex<double> > ubarc = ubar.conjugate();
NRMat<std::complex<double> > ubarc2 =  NRMat<std::complex<double> >(ainv) * u.conjugate() * NRMat<std::complex<double> >(a);
cout <<"error of ubar complex conjugation = "<<(ubarc-ubarc2).norm()<<endl;
NRMat<std::complex<double> > uubarc = uubar.conjugate();
NRMat<std::complex<double> > ubartest2c=ubar*ubarc;
cout <<"nonpseudounitarity of ubar = "<<ubartest2c.norm(1.)<<endl;
NRMat<std::complex<double> > uubartestc=uubarc*uubar;
cout <<"nonpseudounitarity of uubar = "<<uubartestc.norm(1.)<<endl;
NRMat<std::complex<double> > uubartest2c = NRMat<std::complex<double> >(ainv) * u.conjugate() * NRMat<std::complex<double> >(a * ainv) * u * NRMat<std::complex<double> >(a);
cout <<"nonpseudounitarity of explicit uubar = "<<uubartest2c.norm(1.)<<endl;
}

if(0)
{
laerror("test exception");
}

if(0)
{
NRSMat<char> adj;
cin >>adj;
NRVec<int> clique = findclique(adj);
cout <<"First clique is "<<clique<<endl;
cout <<"Check adjacency of the clique: "<<adj.submatrix(clique)<<endl;
NRVec<int> cover = cliquecover(adj);
cout <<"Clique cover is "<<cover<<endl;
NRPerm<int> p(cover.size());
cover.sort(0,p,true);
cout<<"permutation to disentabgle the cliques = "<<p<<endl;
NRSMat<char> adjperm = adj.permuted(p);
cout <<"resorted graph = "<<adjperm<<endl;
}

if(0)
{
int seed;
int f=open("/dev/random",O_RDONLY);
if(sizeof(int)!=read(f,&seed,sizeof(int))) laerror("cannot read /dev/random");
close(f);
srand(seed);

Mat3<double> a;
a.randomize(10.);
cout<<a<<endl; 
Vec3<double> b;
b.randomize(10.);
cout <<b<<endl;
Vec3<double> x = a.linear_solve(b);
cout <<x<<endl;
cout <<"linear solve error = "<<(b-a*x).norm()<<endl;
}

if(0)
{
int seed;
int f=open("/dev/random",O_RDONLY);
if(sizeof(int)!=read(f,&seed,sizeof(int))) laerror("cannot read /dev/random");
close(f);
srand(seed);

Mat3<double> a;
a.randomize(10.);
cout<<a<<endl;
Mat3<double> ata = a.transpose()*a;
Mat3<double> work(ata);
Vec3<double> w;
Mat3<double> v;
work.eivec_sym(w,v,true);
//cout <<w<<endl<<sqrt(w[0])<<" "<<sqrt(w[1])<<" "<<sqrt(w[2])<<" "<<endl<<endl<<v<<endl;

Mat3<double> aat = a*a.transpose();
Mat3<double> work2(aat);
Vec3<double> w2; 
Mat3<double> v2;
work2.eivec_sym(w2,v2,true);
//cout <<w2<<endl<<sqrt(w2[0])<<" "<<sqrt(w2[1])<<" "<<sqrt(w2[2])<<" "<<endl<<endl<<v2<<endl;
//cout <<"dets  "<<v2.determinant()<<" "<<v.determinant()<<endl;

Mat3<double> u3,v3;
Vec3<double> w3;
a.svd(u3,w3,v3);
cout <<"Mat3 SVD\n";
cout <<u3<<endl<<w3<<endl<<endl<<v3.transpose()<<endl<<endl;
cout <<"Mat3 SVD dets "<<u3.determinant()<<" "<<v3.determinant()<<endl;


NRMat<double> aa(a);
NRMat<double> uu(3,3),vv(3,3);
NRVec<double> ss(3);
singular_decomposition(aa,&uu,ss,&vv,0);
cout <<"REGULAR SVD\n";
cout <<uu<<ss<<vv<<endl;

cout <<"svd dets  "<<determinant(uu)<<" "<<determinant(vv)<<endl;

Mat3<double> ainv= a.inverse();
Mat3<double> ainv2= a.svdinverse();
cout <<endl<<"Inverse via svd\n"<<ainv2<<endl;
cout <<"Difference of inverses = "<<(ainv-ainv2).norm()<<endl;
}

if(0)
{
int seed;
int f=open("/dev/random",O_RDONLY);
if(sizeof(int)!=read(f,&seed,sizeof(int))) laerror("cannot read /dev/random");
close(f);
srand(seed);

int n;
cin >>n;
bitvector v(n);
v.randomize();
do{
	cout <<v << " NLZ "<<v.nlz()<<" DEG "<<v.degree()<<" NTZ "<<v.ntz()<<endl;
	v>>=1;
}while(!v.iszero());
	cout <<v << " NLZ "<<v.nlz()<< " DEG "<<v.degree()<<" NTZ "<<v.ntz()<<endl;

v.randomize();

for(int i=0; i<n; ++i) 
	{
	cout <<v << " size "<<v.size()<<" NLZ "<<v.nlz()<< " DEG "<<v.degree()<<" NTZ "<<v.ntz()<<endl;
        v<<=1;
	}
	cout <<v << " size "<<v.size()<<" NLZ "<<v.nlz()<< " DEG "<<v.degree()<<" NTZ "<<v.ntz()<<endl;

v.randomize();
for(int i=0; i<v.size(); ++i) cout <<(v<<i)<<endl;
}

if(0)
{
int seed;
int f=open("/dev/random",O_RDONLY);
if(sizeof(int)!=read(f,&seed,sizeof(int))) laerror("cannot read /dev/random");
close(f);
srand(seed);

int n;
cin >>n;
bitvector v(n),u(n);
u.randomize();
v.randomize();
bitvector w=u*v;
bitvector z=v*u;
cout <<u<<endl;
cout <<v<<endl;
cout <<w<<endl;
if(w!=z) laerror("error in bitvector multiplication");
if(w.degree()!=v.degree()+u.degree()) laerror("error in degree or multiplication");
cout <<w/u-v<<endl;
cout <<w%u<<endl;
cout <<w/v-u<<endl;
cout <<w%v<<endl;
if(!(w/u-v).is_zero()) laerror("error in division");
if(!(w/v-u).is_zero()) laerror("error in division");
if(!(w%u).is_zero()) laerror("error in division");
if(!(w%v).is_zero()) laerror("error in division");
cout <<w.gcd(u)-u<<endl;
cout <<w.gcd(v)-v<<endl;
if(!(w.gcd(u)-u).is_zero()) laerror("error in gcd");
if(!(w.gcd(v)-v).is_zero()) laerror("error in gcd");

u.randomize();
v.randomize();
bitvector g=u.gcd(v);
bitvector l=u.lcm(v);
cout <<u<<endl;
cout <<v<<endl;
cout <<g<<endl;
cout <<l<<endl;
cout <<l/u - v/g<<endl;
cout <<l/v - u/g<<endl;
cout <<l%u<<endl;
cout <<l%v<<endl;
cout <<u%g<<endl;
cout <<v%g<<endl;
if(!(l/u - v/g).is_zero()) laerror("error in gcd");
if(!(l/v - u/g).is_zero()) laerror("error in gcd");
if(!(l%u).is_zero()) laerror("error in gcd");
if(!(l%v).is_zero()) laerror("error in gcd");
if(!(u%g).is_zero()) laerror("error in gcd");
if(!(v%g).is_zero()) laerror("error in gcd");
}

if(0)
{
uint64_t n;
cin >>n;
cout <<factorization(n)<<" phi = "<<eulerphi(n)<<endl;
uint64_t a;
do {
	a=RANDINT64()%n;

	}while(gcd(a,n)!=1);
cout <<"E-F test "<<powmod(a,eulerphi(n),n)<<endl;
}

if(0)
{
bitvector ir; cin >>ir;
if(!ir.is_irreducible()) laerror("input must be an irreducible polynomial");
bitvector a; cin >>a;
bitvector ai = a.field_inv(ir);
cout<< "inverse = "<<ai<<endl;
cout<<"check1 " <<(a*ai)%ir<<endl;
cout<<"check2 " <<a.field_mult(ai,ir)<<endl;

bitvector c=a.composition(ai);
bitvector cc=a.field_composition(ai,ir);
cout <<c<<endl;
cout <<c%ir<<endl;
cout <<cc<<endl;
}

if(0)
{
int seed;
int f=open("/dev/random",O_RDONLY);
if(sizeof(int)!=read(f,&seed,sizeof(int))) laerror("cannot read /dev/random");
close(f);
srand(seed);

int d;
cin >>d;
bitvector ir=find_irreducible(d,3);
if(ir.is_zero()) ir=find_irreducible(d,5);
if(ir.is_zero()) ir=find_irreducible(d,7);
if(ir.is_zero()) {cout<<"cannot find IR polynomial\n"; exit(1);}


cout <<"IR = "<<ir<<endl;
if(!ir.is_irreducible()) laerror("error in IR finder");
else cout <<"IS irreducible\n";
bitvector a(d);
do { a.randomize(); }while(a.is_zero());
cout <<"A = "<<a<<endl;
bitvector aa=a;
cout <<"A+1 = "<<++aa<<endl;
cout <<"A+1-1 = "<<--aa<<endl;
if(a!=aa) laerror("error in inc/dec");
bitvector g=ir.gcd(a);
cout<<"GCD A,IR = "<<g<<endl;
if(!g.is_one() && !a.is_zero()) laerror("ERROR IN is_irreducible\n");
bitvector ai = a.field_inv(ir);
cout <<"I = "<<ai<<endl;
bitvector check = a.field_mult(ai,ir);
cout<<"check " <<check<<endl;
if(!check.is_one()) laerror("error in GF(2^n) inversion");
bitvector s=a.field_sqrt(ir);
cout <<"test a "<<a<<endl;
cout <<"tests2 "<<s.field_mult(s,ir)<<endl;
cout <<"test s "<<s<<endl;
bitvector dif=a - s.field_mult(s,ir);
if(!dif.is_zero()) laerror("error in gf 2^n sqrt");
}


if(0)
{
int n;
cin>>n;
NRPerm<int> p(n);
PermutationAlgebra<int,int> a=p.list_all();
for(int i=0; i<a.size(); ++i)
	if(a[i].weight!=a[i].perm.parity()) laerror("internal error in parity");
}

if(0)
{
/*this is kucharskiP antisymmetrizer just without parsing the input
generate antisymmetrization operator for Brandow diagrams in the Kucharski convention
               #NOTE: permgener conserves order in the subsets,
                #this means that the permuted elements label places
                #in the expression, not the indices, since places
                #correspond to already existing antisymmetry
                #in principle it would be more logical to printout permutation of places rather than indices
                #but this is more convenient for further processing
*/

NRVec<NRVec_from1<int> > groups;
cin >> groups;
int ntot=0;
for(int i=0; i<groups.size(); ++i) ntot+=groups[i].size();
NRVec_from1<string> indices(ntot);
cin >>indices;

PermutationAlgebra<int,int> a=general_antisymmetrizer(groups,-2,true);
for(int i=0; i<a.size(); ++i)
	{
	cout << (a[i].weight==1?'+':'-');
	NRVec_from1<string> pindices=applypermutation(a[i].perm,indices,false);
	for(int j=1; j<=pindices.size(); ++j) cout <<pindices[j];
	cout <<endl;
	}
}

if(0)
{
NRMat<double> a;
cin>>a;
cout <<a.rsum()<<endl;
cout <<a.csum()<<endl;
}

if(0)
{
int n;
cin>>n;
int count=0;
for(int i=4; i<=n; ++i)
for(int j=3; j<i; ++j)
for(int k=2; k<j; ++k)
for(int l=1; l<k; ++l)
	++count;

cout <<count<<" "<<binom(n,4)<<endl;
}

if(0)
{
int d,n;
cin>>d>>n;
unsigned long long s1;
s1=simplicial(d,n);
cout <<s1<<" "<<binom(n+d-1,d)<<endl;
unsigned long long s2 = simplicial(d,n+1);
for(unsigned long long  s=s1; s<s2; ++s) 
	{
	int i=inverse_simplicial(d,s);
	if(i!=n) cout <<"Error "<<s<<" " <<i<<endl;
	}
}

if(0)
{
NRVec<int> d({6,2,1,4,3,5});
d.copyonwrite();
netsort(d.size(),&d[0]);
cout <<d;
}


if(0)
{
INDEXGROUP g;
g.number=3;
g.symmetry= -1;
g.offset=1;
g.range=5;

Tensor<double> epsilon(g);
epsilon.clear();
cout <<epsilon.size()<<endl;

for(LA_largeindex s=0; s<epsilon.size(); ++s)
	{
	SUPERINDEX I = epsilon.inverse_index(s);
	int sign;
	LA_largeindex ss=epsilon.index(&sign,I);
	if(ss!=s || sign!=1) 
		{
		cout <<"offset = "<<s<<endl;
		cout <<"index = "<<I<<endl;
		laerror("Internal error in tensor index calculation");	
		}
	}

NRVec<LA_index> I({1,2,3});
NRVec<NRVec<LA_index> > II(I,1);
epsilon.lhs(II)=1;
II.copyonwrite();
II[0][0]=3;
II[0][1]=2;
II[0][2]=1;

epsilon *= 2.;
epsilon.lhs(1,2,3) -= 1.;

cout <<epsilon(1,2,3)<<" "<<epsilon(3,2,1)<<endl;

for(int i=0; i<epsilon.data.size(); ++i) epsilon.data[i]=10*i;
cout <<epsilon.data;
cout <<epsilon;
}

if(0)
{
int n=3;
NRVec<INDEXGROUP> s(4);
for(int i=0; i<4; ++i) 
	{
	s[i].number=1;
	s[i].symmetry=0;
	s[i].offset=0;
	s[i].range=n;
	}
Tensor<double> t(s);
t.randomize(1.);
cout <<t;
NRPerm<int> p({3,1,4,2});
Tensor<double> tt=t.permute_index_groups(p);
cout <<tt;
for(int i=0; i<n; ++i)
	for(int j=0; j<n; ++j)
		for(int k=0; k<n; ++k)
			for(int l=0; l<n; ++l)
				{
				if(t(i,j,k,l)!=tt(k,i,l,j)) laerror("error in permute_index_groups");
				}
}


if(0)
{
int n=5;
INDEXGROUP g;
g.number=4;
g.symmetry= -1;
g.offset=0;
g.range=n;

Tensor<double> e(g);
e.randomize(1.);
Tensor<double> eu = e.unwind_index(0,1);

for(int i=0; i<n; ++i)
        for(int j=0; j<n; ++j)
                for(int k=0; k<n; ++k)
                        for(int l=0; l<n; ++l)
                                {
				if(e(i,j,k,l)!=eu(j,i,k,l)) laerror("error in unwind_index");
				}
cout <<e;
cout <<eu;
}


if(1)
{
int n=5;
INDEXGROUP ag;
ag.number=4;
ag.symmetry= 1;
ag.offset=0;
ag.range=n;

Tensor<double> a(ag);
a.randomize(1.);

INDEXGROUP bg;
bg.number=3;
bg.symmetry= 0;
bg.offset=0;
bg.range=n;

Tensor<double> b(bg);
b.randomize(1.);

Tensor<double> cc = a.contraction(0,0,b,0,1);
cout <<cc;

INDEXGROUP cga;
cga.number=3;
cga.symmetry= 1;
cga.offset=0;
cga.range=n;

INDEXGROUP cgb;
cgb.number=2;
cgb.symmetry= 0;
cgb.offset=0;
cgb.range=n;

NRVec<INDEXGROUP> shape({cgb,cga});

Tensor<double> c(shape);
c.clear();

for(int i=0; i<n; ++i)
	for(int j=0; j<=i; ++j)
		for(int k=0; k<=j; ++k)
			for(int l=0; l<n; ++l)
				for(int m=0; m<n; ++m)
					{
					for(int p=0; p<n; ++p)
						c.lhs(m,l,k,j,i) += a(p,i,j,k) * b(m,p,l);
					if(abs(c(m,l,k,j,i)-cc(m,l,k,j,i))>1e-13) laerror("internal error in conntraction");
					}

//cout <<c;
}

}
