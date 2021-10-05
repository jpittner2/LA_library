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

using namespace std;
using namespace LA_Vecmat3;
using namespace LA_Quaternion;
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
	return int(random()/(1.+RAND_MAX)*n);
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

static int unitary_n;
static PERM_RANK_TYPE space_dim;

void yyprintme(const NRPerm<int>&p, const int parity, const PERM_RANK_TYPE nterms)
{
cout<< parity<<"/"<<nterms<<" "<<p;
}

void yprintme(const YoungTableaux<int>&y)
{
cout <<y;
if(!y.is_standard()) laerror("internal error in young");
y.young_operator(yyprintme);
}

void pprintme(const Partition<int> &p)
{
CompressedPartition<int> pc(p);
cout<<'['<<pc<<"]\n";

YoungTableaux<int> y(p);
PERM_RANK_TYPE dim=y.generate_all_standard(yprintme);

Partition<int> q=p.adjoint();
PERM_RANK_TYPE snd=p.Sn_irrep_dim();
cout<<"IR dim "<<snd<<endl;
if(dim!=snd) laerror("inconsistency in standard tableaux generation");
PERM_RANK_TYPE und=p.Un_irrep_dim(unitary_n);
cout<<"U("<<unitary_n<<") ir dim "<<und<<endl;
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


NRVec<double> x(1.,10);
NRVec<double> y(2.,10);
NRVec<double> z(-2.,10);

//cout.setf(ios::scientific);
cc:cout.setf(ios::fixed);
cout.precision(10);
cin.exceptions ( ifstream::eofbit | ifstream::failbit | ifstream::badbit );


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
NRMat<double> u(a.nrows(),a.nrows()),v(a.ncols(),a.ncols());
NRVec<double>s(a.ncols()<a.nrows()?a.ncols():a.nrows());
singular_decomposition(a,&u,s,&v,0);
cout <<u;
NRMat<double> sdiag(0., u.ncols(),v.nrows());
sdiag.diagonalset(s);
cout <<sdiag;
cout <<v;
cout << "Error "<<(u*sdiag*v-abak).norm()<<endl;
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
	for(int i=0; i<sparsity;i++) a(randind(n),randind(n))=random()/(1.+RAND_MAX);
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
	for(int i=0; i<sparsity;i++) aa.add(randind(n),randind(n),random()/(1.+RAND_MAX));
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
	for(int i=0; i<sparsity;i++) aa.add(randind(n),randind(n),random()/(1.+RAND_MAX));
	SparseMat<double> bb=exp(aa);
	NRVec<double> v(n);
	 for(int i=0; i<n;++i) v[i]=random()/(1.+RAND_MAX);
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
for(int i=0;i<n;++i) for(int j=0;j<k;++j) a(i,j)= random()/(1.+RAND_MAX);
for(int i=0;i<k;++i) for(int j=0;j<m;++j) b(i,j)= random()/(1.+RAND_MAX);
c.gemm(0., a, 'n', b, 'n', .6);
SparseMat<double> aa(a);
d.gemm(0., aa, 'n', b, 'n', .6);
cout<<c<<d;
cout  <<"test error = "<<(c-d).norm()<<endl;
}
{
NRMat<double> a(k,n),b(k,m),c(n,m),d(n,m);
for(int i=0;i<k;++i) for(int j=0;j<n;++j) a(i,j)= random()/(1.+RAND_MAX);
for(int i=0;i<k;++i) for(int j=0;j<m;++j) b(i,j)= random()/(1.+RAND_MAX);
c.gemm(0., a, 't', b, 'n', .7);
SparseMat<double> aa(a);
d.gemm(0., aa, 't', b, 'n', .7);
cout<<c<<d;
cout  <<"test error = "<<(c-d).norm()<<endl;
}
{
NRMat<double> a(n,k),b(m,k),c(n,m),d(n,m);
for(int i=0;i<n;++i) for(int j=0;j<k;++j) a(i,j)= random()/(1.+RAND_MAX);
for(int i=0;i<m;++i) for(int j=0;j<k;++j) b(i,j)= random()/(1.+RAND_MAX);
c.gemm(0., a, 'n', b, 't', .8);
SparseMat<double> aa(a);
d.gemm(0., aa, 'n', b, 't', .8);
cout<<c<<d;
cout  <<"test error = "<<(c-d).norm()<<endl;
}
{
NRMat<double> a(k,n),b(m,k),c(n,m),d(n,m);
for(int i=0;i<k;++i) for(int j=0;j<n;++j) a(i,j)= random()/(1.+RAND_MAX);
for(int i=0;i<m;++i) for(int j=0;j<k;++j) b(i,j)= random()/(1.+RAND_MAX);
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
	a(i,j)= random()/(1.+RAND_MAX);
	a(j,i)= random()/(1.+RAND_MAX);
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
        a(i,j)= random()/(1.+RAND_MAX);
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
	u[i]=f*random()/(1.+RAND_MAX);
	for(int j=0;j<n;++j)
       		 a(i,j)= f*random()/(1.+RAND_MAX);
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
        a(i,j)= .1*random()/(1.+RAND_MAX);
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
NRMat<double> c=inverse(a,&d);
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
        a.add(i,j,random()/(1.+RAND_MAX));
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
        a(j,i)=a(i,j)=random()/(1.+RAND_MAX)*(i==j?10.:1.);
        }
cout <<a;
NRMat<double> y(1,n);
for(int i=0;i<n;++i) y(0,i)=random()/(1.+RAND_MAX);
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
        for(int i=0; i<spars;i++) a.add(randind(n),randind(n),random()/(1.+RAND_MAX));

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
        a(i,j)= random()/(1.+RAND_MAX);
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
        a(i,j)= random()/(1.+RAND_MAX);
        a(j,i)= random()/(1.+RAND_MAX);
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
	for(int i=0; i<sparsity;i++) aa.add(randind(n),randind(n),random()/(1.+RAND_MAX));
	for(int i=0; i<n; ++i) aa.add(i,i,500*random()/(1.+RAND_MAX));
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
        for(int i=0; i<sparsity;i++) aa.add(randind(n),randind(n),random()/(1.+RAND_MAX));
        for(int i=0; i<n; ++i) aa.add(i,i,500*random()/(1.+RAND_MAX));
NRVec<double> r(m);
NRVec<double> r2(m);
davidson(aa,r,(NRVec<double> *)NULL,"eivecs",m,1,1e-5,1,300,300);
 SparseMat<double> bb(n,n);
for(int i=0; i<sparsity2;i++) bb.add(randind(n),randind(n),random()/(1.+RAND_MAX));
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
		double a=random()/(1.+RAND_MAX);
		double b=random()/(1.+RAND_MAX)-.5;
		aa.add(k,l,a);
		aa.add(l,k,a+b/20);
		}
        for(int i=0; i<n; ++i) aa.add(i,i,500*random()/(1.+RAND_MAX));
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
	a(i,j)= random()/(1.+RAND_MAX);
	if(i==j) a(i,i)+= .5*(i-n);
        }
for(int i=0;i<n;++i) b[i] = i;

NRVec<double> bb=b;
//cout <<a;
//cout <<b;
NRMat<double>aa=a;
linear_solve(aa,bb);
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
        for(int i=0; i<n; ++i) aa.add(i,i,random()/(1.+RAND_MAX));
	for(int i=0; i<n-1; i+=1) aa.add(i,i+1,.002*random()/(1.+RAND_MAX));
	for(int i=0; i<n-1; i+=1) aa.add(i+1,i,.002*random()/(1.+RAND_MAX));

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
for(i=0; i<dim; ++i) deviation[i]= (i&2 ? 1:-1) * random()/(1.+RAND_MAX);
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
		ader(i,j) = 2.*random()/(1.+RAND_MAX) -1.;

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

NRMat<double> vri = inverse(vr);

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
for(int i=0; i<n; ++i) for(int j=i; j<n; ++j) {if((double)random()/RAND_MAX>0.995 || i==j) {++nn; bb(i,j)=bb(j,i)=(double)random()/RAND_MAX*(i==j?10.:.5/(i+j)*(random()>RAND_MAX/2?1:-1));}}
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
for(int i=0; i<=n/400; ++i) for(int j=i; j<n; ++j) {if((double)random()/RAND_MAX>0.995 || i==j) 
	{bh.add(i,j,(double)random()/RAND_MAX*(i==j?10.:(random()>RAND_MAX/2?1:-1)),false);}}
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
CyclePerm<int> c;
cin>>c;
cout<<c<<endl;
NRPerm<int> p(c);
cout <<p;
CyclePerm<int> cc(p);
cout <<cc<<endl;
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
int n,unitary_n;;
cin >>n >>unitary_n;
Partition<int> p(n);
space_dim=0;
int tot=p.generate_all(pprintme,0);
cout <<"partitions generated "<<tot<<endl;
if(tot!=partitions(n)) laerror("internal error in partition generation or enumerations");
if(space_dim!=longpow(unitary_n,n)) {cout<<space_dim<<" "<<ipow(unitary_n,n)<<endl;laerror("integer overflow or internal error in space dimensions");}
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
for(int i=0; i<=p.size(); ++i) if(d!=pp[i]) laerror("error in cycle index");
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

if(1)
{
int n;
cin >>n ;
cout <<"Hermite = "<<hermite_polynomial<int>(n);
}


}
