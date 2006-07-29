// g++ -D _GLIBCPP_NO_TEMPLATE_EXPORT -g testblas.cc testblas2.cc nrutil_modif.cc -L/usr/local/lib/atlas -lstrassen -lf77blas -lcblas -latlas -ltraceback -lbfd -liberty

#include <time.h>
#include "la.h"
#include "sparsemat.h"
#include "matexp.h"
#include "fourindex.h"
#include "davidson.h"
#include "gmres.h"
#include "conjgrad.h"
#include "diis.h"
#include "bitvector.h"
#ifdef USE_TRACEBACK
#include "traceback.h"
#endif


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


int main()
{
#ifdef USE_TRACEBACK
sigtraceback(SIGSEGV,1);
sigtraceback(SIGABRT,1);
sigtraceback(SIGBUS,1);
sigtraceback(SIGFPE,1);
#endif


NRVec<double> x(1.,10);
NRVec<double> y(2.,10);
NRVec<double> z(-2.,10);

cout.setf(ios::fixed);
cout.precision(12);


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
NRMat<double> b=a.transpose();
NRMat<double> u(a.nrows(),a.nrows()),v(a.ncols(),a.ncols());
NRVec<double>s(a.ncols());
singular_decomposition(a,&u,s,&v);
//singular_decomposition(a,NULL,s,NULL); //this does not work when linked with static version of lapack, works with .so.3 version (from suse distrib)
cout <<u;
cout <<s;
cout <<v;
//singular_decomposition(b,&v,s,&u);
//cout <<v;
//cout <<s;
//cout <<u;
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
/*
int n;
cin>>n;
NRMat<double> a(n,n);
for(int i=0;i<n;++i) for(int j=0;j<i;++j)
        {
        a(i,j)= random()/(1.+RAND_MAX);
        a(j,i)= -a(i,j);
        }
NRMat<double> b=exp(a);
cout <<a;
*/
NRMat<double> a,b;
cin >>b;
int n=b.nrows();
cout <<"difference from identity = "<<b.norm(1.)<<endl;

NRMat<double> x(0.,n,n),x0;
	double r;
int i=0;
do
	{
	x0=x;
	NRMat<double> y=exp(x*-.5);
	x+= y*b*y; 
	x-= 1.;
	x=(x-x.transpose())*.5;
	cout <<"matrix x\n"<<x;
	cout <<"iter "<<i <<" residue "<< (r=(exp(x)-b).norm())<<endl;
	cout <<"iter "<<i <<" conv "<<(r=(x-x0).norm())<<endl;
	++i;
	} while(abs(r)>1e-10);
cout <<"result\n"<<x<<endl;
cout <<"exp(result)"<<exp(x)<<endl;
NRMat<double> c=log(b); //matrixfunction(a,&mycident,1);
cout <<c;
NRMat<double> d=exp(c);
cout <<"exp(log(x))\n"<<d;
cout<<(d-b).norm()<<endl;

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
NRMat<double> a;
cin >>a;
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
gmres   (a,b,x,1,1e-10,100,1,0,1,0);
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

if(0)
{
int i;
DIIS<NRVec<double> > diis(5,1);
int dim=8;
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

if(1)
{
NRMat<double> a,b;
cin >>a;
b=realsqrt(a);
cout <<b;
cout <<b*b;
cout <<(b*b-a).norm();
}


}
