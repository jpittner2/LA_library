#ifndef _MATEXP_H_
#define _MATEXP_H_
//general routine for polynomial of a matrix, tuned to minimize the number
//of matrix-matrix multiplications on cost of additions and memory
// the polynom and exp routines will work on any type, for which traits class
// is defined containing definition of an element type, norm and axpy operation

#include "la_traits.h"
#include "laerror.h"
#ifndef NONCBLAS
extern "C" {
#include "cblas.h"
#include "clapack.h"
}
#else
#include "noncblas.h"
#endif


template<class T,class R>
const T polynom0(const T &x, const NRVec<R> &c)
{
int order=c.size()-1;
T z,y;

//trivial reference implementation by horner scheme
if(order==0) {y=x; y=c[0];} //to avoid the problem: we do not know the size of the matrix to contruct a scalar one
else
	{
	int i;
	z=x*c[order];
	for(i=order-1; i>=0; i--)
		{
		if(i<order-1) z=y*x;
		y=z+c[i];
		}
	}

return y;
}


//algorithm which minimazes number of multiplications, at the cost of storage
template<class T,class R>
const T polynom(const T &x, const NRVec<R> &c)
{
int n=c.size()-1;
int i,j,k,m=0,t;

if(n<=4) return polynom0(x,c); //here the horner scheme is optimal

//first find m which minimizes the number of multiplications
j=10*n;
for(i=2;i<=n+1;i++)
    {	
    t=i-2+2*(n/i)-(n%i)?0:1;
    if(t<j)
	{
	j=t;
	m=i;
	}
    }


//allocate array for powers up to m
T *xpows = new T[m];
xpows[0]=x;
for(i=1;i<m;i++) xpows[i]=xpows[i-1]*x;


//run the summation loop
T r,s,f;
k= -1;
for(i=0; i<=n/m;i++)
	{
	for(j=0;j<m;j++)
		{
		k++;
		if(k>n) break;
		if(j==0) {
			  if(i==0) s=x; /*just to get the dimensions of the matrix*/ 
			  s=c[k]; /*create diagonal matrix*/
			  }
		else  
			LA_traits<T>::axpy(s,xpows[j-1],c[k]); //general  s+=xpows[j-1]*c[k]; but more efficient for matrices
		}

	if(i==0) {r=s; f=xpows[m-1];}
	else
		{
		r+= s*f;
		f=f*xpows[m-1];
		}
	}
 
delete[] xpows;
return r;
}


//for general objects
template<class T>
const T ncommutator ( const T &x, const T &y, int nest=1, const bool right=1)
{
T z;
if(right) {z=x; while(--nest>=0) z=z*y-y*z;}
else {z=y; while(--nest>=0) z=x*z-z*x;}
return z;
}

template<class T>
const T nanticommutator ( const T &x, const T &y, int nest=1, const bool right=1)
{
T z;
if(right) {z=x; while(--nest>=0) z=z*y+y*z;}
else {z=y; while(--nest>=0) z=x*z+z*x;}
return z;
}

//general BCH expansion (can be written more efficiently in a specialization for matrices)
template<class T>
const T BCHexpansion (const T &h, const T &t, const int n, const bool verbose=0)\
{
T result=h;
double factor=1.;
T z=h;
for(int i=1; i<=n; ++i)
	{
	factor/=i;
	z= z*t-t*z;
	if(verbose) cerr << "BCH contribution at order "<<i<<" : "<<z.norm()*factor<<endl;
	result+= z*factor; 
	}
return result;
}


template<class T>
const T ipow( const T &x, int i)
{
if(i<0) laerror("negative exponent in ipow");
if(i==0) {T r=x; r=(typename LA_traits<T>::elementtype)1; return r;}//trick for matrix dimension
if(i==1) return x;
T y,z;
z=x;
while(!(i&1))
	{
	z = z*z;
	i >>= 1;
	}
y=z; 
while((i >>= 1)/*!=0*/)
                {
                z = z*z;
                if(i&1) y = y*z;
                }
return y;
}

inline int nextpow2(const double n)
{
const double log2=log(2.);
if(n<=.75) return 0; //try to keep the taylor expansion short
if(n<=1.) return 1;
return int(ceil(log(n)/log2-log(.75)));
}


template<class T, class C>
NRVec<C> exp_aux(const T &x, int &power,int maxpower= -1, int maxtaylor= -1)
{
//should better be computed by mathematica to have accurate last digits, chebyshev instead, see exp in glibc
static double exptaylor[]={
1.,
1.,
0.5,
0.1666666666666666666666,
0.0416666666666666666666,
0.0083333333333333333333,
0.0013888888888888888888,
0.00019841269841269841253,
2.4801587301587301566e-05,
2.7557319223985892511e-06,
2.7557319223985888276e-07,
2.5052108385441720224e-08,
2.0876756987868100187e-09,
1.6059043836821613341e-10,
1.1470745597729724507e-11,
7.6471637318198164055e-13,
4.7794773323873852534e-14,
2.8114572543455205981e-15,
1.5619206968586225271e-16,
8.2206352466243294955e-18,
4.1103176233121648441e-19,
0.};
double mnorm= x.norm();
power=nextpow2(mnorm); 
if(maxpower>=0 && power>maxpower) power=maxpower;
double scale=exp(-log(2.)*power);


//find how long taylor expansion will be necessary
const double precision=1e-14; //decreasing brings nothing
double s,t;
s=mnorm*scale;
int n=0;
t=1.;
do	{
	n++;
	t*=s;
	}
while(t*exptaylor[n]>precision);//taylor 0 will terminate in any case

if(maxtaylor>=0 && n>maxtaylor) n=maxtaylor; //useful e.g. if the matrix is nilpotent in order n+1 as the CC T operator for n electrons


int i; //adjust the coefficients in order to avoid scaling the argument
NRVec<C> taylor2(n+1);
for(i=0,t=1.;i<=n;i++)
	{
	taylor2[i]=exptaylor[i]*t;
	t*=scale;
	}
return taylor2;
}



//it seems that we do not gain anything by polynom vs polynom0, check the m-optimization!
template<class T>
const T exp(const T &x, bool horner=true, int maxpower= -1, int maxtaylor= -1 )
{
int power;

//prepare the polynom of and effectively scale T
NRVec<typename LA_traits<T>::elementtype> taylor2=exp_aux<T,typename LA_traits<T>::elementtype>(x,power,maxpower,maxtaylor);


T r= horner?polynom0(x,taylor2):polynom(x,taylor2); 
//for accuracy summing from the smallest terms up would be better, but this is more efficient for matrices

//power the result back
for(int i=0; i<power; i++) r=r*r;
return r;
}




//this simple implementation seems not to be numerically stable enough
//and probably not efficient either

template<class M, class V>
void exptimesdestructive(const M &mat, V &result, V &rhs,  bool transpose=false, const double scale=1., int maxpower= -1, int maxtaylor= -1) //uses just matrix vector multiplication
{
if(mat.nrows()!=mat.ncols()||(unsigned int) mat.nrows() != (unsigned int)rhs.size()) laerror("inappropriate sizes in exptimes");

int power;
//prepare the polynom of and effectively scale the matrix
NRVec<typename LA_traits<V>::elementtype> taylor2=exp_aux<M,typename LA_traits<V>::elementtype>(mat,power,maxpower,maxtaylor);

V tmp;

for(int i=1; i<=(1<<power); ++i) //unfortunatelly, here we have to repeat it many times, unlike if the matrix is stored explicitly
	{
	if(i>1) rhs=result; //apply again to the result of previous application
	else result=rhs;
	tmp=rhs; //now rhs can be used as scratch	
	result*=taylor2[0];
	for(int j=1; j<taylor2.size(); ++j)
		{
		mat.gemv(0.,rhs,transpose?'t':'n',scale,tmp);
		tmp=rhs;
		result.axpy(taylor2[j],tmp);
		}
	}

return;
}


template<class M, class V>
const V exptimes(const M &mat, V rhs, bool transpose=false, const double scale=1., int maxpower= -1, int maxtaylor= -1 )
{
V result;
exptimesdestructive(mat,result,rhs,transpose,scale,maxpower,maxtaylor);
return result;
}







#ifdef EXPOKIT
//this comes from dgexpv from expokit, it had to be translated to C to make a template
// dgexpv.f -- translated by f2c (version 20030320).

static double dipow(double x,int i)
{
double y=1.0;
if(x==0.0)
        {
        if(i>0) return(0.0);
        if(i==0) laerror("dipow: 0^0");
        return(1./x);
        }
if(i) {
        if(i<0) {i= -i; x= 1.0/x;}
        do
                {
                if(i&1) y *= x;
                x *= x;
                }
        while((i >>= 1)/*!=0*/);
        }
return(y);
}

static int nint(double x)
{
int sgn= x>=0 ? 1 : -1;
return sgn * (int)(abs(x)+.5);
}


#ifdef FORTRAN_
#define FORNAME(x) x##_
#else
#define FORNAME(x) x
#endif

extern "C" void FORNAME(dgpadm)(const int *, int *, double *, double *, int *, double *, int *, int *, int *, int *, int *);
extern "C" void FORNAME(dnchbv)(int *, double *, double *, int *, double *, double *);

//partial template specialization leaving still free room for generic matrix type
template<class M>
extern void dgexpv(const M &mat, NRVec<double> &result, const NRVec<double> &rhs, bool transpose=false, const double t=1., int mxkrylov=0, double tol=1e-14, int mxstep=1000, int mxreject=0, int ideg=6)
{
if(mat.nrows()!= mat.ncols()) laerror("non-square matrix in exptimes");
int n=mat.nrows();
int m;
if(mxkrylov)  m=mxkrylov; else m= n>100? 100: (n>1?n-1:1);
if (m > n || m <= 0) laerror("illegal mxkrylov");
if(result.size()!=n || rhs.size()!=n) laerror("inconsistent vector and matrix size in exptimes");

const double *v=rhs;
double *w=result;
double anorm=mat.norm(); 


int iflag;

    /* System generated locals */
    int i__1, i__2;
    double d__1;

#define  pow_dd(x,y) (exp(y*log(x)))

    /* Local variables */
    int ibrkflag;
    double step_min__, step_max__;
    int i__, j;
    double break_tol__;
    int k1;
    double p1, p2, p3;
    int ih, mh, iv, ns, mx;
    double xm;
    int j1v;
    double hij, sgn, eps, hj1j, sqr1, beta;
    double hump;
    int ifree, lfree;
    double t_old__;
    int iexph;
    double t_new__;
    int nexph;
    double t_now__;
    int nstep;
    double t_out__;
    int nmult;
    double vnorm;
    int nscale;
    double rndoff, t_step__, avnorm;
    int ireject;
    double err_loc__;
    int nreject, mbrkdwn;
    double tbrkdwn, s_error__, x_error__;

/* -----Purpose----------------------------------------------------------| */

/* ---  DGEXPV computes w = exp(t*A)*v - for a General matrix A. */

/*     It does not compute the matrix exponential in isolation but */
/*     instead, it computes directly the action of the exponential */
/*     operator on the operand vector. This way of doing so allows */
/*     for addressing large sparse problems. */

/*     The method used is based on Krylov subspace projection */
/*     techniques and the matrix under consideration interacts only */
/*     via the external routine `matvec' performing the matrix-vector */
/*     product (matrix-free method). */

/* -----Arguments--------------------------------------------------------| */

/*     n      : (input) order of the principal matrix A. */

/*     m      : (input) maximum size for the Krylov basis. */

/*     t      : (input) time at wich the solution is needed (can be < 0). */

/*     v(n)   : (input) given operand vector. */

/*     w(n)   : (output) computed approximation of exp(t*A)*v. */

/*     tol    : (input/output) the requested accuracy tolerance on w. */
/*              If on input tol=0.0d0 or tol is too small (tol.le.eps) */
/*              the internal value sqrt(eps) is used, and tol is set to */
/*              sqrt(eps) on output (`eps' denotes the machine epsilon). */
/*              (`Happy breakdown' is assumed if h(j+1,j) .le. anorm*tol) */

/*     anorm  : (input) an approximation of some norm of A. */

/*   wsp(lwsp): (workspace) lwsp .ge. n*(m+1)+n+(m+2)^2+4*(m+2)^2+ideg+1 */
/*                                   +---------+-------+---------------+ */
/*              (actually, ideg=6)        V        H      wsp for PADE */

/* iwsp(liwsp): (workspace) liwsp .ge. m+2 */

/*     matvec : external subroutine for matrix-vector multiplication. */
/*              synopsis: matvec( x, y ) */
/*                        double precision x(*), y(*) */
/*              computes: y(1:n) <- A*x(1:n) */
/*                        where A is the principal matrix. */


/*     iflag  : (output) exit flag. */
/*              <0 - bad input arguments */
/*               0 - no problem */
/*               1 - maximum number of steps reached without convergence */
/*               2 - requested tolerance was too high */

/* -----Accounts on the computation--------------------------------------| */
/*     Upon exit, an interested user may retrieve accounts on the */
/*     computations. They are located in wsp and iwsp as indicated below: */

/*     location  mnemonic                 description */
/*     -----------------------------------------------------------------| */
/*     iwsp(1) = nmult, number of matrix-vector multiplications used */
/*     iwsp(2) = nexph, number of Hessenberg matrix exponential evaluated */
/*     iwsp(3) = nscale, number of repeated squaring involved in Pade */
/*     iwsp(4) = nstep, number of integration steps used up to completion */
/*     iwsp(5) = nreject, number of rejected step-sizes */
/*     iwsp(6) = ibrkflag, set to 1 if `happy breakdown' and 0 otherwise */
/*     iwsp(7) = mbrkdwn, if `happy brkdown', basis-size when it occured */
/*     -----------------------------------------------------------------| */
/*     wsp(1)  = step_min, minimum step-size used during integration */
/*     wsp(2)  = step_max, maximum step-size used during integration */
/*     wsp(3)  = dummy */
/*     wsp(4)  = dummy */
/*     wsp(5)  = x_error, maximum among all local truncation errors */
/*     wsp(6)  = s_error, global sum of local truncation errors */
/*     wsp(7)  = tbrkdwn, if `happy breakdown', time when it occured */
/*     wsp(8)  = t_now, integration domain successfully covered */
/*     wsp(9)  = hump, i.e., max||exp(sA)||, s in [0,t] (or [t,0] if t<0) */
/*     wsp(10) = ||w||/||v||, scaled norm of the solution w. */
/*     -----------------------------------------------------------------| */
/*     The `hump' is a measure of the conditioning of the problem. The */
/*     matrix exponential is well-conditioned if hump = 1, whereas it is */
/*     poorly-conditioned if hump >> 1. However the solution can still be */
/*     relatively fairly accurate even when the hump is large (the hump */
/*     is an upper bound), especially when the hump and the scaled norm */
/*     of w [this is also computed and returned in wsp(10)] are of the */
/*     same order of magnitude (further details in reference below). */

/* ----------------------------------------------------------------------| */
/* -----The following parameters may also be adjusted herein-------------| */

/* mxstep   = 1000, .           mxreject = 0, .           ideg     = 6, */
/*     mxstep  : maximum allowable number of integration steps. */
/*               The value 0 means an infinite number of steps. */

/*     mxreject: maximum allowable number of rejections at each step. */
/*               The value 0 means an infinite number of rejections. */

/*     ideg    : the Pade approximation of type (ideg,ideg) is used as */
/*               an approximation to exp(H). The value 0 switches to the */
/*               uniform rational Chebyshev approximation of type (14,14) */

/*     delta   : local truncation error `safety factor' */

/*     gamma   : stepsize `shrinking factor' */

/* ----------------------------------------------------------------------| */
/*     Roger B. Sidje (rbs@maths.uq.edu.au) */
/*     EXPOKIT: Software Package for Computing Matrix Exponentials. */
/*     ACM - Transactions On Mathematical Software, 24(1):130-156, 1998 */
/* ----------------------------------------------------------------------| */
    iflag = 0;

    i__1 = m + 2;
    int lwsp= n * (m + 2) + i__1 * i__1 * 5 + ideg + 1;
    double *wsp = new double[lwsp];
    int *iwsp = new int[m+2];


    /* Parameter adjustments */
    --w;
    --v;
    --wsp;
    --iwsp;


/* ---  initialisations ... */

    k1 = 2;
    mh = m + 2;
    iv = 1;
    ih = iv + n * (m + 1) + n;
    ifree = ih + mh * mh;
    lfree = lwsp - ifree + 1;
    ibrkflag = 0;
    mbrkdwn = m;
    nmult = 0;
    nreject = 0;
    nexph = 0;
    nscale = 0;
    t_out__ = abs(t);
    tbrkdwn = 0.;
    step_min__ = t_out__;
    step_max__ = 0.;
    nstep = 0;
    s_error__ = 0.;
    x_error__ = 0.;
    t_now__ = 0.;
    t_new__ = 0.;
    p1 = 1.3333333333333333;
L1:
    p2 = p1 - 1.;
    p3 = p2 + p2 + p2;
    eps = (d__1 = p3 - 1., abs(d__1));
    if (eps == 0.) {
	goto L1;
    }
    if (tol <= eps) {
	tol = sqrt(eps);
    }
    rndoff = eps * anorm;
    break_tol__ = 1e-7;
/* >>>  break_tol = tol */
/* >>>  break_tol = anorm*tol */
    sgn = t>=0?1. : -1.; 
    cblas_dcopy(n, &v[1], 1, &w[1], 1);
    beta = cblas_dnrm2(n, &w[1], 1);
    vnorm = beta;
    hump = beta;

/* ---  obtain the very first stepsize ... */

    sqr1 = sqrt(.1);
    xm = 1. / (double) (m);
    d__1 = (m + 1) / 2.72;
    i__1 = m + 1;
    p1 = tol * dipow(d__1, i__1) * sqrt((m + 1) * 6.28);
    d__1 = p1 / (beta * 4. * anorm);
    t_new__ = 1. / anorm * pow_dd(d__1, xm);
    d__1 = log10(t_new__) - sqr1;
    i__1 = nint(d__1) - 1;
    p1 = dipow(10., i__1);
    d__1 = t_new__ / p1 + .55;
    t_new__ = floor(d__1) * p1;

/* ---  step-by-step integration ... */

L100:
    if (t_now__ >= t_out__) {
	goto L500;
    }
    ++nstep;
/* Computing MIN */
    d__1 = t_out__ - t_now__;
    t_step__ = min(d__1,t_new__);
    p1 = 1. / beta;
    i__1 = n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	wsp[iv + i__ - 1] = p1 * w[i__];
    }
    i__1 = mh * mh;
    for (i__ = 1; i__ <= i__1; ++i__) {
	wsp[ih + i__ - 1] = 0.;
    }

/* ---  Arnoldi loop ... */

    j1v = iv + n;
    i__1 = m;
    for (j = 1; j <= i__1; ++j) {
	++nmult;
	{
	NRVec<double> res(&wsp[j1v],n,true);
	NRVec<double> rhs(&wsp[j1v - n],n,true);
	mat.gemv(0.,res,transpose?'t':'n',1.,rhs);
	}
	i__2 = j;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    hij = cblas_ddot(n, &wsp[iv + (i__ - 1) * n], 1, &wsp[j1v], 1);
	    d__1 = -hij;
	    cblas_daxpy(n, d__1, &wsp[iv + (i__ - 1) * n], 1, &wsp[j1v], 1);
	    wsp[ih + (j - 1) * mh + i__ - 1] = hij;
	}
	hj1j = cblas_dnrm2(n, &wsp[j1v], 1);
/* ---     if `happy breakdown' go straightforward at the end ... */
	if (hj1j <= break_tol__) {
/*           print*,'happy breakdown: mbrkdwn =',j,' h =',hj1j */
	    k1 = 0;
	    ibrkflag = 1;
	    mbrkdwn = j;
	    tbrkdwn = t_now__;
	    t_step__ = t_out__ - t_now__;
	    goto L300;
	}
	wsp[ih + (j - 1) * mh + j] = hj1j;
	d__1 = 1. / hj1j;
	cblas_dscal(n, d__1, &wsp[j1v], 1);
	j1v += n;
/* L200: */
    }
    ++nmult;
    {
    NRVec<double> res(&wsp[j1v],n,true);
    NRVec<double> rhs(&wsp[j1v - n],n,true);
    mat.gemv(0.,res,transpose?'t':'n',1.,rhs);
    }
    avnorm = cblas_dnrm2(n, &wsp[j1v], 1);

/* ---  set 1 for the 2-corrected scheme ... */

L300:
    wsp[ih + m * mh + m + 1] = 1.;

/* ---  loop while ireject<mxreject until the tolerance is reached ... */

    ireject = 0;
L401:

/* ---  compute w = beta*V*exp(t_step*H)*e1 ... */

    ++nexph;
    mx = mbrkdwn + k1;
    if (ideg != 0) {
/* ---     irreducible rational Pade approximation ... */
	d__1 = sgn * t_step__;
	FORNAME(dgpadm)(&ideg, &mx, &d__1, &wsp[ih], &mh, &wsp[ifree], &lfree, &iwsp[1]
		, &iexph, &ns, &iflag);
	iexph = ifree + iexph - 1;
	nscale += ns;
    } else {
/* ---     uniform rational Chebyshev approximation ... */
	iexph = ifree;
	i__1 = mx;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    wsp[iexph + i__ - 1] = 0.;
	}
	wsp[iexph] = 1.;
	d__1 = sgn * t_step__;
	FORNAME(dnchbv)(&mx, &d__1, &wsp[ih], &mh, &wsp[iexph], &wsp[ifree + mx]);
    }
/* L402: */

/* ---  error estimate ... */

    if (k1 == 0) {
	err_loc__ = tol;
    } else {
	p1 = (d__1 = wsp[iexph + m], abs(d__1)) * beta;
	p2 = (d__1 = wsp[iexph + m + 1], abs(d__1)) * beta * avnorm;
	if (p1 > p2 * 10.) {
	    err_loc__ = p2;
	    xm = 1. / (double) (m);
	} else if (p1 > p2) {
	    err_loc__ = p1 * p2 / (p1 - p2);
	    xm = 1. / (double) (m);
	} else {
	    err_loc__ = p1;
	    xm = 1. / (double) (m - 1);
	}
    }

/* ---  reject the step-size if the error is not acceptable ... */

    if (k1 != 0 && err_loc__ > t_step__ * 1.2 * tol && (mxreject == 0 || 
	    ireject < mxreject)) {
	t_old__ = t_step__;
	d__1 = t_step__ * tol / err_loc__;
	t_step__ = t_step__ * .9 * pow_dd(d__1, xm);
	d__1 = log10(t_step__) - sqr1;
	i__1 = nint(d__1) - 1;
	p1 = dipow(10., i__1);
	d__1 = t_step__ / p1 + .55;
	t_step__ = floor(d__1) * p1;
#ifdef DEBUG
/*           print*,'t_step =',t_old */
/*           print*,'err_loc =',err_loc */
/*           print*,'err_required =',delta*t_old*tol */
/*           print*,'stepsize rejected, stepping down to:',t_step */
#endif
	++ireject;
	++nreject;
	if (mxreject != 0 && ireject > mxreject) {
	    iflag = 2;
	laerror("Failure in DGEXPV: The requested tolerance is too high. Rerun with a smaller value.");
	}
	goto L401;
    }

/* ---  now update w = beta*V*exp(t_step*H)*e1 and the hump ... */

/* Computing MAX */
    i__1 = 0, i__2 = k1 - 1;
    mx = mbrkdwn + max(i__1,i__2);
   cblas_dgemv(CblasRowMajor, CblasTrans, n, mx, beta,  &wsp[iv], n, &wsp[iexph], 1,  0.,  &w[1],1);
    beta = cblas_dnrm2(n, &w[1], 1);
    hump = max(hump,beta);

/* ---  suggested value for the next stepsize ... */

    d__1 = t_step__ * tol / err_loc__;
    t_new__ = t_step__ * .9 * pow_dd(d__1, xm);
    d__1 = log10(t_new__) - sqr1;
    i__1 = nint(d__1) - 1;
    p1 = dipow(10., i__1);
    d__1 = t_new__ / p1 + .55;
    t_new__ = floor(d__1) * p1;
    err_loc__ = max(err_loc__,rndoff);

/* ---  update the time covered ... */

    t_now__ += t_step__;

/* ---  display and keep some information ... */

#ifdef DEBUG
/*        print*,'integration',nstep,'---------------------------------' */
/*        print*,'scale-square =',ns */
/*        print*,'step_size =',t_step */
/*        print*,'err_loc   =',err_loc */
/*        print*,'next_step =',t_new */
#endif
    step_min__ = min(step_min__,t_step__);
    step_max__ = max(step_max__,t_step__);
    s_error__ += err_loc__;
    x_error__ = max(x_error__,err_loc__);
    if (mxstep == 0 || nstep < mxstep) {
	goto L100;
    }
    iflag = 1;
L500:
/*     iwsp(1) = nmult */
/*     iwsp(2) = nexph */
/*     iwsp(3) = nscale */
/*     iwsp(4) = nstep */
/*     iwsp(5) = nreject */
/*     iwsp(6) = ibrkflag */
/*     iwsp(7) = mbrkdwn */
/*     wsp(1)  = step_min */
/*     wsp(2)  = step_max */
/*     wsp(3)  = 0.0d0 */
/*     wsp(4)  = 0.0d0 */
/*     wsp(5)  = x_error */
/*     wsp(6)  = s_error */
/*     wsp(7)  = tbrkdwn */
/*     wsp(8)  = sgn*t_now */
/*     wsp(9)  = hump/vnorm */
/*     wsp(10) = beta/vnorm */
    if(iflag) laerror("dgexpv error");
delete[] ++wsp; delete[] ++iwsp;
return;
} 
#undef FORNAME
#undef pow_dd

#endif 
//EXPOKIT


//@@@ power series matrix logarithm?

#endif
