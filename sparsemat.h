#ifndef _SPARSEMAT_H_
#define _SPARSEMAT_H_

#include "la_traits.h"

template<typename T>
inline const T MAX(const T &a, const T &b)
        {return b > a ? (b) : (a);}

template<typename T>
inline void SWAP(T &a, T &b)
        {T dum=a; a=b; b=dum;}


//threshold for neglecting elements, if not defined, no tests are done except exact zero test in simplify - might be even faster
//seems to perform better with a threshold, in spite of abs() tests
#define  SPARSEEPSILON 1e-13 

typedef unsigned int SPMatindex;
typedef int SPMatindexdiff; //more clear would be to use traits

//element of a linked list
template<typename T>
struct matel
        {
        T elem;
        SPMatindex row;
        SPMatindex col;
        matel *next;
        };


template <typename T>
class SparseMat {
protected:
	SPMatindex nn;
        SPMatindex mm;
	bool symmetric;
	unsigned int nonzero;
        int *count;
	matel<T> *list;
private:
	matel<T> **rowsorted; //NULL terminated
	matel<T> **colsorted; //NULL terminated
	void unsort();
	void deletelist();
	void copylist(const matel<T> *l);
public:
	//iterator
        typedef class iterator {
        private:
                matel<T> *p;
        public:
                iterator() {};
                ~iterator() {};
                iterator(matel<T> *list): p(list) {};
                bool operator==(const iterator rhs) const {return p==rhs.p;}
                bool operator!=(const iterator rhs) const {return p!=rhs.p;}
                iterator operator++() {return p=p->next;}
                iterator operator++(int) {matel<T> *q=p; p=p->next; return q;}
                matel<T> & operator*() const {return *p;}
                matel<T> * operator->() const {return p;}
        };
        iterator begin() const {return list;}
        iterator end() const {return NULL;}

	//constructors etc.
	inline SparseMat() :nn(0),mm(0),symmetric(0),nonzero(0),count(NULL),list(NULL),rowsorted(NULL),colsorted(NULL) {};
	inline SparseMat(const SPMatindex n, const SPMatindex m) :nn(n),mm(m),symmetric(0),nonzero(0),count(new int(1)),list(NULL),rowsorted(NULL),colsorted(NULL) {};
	SparseMat(const SparseMat &rhs); //copy constructor
	inline int getcount() const {return count?*count:0;}
	explicit SparseMat(const NRMat<T> &rhs); //construct from a dense one
	explicit SparseMat(const NRSMat<T> &rhs); //construct from a dense symmetric one
	SparseMat & operator=(const SparseMat &rhs);
	SparseMat & operator=(const T a);          //assign a to diagonal
    	SparseMat & operator+=(const T a);         //assign a to diagonal
	SparseMat & operator-=(const T a);         //assign a to diagonal
        SparseMat & operator*=(const T a);         //multiply by a scalar
        SparseMat & operator+=(const SparseMat &rhs);
	SparseMat & addtriangle(const SparseMat &rhs, const bool lower, const char sign);
        SparseMat & join(SparseMat &rhs); //more efficient +=, rhs will be emptied
        SparseMat & operator-=(const SparseMat &rhs);
	inline const SparseMat operator+(const T &rhs) const {return SparseMat(*this) += rhs;}
        inline const SparseMat operator-(const T &rhs) const {return SparseMat(*this) -= rhs;}
        inline const SparseMat operator*(const T &rhs) const {return SparseMat(*this) *= rhs;}
        inline const SparseMat operator+(const SparseMat &rhs) const {return SparseMat(*this) += rhs;} //must not be symmetric+general
        inline const SparseMat operator-(const SparseMat &rhs) const {return SparseMat(*this) -= rhs;} //must not be symmetric+general
	inline const NRVec<T> operator*(const NRVec<T> &rhs) const; // SparseMat * Vec
	inline const NRMat<T> operator*(const NRMat<T> &rhs) const; // SparseMat * Mat
	void diagonalof(NRVec<T> &, const bool divide=0) const; //get diagonal
	const SparseMat operator*(const SparseMat &rhs) const; 
	SparseMat & oplusequal(const SparseMat &rhs); //direct sum
	SparseMat & oplusequal(const NRMat<T> &rhs);
	SparseMat & oplusequal(const NRSMat<T> &rhs);
	const SparseMat oplus(const SparseMat &rhs) const {return SparseMat(*this).oplusequal(rhs);}; //direct sum
	const SparseMat oplus(const NRMat<T> &rhs) const {return SparseMat(*this).oplusequal(rhs);};
	const SparseMat oplus(const NRSMat<T> &rhs) const {return SparseMat(*this).oplusequal(rhs);};
	const SparseMat otimes(const SparseMat &rhs) const; //direct product
        const SparseMat otimes(const NRMat<T> &rhs) const;
        const SparseMat otimes(const NRSMat<T> &rhs) const;
        void gemm(const T beta, const SparseMat &a, const char transa, const SparseMat &b, const char transb, const T alpha);//this := alpha*op( A )*op( B ) + beta*this, if this is symemtric, only half will be added onto it
	const T dot(const SparseMat &rhs) const; //supervector dot product
	const T dot(const NRMat<T> &rhs) const; //supervector dot product
	inline ~SparseMat();
	void axpy(const T alpha, const SparseMat &x, const bool transp=0); // this+= a*x(transposed)
	inline matel<T> *getlist() const {return list;}
	void setlist(matel<T> *l) {list=l;}
	inline SPMatindex nrows() const {return nn;}
        inline SPMatindex ncols() const {return mm;}
	void get(int fd, bool dimensions=1, bool transposed=false);
	void put(int fd, bool dimensions=1, bool transposed=false) const;
	void resize(const SPMatindex n, const SPMatindex m); //destructive
	void incsize(const SPMatindex n, const SPMatindex m); //increase size without destroying the data
	void transposeme();
	const SparseMat transpose() const;
	void permuteindices(const NRVec<SPMatindex> &p);
	void permuterows(const NRVec<SPMatindex> &p);
	void permutecolumns(const NRVec<SPMatindex> &p);
	inline void setsymmetric() {if(nn!=mm) laerror("non-square cannot be symmetric"); symmetric=1;}
	inline void defineunsymmetric() {symmetric=0;} //just define and do nothing with it
	void setunsymmetric();//unwind the matrix assuming it was indeed symmetric
	inline bool issymmetric() const {return symmetric;}
	unsigned int length() const;
	void copyonwrite();
	void simplify();
	const T trace() const;
	const T norm(const T scalar=(T)0) const; //is const only mathematically, not in internal implementation - we have to simplify first
	unsigned int sort(int type) const;
	inline void add(const SPMatindex n, const SPMatindex m, const T elem) {matel<T> *ltmp= new matel<T>; ltmp->next=list; list=ltmp; list->row=n; list->col=m; list->elem=elem;}
	void addsafe(const SPMatindex n, const SPMatindex m, const T elem);
};

//due to mutual includes this has to be after full class declaration
#include "vec.h"
#include "smat.h"
#include "mat.h"

template <typename T>
inline const NRVec<T> SparseMat<T>::operator*(const NRVec<T> &rhs) const
{NRVec<T> result(nn); result.gemv((T)0,*this,'n',(T)1,rhs); return result;};

template <typename T>
inline const NRMat<T> SparseMat<T>::operator*(const NRMat<T> &rhs) const
{NRMat<T> result(nn,rhs.ncols()); result.gemm((T)0,*this,'n',rhs,'n',(T)1); return result;};

template <typename T>
	extern istream& operator>>(istream  &s, SparseMat<T> &x);

template <typename T>
	extern ostream& operator<<(ostream &s, const SparseMat<T> &x);

//destructor
template <typename T>
SparseMat<T>::~SparseMat()
{
	unsort();
        if(!count) return;
        if(--(*count)<=0)
                {
		deletelist();
                delete count;
                }
}

//copy constructor (sort arrays are not going to be copied)
template <typename T>
SparseMat<T>::SparseMat(const SparseMat<T> &rhs)
{
#ifdef debug
if(! &rhs) laerror("SparseMat copy constructor with NULL argument");
#endif
        nn=rhs.nn;
        mm=rhs.mm;
	symmetric=rhs.symmetric;
	if(rhs.list&&!rhs.count) laerror("some inconsistency in SparseMat contructors or assignments");
        list=rhs.list;
        if(list) {count=rhs.count; (*count)++;} else count=new int(1); //make the matrix defined, but empty and not shared
	colsorted=rowsorted=NULL;
	nonzero=0;
}

template <typename T>
const SparseMat<T> SparseMat<T>::transpose() const
{
if(list&&!count) laerror("some inconsistency in SparseMat transpose");
SparseMat<T> result;
result.nn=mm;
result.mm=nn;
result.symmetric=symmetric;
if(result.symmetric) 
	{
	result.list=list;
        if(list) {result.count=count; (*result.count)++;} else result.count=new int(1); //make the matrix defined, but empty and not shared
	}
else //really transpose it
	{
	result.count=new int(1);
	result.list=NULL;
	matel<T> *l =list;
	while(l)
		{
		result.add(l->col,l->row,l->elem);
		l=l->next;
		}
	}
result.colsorted=result.rowsorted=NULL;
result.nonzero=0;
return result;
}



template<typename T>
inline const SparseMat<T> commutator ( const SparseMat<T> &x, const SparseMat<T> &y, const bool trx=0, const bool tryy=0)
{
SparseMat<T> r;
r.gemm((T)0,x,trx?'t':'n',y,tryy?'t':'n',(T)1);
r.gemm((T)1,y,tryy?'t':'n',x,trx?'t':'n',(T)-1); //saves a temporary and simplifies the whole sum
return r;
}

template<typename T>
inline const SparseMat<T> anticommutator ( const SparseMat<T> &x, const SparseMat<T> &y, const bool trx=0, const bool tryy=0)
{
SparseMat<T> r;
r.gemm((T)0,x,trx?'t':'n',y,tryy?'t':'n',(T)1);
r.gemm((T)1,y,tryy?'t':'n',x,trx?'t':'n',(T)1); //saves a temporary and simplifies the whole sum
return r;
}


//add sparse to dense
template<typename T>
NRMat<T> & NRMat<T>::operator+=(const SparseMat<T> &rhs)
{
if((unsigned int)nn!=rhs.nrows()||(unsigned int)mm!=rhs.ncols()) laerror("incompatible matrices in +=");
matel<T> *l=rhs.getlist();
bool sym=rhs.issymmetric();
while(l)
        {
#ifdef MATPTR
        v[l->row][l->col] +=l->elem;
        if(sym && l->row!=l->col) v[l->col][l->row] +=l->elem;
#else
        v[l->row*mm+l->col] +=l->elem;
         if(sym && l->row!=l->col) v[l->col*mm+l->row] +=l->elem;
#endif
        l=l->next;
        }
return *this;
}

#endif
