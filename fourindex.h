#ifndef _fourindex_included
#define _fourindex_included
#include <iostream>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include "la.h"
#include "laerror.h"

//element of a linked list, indices in a portable way, no bit shifts and endianity problems any more!
//note: nn is never compared with individual indices, so indexing from 1 as well as from 0 is possible
//it is actually not needed for the algorithms here, but may be useful for the
//user of this class to keep this piece of information along with the data

//when patient enough, make const_casts for piterators to have pbegin() const

template<class I>
union packed_index {
                I packed[4];
                struct {
                        I i;
                        I j;
                        I k;
                        I l;
                        } indiv;
                };

template<class I, class T>
struct matel4
        {
        T elem;
        matel4 *next;
	union packed_index<I> index;
        };


template<class I, class T>
struct matel4stored
        {
        T elem;
        union packed_index<I> index;
        };


//later add symmetry of complex integrals
typedef enum {undefined_symmetry=-1,nosymmetry=0, twoelectronrealmullikan=1, twoelectronrealdirac=2, T2ijab_real=3, antisymtwoelectronrealmullikan=4,antisymtwoelectronrealdirac=5} fourindexsymtype; //only permutation-nonequivalent elements are stored
// these should actually be static private members of the fourindex class, but leads to an ICE on gcc3.2
static const int fourindex_n_symmetrytypes=6;
static const int fourindex_permnumbers[fourindex_n_symmetrytypes]={1,8,8,4,16,16};
static const int fourindex_permutations[fourindex_n_symmetrytypes][16][5]=
		{
		{{0,1,2,3,1}},
		{{0,1,2,3,1}, {1,0,2,3,1}, {0,1,3,2,1}, {1,0,3,2,1}, {2,3,0,1,1}, {3,2,0,1,1}, {2,3,1,0,1}, {3,2,1,0,1}},
		{{0,1,2,3,1},{2,1,0,3,1},{0,3,2,1,1},{2,3,0,1,1},{1,0,3,2,1},{3,0,1,2,1},{1,2,3,0,1},{3,2,1,0,1}},
		{{0,1,2,3,1},{1,0,2,3,-1},{0,1,3,2,-1},{1,0,3,2,1}},
		{{0,1,2,3,1}, {1,0,2,3,1}, {0,1,3,2,1}, {1,0,3,2,1}, {2,3,0,1,1}, {3,2,0,1,1}, {2,3,1,0,1}, {3,2,1,0,1}, {0,3,2,1,-1}, {1,3,2,0,-1}, {0,2,3,1,-1}, {1,2,3,1,-1}, {2,1,0,3,-1}, {3,1,0,2,-1}, {2,0,1,3,-1}, {3,0,1,2,-1}},
		{{0,1,2,3,1},{2,1,0,3,1},{0,3,2,1,1},{2,3,0,1,1},{1,0,3,2,1},{3,0,1,2,1},{1,2,3,0,1},{3,2,1,0,1}, {0,1,3,2,-1},{2,1,3,0,-1},{0,3,1,2,-1},{2,3,1,0,-1},{1,0,2,3,-1},{3,0,2,1,-1},{1,2,0,3,-1},{3,2,0,1,-1}},
		};


template <class I, class T>
void symmetry_faktor(const fourindexsymtype symmetry,const union packed_index<I> &index, T &elem)
{
switch(symmetry)
	{
	case antisymtwoelectronrealmullikan: //in case of antisymmetry it will vanish anyway, first two  conditions apply the same
	case twoelectronrealmullikan:
		if(index.indiv.i==index.indiv.j) elem*=.5;
		if(index.indiv.k==index.indiv.l) elem*=.5;
		if(index.indiv.i==index.indiv.k && index.indiv.j==index.indiv.l
			|| index.indiv.i==index.indiv.l && index.indiv.j==index.indiv.k		) elem*=.5; 
		break;	
	case antisymtwoelectronrealdirac: //in case of antisymmetry it will vanish anyway, first two  conditions apply the same
	case twoelectronrealdirac:
		if(index.indiv.i==index.indiv.k) elem*=.5;
                if(index.indiv.j==index.indiv.l) elem*=.5;
                if(index.indiv.i==index.indiv.j && index.indiv.k==index.indiv.l
			|| index.indiv.k==index.indiv.j && index.indiv.i==index.indiv.l) elem*=.5; 
		break;
	case T2ijab_real: break; //result will automatically vanish due to generated antisymmetry; i!=a from principle
	case nosymmetry: break;
	default: laerror("illegal symmetry");
	}
}


template <class I, class T>
class fourindex {
protected:
	fourindexsymtype symmetry;
	I nn;
        int *count;
	matel4<I,T> *list;
private:
	void deletelist();
	void copylist(const matel4<I,T> *l);
public:
	//iterator
	typedef class iterator {
	private:
		matel4<I,T> *p;
	public:
		iterator() {};
		~iterator() {};
		iterator(matel4<I,T> *list): p(list) {};
		bool operator==(const iterator &rhs) const {return p==rhs.p;}
		bool operator!=(const iterator &rhs) const {return p!=rhs.p;}
		iterator &operator++() {p=p->next; return *this;}
		iterator operator++(int) {iterator q(p); p=p->next; return q;}
		matel4<I,T> & operator*()  {return *p;}
		matel4<I,T> * operator->()  {return p;}
                const matel4<I,T> * operator->() const {return p;}
                const matel4<I,T> & operator*() const {return *p;}
	};
	iterator begin() const {return list;}
	iterator end() const {return NULL;}
        
        //permiterator ... iterates also over all permutations, with a possibly scaled matrix element or skips permutations yielding equivalent result
        //has to take into account the symmetry type of the fourindex
        typedef class piterator {
	private:
		fourindexsymtype symmetry;
		matel4<I,T> *p;
		matel4<I,T> my;
		int permindex;
		void setup(void) //make a copy of *p to my with scaled element and anti/permuted indices
			{
			if(symmetry==undefined_symmetry) laerror("fourindex symmetry has not been set");
			if(!p) {permindex=0; memset(&my,0,sizeof(my)); return;}
			for(int i=0; i<4; ++i)
				my.index.packed[i] = p->index.packed[fourindex_permutations[symmetry][permindex][i]];
			my.elem = p->elem * fourindex_permutations[symmetry][permindex][4];
			//now treat the redundancy due to possibly equal indices by a scaling factor
			//if the processing of individual term becomes very costly, an alternative would be to screen permutations yielding identical result
			symmetry_faktor(symmetry, p->index, my.elem);
			};
        public:
                piterator() {};
		piterator(matel4<I,T> *pp): symmetry(nosymmetry),p(pp),permindex(0){};
                ~piterator() {};
		piterator(const fourindex &x): symmetry(x.symmetry),p(x.list),permindex(0) {setup();};
		piterator& operator++() {if(++permindex>=fourindex_permnumbers[symmetry]) {permindex=0; p=p->next;} setup(); return *this;}
                const matel4<I,T> & operator*() const {return my;}
                const matel4<I,T> * operator->() const {return &my;}
		piterator operator++(int) {laerror("postincrement not possible on permute-iterator");}
                bool operator!=(const piterator &rhs) const {return p!=rhs.p;} //should only be used for comparison with pend()
		bool end(void) {return !p;}
		bool notend(void) {return p;}
        };
        piterator pbegin() const {return piterator(*this);}
        piterator pend() const {return piterator(NULL);}//inefficient, use end() or notend() instead

	//constructors etc.
	inline fourindex() :symmetry(undefined_symmetry),nn(0),count(NULL),list(NULL) {};
	inline fourindex(const I n) :symmetry(undefined_symmetry),nn(n),count(new int(1)),list(NULL) {};
	fourindex(const fourindex &rhs); //copy constructor
	inline int getcount() const {return count?*count:0;}
	fourindex & operator=(const fourindex &rhs);
        fourindex & operator+=(const fourindex &rhs);
	void setsymmetry(fourindexsymtype s) {symmetry=s;}
	fourindexsymtype getsymmetry() const {return symmetry;}
        fourindex & join(fourindex &rhs); //more efficient +=, rhs will be emptied
	inline ~fourindex();
	inline matel4<I,T> *getlist() const {return list;}
	inline I size() const {return nn;}
	void resize(const I n);
	void copyonwrite();
	unsigned long length() const;
	inline void add(const I i, const I j, const I k, const I l, const T elem) 
		{matel4<I,T> *ltmp= new matel4<I,T>; ltmp->next=list; list=ltmp; list->index.indiv.i=i;list->index.indiv.j=j;list->index.indiv.k=k;list->index.indiv.l=l; list->elem=elem;}

	inline void add(const union packed_index<I> &index , const T elem) 
                {matel4<I,T> *ltmp= new matel4<I,T>; ltmp->next=list; list=ltmp; list->index=index; list->elem=elem;}
	
	inline void add(const I (&index)[4], const T elem)
                {matel4<I,T> *ltmp= new matel4<I,T>; ltmp->next=list; list=ltmp; memcpy(&list->index.packed, &index, sizeof(union packed_index<I>)); list->elem=elem;}
	unsigned long put(int fd,bool withattr=true) const;
	unsigned long get(int fd,bool withattr=true);
};


//and a class for accessing a disc-stored fourindex, taking care of permutational index symmetry
template <class I, class T>
class fourindex_ext {
private: //at the moment for simplicity forbid some operations, otherwise reference counting on the buffer has to be done
	fourindex_ext();
	fourindex_ext(const fourindex_ext &rhs);
	fourindex_ext & operator=(const fourindex_ext &rhs);
protected:
	matel4stored<I,T> *buffer;
	matel4stored<I,T> *current;
	int fd;
	unsigned int bufsize;
	unsigned int nread;
        fourindexsymtype symmetry;
	I nn;

	//methods
	void tryread() const
		{
		const_cast<fourindex_ext<I,T> *>(this)->current=NULL;
		ssize_t r=read(fd,buffer,bufsize*sizeof(matel4stored<I,T>));
		if(r<0) {perror("read error"); laerror("read error in fourindex_ext");}
		if(r%sizeof(matel4stored<I,T>)) laerror("read inconsistency in fourindex_ext");
		const_cast<fourindex_ext<I,T> *>(this)->nread = r/sizeof(matel4stored<I,T>);
		if(nread) const_cast<fourindex_ext<I,T> *>(this)->current=buffer;
		}
	void next() const { if(current && (unsigned int) (++ const_cast<fourindex_ext<I,T> *>(this)->current - buffer) >=nread) tryread(); }
	bool eof() const {return !current;};


public:
        fourindex_ext(const int file, const fourindexsymtype s=undefined_symmetry, const I n=0, const unsigned int b=256) :current(NULL),fd(file),bufsize(b),nread(0),symmetry(s),nn(n) {buffer = new matel4stored<I,T>[bufsize]; }
	~fourindex_ext() {if(buffer) delete[] buffer;}
	void setsymmetry(fourindexsymtype s) {symmetry=s;};
	fourindexsymtype getsymmetry() const {return symmetry;}
	void rewind() const {if(0!=lseek(fd,0L,SEEK_SET)) {perror("seek error"); laerror("cannot seek in fourindex_ext");} };
	inline I size() const {return nn;}
	
//iterator and permute-iterator are both implemented as poiters to the original class, using private functions of this class
//this is possible, since one instance of this class can have only one active iterator at a time
	
//iterator
        typedef class iterator {
        private:
                const fourindex_ext *base; 
        public:
		iterator() {};
                iterator(const fourindex_ext *p): base(p) {};
                ~iterator() {};
                bool operator!=(const iterator &rhs) const {return base!=rhs.base;} //should only be used for comparison with end()
                iterator &operator++() {if(base) base->next(); if(base->eof()) base=NULL; return *this;} 
		iterator operator++(int) {laerror("postincrement not possible");}
                const matel4stored<I,T> * operator->() const {return base->current;}
                const matel4stored<I,T> & operator*() const {return *base->current;}
		bool notNULL() const {return base;}
        };
        iterator begin() const {rewind(); tryread(); if(!eof()) return this; else return NULL;}
        iterator end() const {return iterator(NULL);}

//piterator ... iterate over all allowed permutations; conveniently expressed via the basic iterator which does the block-buffering
        typedef class piterator {
        private:
                fourindex_ext *base;
		matel4stored<I,T> my;
		int permindex;
		fourindex_ext::iterator it;

		//private methods
                void setup(void) //make a copy of *it to my with scaled element and anti/permuted indices
                        {
                        if(base->symmetry==undefined_symmetry) laerror("fourindex symmetry has not been set");
                        if(!it.notNULL()) {permindex=0; memset(&my,0,sizeof(my)); return;} //we rely that end() is NULL
                        for(int i=0; i<4; ++i)
                                my.index.packed[i] = it->index.packed[fourindex_permutations[base->symmetry][permindex][i]];
                        my.elem = it->elem * fourindex_permutations[base->symmetry][permindex][4];
                        //redundancy due to possibly equal indices
                        //if the processing of individual term becomes very costly, an alternative would be to screen permutations yielding identical result
			symmetry_faktor(base->symmetry, it->index, my.elem);
                        };
        public:
		piterator() {};
                piterator(fourindex_ext *p): base(p),permindex(0) {if(p) {it=p->begin(); setup();}};
		piterator(fourindex_ext &x): base(&x),permindex(0) {it= x.begin(); setup();};
                ~piterator() {};
                bool operator!=(const piterator &rhs) const {return base!=rhs.base;} //should only be used for comparison with end()
                piterator &operator++() {if(++permindex>=fourindex_permnumbers[base->symmetry]) {permindex=0; ++it;} if(it.notNULL()) setup(); else base=NULL; return *this;} 
		piterator operator++(int) {laerror("postincrement not possible");}
                const matel4stored<I,T> * operator->() const {return &my;}
                const matel4stored<I,T> & operator*() const {return my;}
		bool end(void) {return !base;}
                bool notend(void) {return base;}
	};
	piterator pbegin() {return piterator(*this);}
	piterator pend() const {return piterator(NULL);} //inefficient, use end() or notend() instead


};



/////////////////////////////implementations///////////////////////////////////

template <class I,class T>
unsigned long fourindex<I,T>::put(int fd, bool withattr) const
{
unsigned long n=0;
matel4<I,T> *l=list;
matel4stored<I,T> buf;
if(withattr)
        {
	union {fourindexsymtype sym; I n; T padding;} u;
	u.sym=symmetry;
	if(sizeof(u)!=write(fd,&u,sizeof(u))) laerror("write error in fourindex::put");
	u.n=nn;
	if(sizeof(u)!=write(fd,&u,sizeof(u))) laerror("write error in fourindex::put");
        }
while(l)
	{
	++n;
	buf.elem= l->elem;
	buf.index= l->index;
	if(sizeof(buf)!=write(fd,&buf,sizeof(buf))) laerror("write error in fourindex::put");
	l=l->next;
	}
return n;
}


template <class I,class T>
unsigned long fourindex<I,T>::get(int fd,bool withattr)
{
unsigned long n=0;
matel4stored<I,T> buf;
if(withattr)
	{
	union {fourindexsymtype sym; I n; T padding;} u;
        if(sizeof(u)!=read(fd,&u,sizeof(u))) laerror("read inconsistency in fourindex::put");
        symmetry=u.sym;
        if(sizeof(u)!=read(fd,&u,sizeof(u))) laerror("read inconsistency in fourindex::put");
        nn=u.n;

	}
while(sizeof(buf)==read(fd,&buf,sizeof(buf))) {++n; add(buf.index,buf.elem);}
return n;
}


//destructor
template <class I,class T>
fourindex<I,T>::~fourindex()
{
        if(!count) return;
        if(--(*count)<=0)
                {
		deletelist();
                delete count;
                }
}

//copy constructor (sort arrays are not going to be copied)
template <class I, class T>
fourindex<I,T>::fourindex(const fourindex<I,T> &rhs)
{
#ifdef debug
if(! &rhs) laerror("fourindex copy constructor with NULL argument");
#endif
        nn=rhs.nn;
	if(rhs.list&&!rhs.count) laerror("some inconsistency in fourindex contructors or assignments");
        list=rhs.list;
        if(list) {count=rhs.count; (*count)++;} else count=new int(1); //make the matrix defined, but empty and not shared
}



//assignment operator
template <class I, class T>
fourindex<I,T> & fourindex<I,T>::operator=(const fourindex<I,T> &rhs)
{
        if (this != &rhs)
                {
                if(count)
                    if(--(*count) ==0) {deletelist(); delete count;} // old stuff obsolete
                list=rhs.list;
                nn=rhs.nn; 
                if(list) count=rhs.count; else count= new int(0); //make the matrix defined, but empty and not shared, count will be incremented below
                if(count) (*count)++;
                }
        return *this;
}


template <class I, class T>
fourindex<I,T> & fourindex<I,T>::operator+=(const fourindex<I,T> &rhs)
{
if(nn!=rhs.nn) laerror("incompatible dimensions for +=");
if(!count) {count=new int;  *count=1; list=NULL;}
else copyonwrite();
register matel4<I,T> *l=rhs.list;
while(l)
        {
        add( l->index,l->elem);
        l=l->next;
        }
return *this;
}

template <class I, class T>
fourindex<I,T> & fourindex<I,T>::join(fourindex<I,T> &rhs)
{
if(nn!=rhs.nn) laerror("incompatible dimensions for join");
if(*rhs.count!=1) laerror("shared rhs in join()");
if(!count) {count=new int;  *count=1; list=NULL;}
else copyonwrite();
matel4<I,T> **last=&list;
while(*last) last= &((*last)->next);
*last=rhs.list;
rhs.list=NULL;
return *this;
}

template <class I, class T>
void fourindex<I,T>::resize(const I n)
{
        if(n<=0 ) laerror("illegal fourindex dimension");
	if(count)
                {
                if(*count > 1) {(*count)--; count=NULL; list=NULL;} //detach from previous
                else if(*count==1) deletelist();
                }
        nn=n;
        count=new int(1); //empty but defined matrix
        list=NULL;
}


template <class I, class T>
void fourindex<I,T>::deletelist()
{
if(*count >1) laerror("trying to delete shared list");
matel4<I,T> *l=list;
while(l)
        {
        matel4<I,T> *ltmp=l;
        l=l->next;
        delete ltmp;
        }
list=NULL;
delete count;
count=NULL;
}

template <class I, class T>
void fourindex<I,T>::copylist(const matel4<I,T> *l)
{
list=NULL;
while(l)
        {
        add(l->index,l->elem);
        l=l->next;
        }
}

template <class I, class T>
void fourindex<I,T>::copyonwrite()
{
        if(!count) laerror("probably an assignment to undefined fourindex");
        if(*count > 1)
                {
                (*count)--;
                count = new int; *count=1;
                if(!list) laerror("empty list with count>1");
                copylist(list);
                }
}

template <class I, class T>
unsigned long fourindex<I,T>::length() const
{
unsigned long n=0;
matel4<I,T> *l=list;
while(l)
	{
	++n;
	l=l->next;
	}
return n;
}

template <class I, class T>
ostream& operator<<(ostream &s, const fourindex_ext<I,T> &x)
                {
                int n;
                n=x.size();
                s << n << '\n';
                typename fourindex<I,T>::iterator it=x.begin();
                while(it!=x.end())
                        {
                        s << (typename LA_traits_io<I>::IOtype)it->index.indiv.i << ' ' << (typename LA_traits_io<I>::IOtype)it->index.indiv.j<<  ' ' <<(typename LA_traits_io<I>::IOtype)it->index.indiv.k << ' ' << (typename LA_traits_io<I>::IOtype)it->index.indiv.l  << ' ' << (typename LA_traits_io<T>::IOtype) it->elem << '\n';
                        ++it;
                        }
                s << "-1 -1 -1 -1\n";
                return s;
                }



template <class I, class T>
ostream& operator<<(ostream &s, const fourindex<I,T> &x)
                {
                int n;
                n=x.size();
                s << n << '\n';
                typename fourindex<I,T>::iterator it=x.begin(),end=x.end();
                while(it!=end)
                        {
                        s << (typename LA_traits_io<I>::IOtype)it->index.indiv.i << ' ' << (typename LA_traits_io<I>::IOtype)it->index.indiv.j<<  ' ' <<(typename LA_traits_io<I>::IOtype)it->index.indiv.k << ' ' << (typename LA_traits_io<I>::IOtype)it->index.indiv.l  << ' ' << (typename LA_traits_io<T>::IOtype) it->elem << '\n';
			++it;
                        }
                s << "-1 -1 -1 -1\n";
                return s;
                }

template <class I, class T>
istream& operator>>(istream  &s, fourindex<I,T> &x)
                {
                typename LA_traits_io<I>::IOtype i,j,k,l;
		typename LA_traits_io<T>::IOtype elem;
		int n;
                s >> n ;
                x.resize(n);
                s >> i >> j >>k >>l;
                while(i!= (typename LA_traits_io<I>::IOtype)-1 && j!= (typename LA_traits_io<I>::IOtype)-1 &&  k != (typename LA_traits_io<I>::IOtype)-1 && l!= (typename LA_traits_io<I>::IOtype)-1)
                        {
			s>>elem;
			x.add((I)i,(I)j,(I)k,(I)l,(T)elem);
			s >> i >> j >>k >>l;
                        }
                return s;
                }

/////////////////////densely stored fourindex///////////////////////////////////
//not all symmetry cases implemented yet, but a general template declaration used
//we use a general template forward declaration, but then it has to be done differently for (almost) each case
//by means of partial template specialization

//note - loops for the twoelectronrealmullikan integral to be unique and in canonical order
// i=1..n, j=1..i, k=1..i, l=1..(i==k?j:k)

template<fourindexsymtype S, class T, class DUMMY> class fourindex_dense;

//make it as a derived class in order to be able to use it in a base class context - "supermatrix" operations
template<class T, class I> 
class fourindex_dense<twoelectronrealmullikan,T,I> : public NRSMat<T> {
public:
	fourindex_dense(): NRSMat<T>() {};
	explicit fourindex_dense(const int n): NRSMat<T>(n*(n+1)/2) {};
	fourindex_dense(const NRSMat<T> &rhs): NRSMat<T>(rhs) {}; //be able to convert the parent class  transparently to this
	fourindex_dense(const T &a, const int n): NRSMat<T>(a,n*(n+1)/2) {};
	fourindex_dense(const T *a, const int n): NRSMat<T>(a,n*(n+1)/2) {};
	//and also construct it from sparse and externally stored fourindex classes
	//it seems not possible to nest template<class I> just for the two constructors
	fourindex_dense(const fourindex<I,T> &rhs);
	fourindex_dense(const fourindex_ext<I,T> &rhs);

	T& operator() (unsigned int i, unsigned int j, unsigned int k, unsigned int l);
	const T& operator() (unsigned int i, unsigned int j, unsigned int k, unsigned int l) const;

};

template<class T, class I> 
fourindex_dense<twoelectronrealmullikan,T,I>::fourindex_dense<twoelectronrealmullikan,T,I>(const fourindex<I,T> &rhs) : NRSMat<T>((T)0,rhs.size()*(rhs.size()+1)/2)
{
if(rhs.getsymmetry() != twoelectronrealmullikan ) laerror("fourindex_dense symmetry mismatch");
typename fourindex<I,T>::iterator p;
for(p=rhs.begin(); p!= rhs.end(); ++p) (*this)(p->index.indiv.i,p->index.indiv.j,p->index.indiv.k,p->index.indiv.l) = p->elem;
}

template<class T, class I>
fourindex_dense<twoelectronrealmullikan,T,I>::fourindex_dense<twoelectronrealmullikan,T,I>(const fourindex_ext<I,T> &rhs) : NRSMat<T>((T)0,rhs.size()*(rhs.size()+1)/2)
{
if(rhs.getsymmetry() != twoelectronrealmullikan ) laerror("fourindex_dense symmetry mismatch");
typename fourindex_ext<I,T>::iterator p;
for(p=rhs.begin(); p!= rhs.end(); ++p) (*this)(p->index.indiv.i,p->index.indiv.j
,p->index.indiv.k,p->index.indiv.l) = p->elem;
}



template<class T, class DUMMY>
T& fourindex_dense<twoelectronrealmullikan,T,DUMMY>::operator() (unsigned int i, unsigned int j, unsigned int k, unsigned int l)
{
unsigned long I = SMat_index_1(i,j);
unsigned long J = SMat_index_1(k,l);
//I,J act as indices of a NRSmat
#ifdef DEBUG
     	if (*count != 1) laerror("lval (i,j,k,l) with count > 1 in fourindex_dense");
       	if (I<0 || I>=(unsigned long)nn || J<0 || J>=(unsigned long)nn) laerror("fourindex_dense index out of range");
       	if (!v) laerror("access to unallocated fourindex_dense");
#endif
return v[SMat_index(I,J)];
}

template<class T, class DUMMY>
const T& fourindex_dense<twoelectronrealmullikan,T,DUMMY>::operator() (unsigned int i, unsigned int j, unsigned int k, unsigned int l) const
{
unsigned long I = SMat_index_1(i,j);
unsigned long J = SMat_index_1(k,l);
//I,J act as indices of a NRSmat
#ifdef DEBUG
       	if (I<0 || I>=(unsigned long)nn || J<0 || J>=(unsigned long)nn) laerror("fourindex_dense index out of range");
       	if (!v) laerror("access to unallocated fourindex_dense");
#endif
return v[SMat_index(I,J)];
}




#endif /*_fourindex_included*/
