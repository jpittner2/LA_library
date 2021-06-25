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

#ifndef _fourindex_included
#define _fourindex_included
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/vfs.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdint.h>
#include <sys/stat.h>
#include "laerror.h"
#include "vec.h"
#include "smat.h"
#include "mat.h"
#include "nonclass.h"

namespace LA {

static unsigned int hcd0(unsigned int big,unsigned int small)
{
register unsigned int help;


if(big==0)
        {
        if(small==0) laerror("bad arguments in hcd");
        return small;
        }
if(small==0) return big;
if(small==1||big==1) return 1;

if(small>big) {help=big; big=small; small=help;}
do      {
        help=small;
        small= big%small;
        big=help;
        }
while(small != 0);
return big;
}


static inline unsigned int lcm0(unsigned int i,unsigned int j)
{
return (i/hcd0(i,j)*j);
}


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
        }
#ifdef __GNUC__
__attribute__((packed))
#endif
;


//later add symmetry of complex integrals
typedef enum {
	undefined_symmetry=-1,
	nosymmetry=0, 
	twoelectronrealmullikan=1, twoelectronrealmullikanAA=1,
	twoelectronrealdirac=2, 
	T2ijab_aces=3, 
	trdm2AA=3, 
	antisymtwoelectronrealdirac=4, rdm2AA=4, 
	T2IjAb_aces=5, trdm2AB=5, 
	twoelectronrealmullikanAB=6, 
	T2ijab_unitary=7, 
	T2IjAb_unitary=8, 
	antisymtwoelectronrealdiracAB=9,rdm2AB=9,
	twoelectronrealmullikanreducedsymAA=10,
	twoelectronrealmullikanreducedsymAB=11,
} fourindexsymtype; //only permutation-nonequivalent elements are stored

// these should actually be static private members of the fourindex class, but leads to an ICE on gcc3.2
static const int fourindex_n_symmetrytypes=12;
static const int fourindex_permnumbers[fourindex_n_symmetrytypes]={1,8,8,4,8,1,4,8,2,2,4,2};
static const int fourindex_permutations[fourindex_n_symmetrytypes][9][5]=
		{
		{{0,1,2,3,1}}, //nosymmetry
		{{0,1,2,3,1}, {1,0,2,3,1}, {0,1,3,2,1}, {1,0,3,2,1}, {2,3,0,1,1}, {3,2,0,1,1}, {2,3,1,0,1}, {3,2,1,0,1}}, //twoelectronrealmullikan
		{{0,1,2,3,1},{2,1,0,3,1},{0,3,2,1,1},{2,3,0,1,1},{1,0,3,2,1},{3,0,1,2,1},{1,2,3,0,1},{3,2,1,0,1}}, //twoelectronrealdirac
		{{0,1,2,3,1},{1,0,2,3,-1},{0,1,3,2,-1},{1,0,3,2,1}}, //T2ijab_aces
		{{0,1,2,3,1},{1,0,2,3,-1},{0,1,3,2,-1},{1,0,3,2,1},{2,3,0,1,1},{3,2,0,1,-1},{2,3,1,0,-1},{3,2,1,0,1}}, //antisymtwoelectronrealdirac
		{{0,1,2,3,1}}, //T2IjAb_aces is like nosymmetry but different index ranges
		{{0,1,2,3,1},{1,0,2,3,1},{0,1,3,2,1},{1,0,3,2,1}}, //twoelectronrealmullikanAB
		{{0,1,2,3,1},{1,0,2,3,-1},{0,1,3,2,-1},{1,0,3,2,1},{2,3,0,1,-1},{3,2,0,1,1},{2,3,1,0,1},{3,2,1,0,-1}}, //T2ijab_unitary (antiherm exponent)
		{{0,1,2,3,1},{2,3,0,1,-1}}, //T2IjAb_unitary (antihermitian exponent)
		{{0,1,2,3,1},{2,3,0,1,1}}, //antisymtwoelectronrealdiracAB
		{{0,1,2,3,1},{2,3,0,1,1},{1,0,3,2,1},{3,2,1,0,1}}, //twoelectronrealmullikanreducedsymAA (e.g. unitary downfolded CC eff. integrals)
		{{0,1,2,3,1},{1,0,3,2,1}}, //twoelectronrealmullikanreducedsymAB (e.g. unitary downfolded CC eff. integrals)
		};


template <class I, class T>
void symmetry_faktor(const fourindexsymtype symmetry,const union packed_index<I> &index, T &elem)
{
switch(symmetry)
	{
	case twoelectronrealmullikanreducedsymAA:
		if(index.indiv.i==index.indiv.j && index.indiv.k==index.indiv.l) elem*=.5;
		if(index.indiv.i==index.indiv.k && index.indiv.j==index.indiv.l) elem*=.5;
		break;
	case twoelectronrealmullikanreducedsymAB:
		if(index.indiv.i==index.indiv.j && index.indiv.k==index.indiv.l) elem*=.5;
		break;
	case antisymtwoelectronrealdirac: 
	case antisymtwoelectronrealdiracAB: 
		laerror("not implemented");
	case twoelectronrealmullikan:
		if(index.indiv.i==index.indiv.j) elem*=.5;
		if(index.indiv.k==index.indiv.l) elem*=.5;
		if(index.indiv.i==index.indiv.k && index.indiv.j==index.indiv.l
			|| index.indiv.i==index.indiv.l && index.indiv.j==index.indiv.k		) elem*=.5; 
		break;	
	case twoelectronrealdirac:
		if(index.indiv.i==index.indiv.k) elem*=.5;
                if(index.indiv.j==index.indiv.l) elem*=.5;
                if(index.indiv.i==index.indiv.j && index.indiv.k==index.indiv.l
			|| index.indiv.k==index.indiv.j && index.indiv.i==index.indiv.l) elem*=.5; 
		break;
	case T2ijab_aces: break; //result will automatically vanish due to generated antisymmetry; i!=a from principle
	case T2IjAb_aces: break; //no actual symmetry
	case nosymmetry: break;
	default: laerror("illegal symmetry or symmetry-redundant scaling factor not implemented");
	}
}


template <class I, class T>
class fourindex {
protected:
	fourindexsymtype symmetry;
	I nn;
        int *count;
	matel4<I,T> *list;
	I terminator;
	bool doscaling;
private:
	void deletelist();
	void copylist(const matel4<I,T> *l);
public:
	//iterator
	class iterator {
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
	void setterminator(const I terminator0) {terminator=terminator0;}
	I getterminator() const {return terminator;}
	void setscaling(const bool doscaling0) {doscaling=doscaling0;}
	iterator begin() const {return list;}
	iterator end() const {return NULL;}
        
        //permiterator ... iterates also over all permutations, with a possibly scaled matrix element for redundant results of the permutations
        //it is assumed the original fourindex is the symmetry-reduced petite list, then the results of piterator contributions can be accumulated
        //has to take into account the symmetry type of the fourindex
        class piterator {
	private:
		fourindexsymtype symmetry;
		matel4<I,T> *p;
		matel4<I,T> my;
		int permindex;
		bool doscaling;
		void setup(void) //make a copy of *p to my with scaled element and anti/permuted indices
			{
			if(symmetry==undefined_symmetry) laerror("fourindex symmetry has not been set");
			if(!p) {permindex=0; memset(&my,0,sizeof(my)); return;}
			for(int i=0; i<4; ++i)
				my.index.packed[i] = p->index.packed[fourindex_permutations[symmetry][permindex][i]];
			my.elem = p->elem * fourindex_permutations[symmetry][permindex][4];
			//now treat the redundancy due to possibly equal indices by a scaling factor
			//if the processing of individual term becomes very costly, an alternative would be to screen permutations yielding identical result
			if(doscaling) symmetry_faktor(symmetry, p->index, my.elem);
			};
        public:
                piterator() {};
		piterator(matel4<I,T> *pp): symmetry(nosymmetry),p(pp),permindex(0),doscaling(true){};
                ~piterator() {};
		piterator(const fourindex &x): symmetry(x.symmetry),p(x.list),permindex(0),doscaling(x.doscaling) {setup();};
		piterator& operator++() {if(++permindex>=fourindex_permnumbers[symmetry]) {permindex=0; p=p->next;} setup(); return *this;}
                const matel4<I,T> & operator*() const {return my;}
                const matel4<I,T> * operator->() const {return &my;}
		piterator operator++(int) {laerror("postincrement not possible on permute-iterator"); return *this;}
                bool operator!=(const piterator &rhs) const {return p!=rhs.p;} //should only be used for comparison with pend()
		bool end(void) {return !p;}
		bool notend(void) {return p;}
        };
        piterator pbegin() const {return piterator(*this);}
        piterator pend() const {return piterator(NULL);}//inefficient, use end() or notend() instead

	//constructors etc.
	inline fourindex() :symmetry(undefined_symmetry),nn(0),count(NULL),list(NULL),terminator(-1),doscaling(true) {};
	inline fourindex(const I n, const fourindexsymtype symmetry0=undefined_symmetry, const I terminator0= -1, const bool doscaling0=true) :nn(n),count(new int(1)),list(NULL),symmetry(symmetry0),terminator(terminator0),doscaling(doscaling0) {};
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
	void dealloc(void) {resize(0);}
	void copyonwrite();
	unsigned long length() const;
	inline void add(const I i, const I j, const I k, const I l, const T elem) 
		{matel4<I,T> *ltmp= new matel4<I,T>; ltmp->next=list; list=ltmp; list->index.indiv.i=i;list->index.indiv.j=j;list->index.indiv.k=k;list->index.indiv.l=l; list->elem=elem;}

	inline void add(const union packed_index<I> &index , const T elem) 
                {matel4<I,T> *ltmp= new matel4<I,T>; ltmp->next=list; list=ltmp; list->index=index; list->elem=elem;}
	
	inline void add(const I (&index)[4], const T elem)
                {matel4<I,T> *ltmp= new matel4<I,T>; ltmp->next=list; list=ltmp; memcpy(&list->index.packed, &index, sizeof(union packed_index<I>)); list->elem=elem;}
	inline void add(const matel4<I,T> &rhs)
		{matel4<I,T> *ltmp= new matel4<I,T>; ltmp->next=list; list=ltmp; memcpy(&list->index.packed, &rhs.index, sizeof(union packed_index<I>)); list->elem=rhs.elem;}
	inline void add(const matel4stored<I,T> &rhs)
		{matel4<I,T> *ltmp= new matel4<I,T>; ltmp->next=list; list=ltmp; memcpy(&list->index.packed, &rhs.index, sizeof(union packed_index<I>)); list->elem=rhs.elem;}
	unsigned long put(int fd,bool withattr=true) const;
	unsigned long get(int fd,bool withattr=true);
	void fscanf(FILE *f); //C-style formatted IO
	void fprintf(FILE *f, char *format) const;
	void permuteme(const NRPerm<int> &p, const bool inverse=false)// index permutation
		{
		if(p.size()!=nn)  laerror("inconsistent dimension in fourindex::permuteme");
		copyonwrite();
		matel4<I,T> *l=list;
		NRPerm<int> pp; if(inverse) pp=p.inverse(); else pp=p;
		while(l)
        		{
        		l->index.indiv.i = (I) pp[(int)l->index.indiv.i];
        		l->index.indiv.j = (I) pp[(int)l->index.indiv.j];
        		l->index.indiv.k = (I) pp[(int)l->index.indiv.k];
        		l->index.indiv.l = (I) pp[(int)l->index.indiv.l];
        		l=l->next;
        		}
		}
};


//and a class for accessing a disc-stored fourindex, taking care of permutational index symmetry
//O_DIRECT approach to avoid filling of the buffer cache when reading
//large file sequentially is implemented:
//the user of the class must open the file with O_DIRECT
//NOTE!!! it will not work on linux 2.4, where O_DIRECT requires filesize to be a multiple of the block; 2.6 kernel is necessary!!!
//it used to work on older 2.6 kernels, but now does not work again since there is restriction to 512-byte alignment

template <class I, class T>
class fourindex_ext {
private: //at the moment for simplicity forbid some operations, otherwise reference counting on the buffer has to be done
	fourindex_ext();
	fourindex_ext(const fourindex_ext &rhs);
	fourindex_ext & operator=(const fourindex_ext &rhs);
protected:
	matel4stored<I,T> *buffer0;
	matel4stored<I,T> *buffer;
	matel4stored<I,T> *current;
	int fd;
	unsigned int bufsize;
	unsigned int nread;
        fourindexsymtype symmetry;
	I nn;
	bool doscaling;
	I terminator;

	//methods
	void tryread() const
		{
		const_cast<fourindex_ext<I,T> *>(this)->current=NULL;
		errno=0;
		//printf("read %d %llx %d\n",fd,buffer,bufsize*sizeof(matel4stored<I,T>));
		ssize_t r=read(fd,buffer,bufsize*sizeof(matel4stored<I,T>));
		if(r<0) {perror("read error"); laerror("read error in fourindex_ext (might be bug of O_DIRECT)");}
		if(r%sizeof(matel4stored<I,T>)) laerror("read inconsistency in fourindex_ext");
		const_cast<fourindex_ext<I,T> *>(this)->nread = r/sizeof(matel4stored<I,T>);
		if(nread) const_cast<fourindex_ext<I,T> *>(this)->current=buffer;
		}
	void next() const { 
			   if(current)
				{
			 	if( (unsigned int) (++ const_cast<fourindex_ext<I,T> *>(this)->current - buffer) >=nread) tryread(); 
				}
			  }
	bool eof() const {return !current;};


public:
        void setterminator(const I terminator0) {terminator=terminator0;}
	I getterminator() const {return terminator;}
       	void setscaling(const bool doscaling0) {doscaling=doscaling0;}
	void resize(I n) {nn=n;}
        fourindex_ext(const int file, const fourindexsymtype s=undefined_symmetry, const I n=0, const unsigned int b=1024, const I terminator0= -1, const bool doscaling0= true) :current(NULL),fd(file),nread(0),symmetry(s),nn(n) 
		{
		struct statfs sfs;
		struct stat64 sf;
		if(fstat64(fd,&sf)) {perror("cannot fstat");laerror("I/O error");}
		if(fstatfs(fd,&sfs)) {perror("cannot fstatfs");laerror("I/O error");}
		const unsigned int pagesize=getpagesize();
		//make bufsize*sizeof(matel4stored<I,T>) a multiple of fs block size and page size
		bufsize=b*sizeof(matel4stored<I,T>);
		bufsize=lcm0(bufsize,pagesize);
		bufsize=lcm0(bufsize,sfs.f_bsize);
		bufsize=lcm0(bufsize,sf.st_blksize);
		buffer0 = new matel4stored<I,T>[(bufsize+pagesize)/sizeof(matel4stored<I,T>)+1]; //ensure alignment at page boundary
		unsigned char *buf= (unsigned char *) buffer0;
		buf= buf + pagesize - ((uint64_t)buf % pagesize);
		buffer = (matel4stored<I,T> *) buf;
		mlock(buf,bufsize); //ignore error when not root, hope we will not be paged out anyway
		bufsize /= sizeof(matel4stored<I,T>);
		terminator=terminator0;
		doscaling=doscaling0;
		}
	~fourindex_ext() {if(buffer0) delete[] buffer0;}
	void setsymmetry(fourindexsymtype s) {symmetry=s;};
	fourindexsymtype getsymmetry() const {return symmetry;}
	void rewind() const {if(0!=lseek64(fd,0L,SEEK_SET)) {perror("seek error"); laerror("cannot seek in fourindex_ext");} };

	//file output
	void put(const matel4stored<I,T> x)
		{
		if(!current) current=buffer;
		*current++ = x;
		if(current-buffer >= bufsize ) flush();
		}
	void put(I i, I j, I k, I l, const T &elem)
		{
		if(!current) current=buffer;
		current->index.indiv.i=i;
		current->index.indiv.j=j;
		current->index.indiv.k=k;
		current->index.indiv.l=l;
		current->elem = elem;
		++current;
		if(current-buffer >= bufsize ) flush();
		}
	void flush()
		{
		if(current)
			{
			ssize_t r=write(fd,buffer,(current-buffer)*sizeof(matel4stored<I,T>));
			if(r!=(current-buffer)*sizeof(matel4stored<I,T>)) laerror("write error in fourindex_ext");
			}
		current=NULL;
		}

	inline I size() const {return nn;}
	

//iterator and permute-iterator are both implemented as poiters to the original class, using private functions of this class
//this is possible, since one instance of this class can have only one active iterator at a time
	
//iterator
        class iterator {
        private:
                const fourindex_ext *base; 
        public:
		iterator() {};
                iterator(const fourindex_ext *p): base(p) {};
                ~iterator() {};
                bool operator!=(const iterator &rhs) const {return base!=rhs.base;} //should only be used for comparison with end()
                iterator &operator++() {if(base) base->next(); if(base->eof()) base=NULL; return *this;} 
		iterator operator++(int) {laerror("postincrement not possible"); return *this;}
                const matel4stored<I,T> * operator->() const {return base->current;}
                const matel4stored<I,T> & operator*() const {return *base->current;}
		bool notNULL() const {return base;}
        };
        iterator begin() const {rewind(); tryread(); if(!eof()) return this; else return NULL;}
        iterator end() const {return iterator(NULL);}


//piterator ... iterate over all allowed permutations; conveniently expressed via the basic iterator which does the block-buffering
        class piterator {
        private:
                fourindex_ext *base;
		matel4<I,T> my;
		int permindex;
		typename fourindex_ext::iterator it;

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
			if(base->doscaling) symmetry_faktor(base->symmetry, it->index, my.elem);
                        };
        public:
		piterator() {};
                piterator(fourindex_ext *p): base(p),permindex(0) {if(p) {it=p->begin(); setup();}};
		piterator(fourindex_ext &x): base(&x),permindex(0) {it= x.begin(); setup();};
                ~piterator() {};
                bool operator!=(const piterator &rhs) const {return base!=rhs.base;} //should only be used for comparison with end()
                piterator &operator++() {if(++permindex>=fourindex_permnumbers[base->symmetry]) {permindex=0; ++it;} if(it.notNULL()) setup(); else base=NULL; return *this;} 
		piterator operator++(int) {laerror("postincrement not possible"); return *this;}
                const matel4<I,T> * operator->() const {return &my;}
                const matel4<I,T> & operator*() const {return my;}
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
	terminator=rhs.terminator;
	doscaling=rhs.doscaling;
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
		terminator=rhs.terminator;
        	doscaling=rhs.doscaling;
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
std::ostream& operator<<(std::ostream &s, const fourindex_ext<I,T> &x)
                {
                int n;
                n=x.size();
                s << n << '\n';
                typename fourindex_ext<I,T>::iterator it=x.begin();
                while(it!=x.end())
                        {
                        s << (typename LA_traits_io<I>::IOtype)it->index.indiv.i << ' ' << (typename LA_traits_io<I>::IOtype)it->index.indiv.j<<  ' ' <<(typename LA_traits_io<I>::IOtype)it->index.indiv.k << ' ' << (typename LA_traits_io<I>::IOtype)it->index.indiv.l  << ' ' << (typename LA_traits_io<T>::IOtype) it->elem << '\n';
                        ++it;
                        }
		s << (typename LA_traits_io<I>::IOtype) x.getterminator() << ' ' << (typename LA_traits_io<I>::IOtype)x.getterminator() <<  ' ' <<(typename LA_traits_io<I>::IOtype)x.getterminator() << ' ' << (typename LA_traits_io<I>::IOtype)x.getterminator()  << ' ' << (typename LA_traits_io<T>::IOtype) 0 << '\n';
                return s;
                }



template <class I, class T>
std::ostream& operator<<(std::ostream &s, const fourindex<I,T> &x)
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
		s << (typename LA_traits_io<I>::IOtype)x.getterminator() << ' ' << (typename LA_traits_io<I>::IOtype)x.getterminator() <<  ' ' <<(typename LA_traits_io<I>::IOtype)x.getterminator() << ' ' << (typename LA_traits_io<I>::IOtype)x.getterminator()  << ' ' << (typename LA_traits_io<T>::IOtype) 0 << '\n';
                return s;
                }

template <class I, class T>
std::istream& operator>>(std::istream  &s, fourindex<I,T> &x)
                {
                typename LA_traits_io<I>::IOtype i,j,k,l;
		typename LA_traits_io<T>::IOtype elem;
		int n;
                s >> n ;
                x.resize(n);
                s >> i >> j >>k >>l;
                while(i!= (typename LA_traits_io<I>::IOtype)x.getterminator() && j!= (typename LA_traits_io<I>::IOtype)x.getterminator() &&  k != (typename LA_traits_io<I>::IOtype)x.getterminator() && l!= (typename LA_traits_io<I>::IOtype)x.getterminator())
                        {
			s>>elem;
			x.add((I)i,(I)j,(I)k,(I)l,(T)elem);
			s >> i >> j >>k >>l;
                        }
                return s;
                }

template <class I, class T>
std::istream& operator>>(std::istream &s, fourindex_ext<I,T> &x)
                {
                int n;
                s >> n;
                x.resize(n);
		typename LA_traits_io<I>::IOtype i,j,k,l;
                typename LA_traits_io<T>::IOtype elem;
		
		s >> i >> j >>k >>l;
                while(i!= (typename LA_traits_io<I>::IOtype)x.getterminator() && j!= (typename LA_traits_io<I>::IOtype)x.getterminator() &&  k != (typename LA_traits_io<I>::IOtype)x.getterminator() && l!= (typename LA_traits_io<I>::IOtype)x.getterminator())
                        {
                        s>>elem;
                        x.put((I)i,(I)j,(I)k,(I)l,(T)elem);
                        s >> i >> j >>k >>l;
                        }
	
		x.flush();
                return s;
                }



/////////////////////densely stored fourindex///////////////////////////////////
//not all symmetry cases implemented yet, but a general template declaration used
//we use a general template forward declaration, but then it has to be done differently for (almost) each case
//by means of partial template specialization

//note - loops for the twoelectronrealmullikan integral to be unique and in canonical order
// i=1..n, j=1..i, k=1..i, l=1..(i==k?j:k)

//general template declaration
template<fourindexsymtype S, class T, class DUMMY> class fourindex_dense;

//traits class
template<fourindexsymtype S, class T, class DUMMY>
struct LA_traits<fourindex_dense<S,T,DUMMY> > {
typedef T elementtype;
typedef typename LA_traits<T>::normtype normtype;
};


//make it as a derived class in order to be able to use it in a base class context - "supermatrix" operations
template<class T, class I> 
class fourindex_dense<twoelectronrealmullikanAB,T,I> : public NRMat<T> {
public:
	fourindex_dense(): NRMat<T>() {};
	explicit fourindex_dense(const int n): NRMat<T>(n*(n+1)/2,n*(n+1)/2) {};
	fourindex_dense(const NRMat<T> &rhs): NRMat<T>(rhs) {}; //be able to convert the parent class  transparently to this
	fourindex_dense(const T &a, const int n): NRMat<T>(a,n*(n+1)/2,n*(n+1)/2) {};
	fourindex_dense(const T *a, const int n): NRMat<T>(a,n*(n+1)/2,n*(n+1)/2) {};
	//and also construct it from sparse and externally stored fourindex classes
	//it seems not possible to nest template<class I> just for the two constructors
	fourindex_dense(const fourindex<I,T> &rhs);
	fourindex_dense(const fourindex_ext<I,T> &rhs);

	T& operator() (unsigned int i, unsigned int j, unsigned int k, unsigned int l);
	const T& operator() (unsigned int i, unsigned int j, unsigned int k, unsigned int l) const;
	void resize(const int n) {(*this).NRMat<T>::resize(n*(n+1)/2,n*(n+1)/2);};
	void putext(int f, T thr=1e-15);
	int nbas() const {return (int)std::sqrt(2*(*this).nrows());};

};

template<class T, class I>
void fourindex_dense<twoelectronrealmullikanAB,T,I>::putext(int f, T thr)
{
T y;
for(int i=1; i<=nbas(); ++i) for(int j=1; j<=i; ++j)
        for(int k=1; k<=nbas(); ++k) for(int l=1; l<=k; ++l)
		if((y=abs((*this)(i,j,k,l))) > thr)
			{
			matel4stored<I,T> x;
			x.elem= y;
			x.index.indiv.i=i;
			x.index.indiv.j=j;
			x.index.indiv.k=k;
			x.index.indiv.l=l;
			if(sizeof(matel4stored<I,T>) != write(f,&x,sizeof(matel4stored<I,T>)) )
				laerror("write error in putext");
			}
}


template<class T, class I> 
fourindex_dense<twoelectronrealmullikanAB,T,I>::fourindex_dense(const fourindex<I,T> &rhs) : NRMat<T>((T)0,rhs.size()*(rhs.size()+1)/2,rhs.size()*(rhs.size()+1)/2)
{
if(rhs.getsymmetry() != twoelectronrealmullikanAB ) laerror("fourindex_dense symmetry mismatch");
typename fourindex<I,T>::iterator p;
#ifdef DEBUG
unsigned int IJ = SMat_index_1(p->index.indiv.i,p->index.indiv.j);
unsigned int KL = SMat_index_1(p->index.indiv.k,p->index.indiv.l);
if (IJ<0 || IJ>=(unsigned int)NRMat<T>::nn || KL<0 || KL>=(unsigned int)NRMat<T>::mm) laerror("fourindex_dense index out of range in constructor");
#endif
for(p=rhs.begin(); p!= rhs.end(); ++p) (*this)(p->index.indiv.i,p->index.indiv.j,p->index.indiv.k,p->index.indiv.l) = p->elem;
}

template<class T, class I>
fourindex_dense<twoelectronrealmullikanAB,T,I>::fourindex_dense(const fourindex_ext<I,T> &rhs) : NRMat<T>((T)0,rhs.size()*(rhs.size()+1)/2,rhs.size()*(rhs.size()+1)/2)
{
if(rhs.getsymmetry() != twoelectronrealmullikanAB ) laerror("fourindex_dense symmetry mismatch");
typename fourindex_ext<I,T>::iterator p;
for(p=rhs.begin(); p!= rhs.end(); ++p) 
	{
#ifdef DEBUG
unsigned int IJ = SMat_index_1(p->index.indiv.i,p->index.indiv.j);
unsigned int KL = SMat_index_1(p->index.indiv.k,p->index.indiv.l);
if (IJ<0 || IJ>=(unsigned int)NRMat<T>::nn || KL<0 || KL>=(unsigned int)NRMat<T>::mm) laerror("fourindex_dense index out of range in constructor");
#endif
	(*this)(p->index.indiv.i,p->index.indiv.j ,p->index.indiv.k,p->index.indiv.l) = p->elem;
	}
}



template<class T, class DUMMY>
T& fourindex_dense<twoelectronrealmullikanAB,T,DUMMY>::operator() (unsigned int i, unsigned int j, unsigned int k, unsigned int l)
{
int I = SMat_index_1(i,j);
int J = SMat_index_1(k,l);
//I,J act as indices of a NRmat
#ifdef DEBUG
     	if (*NRMat<T>::count != 1) laerror("lval (i,j,k,l) with count > 1 in fourindex_dense");
       	if (I<0 || I>=NRMat<T>::nn || J<0 || J>=NRMat<T>::mm) laerror("fourindex_dense index out of range");
       	if (!NRMat<T>::v) laerror("access to unallocated fourindex_dense");
#endif
return NRMat<T>::operator()(I,J);
}


template<class T, class DUMMY>
const T& fourindex_dense<twoelectronrealmullikanAB,T,DUMMY>::operator() (unsigned int i, unsigned int j, unsigned int k, unsigned int l) const
{
int I = SMat_index_1(i,j);
int J = SMat_index_1(k,l);
//I,J act as indices of a NRSmat
#ifdef DEBUG
       	if (I<0 || I>=NRMat<T>::nn || J<0 || J>=NRMat<T>::mm) laerror("fourindex_dense index out of range");
       	if (!NRMat<T>::v) laerror("access to unallocated fourindex_dense");
#endif
return NRMat<T>::operator()(I,J);
}


////////////////////
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
	void resize(const int n) {(*this).NRSMat<T>::resize(n*(n+1)/2);};
	void putext(int f, T thr=1e-15);
	int nbas() const {return (int)std::sqrt(2*(*this).nrows());};

};

template<class T, class I>
void fourindex_dense<twoelectronrealmullikan,T,I>::putext(int f, T thr)
{
T y;
for(int i=1; i<=nbas(); ++i) for(int j=1; j<=i; ++j)
        for(int k=1; k<=i; ++k) for(int l=1; l<=(i==k?j:k); ++l)
		if((y=abs((*this)(i,j,k,l))) > thr)
			{
			matel4stored<I,T> x;
			x.elem= y;
			x.index.indiv.i=i;
			x.index.indiv.j=j;
			x.index.indiv.k=k;
			x.index.indiv.l=l;
			if(sizeof(matel4stored<I,T>) != write(f,&x,sizeof(matel4stored<I,T>)) )
				laerror("write error in putext");
			}
}


template<class T, class I> 
fourindex_dense<twoelectronrealmullikan,T,I>::fourindex_dense(const fourindex<I,T> &rhs) : NRSMat<T>((T)0,rhs.size()*(rhs.size()+1)/2)
{
if(rhs.getsymmetry() != twoelectronrealmullikan ) laerror("fourindex_dense symmetry mismatch");
typename fourindex<I,T>::iterator p;
#ifdef DEBUG
unsigned int IJ = SMat_index_1(p->index.indiv.i,p->index.indiv.j);
unsigned int KL = SMat_index_1(p->index.indiv.k,p->index.indiv.l);
if (IJ<0 || IJ>=(unsigned int)NRSMat<T>::nn || KL<0 || KL>=(unsigned int)NRSMat<T>::nn) laerror("fourindex_dense index out of range in constructor");
#endif
for(p=rhs.begin(); p!= rhs.end(); ++p) (*this)(p->index.indiv.i,p->index.indiv.j,p->index.indiv.k,p->index.indiv.l) = p->elem;
}

template<class T, class I>
fourindex_dense<twoelectronrealmullikan,T,I>::fourindex_dense(const fourindex_ext<I,T> &rhs) : NRSMat<T>((T)0,rhs.size()*(rhs.size()+1)/2)
{
if(rhs.getsymmetry() != twoelectronrealmullikan ) laerror("fourindex_dense symmetry mismatch");
typename fourindex_ext<I,T>::iterator p;
for(p=rhs.begin(); p!= rhs.end(); ++p) 
	{
#ifdef DEBUG
unsigned int IJ = SMat_index_1(p->index.indiv.i,p->index.indiv.j);
unsigned int KL = SMat_index_1(p->index.indiv.k,p->index.indiv.l);
if (IJ<0 || IJ>=(unsigned int)NRSMat<T>::nn || KL<0 || KL>=(unsigned int)NRSMat<T>::nn) laerror("fourindex_dense index out of range in constructor");
#endif
	(*this)(p->index.indiv.i,p->index.indiv.j ,p->index.indiv.k,p->index.indiv.l) = p->elem;
	}
}



template<class T, class DUMMY>
T& fourindex_dense<twoelectronrealmullikan,T,DUMMY>::operator() (unsigned int i, unsigned int j, unsigned int k, unsigned int l)
{
int I = SMat_index_1(i,j);
int J = SMat_index_1(k,l);
//I,J act as indices of a NRSmat
#ifdef DEBUG
     	if (*NRSMat<T>::count != 1) laerror("lval (i,j,k,l) with count > 1 in fourindex_dense");
       	if (I<0 || I>=NRSMat<T>::nn || J<0 || J>=NRSMat<T>::nn) laerror("fourindex_dense index out of range");
       	if (!NRSMat<T>::v) laerror("access to unallocated fourindex_dense");
#endif
return NRSMat<T>::v[SMat_index(I,J)];
}


template<class T, class DUMMY>
const T& fourindex_dense<twoelectronrealmullikan,T,DUMMY>::operator() (unsigned int i, unsigned int j, unsigned int k, unsigned int l) const
{
int I = SMat_index_1(i,j);
int J = SMat_index_1(k,l);
//I,J act as indices of a NRSmat
#ifdef DEBUG
       	if (I<0 || I>=NRSMat<T>::nn || J<0 || J>=NRSMat<T>::nn) laerror("fourindex_dense index out of range");
       	if (!NRSMat<T>::v) laerror("access to unallocated fourindex_dense");
#endif
return NRSMat<T>::v[SMat_index(I,J)];
}

template<class T, class I>
class fourindex_dense<nosymmetry,T,I> : public NRMat<T> {
protected:
	unsigned int nn;
friend class explicit_t2;
public:
        fourindex_dense(): NRMat<T>() {nn=0;};
	void resize(const int nnn) {nn=nnn; (*this).NRMat<T>::resize(nn*nn,nn*nn);};
        explicit fourindex_dense(const int nnn): NRMat<T>(nnn*nnn,nnn*nnn) {nn=nnn;};

        inline T& operator() (unsigned int i, unsigned int j, unsigned int a, unsigned int b)
{
#ifdef DEBUG
if(i<1||i>nn ||j<1||j>nn|| a<1||a>nn||b<1||b>nn) laerror("nosymmetry fourindex out of range");
if (!NRMat<T>::v) laerror("access to unallocated fourindex_dense");
#endif
return (*this).NRMat<T>::operator() ((j-1)*nn+i-1,(b-1)*nn+a-1);
}
        inline const T& operator() (unsigned int i, unsigned int j, unsigned int a, unsigned int b) const
{
#ifdef DEBUG
if(i<1||i>nn ||j<1||j>nn|| a<1||a>nn||b<1||b>nn) laerror("nosymmetry fourindex out of range");
if (!NRMat<T>::v) laerror("access to unallocated fourindex_dense");
#endif
return (*this).NRMat<T>::operator() ((j-1)*nn+i-1,(b-1)*nn+a-1);
}

void print(std::ostream &out) const
		{
		unsigned int i,j,a,b;
		for(i=1; i<=nn; ++i) for(j=1; j<=nn; ++j) for(a=1; a<=nn; ++a) for(b=1; b<=nn; ++b) out << i<<" "<<j<<" "<<a<<" "<<b<<" "<<(*this)(i,j,a,b)<<std::endl;
		}
};


//access to spin-blocks of T2 amplitudes in aces storage order
//both occupied and virtual indices start from 1

template<class T, class I>
class fourindex_dense<T2IjAb_aces,T,I> : public NRMat<T> {
protected:
	unsigned int noca,nocb,nvra,nvrb;
friend class explicit_t2;
public:
        fourindex_dense(): NRMat<T>() {noca=nocb=nvra=nvrb=0;};
	void resize(const int nocca, const int noccb, const int nvrta, const int nvrtb) {noca=nocca; nocb=noccb; nvra=nvrta; nvrb=nvrtb; (*this).NRMat<T>::resize(nocca*noccb,nvrta*nvrtb);};
        explicit fourindex_dense(const int nocca, const int noccb, const int nvrta, const int nvrtb): NRMat<T>(nocca*noccb,nvrta*nvrtb) {noca=nocca; nocb=noccb; nvra=nvrta; nvrb=nvrtb;};

//here i,a are alpha j,b beta
        inline T& operator() (unsigned int i, unsigned int j, unsigned int a, unsigned int b)
{
#ifdef DEBUG
if(i<1||i>noca ||j<1||j>nocb|| a<1||a>nvra||b<1||b>nvrb) laerror("T2IjAb_aces fourindex out of range");
if (!NRMat<T>::v) laerror("access to unallocated fourindex_dense");
#endif
return (*this).NRMat<T>::operator() ((j-1)*noca+i-1,(b-1)*nvra+a-1);
}
        inline const T& operator() (unsigned int i, unsigned int j, unsigned int a, unsigned int b) const
{
#ifdef DEBUG
if(i<1||i>noca ||j<1||j>nocb|| a<1||a>nvra||b<1||b>nvrb) laerror("T2IjAb_aces fourindex out of range");
if (!NRMat<T>::v) laerror("access to unallocated fourindex_dense");
#endif
return (*this).NRMat<T>::operator() ((j-1)*noca+i-1,(b-1)*nvra+a-1);
}

	void print(std::ostream &out) const
		{
		unsigned int i,j,a,b;
		for(i=1; i<=noca; ++i) for(j=1; j<=nocb; ++j) for(a=1; a<=nvra; ++a) for(b=1; b<=nvrb; ++b) out << i<<" "<<j<<" "<<a<<" "<<b<<" "<<(*this)(i,j,a,b)<<std::endl;
		}
};

template<class T, class I>
class fourindex_dense<antisymtwoelectronrealdiracAB,T,I> : public NRSMat<T> {
protected:
	unsigned int nbas;
friend class explicit_t2;
public:
        fourindex_dense(): NRSMat<T>() {nbas=0;};
	void resize(const int n) {nbas=n; (*this).NRSMat<T>::resize(nbas*nbas);};
        explicit fourindex_dense(const int n): NRSMat<T>(n*n) {nbas=n;};

//here i,a are alpha j,b beta
        inline T& operator() (unsigned int i, unsigned int j, unsigned int a, unsigned int b)
{
#ifdef DEBUG
if(i<1||i>nbas ||j<1||j>nbas|| a<1||a>nbas||b<1||b>nbas) laerror("antisymtwoelectronrealdiracAB fourindex out of range");
if (!NRSMat<T>::v) laerror("access to unallocated fourindex_dense");
#endif
return (*this).NRSMat<T>::operator() ((j-1)*nbas+i-1,(b-1)*nbas+a-1);
}
        inline const T& operator() (unsigned int i, unsigned int j, unsigned int a, unsigned int b) const
{
#ifdef DEBUG
if(i<1||i>nbas ||j<1||j>nbas|| a<1||a>nbas||b<1||b>nbas) laerror("antisymtwoelectronrealdiracAB fourindex out of range");
if (!NRSMat<T>::v) laerror("access to unallocated fourindex_dense");
#endif
return (*this).NRSMat<T>::operator() ((j-1)*nbas+i-1,(b-1)*nbas+a-1);
}

	void print(std::ostream &out) const
		{
		unsigned int i,j,a,b;
		for(i=1; i<=nbas; ++i) for(j=1; j<=nbas; ++j)
			 for(a=1; a<=i; ++a) for(b=1; b<= (a<i?nbas:j); ++b) 
			out << i<<" "<<j<<" "<<a<<" "<<b<<" "<<(*this)(i,j,a,b)<<std::endl;
		}
};


template<class T, class I>
class fourindex_dense<T2ijab_aces,T,I> : public NRMat<T> {
protected:
	unsigned int nocc,nvrt,ntri;
friend class explicit_t2;
public:
        fourindex_dense(): NRMat<T>() {nocc=nvrt=ntri=0;};
        explicit fourindex_dense(const int noc, const int nvr): NRMat<T>(noc*(noc-1)/2,nvr*(nvr-1)/2) {nocc=noc; nvrt=nvr; ntri=nvr*(nvr-1)/2;};
	void resize(const int noc, const int nvr) {(*this).NRMat<T>::resize(noc*(noc-1)/2,nvr*(nvr-1)/2); nocc=noc; nvrt=nvr; ntri=nvr*(nvr-1)/2;};

//we cannot return reference due to the possible sign change
//stored values are for i>j a>b
        inline T operator() (unsigned int i, unsigned int j, unsigned int a, unsigned int b) const
{
#ifdef DEBUG
if(i<1||i>nocc ||j<1||j>nocc|| a<1||a>nvrt||b<1||b>nvrt) laerror("T2ijab_aces fourindex out of range");
if (!NRMat<T>::v) laerror("access to unallocated fourindex_dense");
#endif
int minus=0;
if(i==j||a==b) return (T)0; //important, needed
if(i<j) {minus++; unsigned int t=i; i=j; j=t;}
if(a<b) {minus++; unsigned int t=a; a=b; b=t;}
T val=(*this).NRMat<T>::operator() ((i-2)*(i-1)/2+j-1, (a-2)*(a-1)/2+b-1);
return (minus&1)? -val:val;
}
	inline void set(unsigned int i, unsigned int j, unsigned int a, unsigned int b, T elem)
{
#ifdef DEBUG
if(i<1||i>nocc ||j<1||j>nocc|| a<1||a>nvrt||b<1||b>nvrt) laerror("T2ijab_aces fourindex out of range");
if (!NRMat<T>::v) laerror("access to unallocated fourindex_dense");
if(i==j||a==b && elem) laerror("antisymmetry violation in fourindex_dense<T2ijab_aces>");
#endif
int minus=0;
if(i<j) {minus++; unsigned int t=i; i=j; j=t;}
if(a<b) {minus++; unsigned int t=a; a=b; b=t;}
(*this).NRMat<T>::operator() ((i-2)*(i-1)/2+j-1, (a-2)*(a-1)/2+b-1) = minus? -elem : elem;
}
        inline void add(unsigned int i, unsigned int j, unsigned int a, unsigned int b, T elem)
{
#ifdef DEBUG
if(i<1||i>nocc ||j<1||j>nocc|| a<1||a>nvrt||b<1||b>nvrt) laerror("T2ijab_aces fourindex out of range");
if (!NRMat<T>::v) laerror("access to unallocated fourindex_dense");
if(i==j||a==b && elem) laerror("antisymmetry violation in fourindex_dense<T2ijab_aces>");
#endif
int minus=0;
if(i<j) {minus++; unsigned int t=i; i=j; j=t;}
if(a<b) {minus++; unsigned int t=a; a=b; b=t;}
(*this).NRMat<T>::operator() ((i-2)*(i-1)/2+j-1, (a-2)*(a-1)/2+b-1) += minus? -elem : elem;
}


        void print(std::ostream &out) const
                {
                unsigned int i,j,a,b;
                for(i=1; i<=nocc; ++i) for(j=1; j<i; ++j) for(a=1; a<=nvrt; ++a) for(b=1; b<a; ++b) out << i<<" "<<j<<" "<<a<<" "<<b<<" "<<(*this)(i,j,a,b)<<std::endl;
                }


};



//compact in-core storage of antisymmetrized two-electron integrals

template<class T, class I> 
class fourindex_dense<antisymtwoelectronrealdirac,T,I> : public NRSMat<T> {
private: 
	int nbas;
public:
	fourindex_dense(): NRSMat<T>() {};
	explicit fourindex_dense(const int n): nbas(n), NRSMat<T>(n*(n-1)/2) {};
	fourindex_dense(const T &a, const int n): nbas(n), NRSMat<T>(a,n*(n-1)/2) {};
	fourindex_dense(const T *a, const int n): nbas(n), NRSMat<T>(a,n*(n-1)/2) {};
	//and also construct it from sparse and externally stored fourindex classes
	//it seems not possible to nest template<class I> just for the two constructors
	fourindex_dense(const fourindex<I,T> &rhs);
	fourindex_dense(const fourindex_ext<I,T> &rhs);

	void set(unsigned int i, unsigned int j, unsigned int k, unsigned int l, T elem);
	void add(unsigned int i, unsigned int j, unsigned int k, unsigned int l, T elem);
	void add_unique(unsigned int i, unsigned int j, unsigned int k, unsigned int l, T elem);
	const T operator() (unsigned int i, unsigned int j, unsigned int k, unsigned int l) const;
	void resize(const int n) {nbas=n; (*this).NRSMat<T>::resize(n*(n-1)/2);};
        void print(std::ostream &out) const
                {
                unsigned int i,j,k,l;
		for(i=1; i<=nbas; ++i)
       			 for(k=1;k<i; ++k)
                		for(j=1; j<=i; ++j)
                        		for(l=1; l<j && (j==i ? l<=k : 1); ++l)
                				std::cout << i<<" "<<k<<" "<<j<<" "<<l<<" "<<(*this)(i,k,j,l)<<std::endl;
                }


};




//WARNING: assignment by regular pointer not possible due to the sign tracking - returns const value to prevent it, use set() for assignment
//therefore also operator() cannot be T& but T only  - no legal lvalue
template<class T, class DUMMY>
const T fourindex_dense<antisymtwoelectronrealdirac,T,DUMMY>::operator() (unsigned int i, unsigned int j, unsigned int k, unsigned int l) const
{
int I = ASMat_index_1(i,j);
int J = ASMat_index_1(k,l);
if (I<0 || J<0) return 0.; 
int sign=1;
if(i<j) sign = -sign;
if(k<l) sign = -sign;

#ifdef DEBUG
if (I>=(unsigned int)NRSMat<T>::nn || J>=(unsigned int)NRSMat<T>::nn) laerror("index out of range");
if (!NRSMat<T>::v) laerror("access to unallocated fourindex_dense");
#endif
return sign==1? NRSMat<T>::v[SMat_index(I,J)] : -NRSMat<T>::v[SMat_index(I,J)];
}


template<class T, class DUMMY>
void fourindex_dense<antisymtwoelectronrealdirac,T,DUMMY>::set(unsigned int i, unsigned int j, unsigned int k, unsigned int l, T elem)
{
if(i<j) elem = -elem;
if(k<l) elem = -elem;
int I = ASMat_index_1(i,j);
int J = ASMat_index_1(k,l);
if (I<0 || J<0) laerror("assignment to nonexisting element");
#ifdef DEBUG
if (I>=NRSMat<T>::nn || J>=NRSMat<T>::nn) laerror("index out of range");
if (!NRSMat<T>::v) laerror("access to unallocated fourindex_dense");
#endif
NRSMat<T>::v[SMat_index(I,J)] = elem;
}


template<class T, class DUMMY>
void fourindex_dense<antisymtwoelectronrealdirac,T,DUMMY>::add(unsigned int i, unsigned int j, unsigned int k, unsigned int l, T elem)
{
if(i<j) elem = -elem;
if(k<l) elem = -elem;
int I = ASMat_index_1(i,j);
int J = ASMat_index_1(k,l);
if (I<0 || J<0) laerror("assignment to nonexisting element");
#ifdef DEBUG
if (I>=NRSMat<T>::nn || J>=NRSMat<T>::nn) laerror("index out of range");
if (!NRSMat<T>::v) laerror("access to unallocated fourindex_dense");
#endif
NRSMat<T>::v[SMat_index(I,J)] += elem;
}

template<class T, class DUMMY>
void fourindex_dense<antisymtwoelectronrealdirac,T,DUMMY>::add_unique(unsigned int i, unsigned int j, unsigned int k, unsigned int l, T elem)
{
if(i<=j || k<=l) return;
int I = ASMat_index_1(i,j);
int J = ASMat_index_1(k,l);
if (I<0 || J<0 || I<J) return;
NRSMat<T>::v[SMat_index(I,J)] += elem;
}


template<class T, class I> 
fourindex_dense<antisymtwoelectronrealdirac,T,I>::fourindex_dense(const fourindex<I,T> &rhs) : nbas(rhs.size()), NRSMat<T>((T)0,rhs.size()*(rhs.size()-1)/2)
{
if(rhs.getsymmetry() != twoelectronrealmullikan ) laerror("fourindex_dense symmetry mismatch");
typename fourindex_ext<I,T>::piterator p; //we have to run over equivalents in non-canonical order to build the antisymmetrization properly; it could be done less elegantly but more efficiently moving the if's to outer parts of the piterator loop, if needed
for(p= const_cast<fourindex_ext<I,T> *>(&rhs)->pbegin(); p.notend(); ++p)
        {
        I i=p->index.indiv.i;
        I j=p->index.indiv.j;
        I k=p->index.indiv.k;
        I l=p->index.indiv.l;
        add_unique(i,k,j,l,p->elem);
        add_unique(i,k,l,j,-p->elem);
        }
}


template<class T, class I>
fourindex_dense<antisymtwoelectronrealdirac,T,I>::fourindex_dense(const fourindex_ext<I,T> &rhs) : nbas(rhs.size()), NRSMat<T>((T)0,rhs.size()*(rhs.size()-1)/2)
{
if(rhs.getsymmetry() != twoelectronrealmullikan ) laerror("fourindex_dense symmetry mismatch");
typename fourindex_ext<I,T>::piterator p; //we have to run over equivalents in non-canonical order to build the antisymmetrization properly; it could be done less elegantly but more efficiently moving the if's to outer parts of the piterator loop, if needed
for(p= const_cast<fourindex_ext<I,T> *>(&rhs)->pbegin(); p.notend(); ++p) 
        {
	I i=p->index.indiv.i;
	I j=p->index.indiv.j;
	I k=p->index.indiv.k;
	I l=p->index.indiv.l;
        add_unique(i,k,j,l,p->elem);
        add_unique(i,k,l,j,-p->elem);
        }
}


}//namespace

//TODO:
//implement fourindex_dense for unitary t2 symmetry types

#endif /*_fourindex_included*/
