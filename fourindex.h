#ifndef _fourindex_included
#define _fourindex_included

//element of a linked list, indices in a portable way, no bit shifts and endianity problems any more!


template<class I, class T>
struct matel4
        {
        T elem;
        matel4 *next;
	typedef union {
		I packed[4];
		struct {
			I i;
			I j;
			I k;	
			I l;
			} indiv;
		} packedindex;
	packedindex index;
        };

typedef enum {nosymmetry=0, twoelectronreal=1, twoelectroncomplex=2, twobodyantisym=3} fourindexsymtype; //if twoelectron, only permutation-nonequivalent elements are stored

template <class I, class T>
class fourindex {
protected:
	I nn;
	fourindexsymtype symmetry;
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
		bool operator==(const iterator rhs) const {return p==rhs.p;}
		bool operator!=(const iterator rhs) const {return p!=rhs.p;}
		iterator operator++() {return p=p->next;}
		iterator operator++(int) {matel4<I,T> *q=p; p=p->next; return q;}
		matel4<I,T> & operator*() const {return *p;}
		matel4<I,T> * operator->() const {return p;}
	};
	iterator begin() const {return list;}
	iterator end() const {return NULL;}

	//constructors etc.
	inline fourindex() :nn(0),count(NULL),list(NULL) {};
	inline fourindex(const I n) :nn(n),count(new int(1)),list(NULL) {};
	fourindex(const fourindex &rhs); //copy constructor
	inline int getcount() const {return count?*count:0;}
	fourindex & operator=(const fourindex &rhs);
        fourindex & operator+=(const fourindex &rhs);
	inline void setsymmetry(fourindexsymtype s) {symmetry=s;}
        fourindex & join(fourindex &rhs); //more efficient +=, rhs will be emptied
	inline ~fourindex();
	inline matel4<I,T> *getlist() const {return list;}
	inline I size() const {return nn;}
	void resize(const I n);
	void copyonwrite();
	int length() const;
	inline void add(const I i, const I j, const I k, const I l, const T elem) 
		{matel4<I,T> *ltmp= new matel4<I,T>; ltmp->next=list; list=ltmp; list->index.indiv.i=i;list->index.indiv.j=j;list->index.indiv.k=k;list->index.indiv.l=l; list->elem=elem;}

	inline void add(const typename matel4<I,T>::packedindex &index , const T elem) 
                {matel4<I,T> *ltmp= new matel4<I,T>; ltmp->next=list; list=ltmp; list->index=index; list->elem=elem;}
	
	inline void add(const I (&index)[4], const T elem)
                {matel4<I,T> *ltmp= new matel4<I,T>; ltmp->next=list; list=ltmp; memcpy(&list->index.packed, &index, sizeof(typename matel4<I,T>::packedindex)); list->elem=elem;}

		

};


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
int fourindex<I,T>::length() const
{
int n=0;
matel4<I,T> *l=list;
while(l)
	{
	++n;
	l=l->next;
	}
return n;
}


template <class I, class T>
ostream& operator<<(ostream &s, const fourindex<I,T> &x)
                {
                int n;
                n=x.size();
                s << n << '\n';
                typename fourindex<I,T>::iterator it=x.begin();
                while(it!=x.end())
                        {
                        s << (int)it->index.indiv.i << ' ' << (int)it->index.indiv.j<<  ' ' <<(int)it->index.indiv.k << ' ' << (int)it->index.indiv.l  << ' ' << it->elem << '\n';
			++it;
                        }
                s << "-1 -1 -1 -1\n";
                return s;
                }

template <class I, class T>
istream& operator>>(istream  &s, fourindex<I,T> &x)
                {
                int i,j,k,l;
		T elem;
		int n;
                s >> n ;
                x.resize(n);
                s >> i >> j >>k >>l;
                while(i>=0 && j>=0 &&k>=0 &&l>=0)
                        {
			s>>elem;
			x.add(i,j,k,l,elem);
			s >> i >> j >>k >>ll;
                        }
                return s;
                }


#endif /*_fourindex_included*/
