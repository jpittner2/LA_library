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


#ifndef _PERMUTATION_H
#define _PERMUTATION_H

#include "la_traits.h"
#include "vec.h"
#include "polynomial.h"
#include "nonclass.h"

typedef   unsigned long long PERM_RANK_TYPE;

//permutations are always numbered from 1; offset is employed when applied to vectors and matrices


namespace LA {


//forward declaration
template <typename T> class CyclePerm;
template <typename T> class Partition;
template <typename T> class CompressedPartition;
template <typename T> class YoungTableaux;
template <typename T, typename R> class WeightPermutation;
template <typename T, typename R> class PermutationAlgebra;


//operator== != < > inherited from NRVec

template <typename T>
class NRPerm : public NRVec_from1<T> {
public:
	//basic constructors
	NRPerm(): NRVec_from1<T>() {};
	template<int SIZE> explicit NRPerm(const T (&a)[SIZE]) : NRVec_from1<T>(a) {};
	NRPerm(const int n) : NRVec_from1<T>(n) {};
	NRPerm(const NRVec_from1<T> &rhs): NRVec_from1<T>(rhs) {};
        NRPerm(const T *a, const int n): NRVec_from1<T>(a, n) {};
	explicit NRPerm(const CyclePerm<T> &rhs, const int n=0); 

	//specific operations
	int size() const {return NRVec_from1<T>::size();};
	void identity();
	bool is_valid() const; //is it really a permutation
	bool is_identity() const;
	CompressedPartition<T> cycles() const {return CyclePerm<T>(*this).cycles(size());};
	NRPerm inverse() const;
	NRPerm reverse() const; //backward order
	NRPerm operator&(const NRPerm &rhs) const; //concatenate the permutations this,rhs, renumbering rhs (not commutative)
	NRPerm operator|(const NRPerm &rhs) const; //concatenate the permutations rhs,this, renumbering rhs (not commutative)
	NRPerm operator*(const NRPerm &q) const; //q is rhs and applied first, this applied second
	NRPerm operator*(const CyclePerm<T> &r) const;
	NRPerm multiply(const NRPerm<T> &q, bool inverse) const; //multiplication but optionally q inversed
	NRPerm conjugate_by(const NRPerm &q, bool reverse=false) const; //q^-1 p q or q p q^-1
	NRPerm commutator(const NRPerm &q, bool inverse=false) const;  //p^-1 q^-1 p q or q^-1 p^-1 q p
	int parity() const; //returns +/- 1
	void randomize(void); //uniformly random by Fisher-Yates shuffle
	bool next(); //generate next permutation in lex order
	PERM_RANK_TYPE generate_all(void (*callback)(const NRPerm<T>&), int parity_select=0); //Algorithm L from Knuth's vol.4, efficient but not in lex order!
	PermutationAlgebra<T,T>  list_all(int parity_select=0);
	PermutationAlgebra<T,T>  list_all_lex();
	PERM_RANK_TYPE generate_all_multi(void (*callback)(const NRPerm<T>&));  //Algorithm L2 from Knuth's vol.4, for a multiset (repeated numbers, not really permutations)
	PERM_RANK_TYPE generate_all2(void (*callback)(const NRPerm<T>&)); //recursive method, also not lexicographic
	PERM_RANK_TYPE generate_all_lex(void (*callback)(const NRPerm<T>&)); //generate in lex order using next()
	PERM_RANK_TYPE generate_restricted(void (*callback)(const NRPerm<T>&), const NRVec_from1<T> &classes, int restriction_type=0);
	PermutationAlgebra<T,T>  list_restricted(const NRVec_from1<T> &classes, int restriction_type=0, bool inverted=false); //weight is set to parity (antisymmetrizer) by default
	PERM_RANK_TYPE rank() const; //counted from 0 to n!-1
	NRVec_from1<T> inversions(const int type, PERM_RANK_TYPE *prank=NULL) const; //inversion tables
	explicit NRPerm(const int type, const NRVec_from1<T> &inversions); //compute permutation from inversions
	explicit NRPerm(const int n, const PERM_RANK_TYPE rank); //compute permutation from its rank
	NRPerm pow(const int n) const {return power(*this,n);};
};


//this is not a class memeber due to double templating
//it is also possible to use member function permuted of NRVec(_from1)
template <typename T, typename X>
NRVec_from1<X> applypermutation(const NRPerm<T> &p, const NRVec_from1<X> &set, bool inverse=false)
{
#ifdef DEBUG
if(p.size()!=set.size()) laerror("size mismatch in applypermutation");
#endif
NRVec_from1<X> r(set.size());
if(inverse) for(int i=1; i<=p.size(); ++i) r[p[i]] = set[i];
else for(int i=1; i<=p.size(); ++i) r[i] = set[p[i]];
return r;
}


template <typename T, typename R> 
class WeightPermutation {
public:
	R weight;
	NRPerm<T> perm;
	
	int size() const {return perm.size();};
	bool is_zero() const {return weight==0;}
	bool is_scaledidentity() const {return perm.is_identity();}
	bool is_identity() const {return weight==1 && is_scaledidentity();}
	bool is_plaindata() const {return false;};
	WeightPermutation() : weight(0) {};
	WeightPermutation(const R w, const NRPerm<T> &p) : weight(w), perm(p) {};
	WeightPermutation(const NRPerm<T> &p) : perm(p) {weight= p.parity();};
	void copyonwrite() {perm.copyonwrite();};
	WeightPermutation  operator&(const WeightPermutation &rhs) const {return WeightPermutation(weight*rhs.weight,perm&rhs.perm);};
	WeightPermutation  operator|(const WeightPermutation &rhs) const {return WeightPermutation(weight*rhs.weight,perm|rhs.perm);};
	WeightPermutation  operator*(const WeightPermutation &rhs) const {return WeightPermutation(weight*rhs.weight,perm*rhs.perm);};
	WeightPermutation operator*(const R &x) const {return WeightPermutation(weight*x,perm); }

	bool operator==(const WeightPermutation &rhs) const {return this->perm == rhs.perm;}; //NOTE for sorting, compares only the permutation not the weight!
	bool operator!=(const WeightPermutation &rhs) const {return !(*this==rhs);} //NOTE: compares only the permutation
	bool operator>(const WeightPermutation &rhs) const {return  this->perm > rhs.perm;};
        bool operator<(const WeightPermutation &rhs) const {return  this->perm < rhs.perm;};
        bool operator>=(const WeightPermutation &rhs) const {return !(*this < rhs);};
        bool operator<=(const WeightPermutation &rhs) const {return !(*this > rhs);};
	WeightPermutation & operator=(const WeightPermutation &rhs) {weight=rhs.weight; perm=rhs.perm; return *this;};

	
};


//some necessary traits of the non-scalar class to be able to use LA methods
template<typename T, typename R>
class LA_traits<WeightPermutation<T,R> > {
public: 
        static bool is_plaindata() {return false;};
        static void copyonwrite(WeightPermutation<T,R>& x) {x.perm.copyonwrite();};
	typedef typename LA_traits<R>::normtype  normtype;
	typedef R coefficienttype;
	typedef NRPerm<T> elementtype;
	static inline bool smaller(const WeightPermutation<T,R>& x, const WeightPermutation<T,R>& y) {return x.perm<y.perm;};
	static inline bool bigger(const WeightPermutation<T,R>& x, const WeightPermutation<T,R>& y) {return x.perm>y.perm;};
	static R coefficient(const WeightPermutation<T,R>& x){return x.weight;};
	static R& coefficient(WeightPermutation<T,R>& x) {return x.weight;};
	static typename LA_traits<R>::normtype abscoefficient(const WeightPermutation<T,R>& x){return LA_traits<R>::abs2(x.weight);};
	static void clear(WeightPermutation<T,R> *v, int nn) {for(int i=0; i<nn; ++i) {v[i].weight=0; v[i].perm.clear();}}
};




template <typename T, typename R>
std::istream & operator>>(std::istream &s, WeightPermutation<T,R> &x)
{
s>>x.weight>>x.perm;
return s;
}

template <typename T, typename R>
std::ostream & operator<<(std::ostream &s, const WeightPermutation<T,R> &x)
{
s<<x.weight<<' '<<x.perm<<' ';
return s;
}
        


template <typename T, typename R>
class PermutationAlgebra : public NRVec<WeightPermutation<T,R> >
{
public:
	PermutationAlgebra() {};
	PermutationAlgebra(int n) : NRVec<WeightPermutation<T,R> >(n) {};
	PermutationAlgebra(const NRVec<WeightPermutation<T,R> > &x) :  NRVec<WeightPermutation<T,R> >(x) {};
	int size() const {return NRVec<WeightPermutation<T,R> >::size();};
	void copyonwrite() {NRVec<WeightPermutation<T,R> >::copyonwrite();};
	int sort(int direction = 0, int from = 0, int to = -1, int *permut = NULL, bool stable=false) {return NRVec<WeightPermutation<T,R> >::sort(direction,from, to,permut,stable);};

	PermutationAlgebra operator&(const PermutationAlgebra &rhs) const;
	PermutationAlgebra operator|(const PermutationAlgebra &rhs) const;
	PermutationAlgebra operator*(const PermutationAlgebra &rhs) const; 
	PermutationAlgebra operator+(const PermutationAlgebra &rhs) const;
	PermutationAlgebra &operator*=(const R &x) {this->copyonwrite(); for(int i=1; i<=size(); ++i) (*this)[i].weight *= x; return *this;};
	PermutationAlgebra operator*(const R &x) const {PermutationAlgebra r(*this); return r*=x;};
	void simplify(const typename LA_traits<R>::normtype thr=0) {NRVec_simplify(*this,thr);};
	bool operator==(PermutationAlgebra &rhs); //do NOT inherit from NRVec, as the underlying one ignores weights for the simplification; also we have to simplify before comparison
	bool is_zero() const {return size()==0;}; //assume it was simplified
	bool is_scaled_identity() const {return size()==1 && (*this)[0].is_scaledidentity();}; //assume it was simplified
	bool is_identity() const {return size()==1 && (*this)[0].is_identity();}; //assume it was simplified
};




//permutations represented in the cycle format
template <typename T>
class CyclePerm : public NRVec_from1<NRVec_from1<T> > {
public:
	CyclePerm() :  NRVec_from1<NRVec_from1<T> >() {};
	template<int SIZE> explicit CyclePerm(const NRVec_from1<T> (&a)[SIZE]) : NRVec_from1<NRVec_from1<T> >(a) {};
//NOTE - how to do it so that direct nested brace initializer would work?
	explicit CyclePerm(const NRPerm<T> &rhs);

	bool is_valid() const; //is it really a permutation
	bool is_identity() const; //no cycles of length > 1
	void identity() {this->resize(0);};
	CyclePerm inverse() const; //reverse all cycles
	int parity() const; //negative if having odd number of even-length cycles
	T max() const {T m=0; for(int i=1; i<=this->size(); ++i) {T mm= (*this)[i].max(); if(mm>m) m=mm;} return m;}
	CompressedPartition<T> cycles(T n = 0) const;
	void readfrom(const std::string &line);
	CyclePerm operator*(const CyclePerm &q) const; //q is rhs and applied first, this applied second
	NRPerm<T> operator*(const NRPerm<T> &r) const;
	CyclePerm conjugate_by(const CyclePerm &q) const; //q^-1 p q
	PERM_RANK_TYPE order() const; //lcm of cycle lengths
	bool operator==(const CyclePerm &rhs) const {return NRPerm<T>(*this) == NRPerm<T>(rhs);}; //cycle representation is not unique, cannot inherit operator== from NRVec
	void simplify(bool keep1=false); //remove cycles of size 0 or 1
	CyclePerm pow_simple(const int n) const {return CyclePerm(NRPerm<T>(*this).pow(n));}; //do not call power() with our operator*
	CyclePerm pow(const int n, const bool keep1=false) const; //a more efficient algorithm
};


template <typename T>
T gcd(T big, T small)
{
if(big==0)
        {
        if(small==0) laerror("bad arguments in gcd");
        return small;
        }
if(small==0) return big;
if(small==1||big==1) return 1;

T help;
if(small>big) {help=big; big=small; small=help;}
do      {
        help=small;
        small= big%small;
        big=help;
        }
while(small != 0);
return big;
}


template <typename T> 
inline T lcm(T a, T b)
{
return (a/gcd(a,b))*b;
}


template <typename T>
std::istream & operator>>(std::istream &s, CyclePerm<T> &x);

template <typename T>
std::ostream & operator<<(std::ostream &s, const CyclePerm<T> &x);



//compressed partitions stored as #of 1s, #of 2s, etc.
template <typename T>
class CompressedPartition : public NRVec_from1<T> {
public:
	CompressedPartition(): NRVec_from1<T>() {};
	template<int SIZE> explicit CompressedPartition(const T (&a)[SIZE]) : NRVec_from1<T>(a) {};
	CompressedPartition(const int n) : NRVec_from1<T>(n) {};
	T sum() const {T s=0; for(int i=1; i<=this->size(); ++i) s += i*(*this)[i]; return s;}
	T nparts() const {T s=0; for(int i=1; i<=this->size(); ++i) s += (*this)[i]; return s;}
	T nclasses() const {T s=0; for(int i=1; i<=this->size(); ++i) if((*this)[i]) ++s; return s;}
	bool is_valid() const {return this->size() == this->sum();}
	explicit CompressedPartition(const Partition<T> &rhs) : NRVec_from1<T>(rhs.size()) {this->clear(); for(int i=1; i<=rhs.size(); ++i) if(!rhs[i]) break; else (*this)[rhs[i]]++; }
	PERM_RANK_TYPE Sn_class_size() const; 
	int parity() const; //of a permutation with given cycle lengths, returns +/- 1

};

template <typename T>
std::ostream & operator<<(std::ostream &s, const CompressedPartition<T> &x);


template <typename T>
class Partition : public NRVec_from1<T> {
public:
        Partition(): NRVec_from1<T>() {};
	template<int SIZE> explicit Partition(const T (&a)[SIZE]) : NRVec_from1<T>(a) {};
        Partition(const int n) : NRVec_from1<T>(n) {};
        T nparts() const {T s=0; for(int i=1; i<=this->size(); ++i) if((*this)[i]) ++s; return s;}
        bool is_valid() const {if(this->size() != this->sum()) return false; for(int i=2; i<=this->size(); ++i) if((*this)[i]>(*this)[i-1]) return false; return true; }
	explicit Partition(const CompressedPartition<T>  &rhs) : NRVec_from1<T>(rhs.size()) {this->clear(); int ithru=0; for(int i=rhs.size(); i>=1; --i) for(int j=0; j<rhs[i]; ++j) (*this)[++ithru]=i; }
	explicit Partition(const YoungTableaux<T> &x); //extract a partition  as a shape of Young tableaux
	Partition adjoint() const; //also called conjugate partition
	PERM_RANK_TYPE Sn_irrep_dim() const;
	PERM_RANK_TYPE Un_irrep_dim(const int n) const;
	PERM_RANK_TYPE generate_all(void (*callback)(const Partition<T>&), int nparts=0); //nparts <0 means at most to -nparts
	int parity() const; //of a permutation with given cycle lengths, returns +/- 1

};

template <typename T>
extern T Sn_character(const Partition<T> &irrep, const  Partition<T> &cclass);

template <typename T>
inline T Sn_character(const CompressedPartition<T> &irrep, const  CompressedPartition<T> &cclass)
{
return Sn_character(Partition<T>(irrep),Partition<T>(cclass));
}


template <typename T>
class YoungTableaux : public NRVec_from1<NRVec_from1<T> > {
public:
        YoungTableaux() :  NRVec_from1<NRVec_from1<T> >() {};
        explicit YoungTableaux(const Partition<T> &frame);
        template<int SIZE> explicit YoungTableaux(const NRVec_from1<T> (&a)[SIZE]) : NRVec_from1<NRVec_from1<T> >(a) {};
//NOTE - how to do it so that direct nested brace initializer would work?

	bool is_valid() const; //check whether its shape forms a partition
	int nrows() const {return this->size();}
	int ncols() const {return (*this)[1].size();}
        bool is_standard() const; //is it filled in standard way (possibly with repeated numbers)
	T sum() const; //get back sum of the partition
	T max() const; //get back highest number filled in
	NRVec_from1<T> yamanouchi() const; //yamanouchi symbol
	T character_contribution(int ncyc=0) const; //contribution of filled tableaux to Sn character
	PERM_RANK_TYPE generate_all_standard(void (*callback)(const YoungTableaux<T>&));
	PermutationAlgebra<T,T> young_operator() const; //generate young operator for a standard tableaux

};

template <typename T>
std::ostream & operator<<(std::ostream &s, const YoungTableaux<T> &x);


extern PERM_RANK_TYPE partitions(int n, int k= -1); //enumerate partitions to k parts; k== -1 for total # of partitions


//Sn character table
template <typename T>
class Sn_characters {
public:
T n;
NRVec_from1<CompressedPartition<T> > classes;
NRVec_from1<CompressedPartition<T> > irreps; //can be in different order than classes
NRVec_from1<PERM_RANK_TYPE> classsizes;
NRMat_from1<T> chi; //characters

	Sn_characters(const int n0); //compute the table
	bool is_valid() const; //check internal consistency
	T irrepdim(T i) const {return chi(i,1);};
	T sumirrepdims() const {T s=0; for(T i=1; i<=chi.nrows(); ++i) s+=irrepdim(i); return s;};
};

template <typename T>  class Polynomial; //forward declaration

template <typename T>
class CycleIndex {
public:
NRVec_from1<CompressedPartition<T> > classes;
NRVec_from1<PERM_RANK_TYPE> classsizes;

	CycleIndex(const Sn_characters<T> &rhs): classes(rhs.classes),classsizes(rhs.classsizes) {};
	bool is_valid() const; //check internal consistency
	Polynomial<T> substitute(const Polynomial<T> &p, PERM_RANK_TYPE *denom) const;

};

template <typename T>
extern std::ostream & operator<<(std::ostream &s, const Sn_characters<T> &c);


template<typename T>
const NRVec<T> NRVec<T>::permuted(const NRPerm<int> &p, const bool inverse) const
{
#ifdef DEBUG
if(!p.is_valid()) laerror("invalid permutation of vector");
#endif
int n=p.size();
if(n!=(*this).size()) laerror("incompatible permutation and vector");
#ifdef CUDALA
        if(this->getlocation() != cpu || p.getlocation() != cpu ) laerror("permutations can be done only in CPU memory");
#endif
NRVec<T> r(n);
if(inverse) for(int i=1; i<=n; ++i) r[i-1] = v[p[i]-1];
else for(int i=1; i<=n; ++i) r[p[i]-1] = v[i-1];
return r;
}

template<typename T>
NRMat<PERM_RANK_TYPE> Multable(T n);

template<typename T, typename R>
NRMat<R> RegularRepresentation(const PermutationAlgebra<T,R> &a, const NRMat<PERM_RANK_TYPE> &mtable);


template<typename T>
PermutationAlgebra<T,T> general_antisymmetrizer(const NRVec<NRVec_from1<T> > &groups, int restriction_type=0, bool inverted=false);


}//namespace
#endif
