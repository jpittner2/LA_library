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

typedef   unsigned long long PERM_RANK_TYPE;

//permutations are always numbered from 1; offset is employed when applied to vectors and matrices

namespace LA {

//forward declaration
template <typename T> class CyclePerm;
template <typename T> class Partition;
template <typename T> class CompressedPartition;
template <typename T> class YoungTableaux;

template <typename T>
class NRPerm : public NRVec_from1<T> {
public:
	//basic constructors
	NRPerm(): NRVec_from1<T>() {};
	NRPerm(const int n) : NRVec_from1<T>(n) {};
	NRPerm(const NRVec_from1<T> &rhs): NRVec_from1<T>(rhs) {};
        NRPerm(const T *a, const int n): NRVec_from1<T>(a, n) {};
	explicit NRPerm(const CyclePerm<T> &rhs, const int n=0); 

	//specific operations
	void identity();
	bool is_valid() const; //is it really a permutation
	bool is_identity() const;
	NRPerm inverse() const;
	NRPerm operator*(const NRPerm q) const; //q is rhs and applied first, this applied second
	NRPerm conjugate_by(const NRPerm q) const; //q^-1 p q
	int parity() const;
	void randomize(void); //uniformly random by Fisher-Yates shuffle
	bool next(); //generate next permutation in lex order
	PERM_RANK_TYPE generate_all(void (*callback)(const NRPerm<T>&), int parity_select=0); //Algorithm from Knuth's vol.4, efficient but not in lex order!
	PERM_RANK_TYPE generate_all2(void (*callback)(const NRPerm<T>&)); //recursive method, also not lexicographic
	PERM_RANK_TYPE generate_all_lex(void (*callback)(const NRPerm<T>&)); //generate in lex order using next()
	PERM_RANK_TYPE rank() const; //counted from 0 to n!-1
	NRVec_from1<T> inversions(const int type, PERM_RANK_TYPE *prank=NULL) const; //inversion tables
	explicit NRPerm(const int type, const NRVec_from1<T> &inversions); //compute permutation from inversions
	explicit NRPerm(const int n, const PERM_RANK_TYPE rank); //compute permutation from its rank
};

extern PERM_RANK_TYPE factorial(const int n);
extern PERM_RANK_TYPE longpow(PERM_RANK_TYPE x, int i);

//permutations represented in the cycle format
template <typename T>
class CyclePerm : public NRVec_from1<NRVec_from1<T> > {
public:
	CyclePerm() :  NRVec_from1<NRVec_from1<T> >() {};
	explicit CyclePerm(const NRPerm<T> &rhs);

	bool is_valid() const; //is it really a permutation
	bool is_identity() const; //no cycles of length > 1
	void identity() {this->resize(0);};
	CyclePerm inverse() const; //reverse all cycles
	int parity() const; //negative if having odd number of even-length cycles
	T max() const {T m=0; for(int i=1; i<=this->size(); ++i) {T mm= (*this)[i].max(); if(mm>m) m=mm;} return m;}
	CompressedPartition<T> cycles(const T n) const;
	void readfrom(const std::string &line);
	CyclePerm operator*(const CyclePerm q) const; //q is rhs and applied first, this applied second
	PERM_RANK_TYPE order() const; //lcm of cycle lengths
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
return (a*b)/gcd(a,b);
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
	CompressedPartition(const int n) : NRVec_from1<T>(n) {};
	T sum() const {T s=0; for(int i=1; i<=this->size(); ++i) s += i*(*this)[i]; return s;}
	T nparts() const {T s=0; for(int i=1; i<=this->size(); ++i) s += (*this)[i]; return s;}
	T nclasses() const {T s=0; for(int i=1; i<=this->size(); ++i) if((*this)[i]) ++s; return s;}
	bool is_valid() const {return this->size() == this->sum();}
	explicit CompressedPartition(const Partition<T> &rhs) : NRVec_from1<T>(rhs.size()) {this->clear(); for(int i=1; i<=rhs.size(); ++i) if(!rhs[i]) break; else (*this)[rhs[i]]++; }
	PERM_RANK_TYPE Sn_class_size() const; 
	int parity() const; //of a permutation with given cycle lengths

};

template <typename T>
std::ostream & operator<<(std::ostream &s, const CompressedPartition<T> &x);


template <typename T>
class Partition : public NRVec_from1<T> {
public:
        Partition(): NRVec_from1<T>() {};
        Partition(const int n) : NRVec_from1<T>(n) {};
        T nparts() const {T s=0; for(int i=1; i<=this->size(); ++i) if((*this)[i]) ++s; return s;}
        bool is_valid() const {if(this->size() != this->sum()) return false; for(int i=2; i<=this->size(); ++i) if((*this)[i]>(*this)[i-1]) return false; return true; }
	explicit Partition(const CompressedPartition<T>  &rhs) : NRVec_from1<T>(rhs.size()) {this->clear(); int ithru=0; for(int i=rhs.size(); i>=1; --i) for(int j=0; j<rhs[i]; ++j) (*this)[++ithru]=i; }
	explicit Partition(const YoungTableaux<T> &x); //extract a partition  as a shape of Young tableaux
	Partition adjoint() const; //also called conjugate partition
	PERM_RANK_TYPE Sn_irrep_dim() const;
	PERM_RANK_TYPE Un_irrep_dim(const int n) const;
	PERM_RANK_TYPE generate_all(void (*callback)(const Partition<T>&), int nparts=0); //nparts <0 means at most to -nparts
	int parity() const; //of a permutation with given cycle lengths

};


template <typename T>
class YoungTableaux : public NRVec_from1<NRVec_from1<T> > {
public:
        YoungTableaux() :  NRVec_from1<NRVec_from1<T> >() {};
        explicit YoungTableaux(const Partition<T> &frame);

	bool is_valid() const; //check whether its shape forms a partition
	int nrows() const {return this->size();}
	int ncols() const {return (*this)[1].size();}
        bool is_standard() const; //is it filled in standard way (possibly with repeated numbers)
	T sum() const; //get back sum of the partition
	T max() const; //get back highest number filled in
	NRVec_from1<T> yamanouchi() const; //yamanouchi symbol
	T character_contribution(int ncyc=0) const; //contribution of filled tableaux to Sn character

		//@@@ ??>young operator as a linear comb of permutations - maybe a class for itself, i.e. element of the Sn group algebra? or maybe as a vector with index being the rank of the permutation(n! length) or as a sparsemat thereof or maybe a list???
};

template <typename T>
std::ostream & operator<<(std::ostream &s, const YoungTableaux<T> &x);


extern PERM_RANK_TYPE partitions(int n, int k= -1); //enumerate partitions to k parts; k== -1 for total # of partitions

template <typename T>
extern T Sn_character(const Partition<T> &irrep, const  Partition<T> &cclass);

template <typename T>
inline T Sn_character(const CompressedPartition<T> &irrep, const  CompressedPartition<T> &cclass)
{
return Sn_character(Partition<T>(irrep),Partition<T>(cclass));
}


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
};

template <typename T>
extern std::ostream & operator<<(std::ostream &s, const Sn_characters<T> &c);


}//namespace
#endif
