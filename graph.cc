/*
    LA: linear algebra C++ interface library
    Copyright (C) 2023 Jiri Pittner <jiri.pittner@jh-inst.cas.cz> or <jiri@pittnerovi.com>

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


//this is a haphazard collection of graph theoretical algorithms I happened to need

#include "graph.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

namespace LA {

template<typename G>
NRVec<int> neighbors(const G &adjacency, const int v)
{
if(v<0||v>=adjacency.nrows()) laerror("invalid vertex number in neighbors");
int n=0; 
for(int i=0; i<adjacency.nrows(); ++i) if(adjacency(v,i) && v!=i) ++n;
int ii=0;
NRVec<int> r(n);
for(int i=0; i<adjacency.nrows(); ++i) if(adjacency(v,i) && v!=i) r[ii++]=i;
return r;
}



template<typename G>
NRVec<int> nonneighbors(const G &adjacency, const int v)
{
if(v<0||v>=adjacency.nrows()) laerror("invalid vertex number in nonneighbors");
int n=0;
for(int i=0; i<adjacency.nrows(); ++i) if(!adjacency(v,i) && v!=i) ++n;
int ii=0;
NRVec<int> r(n);
for(int i=0; i<adjacency.nrows(); ++i) if(!adjacency(v,i) && v!=i) r[ii++]=i;
return r;
}



//heuristic algorithm by Boppana and Halldorsson, BIT 32, 180, (1992)
template<typename G>
NRVec<int> findclique(const G &adjacency, const int v)
{
NRVec<int> r(0); 
if(adjacency.nrows())
	{
	if(v<0||v>=adjacency.nrows()) laerror("invalid vertex number in findclique");
	NRVec<int> neighb = neighbors(adjacency,v);
	NRVec<int> nonneighb = nonneighbors(adjacency,v);
	NRVec<int> c1(0),c2(0);
	if(neighb.size())
		{
		G subg1 = adjacency.submatrix(neighb);
		c1 = findclique(subg1);
		//reindex to original graph
		for(int i=0; i<c1.size(); ++i) c1[i] = neighb[c1[i]];
		}
	if(nonneighb.size())
		{
		G subg2 = adjacency.submatrix(nonneighb);
		c2 = findclique(subg2);
		//reindex to original graph
                for(int i=0; i<c2.size(); ++i) c2[i] = nonneighb[c2[i]];
		}
	c1.append(v);
	c1.sort();
	c2.sort();
	if(c1.size()>=c2.size()) r=c1; else r=c2;
	}
return (r);
}


template<typename G>
NRVec<int> complement(const G &adjacency, const NRVec<int> & vertexlist)
{
int n=adjacency.nrows();
NRVec<char> occ(n);
occ.clear();
for(int i=0; i<vertexlist.size(); ++i) 
	{
	if(vertexlist[i]<0||vertexlist[i]>=n) laerror("illegal vertex in complement");
	occ[vertexlist[i]]=1;
	}
int m=0;
for(int i=0; i<n; ++i) if(!occ[i]) ++m;
NRVec<int>  r(m);
m=0;
for(int i=0; i<n; ++i) if(!occ[i]) r[m++]=i;
return r;
}



//found cliques releatedly until the whole graph is covered
//could be recursive but we made it iterative
template<typename G>
NRVec<int> cliquecover(const G &adjacency)
{
int n=adjacency.nrows();
NRVec<int> r(n); r.clear();
G graph = adjacency;
int cliquenumber=1;
int remaining=n;
NRVec<int> cliques(0);
NRVec<int> notincliques(n);
for(int i=0; i<n; ++i) notincliques[i]=i;
while(remaining)
	{
	NRVec<int> clique = findclique(graph);
	NRVec<int> cliqueorignum(clique.size());
	for(int i=0; i<clique.size(); ++i)  cliqueorignum[i] = notincliques[clique[i]];
	cliques.concatme(cliqueorignum);
	for(int i=0; i<clique.size(); ++i) r[cliqueorignum[i]] = cliquenumber;
	++cliquenumber;
	notincliques = complement(adjacency,cliques);
	remaining=notincliques.size();
	if(remaining)
		{
		graph= adjacency.submatrix(notincliques);
		}
	}
return r;
}


//force instantization
#define INSTANTIZE(T) \
template NRVec<int> neighbors(const NRSMat<T> &adjacency, const int v); \
template NRVec<int> nonneighbors(const NRSMat<T> &adjacency, const int v); \
template NRVec<int> findclique(const NRSMat<T> &adjacency, const int v); \
template NRVec<int> complement(const NRSMat<T> &adjacency, const NRVec<int> & vertexlist); \
template NRVec<int> cliquecover(const NRSMat<T> &adjacency); \



INSTANTIZE(char)

}//namespace
