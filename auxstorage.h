#ifndef _AUXSTORAGE_H_
#define _AUXSTORAGE_H_
#include "laerror.h"
#include "vec.h"
#include "mat.h"
#include "smat.h"

#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

//CAUTION:
//it will not work if T is itself a class with dynamically allocated components
//it cannot be implemented for SparseMat, which lacks fixed record length


template<typename T>
class AuxStorage
	{
	char filename[32];
	int fd;
	bool deleteme;
	size_t recl;
public:
	AuxStorage(void);
	AuxStorage(const char *name);
	~AuxStorage(void) {close(fd); if(deleteme) unlink(filename);};
	void get(NRVec<T> &x, const int pos) const;
	void put(const NRVec<T> &x, const int pos);
        void get(NRMat<T> &x, const int pos) const;
        void put(const NRMat<T> &x, const int pos);
        void get(NRSMat<T> &x, const int pos) const;
        void put(const NRSMat<T> &x, const int pos);
	};

template<typename T>
AuxStorage<T>::AuxStorage(void)
{
strcpy(filename,"AUX_XXXXXX"); 
mktemp(filename); 
unlink(filename);
fd=open(filename,O_CREAT|O_LARGEFILE|O_RDWR,0777);
if(fd<0) {perror(""); laerror("open failed in AuxStorage");}
recl=0;
deleteme=1;
}

template<typename T>
AuxStorage<T>::AuxStorage(const char *name)
{
strcpy(filename,name);
unlink(filename);
fd=open(filename,O_CREAT|O_LARGEFILE|O_RDWR,0777);
if(fd<0) {perror(""); laerror("open failed in AuxStorage");}
recl=0;
deleteme=0;
}


//vectors
template<typename T>
void AuxStorage<T>::get(NRVec<T> &x, const int pos) const
{
if(recl==0) laerror("get from an empty file in AuxStorage");
if((off64_t)-1 == lseek64(fd,pos*((off64_t)recl),SEEK_SET)) {perror(""); laerror("seek failed in AuxStorage");}
if((ssize_t)recl!=read(fd,&x[0],recl)) {perror(""); laerror("read failed in AuxStorage");}
}

template<typename T>
void AuxStorage<T>::put(const NRVec<T> &x, const int pos)
{
if(recl) {if(recl!=x.size()*sizeof(T)) laerror("attempt to write objects of different size to one AuxStorage");}
else recl=x.size()*sizeof(T);
if((off64_t)-1 == lseek64(fd,pos*((off64_t)recl),SEEK_SET)) {perror(""); laerror("seek failed in AuxStorage");}
if(0>write(fd,&x[0],recl)) {perror(""); laerror("write failed in AuxStorage");}
}

//matrices
template<typename T>
void AuxStorage<T>::get(NRMat<T> &x, const int pos) const
{
if(recl==0) laerror("get from an empty file in AuxStorage");
if((off64_t)-1 == lseek64(fd,pos*((off64_t)recl),SEEK_SET)) {perror(""); laerror("seek failed in AuxStorage");}
if((ssize_t)recl!=read(fd,&x(0,0),recl)) {perror(""); laerror("read failed in AuxStorage");}
}

template<typename T>
void AuxStorage<T>::put(const NRMat<T> &x, const int pos)
{
if(recl) {if(recl!=x.nrows()*x.ncols()*sizeof(T)) laerror("attempt to write objects of different size to one AuxStorage");}
else recl=x.nrows()*x.ncols()*sizeof(T);
if((off64_t)-1 == lseek64(fd,pos*((off64_t)recl),SEEK_SET)) {perror(""); laerror("seek failed in AuxStorage");}
if(0>write(fd,&x(0,0),recl)) {perror(""); laerror("write failed in AuxStorage");}
}

//packed symmetric matrices
template<typename T>
void AuxStorage<T>::get(NRSMat<T> &x, const int pos) const
{
if(recl==0) laerror("get from an empty file in AuxStorage");
if((off64_t)-1 == lseek64(fd,pos*((off64_t)recl),SEEK_SET)) {perror(""); laerror("seek failed in AuxStorage");}
if((ssize_t)recl!=read(fd,&x(0,0),recl)) {perror(""); laerror("read failed in AuxStorage");}
}

template<typename T>
void AuxStorage<T>::put(const NRSMat<T> &x, const int pos)
{
if(recl) {if(recl!=x.nrows()*(x.nrows()+1)/2*sizeof(T)) laerror("attempt to write objects of different size to one AuxStorage");}
else recl=x.nrows()*(x.nrows()+1)/2*sizeof(T);
if((off64_t)-1 == lseek64(fd,pos*((off64_t)recl),SEEK_SET)) {perror(""); laerror("seek failed in AuxStorage");}
if(0>write(fd,&x(0,0),recl)) {perror(""); laerror("write failed in AuxStorage");}
}




#endif
