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


template<typename T>
class AuxStorage
	{
	char filename[16];
	int fd;
	off64_t recl;
public:
	AuxStorage(void);
	~AuxStorage(void) {close(fd);};
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
strcpy(filename,"AuxStorageXXXXXX"); 
mktemp(filename); 
unlink(filename);
fd=open(filename,O_CREAT|O_LARGEFILE);
if(fd<0) {perror(""); laerror("open failed in AuxStorage");}
recl=0;
}

template<typename T>
void AuxStorage<T>::get(NRVec<T> &x, const int pos) const
{
if(recl==0) laerror("get from an empty file in AuxStorage");
if((off64_t)-1 == lseek64(fd,pos*recl,SEEK_SET)) {perror(""); laerror("seek failed in AuxStorage");}
if(recl!=read(fd,&x[0],recl)) {perror(""); laerror("read failed in AuxStorage");}
}

template<typename T>
void AuxStorage<T>::put(const NRVec<T> &x, const int pos) const
{
if(recl) {if(recl!=x.size()*sizeof(T)) laerror("attempt to write objects of different size to one AuxStorage");}
else recl=x.size()*sizeof(T);
if((off64_t)-1 == lseek64(fd,pos*recl,SEEK_SET)) {perror(""); laerror("seek failed in AuxStorage");}
if(0>write(fd,&x[0],recl)) {perror(""); laerror("write failed in AuxStorage");}
}


#endif
