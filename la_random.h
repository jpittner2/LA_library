#ifndef _LA_RANDOM_H
#define _LA_RANDOM_H

#include <stdint.h>

namespace LA {

extern double randdouble();
extern double randdoublesigned();
extern int randint32();
extern uint64_t randint64();

//RANDOM numbers defaulting to standard library but switchable to user's functions
#ifndef RANDDOUBLE
#define RANDDOUBLE LA::randdouble
#endif

#ifndef RANDDOUBLESIGNED
#define RANDDOUBLESIGNED LA::randdoublesigned
#endif

#ifndef RANDINT32
#define RANDINT32 LA::randint32
#endif

#ifndef RANDINT64
#define RANDINT64 LA::randint64
#endif


#ifdef __GNUC__
#define WEAK_SYMBOL __attribute__((weak)) 
#else
#define WEAK_SYMBOL 
#endif

}//namespace

#endif
