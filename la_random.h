#ifndef _LA_RANDOM_H
#define _LA_RANDOM_H

namespace LA {

extern double randdouble();
extern double randdoublesigned();
extern int randint32();

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

#ifdef __GNUC__
#define WEAK_SYMBOL __attribute__((weak)) 
#else
#define WEAK_SYMBOL 
#endif

}//namespace

#endif
