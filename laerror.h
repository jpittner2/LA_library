#ifndef _LAERROR_H_
#define _LAERROR_H_

//exception class for laerror
class LAerror
	{
	const char *msg;
	public:
	LAerror(const char *s) {msg=s;};
	};

extern void laerror(const char *);

#endif
