/*
    LA: linear algebra C++ interface library graphical test program
    Copyright (C) 2022 Jiri Pittner <jiri.pittner@jh-inst.cas.cz> or <jiri@pittnerovi.com>

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


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <time.h>

#include "la.h"
#include "vecmat3.h"
#include "quaternion.h"
#include "permutation.h"
#include "polynomial.h"
#include "contfrac.h"
#include "simple.h"


#include <X11/Xlib.h>
#include <X11/keysym.h>
#include <X11/Xos.h>
#include <X11/Xutil.h>
#include <X11/Xatom.h>
#include <X11/Xresource.h>

using namespace std;
using namespace LA_Vecmat3;
using namespace LA_Quaternion;
using namespace LA_Simple;
using namespace LA;



/*
* function: create_simple_window. Creates a window with a white background
*
in the given size.
* input:
display, size of the window (in pixels), and location of the window
*
(in pixels).
* output:
the window's ID.
* notes:
window is created with a black border, 2 pixels wide.
*
the window is automatically mapped after its creation.
*/


Window create_simple_window(Display* display, int width, int height, int x, int y)
{
int screen_num = DefaultScreen(display);
int win_border_width = 2;
Window win;
/* create a simple window, as a direct child of the screen's */
/* root window. Use the screen's black and white colors as
*/
/* the foreground and background colors of the window,
*/
/* respectively. Place the new window's top-left corner at
*/
/* the given 'x,y' coordinates.
*/
win = XCreateSimpleWindow(display, RootWindow(display, screen_num),
x, y, width, height, win_border_width,
BlackPixel(display, screen_num),
WhitePixel(display, screen_num));
/* make the window actually appear on the screen. */
XMapWindow(display, win);
/* flush all pending requests to the X server. */
XFlush(display);
return win;
}



GC create_gc(Display* display, Window win)
{
GC gc;
/* handle of newly created GC. */
unsigned long valuemask = 0;
/* which values in 'values' to */
/* check when creating the GC. */
XGCValues values;
/* initial values for the GC.
*/
unsigned int line_width = 1;
/* line width for the GC.
*/
int line_style = LineSolid;
/* style for lines drawing and */
int cap_style = CapButt;
/* style of the line's edje and */
int join_style = JoinBevel;
/* joined lines.
*/
int screen_num = DefaultScreen(display);
gc = XCreateGC(display, win, valuemask, &values);
if (!gc) {
fprintf(stderr, "error XCreateGC: \n");
}
/* define the style of lines that will be drawn using this GC. */
XSetLineAttributes(display, gc,
line_width, line_style, cap_style, join_style);
/* define the fill style for the GC. to be 'solid filling'. */
XSetFillStyle(display, gc, FillSolid);




return gc;
}


typedef struct {
Vec3<double> point0;
Vec3<double> point1;
} LINE;

template<>
class LA::LA_traits<LINE> {
public:
        static bool is_plaindata() {return true;};
	typedef double normtype;
};



void plotobject(Display* display, Pixmap pixmap, GC gc, int width,int height, const NRVec<LINE> &lines, const Mat3<double> &camera_angle, const Vec3<double> &camera, const Vec3<double> &plane_to_camera, double xod,double xdo,double yod,double ydo)
{
for(int i=0; i<lines.size(); ++i)
	{
	double xy0[2],xy1[2];
	perspective(&xy0[0], lines[i].point0, camera_angle, camera, plane_to_camera);
	perspective(&xy1[0], lines[i].point1, camera_angle, camera, plane_to_camera);
	int x0,x1,y0,y1;
        x0=width*(xy0[0]-xod)/(xdo-xod);
        x1=width*(xy1[0]-xod)/(xdo-xod);
        y0=height-height*(xy0[1]-yod)/(ydo-yod);
        y1=height-height*(xy1[1]-yod)/(ydo-yod);
        XDrawLine(display, pixmap, gc, x0, y0, x1, y1);
	}
}


int main(int argc, char **argv)
{
sigtraceback(SIGSEGV,1);
sigtraceback(SIGABRT,1);
sigtraceback(SIGBUS,1);
sigtraceback(SIGFPE,1);



//cout.setf(ios::scientific);
cc:cout.setf(ios::fixed);
cout.precision(10);
cin.exceptions ( ifstream::eofbit | ifstream::failbit | ifstream::badbit );


Display* display;
// pointer to X Display structure.
int screen_num;
// number of screen to place the window on.
Window win;
// pointer to the newly created window.
unsigned int display_width,
display_height; // height and width of the X display.
unsigned int width, height;
// height and width for the new window.
char *display_name = getenv("DISPLAY"); // address of the X display.
GC gc;
// GC (graphics context) used for drawing in our window.
// open connection with the X server. 
display = XOpenDisplay(display_name);
if (display == NULL) {
fprintf(stderr, "%s: cannot connect to X server '%s'\n",
argv[0], display_name);
exit(1);
}
/* get the geometry of the default screen for our display. */
screen_num = DefaultScreen(display);
display_width = DisplayWidth(display, screen_num);
display_height = DisplayHeight(display, screen_num);
/* make the new window occupy 1/9 of the screen's size. */
width = (display_width *.9);
height = (display_height *.9);
printf("window width - '%d'; height - '%d'\n", width, height);
/* create a simple window, as a direct child of the screen's
*/
/* root window. Use the screen's white color as the background */
/* color of the window. Place the new window's top-left corner */
/* at the given 'x,y' coordinates.
*/
win = create_simple_window(display, width, height, 0, 0);

//create pixmap and map it as window background - trick to avoid need of treating expose events
Pixmap pixmap;
pixmap = XCreatePixmap(display, RootWindow(display, DefaultScreen(display)), width, height, DefaultDepth(display, DefaultScreen(display)));

XSetWindowBackgroundPixmap(display, win, pixmap);

/* allocate a new GC (graphics context) for drawing in the window. */
gc = create_gc(display, pixmap);
XSetBackground(display, gc, WhitePixel(display, DefaultScreen(display)));
XSetForeground(display, gc, WhitePixel(display, DefaultScreen(display)));
XFillRectangle(display, pixmap, gc, 0, 0, width, height);
XSetForeground(display, gc, BlackPixel(display, DefaultScreen(display)));

//display it
XFlush(display); XSync(display, False);

//colors
Colormap screencolormap;
XColor mycolor;
Status rc;
screencolormap= DefaultColormap(display,DefaultScreen(display));
rc= XAllocNamedColor(display,screencolormap,"blue",&mycolor,&mycolor);
if(rc==0) fprintf(stderr,"selhal xallocnamedcolor\n");
XSetForeground(display,gc,mycolor.pixel);
XFlush(display); XSync(display, False);



XSetForeground(display, gc, BlackPixel(display, DefaultScreen(display)));//vratit cernou 
//XDrawLine(display, pixmap, gc, 0, height/2, width, height/2);
//XDrawLine(display, pixmap, gc, width/2, 0, width/2, height);
//XSetForeground(display,gc,mycolor.pixel);



double xod, xdo, yod, ydo;
xod = -100; xdo=100;
yod = -100; xdo=100;

LINE lines0[]= {
{{0,0,0},{0,0,5}},
{{0,5,5},{0,0,5}},
{{0,5,5},{0,5,0}},
{{0,5,0},{0,0,0}},
{{5,0,0},{5,0,5}},
{{5,5,5},{5,0,5}},
{{5,5,5},{5,5,0}},
{{5,5,0},{5,0,0}},
{{0,0,0},{5,0,0}},
{{0,0,5},{5,0,5}},
{{0,5,0},{5,5,0}},
{{0,5,5},{5,5,5}},
};
NRVec<LINE> lines(lines0);

Mat3<double> camera_angle;
double eul[3]={40.*M_PI/180.,40.*M_PI/180.,10.*M_PI/180.};
euler2rotmat(eul,camera_angle,"xyz");
Vec3<double> camera={50,30,40};
Vec3<double> plane_to_camera={-20,-20,-20};


XSetForeground(display,gc,mycolor.pixel);
plotobject(display, pixmap, gc, width,height, lines,camera_angle,camera,plane_to_camera,xod,xdo,yod,ydo);

XClearWindow(display,win); //expose the pixmap
XSetForeground(display, gc, BlackPixel(display, DefaultScreen(display)));//vratit cernou			   
XFlush(display); XSync(display, False);

//prepare to receive events
XSelectInput(display, win, KeyPressMask|KeyReleaseMask|ButtonPressMask|ButtonReleaseMask|Button1MotionMask|PointerMotionMask);
XEvent event;
int ispressed1=0;
int ispressed2=0;
int ispressed3=0;
int x,y,x0,y0;
while(1) //LOOP ....  process events 
	{
	XNextEvent(display, &event);
		switch (event.type)
			{
			case KeyPress:
				{
				char text[16];
				KeySym mykey;
				int n=XLookupString(&event.xkey, text,16,&mykey,0);
				if(n==1&&text[0]=='q') goto done;
				}
				break;
			case ButtonPress:
				if(event.xbutton.button==Button1) ispressed1=1;			
				if(event.xbutton.button==Button2) ispressed2=1;			
				if(event.xbutton.button==Button3) ispressed3=1;			
				break;
			case MotionNotify:
				x = event.xbutton.x;
				y = event.xbutton.y;
				if(ispressed1) 
					{
					if(x0>x) eul[0]+=0.01;
                                	if(x0<x) eul[0]-=0.01;
                                	if(y0>y) eul[1]+=0.01;
                                	if(y0<y) eul[1]-=0.01;
					}
				if(ispressed2)
                                        {
                                        }
				if(ispressed3)
                                        {
                                        }
				if(ispressed1||ispressed2||ispressed3)
					{
					XSetForeground(display, gc, WhitePixel(display, DefaultScreen(display)));
					XFillRectangle(display, pixmap, gc, 0, 0, width, height);
					XSetForeground(display, gc, BlackPixel(display, DefaultScreen(display)));

					euler2rotmat(eul,camera_angle,"xyz");
					plotobject(display, pixmap, gc, width,height, lines,camera_angle,camera,plane_to_camera,xod,xdo,yod,ydo);
					}
				x0=x;
				y0=y;
				break;
			case ButtonRelease:
				if(event.xbutton.button==Button1) ispressed1=0;
				if(event.xbutton.button==Button2) ispressed2=0;
				if(event.xbutton.button==Button3) ispressed3=0;
                                break;
			case Expose:
			default: 
				break;
			}
	XClearWindow(display,win); //expose the pixmap
	XFlush(display); XSync(display, False);
	}

done:

/* close the connection to the X server. */
XCloseDisplay(display);


}
