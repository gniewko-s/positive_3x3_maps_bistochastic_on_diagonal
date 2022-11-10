import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def satisfy_original(a,b,c,phi,r,tol=1e-6):
	res = ((a-1-r*np.cos(phi))**2 - 3*r**2*np.sin(phi)**2)**.5 + ( ((b+c)/2 + 2*r*np.cos(phi))**2 - ((b-c)/2)**2 )**.5 - 1
	return np.abs(res) < tol 

def in_triangle(a,b,c,phi,r):
	return r <= .6 * min(a-1,b,c) / max(-np.cos(phi),-np.cos(phi+2*np.pi/3),-np.cos(phi-2*np.pi/3))	# .5 to fitt the cut off to triangle

def on_roots(a,b,c,phi,roots):
	roots = [r.real if r.imag == 0 and r>0 and satisfy_original(a,b,c,phi,r) and in_triangle(a,b,c,phi,r) else np.NaN for r in roots] 
	m = np.nanmin(roots)
	return m

def arc_polar(a,b,c):
	def inner(phi):
		coeff4 = 9/4; coeff4 = coeff4*np.ones(phi.shape)
		coeff3 = 3*(a+b+c-1) * np.cos(phi)
		coeff2 = ((a+b+c-1)**2-4)*np.cos(phi)**2 - 1.5*((a-1)**2-b*c-1)
		coeff1 = -((a+b+c-1)*((a-1)**2-b*c-1)+2*(b+c))*np.cos(phi)
		coeff0 = ((a-1)**2-b*c-1)**2/4-b*c; coeff0 = coeff0*np.ones(phi.shape)
		P = np.array([coeff4,coeff3,coeff2,coeff1,coeff0]).transpose()
		P = np.array([on_roots(a,b,c,phii,np.roots(p))*6**.5 for phii, p in zip(phi,P)])
		return P
	return inner

def plot(a,b,c):
	mu = min(a-1,b,c)
	r = 6**.5 * mu
	xmax = 3**.5/2*r
	x = np.linspace(-xmax,xmax,101)
	XV = np.concatenate((x,x))
	YV = np.concatenate((-r/2*np.ones(x.shape),r-3**.5*np.abs(x)))
	phi = np.linspace(np.pi/3,5*np.pi/3,201)
	r = arc_polar(a,b,c)(phi)
	phi += np.pi/2
	XE = [r*np.cos(phi),	r*np.cos(phi+2*np.pi/3),	r*np.cos(phi-2*np.pi/3)]
	YE = [r*np.sin(phi),	r*np.sin(phi+2*np.pi/3),	r*np.sin(phi-2*np.pi/3)]
	r = (a+b+c-((a-2*b-2*c)**2+3*(b-c)**2)**.5)/6**.5
	XP = r*np.cos(np.linspace(0,2*np.pi,601)) 
	YP= r*np.sin(np.linspace(0,2*np.pi,601))
	return XV,YV,XE,YE,XP,YP

fig, ax = plt.subplots()
ax.set_ylim((-1.35,2.6))
ax.set_xlim((-2.25,2.25))
ax.set_aspect('equal')
ax.axis('off')
for x,y in ((0,1),(3**.5/2,-.5),(-3**.5/2,-.5)):
	ax.arrow(0,0,2.5*x,2.5*y,head_width=5e-2)
for x,y,t in ((0, 1.5**.5,'$d=1$'),(3**.5*1.5**.5/2,-.5*1.5**.5,'$e=1$'),(-3**.5*1.5**.5/2,-.5*1.5**.5,'$f=1$')):
	plt.plot(x,y, marker="o", markersize=4,markerfacecolor='black', markeredgecolor='black') 
	ax.text(x+.1, y-.1, t)

ai = 3; bi = 0; ci = 0;
si1 = ai+bi+ci; si2 = bi+ci; si3 = bi-ci;
XV,YV,XE,YE,XP,YP = plot(ai,bi,ci)
V, = ax.plot(XV,YV,'b')
E = [ax.plot(X,Y,'r')[0] for X,Y in zip(XE,YE)]
P, =  ax.plot(XP,YP,'g')

fig.subplots_adjust(left=0.25, bottom=0.2)
ax.set_title(f'$a = {ai:.2f}, b = {bi:.2f}, c = {ci:.2f}, \sqrt{{bc}}+a-2 = {(bi*ci)**.5+ai-2:.2f}$',y=1.1, pad=-14)
slider1 = Slider(ax = fig.add_axes([0.25, 0.15, 0.65, 0.03]), label='$a+b+c$', valmin=3, valmax=4, valinit=si1)
slider2 = Slider(ax = fig.add_axes([0.25, 0.1, 0.65, 0.03]), label='b+c', valmin=0, valmax=2, valinit=si2)
slider3 = Slider(ax = fig.add_axes([0.25, 0.05, 0.65, 0.03]), label='b-c', valmin=0, valmax=2, valinit=si3)

def update_1(val):
	s1 = slider1.val; s2 = slider2.val; s3 = slider3.val
	if (s2**2-s3**2)**.5/2+s1-s2 < 2:
		s3 = (-(4-2*s1+2*s2)**2+s2**2)**.5
		slider3.set_val(s3)
	if s3>s2:
		s2=s3
		slider2.set_val(s2)
	redraw(s1,s2,s3)

def update_2(val):
	s1 = slider1.val; s2 = slider2.val; s3 = slider3.val
	if s2<s3:
		s3=s2
		slider3.set_val(s3)
	if (s2**2-s3**2)**.5/2+s1-s2 < 2:
		s1 = 2 + s2 - (s2**2-s3**2)**.5/2
		slider1.set_val(s1)
	redraw(s1,s2,s3)

def update_3(val):
	s1 = slider1.val; s2 = slider2.val; s3 = slider3.val
	if s3>s2:
		s2=s3
		slider2.set_val(s2)
	if (s2**2-s3**2)**.5/2+s1-s2 < 2:
		s1 = 2 + s2 - (s2**2-s3**2)**.5/2
		slider1.set_val(s1)
	redraw(s1,s2,s3)

def redraw(s1,s2,s3):
	global V,E
	b = (s2+s3)/2; c = (s2-s3)/2; a = s1-s2
	XV,YV,XE,YE ,XP,YP= plot(a,b,c)
	V.set_xdata(XV)
	V.set_ydata(YV)
	P.set_xdata(XP)
	P.set_ydata(YP)
	for e,x in zip(E,XE):
		e.set_xdata(x)
	for e,y in zip(E,YE):
		e.set_ydata(y)
	ax.set_title(f'$a = {a:.2f}, b = {b:.2f}, c = {c:.2f}, \sqrt{{bc}}+a-2 = {(b*c)**.5+a-2:.2f}$')
	fig.canvas.draw_idle()

slider1.on_changed(update_1); slider2.on_changed(update_2); slider3.on_changed(update_3)

plt.show()
