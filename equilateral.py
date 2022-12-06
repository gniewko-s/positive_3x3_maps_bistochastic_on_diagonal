import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from typing import Callable, Sequence

def satisfy_original(a: float,b: float,c: float,phi: float,r: float,tol: float=1e-6) -> bool:
	"""satisfy_original(a,b,c,phi,r,tol=1e-6) checks if the point (phi,r) saturates the boundary condition for parameters (a,b,c) with tolerance tol"""
	res = ((a-1-r*np.cos(phi))**2 - 3*r**2*np.sin(phi)**2)**.5 + ( ((b+c)/2 + 2*r*np.cos(phi))**2 - ((b-c)/2)**2 )**.5 - 1
	return np.abs(res) < tol 

def in_triangle(a: float,b: float,c: float,phi: float,r: float) -> bool:
	"""For the value of cutoff = .5 check if the solution contains in the blue triangle implied by the vetres conditions.
For higher values of the cutoff parameter, the solution will be drawn also outside the blue triangle"""
	return r <= .5* min(a-1,b,c) / max(-np.cos(phi),-np.cos(phi+2*np.pi/3),-np.cos(phi-2*np.pi/3))

def on_roots(a: float,b: float,c: float,phi: float,roots: np.ndarray) -> float:
	"""For the array of four roots calculated for the given phi from the edge condition with parameters a,b,c, put np.NaN for roots non-positive, out of trangle or not satisfying the original conditions and returns their mininum, possibly being a np.NaN value"""
	roots = [r.real if r.imag == 0 and r>0 and satisfy_original(a,b,c,phi,r) and in_triangle(a,b,c,phi,r) else np.NaN for r in roots] 
	m = np.nanmin(roots)
	return m

def arc_polar(a: float,b: float,c: float) -> Callable[[np.ndarray], np.ndarray]:
	"""For given a,b,c returns a function inner
Function inner returns a solution (1-d np.array) of the corresponding edge condition for a given range of angle (1-d np.array)"""
	def inner(phi: np.ndarray) -> np.ndarray:
		coeff4 = 9/4; coeff4 = coeff4*np.ones(phi.shape)
		coeff3 = 3*(a+b+c-1) * np.cos(phi)
		coeff2 = ((a+b+c-1)**2-4)*np.cos(phi)**2 - 1.5*((a-1)**2-b*c-1)
		coeff1 = -((a+b+c-1)*((a-1)**2-b*c-1)+2*(b+c))*np.cos(phi)
		coeff0 = ((a-1)**2-b*c-1)**2/4-b*c; coeff0 = coeff0*np.ones(phi.shape)
		P = np.array([coeff4,coeff3,coeff2,coeff1,coeff0]).transpose()
		P = np.array([on_roots(a,b,c,phii,np.roots(p))*6**.5 for phii, p in zip(phi,P)])
		return P
	return inner

def on_CP_roots(roots: np.ndarray, bound: float):
	"""elements of the array roots which are not positive are mapped to np.NaN. Elements greater than bound are restricted to the value of bound."""
	roots = [(r.real if r < bound else  bound) if r.imag == 0 and r>0 else np.NaN  for r in roots] 
	m = np.nanmin(roots)
	return m

def CP(a:float, b:float, c:float) -> Callable[[np.ndarray], np.ndarray]:
	"""For given a,b,c returns a function inner
Function inner returns a solution (1-d np.array) of the corresponding CP condition for a given range of angle (1-d np.array)"""
	if a <= 3:
		return lambda phi: np.nan * phi
	def inner(phi: np.ndarray) -> np.ndarray:
		coeff3 = np.cos(phi)*(np.cos(phi)**2-.75) * (2/3)**1.5
		coeff2 = -(a-1)/2 * np.ones(phi.shape)
		coeff1 = np.zeros(phi.shape)
		coeff0 = a**2*(a-3) * np.ones(phi.shape)
		P = np.array([coeff3,coeff2,coeff1,coeff0]).transpose()
		tri = (1.5**.5 * min(a-1,b,c) / max(-np.cos(phii),-np.cos(phii+2*np.pi/3),-np.cos(phii-2*np.pi/3)) for phii in phi)
		P = np.array([on_CP_roots(np.roots(p),bound) for p,bound in zip(P, tri)])
		return P
	return inner

def plot(a:float,b:float,c:float) -> Sequence[np.ndarray]:
	"""For a given values of parameters a,b,c, the function returns the arrays of points of the plots:
* XV,YV - the 1-d numpy arrays of respectively x and y coordinates of the blue triangle arising from the vertex conditions.
* XE,YE - the 1-d numpy arrays of respectively x and y coordinates of the red triangle arising from the edge conditions.
* XP,YP - the 1-d numpy arrays of respectively x and y coordinates of the green circle arising from the hessian positivity condition.
* XC,YC - the 1-d numpy arrays of respectively x and y coordinates of the black shape arising from the CP condition.
	"""
	mu = min(a-1,b,c)
	r = 6**.5 * mu
	xmax = 3**.5/2*r
	x = np.linspace(-xmax,xmax,101)
	XV = np.concatenate((x,x))
	YV = np.concatenate((-r/2*np.ones(x.shape),r-3**.5*np.abs(x)))
	#########################################
	phi = np.linspace(np.pi/3,np.pi,201);	r = arc_polar(a,b,c)(phi);	# twice prolonged half of arc
	r = np.nanmin(np.vstack((r[:101][::-1],r[100:])), axis = 0);			# folding prolongation and taking minimum
	r = np.hstack((r,r[:0:-1]))																				# mirror to the second half
	phi = np.pi/2 + np.linspace(2*np.pi/3,4*np.pi/3,201)						# proper range of phi
	XE = np.hstack((r*np.cos(phi),	r*np.cos(phi+2*np.pi/3),	r*np.cos(phi-2*np.pi/3)))
	YE = np.hstack((r*np.sin(phi),	r*np.sin(phi+2*np.pi/3),	r*np.sin(phi-2*np.pi/3)))
	#########################################
	r = (a+b+c-((a-2*b-2*c)**2+3*(b-c)**2)**.5)/6**.5
	XP = r*np.cos(np.linspace(0,2*np.pi,601)) 
	YP= r*np.sin(np.linspace(0,2*np.pi,601))
	#########################
	phi = np.linspace(0,2*np.pi,601)
	r = CP(a,b,c)(phi)
	phi = np.pi/2 + phi
	XC = r*np.cos(phi)
	YC = r*np.sin(phi)
	return XV,YV,XE,YE,XP,YP,XC,YC

fig, ax0 = plt.subplots()
mng = plt.get_current_fig_manager()
mng.resize(*mng.window.maxsize())

ax0.set_ylim((-1.25,2.5));	ax0.set_xlim((-2.25,2.25));		ax0.set_aspect('equal');		#ax0.axis('off')
for x,y in ((0,1),(3**.5/2,-.5),(-3**.5/2,-.5)):
	ax0.arrow(0,0,1.6*x,1.6*y,head_width=5e-2)
for x,y,t in ((0, 1.5**.5,'$d=1$'),(3**.5*1.5**.5/2,-.5*1.5**.5,'$e=1$'),(-3**.5*1.5**.5/2,-.5*1.5**.5,'$f=1$')):
	ax0.plot(x,y, marker="o", markersize=4,markerfacecolor='black', markeredgecolor='black') 
	ax0.text(x+.1, y-.04, t)

ai = 3; bi = 0; ci = 0;
si1 = ai+bi+ci; si2 = bi+ci; si3 = bi-ci;
XV,YV,XE,YE,XP,YP,XC,YC = plot(ai,bi,ci)
V, = ax0.plot(XV,YV,'b',label='vertex conditions')
E, = ax0.plot(XE,YE,'r',label='edge conditions')
P, =  ax0.plot(XP,YP,'g',label='hessian positivity condition')
C, = ax0.plot(XC,YC,'k',label='complete positivity condition')

fig.subplots_adjust(left=0, bottom=0.11)
ax0.set_title(f'$a = {ai:.2f}, b = {bi:.2f}, c = {ci:.2f}, \sqrt{{bc}}+a-2 = {(bi*ci)**.5+ai-2:.2f}$',y=1.1)
slider1 = Slider(ax = fig.add_axes([0.07, 0.08, 0.86, 0.02]), label='$a+b+c$', valmin=3, valmax=4, valinit=si1)
slider2 = Slider(ax = fig.add_axes([0.07, 0.05, 0.86, 0.02]), label='b+c', valmin=0, valmax=2, valinit=si2)
slider3 = Slider(ax = fig.add_axes([0.07, 0.02, 0.86, 0.02]), label='b-c', valmin=0, valmax=2, valinit=si3)

# Functions updating the values of parameters when sliders change, taking care of positivity of parameters and $2-a < \sqrt{bc}$ to be satisfied.

def update_1(val: float):
	s1 = slider1.val; s2 = slider2.val; s3 = slider3.val
	for sli in (slider1, slider2, slider3):
		sli.eventson = False
	if (s2**2-s3**2)**.5/2+s1-s2 < 2:
		s3 = (-(4-2*s1+2*s2)**2+s2**2)**.5
		slider3.set_val(s3)
	if s3>s2:
		s2=s3
		slider2.set_val(s2)
	redraw(s1,s2,s3)
	for sli in (slider1, slider2, slider3):
		sli.eventson = True
	

def update_2(val: float):
	s1 = slider1.val; s2 = slider2.val; s3 = slider3.val
	for sli in (slider1, slider2, slider3):
		sli.eventson = False
	if s2<s3:
		s3=s2
		slider3.set_val(s3)
	if (s2**2-s3**2)**.5/2+s1-s2 < 2:
		s1 = 2 + s2 - (s2**2-s3**2)**.5/2
		slider1.set_val(s1)
	redraw(s1,s2,s3)
	for sli in (slider1, slider2, slider3):
		sli.eventson = True

def update_3(val: float):
	s1 = slider1.val; s2 = slider2.val; s3 = slider3.val
	for sli in (slider1, slider2, slider3):
		sli.eventson = False
	if s3>s2:
		s2=s3
		slider2.set_val(s2)
	if (s2**2-s3**2)**.5/2+s1-s2 < 2:
		s1 = 2 + s2 - (s2**2-s3**2)**.5/2
		slider1.set_val(s1)
	redraw(s1,s2,s3)
	for sli in (slider1, slider2, slider3):
		sli.eventson = True

# generate new data for plots and redraw; a common callback of the above slider functions.

def redraw(s1:float,s2:float,s3:float):
	global V,E,P,K
	b = (s2+s3)/2; c = (s2-s3)/2; a = s1-s2
	XV,YV,XE,YE,XP,YP,XC,YC= plot(a,b,c)
	V.set_xdata(XV)
	V.set_ydata(YV)
	P.set_xdata(XP)
	P.set_ydata(YP)
	C.set_xdata(XC)
	C.set_ydata(YC)
	E.set_xdata(XE)
	E.set_ydata(YE)
	ax0.set_title(f'$a = {a:.2f}, b = {b:.2f}, c = {c:.2f}, \sqrt{{bc}}+a-2 = {(b*c)**.5+a-2:.2f}$',y=1.1)
	fig.canvas.draw_idle()

slider1.on_changed(update_1); slider2.on_changed(update_2); slider3.on_changed(update_3)

ax0.legend(loc='upper left')
plt.show()
