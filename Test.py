from Macroelement_Driver import Macroelement
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time
from PIL import Image

# Process profile content: generate a cachegrind file and send it to user.

# You can also write the result to the console:

matplotlib.use('tkagg')
plt.rcParams.update({
"text.usetex": True,
"font.family": "serif"})
matplotlib.rcParams['font.sans-serif']

KN = 20732
KV = 20691
KM = 18960
mu = 0.48
phi = 0.44
beta = 0.95
m_r = 4
m_t = 2
Br  = 0.35
X   = 1
R   = 0.0002
B   = 0.85
A   = 1
chi = 3020
lh  = 0.5
lm  = 0.5
k   = 1
t0  = 0.0024
m0 	= 0.1201
b3  = 0.8820
b4  = 0.9704

max_ksubstep = 1000
error_tolerance= 1e-4
tol_MNRiter = 1e-10
control 	= 1
nstep	    = 500


Elastic  	 = np.array([KN*(200/200)**0.3,KV*(200/200)**0.3,KM*(200/200)**0.3,0], np.float64)
YieldSurface = [0.098,0.1201,0.9001,1,0.8820,0.9704,-0.0829,0.0024]
PlasticFlow  = np.array([0.2,0.2,1,1,1,1,-0.0829,1])
ElasticNucleous = np.array([m_r, m_t, Br, X ,R])
uplift	 	 = np.array([4,1])
Geometry 	 = np.array([B,A])
Settlement   = np.array([chi])
Nonlinearity = np.array([1.2*(200/200)**0.5,2.5])
Integration  = np.array([max_ksubstep, error_tolerance, tol_MNRiter])
Lep 		 = (1/ElasticNucleous[0])*np.array([[Elastic[0],0,0],[0,Elastic[1],-Elastic[3]],[0,-Elastic[3],Elastic[2]]], np.float64)

timel 	 	 = np.linspace(0,3,round(3/0.001))
vload 		 = (np.array(np.linspace(0,0,100)))
hload 		 = (np.array(np.linspace(0,0.01,100)))
mload		 = (np.array(np.linspace(0,-0.1,100)))

linload      = np.append(np.array(np.linspace(0,1,4000)) , np.array(np.linspace(1,1,4000)))
vload 		 = 0*(np.sin(1*np.pi*timel))
hload 		 = 0.01*(np.sin(1*np.pi*timel))
mload		 = 0.03*(np.sin(1*np.pi*timel))

vload_total  = np.diff(vload)
hload_total  = np.diff(hload)
mload_total  = np.diff(mload)

Load  		 = np.column_stack([vload_total, hload_total, mload_total])


sig0 	     = np.array([200,0,0], np.float64)
eps0 		 = np.array([0,0,0], np.float64)
istrain0     = np.array([0,0,0], np.float64)
Vf 			 = np.array(2075, np.float64)
Constraints  = [2]
uplift_code  = 'Active'
#print(type(eps0.astype(float)))

Initial_state = Macroelement.gety(eps0,sig0,istrain0,Vf)
Test1         = Macroelement(Elastic, YieldSurface, PlasticFlow, Nonlinearity, ElasticNucleous, uplift,
                 Geometry, Settlement, Integration, Load, Initial_state, Constraints, uplift_code)


SS , EE, HARD, State_vars = Test1.UpdateHypo(Initial_state, [4,1,t0,B,m0,b3,b4])


w = EE[:,0]
v = EE[:,1]
r = EE[:,2]
N = SS[:,0]
V = SS[:,1]
M = SS[:,2]


fig = plt.figure(figsize= (9,3), dpi = 200)
ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3)
ax1.grid(color='b', ls = '-.', lw = 0.25)
ax2.grid(color='b', ls = '-.', lw = 0.25)
ax3.grid(color='b', ls = '-.', lw = 0.25)
ax1.plot(r,w, 'g-')
ax1.set_title("Vertical")
ax2.plot(v,V, 'b-')
ax2.set_title("Horizontal")
ax3.plot(r,M, 'r-')
ax3.set_title("Móment")
plt.savefig('Figure-1.png')
plt.show()


# im = Image.open(r"Figure-1.png")
# im.show()

