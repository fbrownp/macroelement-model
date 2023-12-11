import numpy as np
import numba as nb

class Macroelement():


	def __init__(self, Elastic, YieldSurface, PlasticFlow, Nonlinearity, ElasticNucleous, uplift, Geometry, Settlement, Integration, Load, Initial_state, Constraints, upliftcode):
		self.elastic 			= np.array(Elastic , np.float64)
		self.yieldsurface 		= np.array(YieldSurface , np.float64)
		self.plasticflow 		= np.array(PlasticFlow , np.float64)
		self.elasticnucleous 	= np.array(ElasticNucleous , np.float64)
		self.geometry 			= np.array(Geometry , np.float64)
		self.settlement 		= np.array(Settlement , np.float64)
		self.integration 	    = np.array(Integration , np.float64)
		self.load 				= np.array(Load , np.float64)
		self.initial_state    	= np.array(Initial_state , np.float64)
		self.nonlinearity   	= np.array(Nonlinearity , np.float64)
		self.constraints   		= np.array(Constraints , np.float64)
		self.uplift 			= np.array(uplift , np.float64)
		self.upliftcode 		= upliftcode
		self.normvarF			= np.array([1, YieldSurface[0], YieldSurface[1]*Geometry[0] ], np.float64)
		self.normvarP 			= np.array([1, PlasticFlow[0]*YieldSurface[0],  PlasticFlow[1]*Geometry[0]*YieldSurface[1]  ], np.float64)
		self.b12 				= np.array(((YieldSurface[2]+YieldSurface[3])**(YieldSurface[2]+YieldSurface[3]))/(YieldSurface[2]**YieldSurface[2]*YieldSurface[3]**YieldSurface[3]), np.float64)
		self.b34 				= np.array(((YieldSurface[4]+YieldSurface[5])**(YieldSurface[4]+YieldSurface[5]))/(YieldSurface[4]**YieldSurface[4]*YieldSurface[5]**YieldSurface[5]), np.float64)
		self.b12p 				= np.array(((PlasticFlow[2]+PlasticFlow[3])**(PlasticFlow[2]+PlasticFlow[3]))/(PlasticFlow[2]**PlasticFlow[2]*PlasticFlow[3]**PlasticFlow[3]), np.float64)
		self.b34p 				= np.array(((PlasticFlow[4]+PlasticFlow[5])**(PlasticFlow[4]+PlasticFlow[5]))/(PlasticFlow[4]**PlasticFlow[4]*PlasticFlow[5]**PlasticFlow[5]), np.float64)
		# self.Mmatrix 			= np.diag([1,1/(YieldSurface[0])**2,1/(YieldSurface[1]*Geometry[0])**2])
		self.Mmatrix 			= np.identity(3, np.float64)
		self.Imatrix			= np.identity(3, np.float64)
		self.Lep 				= (1/ElasticNucleous[0])*np.array([[Elastic[0],0,0],[0,Elastic[1],-Elastic[3]],[0,-Elastic[3],Elastic[2]]], np.float64)
		self.Kelup 				= np.array([[0,0,0],[0,0,0],[0,0,0]], np.float64)
		self.Tmatrix 			= np.array([[1,0,1],[0,0,0],[1,0,1]], np.float64)
		self.THmatrix 			= np.array([[0,1,0],[1,1,1],[0,1,0]], np.float64)

	def getloads(self, loadstep):
		Load_vector = np.array(self.load[loadstep])
		nspb 		= len(self.load)
		return Load_vector, nspb

	@staticmethod
	@nb.jit(nopython=True,fastmath=True,error_model="numpy", cache = True)
	def gety(eps0, sig0, istrain0, Vf):
			y = np.append(eps0, sig0)
			y = np.append(y, Vf)
			y = np.append(y, istrain0)
			return y


	def RFK24(self,y,V,State_vars,uplift_parameters):
		tiny = 1e-10
		max_ksub = self.integration[0]
		err_tol  = self.integration[1]
		
		if np.linalg.norm(V) == 0:
			code = "None"
			return y, code, State_vars

		if self.constraints[0] == 1:
			S = np.array([[1,0,0],[0,1,0],[0,0,1]], np.float64)
			E = np.zeros((3,3), np.float64)

		elif self.constraints[0] == 2:
			S = np.array([[1,0,0],[0,0,0],[0,0,0]], np.float64)
			E = np.array([[0,0,0],[0,1,0],[0,0,1]], np.float64)
		elif self.constraints[0] == 3:
			S = np.zeros((3,3), np.float64)
			E = np.array([[1,0,0],[0,1,0],[0,0,1]], np.float64)
			
		T_j  = 0
		DT_j = 1
		y_j  = y
		ksub = 0
		Res = np.zeros(np.size(y_j), np.float64)
		code = "None"
		while T_j < 1:
			ksub += 1
			if ksub > max_ksub:
				print('--- ERROR: substep number exceeding maximum in trial state evaluation ---')
				print('execution stopped in HYPO_UPDATE')
				code = "break"
				break

			KRK_1 , State_vars_1 = self.NewtonRaphson(y_j, V, S, E,State_vars,uplift_parameters)
			y_2   = y_j + (DT_j/2)*KRK_1

			KRK_2, State_vars_2 = self.NewtonRaphson(y_2, V, S, E,State_vars_1,uplift_parameters)

			y_3   = y_j - DT_j*KRK_1 + 2*DT_j*KRK_2

			KRK_3, State_vars = self.NewtonRaphson(y_3, V, S, E, State_vars_2,uplift_parameters)

			y_hat = y_j + DT_j*KRK_2
			y_til = y_j + DT_j*((1.0/6.0)*KRK_1+(2.0/3.0)*KRK_2+(1.0/6.0)*KRK_3)

			sig_til  = y_til[3:6]
			e_til    = y_til[6]
			qint_til = y_til[7:]

			delta_sig  = y_til[3:6]-y_hat[3:6]
			delta_e	   = y_til[6]-y_hat[6]
			delta_qint = y_til[7:]-y_hat[7:]

			norm_sig = np.linalg.norm(sig_til)
			norm_e   = np.linalg.norm(e_til)
			norm_qint= np.linalg.norm(qint_til)

			if norm_qint < tiny:
				norm_qint = tiny
			if norm_sig < tiny:
				norm_sig = tiny


			Res[3:6] = (delta_sig/norm_sig)
			Res[6]   = (delta_e/norm_e)
			Res[7:]  = (delta_qint/norm_qint)

			norm_Res = np.linalg.norm(Res)
			if norm_Res < tiny or np.isnan(norm_Res) == 1:
				norm_Res = tiny

			NSS = 0.9*DT_j*(err_tol/norm_Res)**(1.0/3.0)

			if norm_Res < err_tol:
				y_j = y_til
				T_j  = T_j + DT_j
				DT_j = min(4.0*DT_j,NSS)
				DT_j = min((1.0-T_j),DT_j)
			else:
				DT_j = max(0.25*DT_j,NSS)

			#f_un , State_vars = NewtonRaphson(y_j,V,S,E,Macroelement,State_vars,uplift_parameters)
		return y_j, code, State_vars


	def UpdateHypo(self,y0,uplift_parameters):
		eps 	= np.array(y0[0:3])
		sig 	= np.array(y0[3:6])
		qint 	= np.array(y0[6:])
		istrain = np.array(y0[7:])
		Vf      = np.array(y0[6])
		State 	= np.array([0,0,0,0,0,0,0,0,0,0], np.float64)
		kstep 	= 1
		SS   	= sig.reshape(1,3)
		EE  	= eps.reshape(1,3)
		HARD    = qint.reshape(1,4)
		State_  = State.reshape(1,10)
		State_vars = State
		# nspb  	= len(self.load)
		#Load_vector, nspb = Macroelement.getloads(0)
		for V in self.load:
			#print('STRESS PATH BRANCH # ' + str(i))
			#n  = nstep[i]
			#Qk , npsb = Macroelement.getloads(i)

			#for j in range(0,n):
			kstep += 1
			print("Step" + str(kstep-1))
			#dx     = 1/n
			#V 	   = np.multiply(Qk, dx)
			#V 	   = Macroelement.load[i]
			y_k    = self.gety(eps,sig,istrain,Vf)
			y_k, code, State    = self.RFK24(y_k, V, State_vars, uplift_parameters)
			if code == "break":
				break

			eps 	= y_k[0:3]
			sig 	= y_k[3:6]
			qint 	= y_k[6:]
			istrain = y_k[7:]
			Vf      = y_k[6]
			State_vars = State

			SS      = np.append(SS , sig.reshape(1,3), axis = 0)
			EE      = np.append(EE , eps.reshape(1,3), axis = 0)
			HARD    = np.append(HARD , qint.reshape(1,4), axis = 0)
			State_ 	= np.append(State_ , State.reshape(1,10), axis = 0)

		return SS , EE , HARD, State_

	def UpdateHypoUI(self,y0,ui, uplift_parameters):
		eps 	= np.array(y0[0:3])
		sig 	= np.array(y0[3:6])
		qint 	= np.array(y0[6:])
		istrain = np.array(y0[7:])
		Vf      = np.array(y0[6])
		State 	= np.array([0,0,0,0,0,0,0,0,0,0] , np.float64)
		kstep 	= 1
		SS   	= sig.reshape(1,3)
		EE  	= eps.reshape(1,3)
		HARD    = qint.reshape(1,4)
		State_  = State.reshape(1,10)
		State_vars = State
		# nspb  	= len(Macroelement.load)
		#Load_vector, nspb = Macroelement.getloads(0)

		for V in self.load:
			#print('STRESS PATH BRANCH # ' + str(i))
			#n  = nstep[i]
			#Qk , npsb = Macroelement.getloads(i)

			#for j in range(0,n):
			kstep += 1
			print("Step" + str(kstep-1))
			#ui.progressBar.setValue(kstep/self.load.shape[0]*100)
			#dx     = 1/n
			#V 	   = np.multiply(Qk, dx)
			# V 	   = self.load[i]
			y_k    = self.gety(eps,sig,istrain,Vf)

			y_k, code, State   = self.RFK24(y_k, V, State_vars, uplift_parameters)
			if code == "break":
				break

			eps 	= y_k[0:3]
			sig 	= y_k[3:6]
			qint 	= y_k[6:]
			istrain = y_k[7:]
			Vf      = y_k[6]
			State_vars = State

			SS      = np.append(SS , sig.reshape(1,3), axis = 0)
			EE      = np.append(EE , eps.reshape(1,3), axis = 0)
			HARD    = np.append(HARD , qint.reshape(1,4), axis = 0)
			State_ 	= np.append(State_ , State.reshape(1,10), axis = 0)

		return SS , EE , HARD, State_


	@staticmethod
	@nb.jit(nopython=True,fastmath=True,error_model="numpy", cache = True)
	def NewtonRaphson_calculations_lin(Lep, S, E, V):
		A 		= np.dot(S,Lep) + E
		deps 	= np.linalg.solve(A,V)
		F 		= np.dot(Lep,deps)
		return A, deps, F

	@staticmethod
	@nb.jit(nopython=True,fastmath=True,error_model="numpy", cache = True)
	def NewtonRaphson_calculations_nonlin_uplift(Lep, S, E, V, xi, Hdel,F, T, Th):
		# Lep 	= np.multiply(Nop,T)*(1-cx) + Ku*cx + np.multiply(Lop,T) + np.multiply(Mep,Th)

		A 		= np.dot(S,Lep) + E
		F[0:3] 	= np.linalg.solve(A,V) # ddeps
		F[3:6]	= np.dot(Lep,F[0:3])	   # dsig
		F[7:10] = np.dot(Hdel,F[0:3]) 	   # dqint
		F[6] 	= np.multiply(xi,F[0]) # dVf
		return F

	@staticmethod
	@nb.jit(nopython=True,fastmath=True,error_model="numpy", cache = True)
	def NewtonRaphson_calculations_nonlin(Lep, S, E, V, xi, Hdel,F):
		A 		= np.dot(S,Lep) + E
		F[0:3] 	= np.linalg.solve(A,V) # ddeps
		F[3:6]	= np.dot(Lep,F[0:3])	   # dsig
		F[7:10] = np.dot(Hdel,F[0:3]) 	   # dqint
		F[6] 	= np.multiply(xi,F[0]) # dVf
		return F

	def NewtonRaphson(self,y,V,S,E,State_vars,uplift_parameters):
		xi 		= self.settlement[0]
		sig     = y[3:6]
		qint    = y[6:]
		Vf      = y[6]
		istrain = y[7:]
		F 		= np.zeros(y.shape[0])
		Nep 	= self.Hypo_Stiff(sig , Vf)
		#Compute A matrix and solve for deps
		A, depsaux, Vi = self.NewtonRaphson_calculations_lin(self.Lep, S, E, V)

		if self.upliftcode == 'Active':
			State_vars , Ku, cx				= Grange_Uplift_Stiffness2(y,self.Lep,Vi,State_vars,*uplift_parameters)
			d1 								= max(abs(State_vars[4:6]))
			Mep,Hdel, N_op, L_op			= self.M_istr(self.Lep, Nep, depsaux, istrain, d1)
			y_j 							= self.NewtonRaphson_calculations_nonlin_uplift(Mep+Ku , S, E, V, xi, Hdel,F, self.Tmatrix, self.THmatrix)
			return y_j, State_vars
		else:
			Mep,Hdel,*_ = self.M_istr(self.Lep, Nep, depsaux, istrain,0)
			y_j 	 = self.NewtonRaphson_calculations_nonlin(Mep, S, E, V, xi, Hdel,F)
			return y_j, State_vars


	def Hypo_Stiff(self,sig,Vf):
		Y    = self.FunctionY(np.divide(sig,self.normvarF*Vf))
		m    = self.PlasticP(np.divide(sig,self.normvarP*Vf))
		Nep  = np.dot(np.multiply(Y,self.Lep),m)
		return Nep


	def FunctionY(self,T):

		b1  = self.yieldsurface[2]
		b2  = self.yieldsurface[3]
		b3  = self.yieldsurface[4]
		b4  = self.yieldsurface[5]
		ec  = self.yieldsurface[6]
		t0  = self.yieldsurface[7]
		Y   = self.FailureLocus(T, b1, b2, b3, b4, ec, self.b12, self.b34, t0,
						   self.nonlinearity[0],self.nonlinearity[1],self.upliftcode)
		return Y

	@staticmethod
	@nb.jit(nopython=True,fastmath=True,error_model="numpy", cache = True)
	def FailureLocus(T, b1, b2, b3, b4, ec, b12, b34, t0,k1,k2,code):

		if abs(T[1]) > 1e-15 or abs(T[2]) > 1e-15:
			TOL = 1e-15
			b   = 1
			a   = T[0] - 1e-15
			itt = 1
			# Init of bisection method
			shi  = (a + b)/2
			fv1p = (b12/((t0+1)**(b1+b2)))*(T[0]/shi+t0)**(b1)*(1-T[0]/shi)**(b2)
			fv2p = (b34/((t0+1)**(b3+b4)))*(T[0]/shi+t0)**(b3)*(1-T[0]/shi)**(b4)
			Hxp  = T[1]/shi
			Mxp  = T[2]/shi
			err = abs(fy(Hxp,Mxp,fv1p,fv2p,ec))
			fv1a = (b12/((t0+1)**(b1+b2)))*(T[0]/a+t0)**(b1)*(1-T[0]/a)**(b2)
			fv2a = (b34/((t0+1)**(b3+b4)))*(T[0]/a+t0)**(b3)*(1-T[0]/a)**(b4)
			Hxa  = T[1]/a
			Mxa  = T[2]/a
			while err > TOL and itt < 200:
				if  fy(Hxa,Mxa,fv1a,fv2a,ec)*fy(Hxp,Mxp,fv1p,fv2p,ec)<0:
					b = shi
				else:
					a = shi

				shi = (a + b)/2
				fv1p = (b12/((t0+1)**(b1+b2)))*(T[0]/shi+t0)**(b1)*(1-T[0]/shi)**(b2)
				fv2p = (b34/((t0+1)**(b3+b4)))*(T[0]/shi+t0)**(b3)*(1-T[0]/shi)**(b4)
				Hxp  = T[1]/shi
				Mxp  = T[2]/shi
				err = abs(fy(Hxp,Mxp,fv1p,fv2p,ec))
				itt += 1
		else:
			shi  = T[0]

		if shi <= 1:
			Y = np.array([[shi**k1,0,0],[0,shi**k1,0],[0,0,shi**k2]] , np.float64)
			return Y





	def PlasticP(self,T):

		if abs(T[1]) > 1e-15 or abs(T[2]) > 1e-15:

			b1      = self.plasticflow[2]
			b2      = self.plasticflow[3]
			b3      = self.plasticflow[4]
			b4      = self.plasticflow[5]
			e       = self.plasticflow[6]
			t0      = self.plasticflow[7]
			b12 	= self.b12p
			b34		= self.b34p
			g    = self.PlasticG(T,  b1, b2 , b3, b4, e, self.b12p, self.b34p, t0)
			tt   = T/g
			v  = tt[0]
			h  = tt[1]
			m  = tt[2]

			plastic_direction 	 = self.PlasticP_Calculation(v,h,m,b1,b2,b3,b4,b12,b34,t0,e)
			return plastic_direction
		else:
			plastic_direction = [np.sign(T[0]),0,0]
			return plastic_direction

	@staticmethod
	@nb.jit(nopython=True,fastmath=True,error_model="numpy", cache = True)
	def PlasticP_Calculation(v,h,m,b1,b2,b3,b4,b12,b34,t0,e):
		G  = np.zeros(3)
		f1 = (b12/((t0+1)**(b1+b2)))*(v+t0)**(b1)*(1-v)**(b2)
		f2 = (b34/((t0+1)**(b3+b4)))*(v+t0)**(b3)*(1-v)**(b4)

		if e != 0:
			G[0]= ((2*b1**(2*b1)*b2*b2**(2*b2)*h**2/(t0 + v)**(2*b1)*(t0 + 1)**(2*b1 + 2*b2))/((b1 + b2)**(2*b1 + 2*b2)*(1 - v)**(2*b2 + 1)) - (2*b1*b1**(2*b1)*b2**(2*b2)*h**2*(t0 + 1)**(2*b1 + 2*b2)/(1 - v)**(2*b2))/((b1 + b2)**(2*b1 + 2*b2)*(t0 + v)**(2*b1 + 1)) - (2*b3*b3**(2*b3)*b4**(2*b4)*m**2*(t0 + 1)**(2*b3 + 2*b4)/(1 - v)**(2*b4))/((b3 + b4)**(2*b3 + 2*b4)*(t0 + v)**(2*b3 + 1)) + (2*b3**(2*b3)*b4*b4**(2*b4)*m**2/(t0 + v)**(2*b3)*(t0 + 1)**(2*b3 + 2*b4))/((b3 + b4)**(2*b3 + 2*b4)*(1 - v)**(2*b4 + 1)) - (2*b1*b1**b1*b2**b2*b3**b3*b4**b4*e*h*m*(t0 + 1)**(b1 + b2)*(t0 + 1)**(b3 + b4))/((b1 + b2)**(b1 + b2)*(b3 + b4)**(b3 + b4)*(t0 + v)**b3*(t0 + v)**(b1 + 1)*(1 - v)**b2*(1 - v)**b4) + (2*b1**b1*b2*b2**b2*b3**b3*b4**b4*e*h*m*(t0 + 1)**(b1 + b2)*(t0 + 1)**(b3 + b4))/((b1 + b2)**(b1 + b2)*(b3 + b4)**(b3 + b4)*(t0 + v)**b1*(t0 + v)**b3*(1 - v)**b4*(1 - v)**(b2 + 1)) - (2*b1**b1*b2**b2*b3*b3**b3*b4**b4*e*h*m*(t0 + 1)**(b1 + b2)*(t0 + 1)**(b3 + b4))/((b1 + b2)**(b1 + b2)*(b3 + b4)**(b3 + b4)*(t0 + v)**b1*(t0 + v)**(b3 + 1)*(1 - v)**b2*(1 - v)**b4) + (2*b1**b1*b2**b2*b3**b3*b4*b4**b4*e*h*m*(t0 + 1)**(b1 + b2)*(t0 + 1)**(b3 + b4))/((b1 + b2)**(b1 + b2)*(b3 + b4)**(b3 + b4)*(t0 + v)**b1*(t0 + v)**b3*(1 - v)**b2*(1 - v)**(b4 + 1)))
			G[1]= ((2*h)/(f1**2) + (2*e*m)/(f1*f2))
			G[2]= ((2*m)/(f2**2) + (2*e*h)/(f1*f2))

		else:
			G[0]= ((2*b1**(2*b1)*b2*b2**(2*b2)*h**2/(t0 + v)**(2*b1)*(t0 + 1)**(2*b1 + 2*b2))/((b1 + b2)**(2*b1 + 2*b2)*(1 - v)**(2*b2 + 1)) - (2*b1*b1**(2*b1)*b2**(2*b2)*h**2*(t0 + 1)**(2*b1 + 2*b2)/(1 - v)**(2*b2))/((b1 + b2)**(2*b1 + 2*b2)*(t0 + v)**(2*b1 + 1)) - (2*b3*b3**(2*b3)*b4**(2*b4)*m**2*(t0 + 1)**(2*b3 + 2*b4)/(1 - v)**(2*b4))/((b3 + b4)**(2*b3 + 2*b4)*(t0 + v)**(2*b3 + 1)) + (2*b3**(2*b3)*b4*b4**(2*b4)*m**2/(t0 + v)**(2*b3)*(t0 + 1)**(2*b3 + 2*b4))/((b3 + b4)**(2*b3 + 2*b4)*(1 - v)**(2*b4 + 1)))
			G[1]= (2*h)/f1**2
			G[2]= (2*m)/f2**2

		plastic_direction    = G/np.linalg.norm(G)
		return plastic_direction

	@staticmethod
	@nb.jit(nopython=True,fastmath=True,error_model="numpy", cache = True)
	def PlasticG(T, b1, b2 , b3, b4, ec, b12, b34, t0):
		TOL = 1e-15
		b   = 1
		a   = T[0] - 1e-15
		itt = 1
		# Init of bisection method
		p = (a + b)/2

		fv1p = (b12/((t0+1)**(b1+b2)))*(T[0]/p+t0)**(b1)*(1-T[0]/p)**(b2)
		fv2p = (b34/((t0+1)**(b3+b4)))*(T[0]/p+t0)**(b3)*(1-T[0]/p)**(b4)
		Hxp  = T[1]/p
		Mxp  = T[2]/p
		err = abs( gy(Hxp,Mxp,fv1p,fv2p,ec) )

		fv1a = (b12/((t0+1)**(b1+b2)))*(T[0]/a+t0)**(b1)*(1-T[0]/a)**(b2)
		fv2a = (b34/((t0+1)**(b3+b4)))*(T[0]/a+t0)**(b3)*(1-T[0]/a)**(b4)
		Hxa  = T[1]/a
		Mxa  = T[2]/a
		while err > TOL and itt < 200:
			if  gy(Hxa,Mxa,fv1a,fv2a,ec)*gy(Hxp,Mxp,fv1p,fv2p,ec) < 0:
				b = p
			else:
				a = p

			p = (a + b)/2
			fv1p = (b12/((t0+1)**(b1+b2)))*(T[0]/p+t0)**(b1)*(1-T[0]/p)**(b2)
			fv2p = (b34/((t0+1)**(b3+b4)))*(T[0]/p+t0)**(b3)*(1-T[0]/p)**(b4)
			Hxp  = T[1]/p
			Mxp  = T[2]/p
			err = abs(gy(Hxp,Mxp,fv1p,fv2p,ec))
			itt += 1
		return p


	def M_istr(self,Lep, Nep, deps, istrain, d1):
		mu  = self.yieldsurface[0]
		phi = self.yieldsurface[1]
		m_r = self.elasticnucleous[0]
		m_t = self.elasticnucleous[1]
		Br  = self.elasticnucleous[2]
		X   = self.elasticnucleous[3]
		R   = self.elasticnucleous[4]*(1-d1)
		#M   = np.diag([1,16/(mu**2),16/(phi**2)])
		M   = self.Mmatrix
		I   = self.Imatrix
		Mep,Hdel,N_op,L_op = self.M_istr_operations(Lep,Nep,deps,istrain,M,I,m_t,m_r,X,Br,R)
		return Mep, Hdel, N_op, L_op

	@staticmethod
	@nb.jit(nopython=True,fastmath=True,error_model="numpy", cache = True)
	def M_istr_operations(Lep,Nep,deps,istrain,M,I,m_t,m_r,X,Br,R):
		istrain_norm = np.linalg.norm(istrain)
		TINY = 1e-20

		if istrain_norm > TINY:
			istrain_dir = np.divide(istrain,istrain_norm)
		else:
			istrain_dir = np.ones(3)*TINY

		istrain_dir_D = np.dot(istrain_dir,deps)

		if np.sqrt(np.dot(np.dot(istrain, M), istrain))/R > 1:
			rho = 1
		else:
			rho = np.sqrt(np.dot(np.dot(istrain, M), istrain))/R

		LijmnSmn 	= np.dot(Lep,istrain_dir)
		LijmnSmnSkl = np.outer(LijmnSmn,istrain_dir)
		NijSkl		= np.outer(Nep,istrain_dir)



		if istrain_dir_D >= 0:
			L_op = (m_t*rho**X + m_r*(1 - rho**X))*Lep + (1-m_t)*rho**X*LijmnSmnSkl
			N_op = -rho**X*NijSkl
			Mep  = L_op + N_op
			Hdel = I - rho**Br*np.outer(istrain_dir,istrain_dir)
			return Mep, Hdel, N_op, L_op
		else:
			Mep  = (m_t*rho**X + m_r*(1 - rho**X))*Lep + (m_r-m_t)*rho**X*LijmnSmnSkl
			L_op = Mep
			N_op = np.zeros((3,3))
			Hdel = I
			return Mep, Hdel, N_op, L_op
#------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------Bij Calculation----------------------------------------------------------

# @nb.jit(nopython=True,fastmath=True,error_model="numpy", cache = True)
# def bijCalculation(b12,b1,b2,b34,b3,b4,t0,x,T):
# 	fv1 = (b12/((t0+1)**(b1+b2)))*(T[0]/x+t0)**(b1)*(1-T[0]/x)**(b2)
# 	fv2 = (b34/((t0+1)**(b3+b4)))*(T[0]/x+t0)**(b3)*(1-T[0]/x)**(b4)
# 	Hx  = T[1]/x
# 	Mx  = T[2]/x
# 	return fv1,fv2, Hx, Mx

@nb.jit(nopython=True,fastmath=True,error_model="numpy", cache = True)
def fy(Hx,Mx,fv1,fv2,ec):
	if fv1 != 0 or fv2 != 0:
		f  = (Hx/fv1)**2 + (Mx/fv2)**2 + 2*ec*Hx*Mx/(fv1*fv2) - 1
		return f
	else:
		f  = 1
		return f

@nb.jit(nopython=True,fastmath=True,error_model="numpy", cache = True)
def gy(Hx,Mx,fv1,fv2,ec):
	if fv1 != 0 or fv2 != 0:
		g  = (Hx/fv1)**2 + (Mx/fv2)**2 + 2*ec*Hx*Mx/(fv1*fv2) - 1
		return g
	else:
		g  = 1
		return g

@nb.jit(nopython=True,fastmath=True,error_model="numpy", cache = True)
def Grange_Uplift_Stiffness2_Calculation_1(d1,d2,d_max,b_max,d_pl_max,n,beta):
	if d1 >= d_pl_max:
		d1 = d_pl_max

	if d2 <= -d_pl_max:
		d2 = -d_pl_max

	d_max[0]  = max(d_max[0],d1)
	d_max[1]  = min(d_max[1],d2)

	beta[0] = d_max[0]*(1-n)+n*d1
	beta[1] = d_max[1]*(1-n)+n*d2

	dx1 	= beta[0] if beta[0] >= 0 else 0
	dx2 	= beta[1] if beta[1] <= 0 else 0

	b_max[0] = max(b_max[0] , beta[0])
	b_max[1] = min(b_max[1] , beta[1])
	#% Computation of incremental beta1 and beta2 terms

	db1 = beta_comp1(n,beta[0],b_max[0],d_pl_max)
	db2 = beta_comp2(n,beta[1],b_max[1],d_pl_max)

	if db1 is None:
		db1 = np.inf
	if db2 is None:
		db2 = np.inf
	return d_max, b_max, db1, db2, dx1, dx2

@nb.jit(nopython=True,fastmath=True,error_model="numpy", cache = True)
def Grange_Uplift_Stiffness2_Calculation_2(v,m,t0,beta,Lep,d1,d2,q1,q2,A,d_pl_max,db1,db2,b_max,Mmax,d_max,dx1,dx2):
	mo1 =  (v+t0)/q1*np.exp(-A*(v+t0))
	mo2 =  (v+t0)/q1*np.exp(-A*(v+t0))

	# Plastic potential derivatives of F1
	bi1   = beta[0]
	fg    = m-(v+t0)/q1*(np.exp(-A*(v+t0))+q2*bi1)
	dgdm  = np.sign(fg)
	dgdv  = -0.5*d1*np.sign(fg)
	#dgdv  = -parameters.uplift(5)*d1*np.sign(fg)*(1-(v+t0))
	#dgdv  = -parameters.uplift(4)*d1*np.sign(fg)
	dgdF1 = np.array([dgdv,dgdm])

	#% Plastic potential derivatives of F2
	bi2   = beta[1]
	fg    = m+(v+t0)/q1*(np.exp(-A*(v+t0))-q2*bi2)
	dgdm  = np.sign(fg)
	dgdv  = -0.5*d2*np.sign(fg)
	#%     dgdv  = -parameters.uplift(5)*d2*np.sign(fg)*(1-(v+t0))
	#dgdv  = -parameters.uplift(4)*d2*np.sign(fg)
	dgdF2 = np.array([dgdv,dgdm])

	#% Derivatives respect to the state variable beta_i(1..2)
	dfdh1  = -(q2*(v+t0)*np.sign(m - ((v+t0)*(np.exp(-A*(v+t0)) + bi1*q2))/q1))/q1
	dfdh2  = -(q2*(v+t0)*np.sign(m + ((v+t0)*(np.exp(-A*(v+t0)) - bi2*q2))/q1))/q1

	#% Calculation of terms theta0 and theta_dot_up, 1 for positive surface, 2 for negative

	theta01   = mo1/Lep[1,1]
	theta02   = mo2/Lep[1,1]

	theta_up1 = dgdF1[1]
	theta_up2 = dgdF2[1]
	#% Calculation of hardening terms
	h1     = -d_pl_max*theta_up1/(theta01*(1-v-t0))*db1
	h2     = -d_pl_max*theta_up2/(theta02*(1-v-t0))*db2
	#% Calculation of plastic modulus
	Hup1   = dfdh1*h1
	Hup2   = dfdh2*h2

	#% Yield Surface derivatives for F1
	dFdv = -np.sign(m - ((v+t0)*(np.exp(-A*(v+t0)) + bi1*q2))/q1)*((np.exp(-A*(v+t0)) + bi1*q2)/q1 - (A*(v+t0)*np.exp(-A*(v+t0)))/q1)
	dFdm =  np.sign(m - ((v+t0)*(np.exp(-A*(v+t0)) + bi1*q2))/q1)
	dfdF1 = np.array([dFdv,dFdm])
	#% Yield Surface derivatives for F2
	dFdv = np.sign(m + ((v+t0)*(np.exp(-A*(v+t0)) - bi2*q2))/q1)*((np.exp(-A*(v+t0)) - bi2*q2)/q1 - (A*(v+t0)*np.exp(-A*(v+t0)))/q1)
	dFdm = np.sign(m + ((v+t0)*(np.exp(-A*(v+t0)) - bi2*q2))/q1)
	dfdF2 = np.array([dFdv,dFdm])



	#% H0 terms for plastic modulus
	H01  = np.dot(np.dot(dfdF1,Lep),dgdF1)
	H02  = np.dot(np.dot(dfdF2,Lep),dgdF2)

	#% Calculation of 1/Kp
	HT1 = np.array(1/(H01+Hup1))
	HT2 = np.array(1/(H02+Hup2))

	#hh1 =  np.dot(HT1,dfdF1)
	#hh2 =  np.dot(HT2,dfdF2)

	#% Determination of Stiffness matrix from plastic processes
	#%
	#%       Kelup= Lep - (HT1*(Lep*dgdF1)*(dfdF1'*Lep) + HT2*(Lep*dgdF2)*(dfdF2'*Lep))
	cx 	  = max(abs(dx1), abs(dx2))
	Kelup = -HT1*np.outer(np.dot(Lep,dgdF1),np.dot(dfdF1,Lep)) -HT2*np.outer(np.dot(Lep,dgdF2),np.dot(dfdF2,Lep))
	State_vector = np.array([beta[0], beta[1], b_max[0], b_max[1], d1, d2, d_max[0], d_max[1], d_pl_max, Mmax], np.float64)
	return State_vector , Kelup, cx

def Grange_Uplift_Stiffness2(y_k,Lep,V,State_vars,q1,q2,t0,B,m0,b3,b4):
	# q2 = 0.9*q2 # for safety reason
	A  = 2.5
	Kelupg 		= np.zeros((3,3),np.float64)
	Vf          = y_k[6]
	m           = y_k[5]/(Vf*B)
	v           = y_k[3]/Vf

	b34 = ((b3+b4)**(b3+b4))/(b3**b3*b4**b4)
	Fv2 = (b34/((t0+1)**(b3+b4)))*(v+t0)**(b3)*(1-v)**(b4)
	Mmax = m0*Fv2
	#     V       = [cos(y_k(3,1)) sin(y_k(3,1)) 0  -sin(y_k(3,1)) cos(y_k(3,1)) 0 0 0 1]*V
	Lep     = np.array([[Lep[0,0], Lep[0,2]],  [Lep[2,0], Lep[2,2]]])
	#     Lep     = sparse([1 2],[1 2],[Lep(1,1) Lep(3,3)])

	#if m > (v+t0)/q1*(np.exp(-A*(v+t0))+q2)*0.98:
	#	m = (v+t0)/q1*(np.exp(-A*(v+t0))+q2)*0.98
	#
	#     plot(m,Lep(2,2),'ro') hold on

	y_j     = [v,m]
	Vnew    = np.divide([V[0] , V[2]], [Vf, Vf*B])
	beta    = State_vars[0:2]
	b_max   = State_vars[2:4]
	d_max   = State_vars[6:8]

	if Mmax > (v+t0)/q1*(np.exp(-A*(v+t0))+q2):
		d_pl_max = 1
	else:
		mo = (v+t0)/q1*np.exp(-A*(v+t0))
		d_pl_max = q1/(q2*(v+t0))*abs(Mmax-mo)

	# Initial computation of state variable delta1 and delta2
	y_jk	= y_j + Vnew
	d1 ,  n  = delta_comp1(y_jk[0],y_jk[1],q1, q2, A, B, t0, beta[0], b_max[0])
	d2 		 = delta_comp2(y_jk[0],y_jk[1],q1, q2, A, B, t0, beta[1], b_max[1])
	cx 		 = np.zeros(1)

	if abs(d1) != 0 or abs(d2) != 0:
		d_max, b_max, db1, db2, dx1, dx2 = Grange_Uplift_Stiffness2_Calculation_1(d1,d2,d_max,b_max,d_pl_max,n,beta)
		State_vector , Kelup, cx   = Grange_Uplift_Stiffness2_Calculation_2(v,m,t0,beta,Lep,d1,d2,q1,q2,A,d_pl_max,db1,db2,b_max,Mmax,d_max, dx1,dx2)
		Kelupg[[0,2,0,2],[0,2,2,0]] = [Kelup[0,0],Kelup[1,1],Kelup[0,1],Kelup[1,0]]
		return State_vector , Kelupg, cx
	else:
		return State_vars , Kelupg, cx
	#%     State_vector'
	#%     pause()

@nb.jit(nopython=True,fastmath=True,error_model="numpy", cache = True)
def delta_comp1(V,M,q1,q2,A,b,t0,B,Bmax):

	n  = 4-3*np.exp(-4*(V+t0))

	if  f_load_p(M , V , q1, q2, A, b, t0, B , Bmax) <= 0:
		mo = (V+t0)/q1*np.exp(-A*(V+t0))
		d = 0
	elif B == Bmax:
		mo = (V+t0)/q1*np.exp(-A*(V+t0))
		d = q1/(q2*(V+t0))*abs((M-mo))
	elif B < Bmax:
		mo = (V+t0)/q1*q2*Bmax*(1-n) + (V+t0)/q1*np.exp(-A*(V+t0))
		d  = q1/(q2*n*(V+t0))*abs(M-mo)
	return d,n


@nb.jit(nopython=True,fastmath=True,error_model="numpy", cache = True)
def delta_comp2(V,M,q1,q2,A,b,t0,B,Bmax):

	n  = 4-3*np.exp(-4*(V+t0))
	if  f_load_n(M , V , q1,q2,A,b,t0 , B , Bmax) >= 0:
		mo = -(V+t0)/q1*np.exp(-A*(V+t0))
		d = 0
		return d
	elif B == Bmax:
		mo = -(V+t0)/q1*np.exp(-A*(V+t0))
		d  = -q1/(q2*(V+t0))*abs((M-mo))
		return d
	elif B > Bmax:
		mo = (V+t0)/q1*q2*Bmax*(1-n) - (V+t0)/q1*np.exp(-A*(V+t0))
		d  = -q1/(q2*n*(V+t0))*abs(M-mo)
		return d


@nb.jit(nopython=True,fastmath=True,error_model="numpy", cache = True)
def beta_comp1(n,B,Bmax,d_pl_max):

	if np.isnan(B) == True:
		B = 1

	if B == Bmax:
		if B == 0:
			db = np.inf
			return db
		else:
			db = (1-B/d_pl_max)**2/(B/d_pl_max*(2-B/d_pl_max))
			return db
	elif B <= Bmax:
		term1 = (B-(1-n)*Bmax)/(n*d_pl_max)
		if term1 == 0:
			db = np.inf
		else:
			db = n*(1-term1)**2/(term1*(2-term1))
			return db




@nb.jit(nopython=True,fastmath=True,error_model="numpy", cache = True)
def beta_comp2(n,B,Bmax,d_pl_max):

	if np.isnan(B) == True:
		B = 1

	if B == Bmax:
		if B == 0:
			db = np.inf
			return	db
		else:
			db = (1+B/d_pl_max)**2/(-B/d_pl_max*(2+B/d_pl_max))
			return db
	elif B >= Bmax:
		term1 = (B-(1-n)*Bmax)/(n*d_pl_max)
		if term1 == 0:
			db = np.inf
			return db
		else:
			db = n*(1+term1)**2/(-term1*(2+term1))
			return db



@nb.jit(nopython=True,fastmath=True,error_model="numpy", cache = True)
def f_load_p(M , V , q1,q2,A,b,t0 , B , Bmax):

	n    = 4-3*np.exp(-4*(V+t0))
	f_el = M - (V+t0)/q1*q2*Bmax*(1-n) - (V+t0)/q1*np.exp(-A*(V+t0))
	return f_el


@nb.jit(nopython=True,fastmath=True,error_model="numpy", cache = True)
def f_load_n(M , V , q1,q2,A,b,t0 , B , Bmax):
	n    = 4-3*np.exp(-4*(V+t0))
	f_el = M - (V+t0)/q1*q2*Bmax*(1-n) + (V+t0)/q1*np.exp(-A*(V+t0))
	return f_el
