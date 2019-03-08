import numpy as np
import matplotlib.pyplot as plt
import random as random
import time

#Function to calculate true value of f(x)
def f(x,y,a,b):
	fxy = (a-x)**2 + b*((y-x**2)**2)							#Rosenbrock [-3,3] - ABS Minimum at f(1,1) = 0
	#fxy = (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 +(		#Beale [-4.5,4.5] - ABS Minumum at f(3,0.5) = 0
	#	2.625 - x + x*y**3)**2
	#fxy = 5 + 10 * (x-1)**2 - 5 * np.cos(4*(x-1)*np.pi)		#Test [-1,1] ABS Minimum at f(1)   = 0
	return fxy	
#Function to calculate noisy value of f(x)
def f_noise(x,y,a,b,mean,sd):
	fxy = f(x,y,a,b) + np.random.normal(mean,sd)
	return fxy
#Function for Annealing Schedule
def temp(T, tempfactor):
	new_T = T * tempfactor	
	#print('T=',new_T)										#Geometric Annealing
	return new_T
#Function to generate Acceptance probabilities
def acceptance (sd, cn, cn_previous, beta):
	A = np.minimum(1, np.exp((-2/sd**2)*((cn+beta*sd**2 )/2)*((cn_previous+beta*sd**2)/2)))
	return A
#Function to generate new x solutions
def new_neighbor(potential_steps, current_coord, f_domain):
	step = potential_steps[random.randint(0,len(potential_steps)-1)]
	proposed_coord = step + current_coord
	if abs(proposed_coord) > abs(f_domain):
		return(new_neighbor(potential_steps,current_coord,f_domain))
	else:
		return(proposed_coord)

#Output Control
report = True
plot = True
plot_maximum = False
plot_acceptance = True
a_factor = 10
random_initialisation = True
logtofile = True
repetitions = 2000
#Solutions Control
f_domain = 3									#Parametric Space width: Symmetrical around zero
a = 1											#Geometry of objective function I
b = 100											#Geometry of objective function II
initial_x = -3
initial_y = 3
#SA Paramenters
T_0 = 20	  									#Initial temperature
temperature_drops = 100							#Number of times to drop the temperature
max_scenarios = 2000							#Maximum number of evaluated solutions
tempfactor = 0.95								#Geometric annealing parameter
potential_solutions = np.int64(1000)				#Number of potential discrete solutions
max_resample_count = 1000000						#Number of maximum potential resamples of a given solution
#Noise Parameters
mean = 0
sd = 1
#OSA Paramenters
c_reject = 0									#Cumulative variable rejection threshold
potential_steps = [1,1e-1,1e-2,1e-3,-1,			#Potential Step Sizes
	-1e-1,-1e-2,-1e-3]

steps_per_temp =max_scenarios/temperature_drops #Number per iterations in a given temperature
R = np.zeros(repetitions)


#Loop through repetitions
for i in range(repetitions):
	cn = 0											#Cumulative variable Cn
	cn_previous = 0									#Cumulative variable cn_{-1}
	count = 0										#Number of proposed moves
	temp_count = 0									#Number of reductions in Temp (accepted moves)
	resample_abs_count = 0							#Number of times a the objective function is resampled
	resample_count = 0								#Number of times a solution is resampled
	reject_count = 0								#Number of rejected solutions
	#Algorithm Variables Initialisation
	if random_initialisation == True:
		current_x = f_domain * np.round ((np.random.uniform(-1,1)/potential_solutions), decimals = np.int64(np.log10(potential_solutions)))
		current_y = f_domain * np.round ((np.random.uniform(-1,1)/potential_solutions), decimals = np.int64(np.log10(potential_solutions)))
	elif random_initialisation == False:
		current_x = initial_x
		current_y = initial_y
	current_f = f_noise(current_x,current_y,a,b,mean,sd)
	#Auxiliar Variables Initialisation
	T = T_0											#Temperature
	beta = np.float64(1/T)							#Beta parameter
	print(i)
	


	#Start Annealing
	while count<max_scenarios:
		#Regularly decrease temperature & recalculate beta
		if count % steps_per_temp == 0:
			T = temp(T, tempfactor)
			beta = np.float64(1/ T)						
			temp_count = temp_count + 1

		#Propose new solutions
		new_x = np.round (new_neighbor(potential_steps, current_x, f_domain), decimals = np.int64(np.log10(potential_solutions)))
		new_y = np.round (new_neighbor(potential_steps, current_y, f_domain), decimals = np.int64(np.log10(potential_solutions)))	

		#Process them
		new_f = f_noise(new_x,new_y,a,b,mean,sd)
		delta = new_f - current_f
		R[i] = R[i] + 1
		count = count + 1								#Increase counter for proposed solutions
	
		#Update cn
		cn_intermediate = cn
		cn = cn + delta
		cn_previous = cn_intermediate

		#Calculate beta and its associated acceptance probability
		A = acceptance (sd, cn, cn_previous, beta)

		#Accept with probability A
		if np.random.random() < A:
			#Update values
			current_f = new_f							
			current_x = new_x
			current_y = new_y

			#Do maintenance
			cn = 0										#Reset cumulative variables upon acceptance
			cn_previous = 0								#Reset cumulative variables upon acceptance
			#print('accepted')

		elif cn < c_reject:								#If not yet in the threshold for rejection, resample
			resample = True 							#Create resample flag
			#print('resampled')
			
			while resample == True:										#While the flag is on
				new_f = f_noise(new_x,new_y,a,b,mean,sd)				#Take new sample with same proposed solution
				resample_f = f_noise(current_x,current_y,a,b,mean,sd)	#Resample the previous solution
				delta = new_f - resample_f								#Calculate delta energy level
				R[i] = R[i]+ 1

				resample_count = resample_count + 1				#Increase counter for solution resampling
				resample_abs_count = resample_abs_count + 1		#Increase counter for global resampling
				
				#Update cumulative variable
				cn_intermediate = cn
				cn = cn + delta
				cn_previous = cn_intermediate
				#print('Cn=',np.round(cn, decimals=2))

				#Generate acceptance probability
				A = acceptance (sd, cn, cn_previous, beta)	

				#Accept with probability A
				if np.random.random() < A:
					#print('Accepted resample')
					#time.sleep(1)
					#Update values
					current_f = new_f							
					current_x = new_x
					current_y = new_y

					#Maintenance
					cn = 0										#Reset cumulative variable upon acceptance
					cn_previous = 0								#Reset cumulative variable upon acceptance
					resample_count = 0							#Reset solution resampling count upon acceptance
					resample = False 							#Deactivate infinite resampling loop upon acceptance
				#Reject after resampling
				elif cn > c_reject:
					resample = False
					#print('Rejected resample')
					#time.sleep(1)
					cn = 0
					cn_previous = 0
				#Reject due to too many resamples
				elif resample_count > max_resample_count:
					print('Maximum Number of Resamples reached')
					resample_count=0
					break
		#Reject
		else:													
			cn=0
			cn_previous=0
			reject_count = reject_count + 1
			#print('rejected')	


	R[i] = 2 * R[i] / max_scenarios	

print(R)
print('Average R=',np.average(R))
