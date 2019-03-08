import numpy as np
import matplotlib.pyplot as plt
import random as random
import time
import csv

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
	print('T=',new_T)										#Geometric Annealing
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
plot = False
plot_maximum = False
plot_acceptance = True
a_factor = 100
random_initialisation = True
logtofile = True
#Solutions Control
f_domain = 3									#Parametric Space width: Symmetrical around zero
a = 1											#Geometry of objective function I
b = 100											#Geometry of objective function II
initial_x = -3
initial_y = 3
#SA Paramenters
T_0 = 20	  									#Initial temperature
temperature_drops = 200							#Number of times to drop the temperature
max_scenarios = 1500							#Maximum number of evaluated solutions
tempfactor = 0.95								#Geometric annealing parameter
max_trials = 10000000							#Number of maximum consecutive failed solutions allowed
potential_solutions = np.int64(1000)				#Number of potential discrete solutions
max_resample_count = 10000000						#Number of maximum potential resamples of a given solution
#Noise Parameters
mean = 0
sd = 1

#OSA Paramenters
c_reject = 0									#Cumulative variable rejection threshold
potential_steps = [1,1e-1,1e-2,1e-3,-1,			#Potential Step Sizes
	-1e-1,-1e-2,-1e-3]
#Auxiliar Variables Initialisation
T = T_0											#Temperature
beta = np.float64(1/T)							#Beta parameter
steps_per_temp =np.floor(max_scenarios/temperature_drops) #Number per iterations in a given temperature
print(steps_per_temp)
cn = 0											#Cumulative variable Cn
cn_previous = 0									#Cumulative variable cn_{-1}

count = 0										#Number of proposed moves
accept_count = 0								#Number of accepted solutions
max_count = 1000
trials = 0										#Number of attempts to improve over a given solution

temp_count = 0									#Number of reductions in Temp (accepted moves)
resample_abs_count = 0							#Number of times a the objective function is resampled
resample_count = 0								#Number of times a solution is resampled
reject_count = 0								#Number of rejected solutions

x_storage = []									#Collection of accepted solutions
y_storage = []
f_noise_storage = []							#Collection of accepted noisy values for solutions
f_real_storage = []								#Collection of accepted actual values for solutions
x_min_storage = []								#Solution x for minimum objective so far
y_min_storage = []								#Solution y for minimum objective so far
f_noise_min_storage = []						#Minimum objective function value so far
f_min_storage = []								#Minimum actual value of the objective function so far
A_storage = []									#Storage of transition probabilities
all_x = []
all_y = []
all_f_noise = []
all_f = []


#Algorithm Variables Initialisation
if random_initialisation == True:
	current_x = f_domain * np.round ((np.random.uniform(-1,1)/potential_solutions), decimals = np.int64(np.log10(potential_solutions)))
	current_y = f_domain * np.round ((np.random.uniform(-1,1)/potential_solutions), decimals = np.int64(np.log10(potential_solutions)))
elif random_initialisation == False:
	current_x = initial_x
	current_y = initial_y

current_f = f_noise(current_x,current_y,a,b,mean,sd)

x_storage.append(current_x)								#Collection of accepted solutions
y_storage.append(current_y)								#Collection of accepted solutions
f_noise_storage.append(current_f)						#Collection of accepted noisy values for solutions
f_real_storage.append(f(current_x,current_y,a,b))		#Collection of accepted actual values for solutions
f_min_storage.append(f(current_x,current_y,a,b))		#Minimum actual value of the objective function so far
f_noise_min_storage.append(current_f)					#Minimum objective function value so far
x_min_storage.append(current_x)							#Solution x for minimum objective so far
y_min_storage.append(current_y)							#Solution y for minimum objective so far
all_x.append(current_x)
all_y.append(current_y)
all_f_noise.append(current_f)
all_f.append(f(current_x,current_y,a,b))

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
	count = count + 1								#Increase counter for proposed solutions
	trials = trials + 1								#Increase counter for improvement attempts

	all_x.append(new_x)
	all_y.append(new_y)
	all_f_noise.append(new_f)
	all_f.append(f(new_x,new_y,a,b))
	
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
		#Store
		x_storage.append(current_x)								#Store accepted solution value
		y_storage.append(current_y)								#Store accepted solution value
		f_noise_storage.append(current_f)						#Store accepted noisy value of objective function
		f_real_storage.append(f(current_x,current_y,a,b))		#Store accepted actual value of objective function
		A_storage.append(a_factor*A)							#Store the accepted transition probability
		#Keep track of minimum values so far
		if current_f<f_noise_min_storage[len(f_min_storage)-1]:	
			x_min_storage.append(current_x)
			y_min_storage.append(current_y)
			f_noise_min_storage.append(current_f)
			f_min_storage.append(f(current_x,current_y,a,b))
		else:
			x_min_storage.append(x_min_storage[len(x_min_storage)-1])
			y_min_storage.append(y_min_storage[len(y_min_storage)-1])
			f_noise_min_storage.append(f_noise_min_storage[len(f_noise_min_storage)-1])
			f_min_storage.append(f(x_min_storage[len(x_min_storage)-1],y_min_storage[len(y_min_storage)-1],a,b))


		#Do maintenance
		cn = 0										#Reset cumulative variables upon acceptance
		cn_previous = 0								#Reset cumulative variables upon acceptance
		trials = 0									#Reset counter for failed trials
		accept_count = accept_count + 1
		#print('accepted')

	elif cn < c_reject:								#If not yet in the threshold for rejection, resample
		resample = True 							#Create resample flag
		#print('resampled')
		
		while resample == True:										#While the flag is on
			new_f = f_noise(new_x,new_y,a,b,mean,sd)				#Take new sample with same proposed solution
			resample_f = f_noise(current_x,current_y,a,b,mean,sd)	#Resample the previous solution
			delta = new_f - resample_f								#Calculate delta energy level

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
				#Store
				x_storage.append(current_x)								#Store accepted solution value
				y_storage.append(current_y)								#Store accepted solution value
				f_noise_storage.append(current_f)						#Store accepted noisy value of objective function
				f_real_storage.append(f(current_x,current_y,a,b))		#Store accepted actual value of objective function
				A_storage.append(a_factor*A)							#Store the accepted transition probability
				#Keep track of minimum values so far
				if current_f<f_noise_min_storage[len(f_min_storage)-1]:
					x_min_storage.append(current_x)
					y_min_storage.append(current_y)	
					f_noise_min_storage.append(current_f)
					f_min_storage.append(f(current_x,current_y,a,b))
					
				else:
					x_min_storage.append(x_min_storage[len(x_min_storage)-1])
					y_min_storage.append(y_min_storage[len(y_min_storage)-1])
					f_noise_min_storage.append(f_noise_min_storage[len(f_noise_min_storage)-1])
					f_min_storage.append(f(x_min_storage[len(x_min_storage)-1],y_min_storage[len(y_min_storage)-1],a,b))
					
				#Maintenance
				cn = 0										#Reset cumulative variable upon acceptance
				cn_previous = 0								#Reset cumulative variable upon acceptance
				trials = 0									#Reset counter for failed trials
				resample_count = 0							#Reset solution resampling count upon acceptance
				accept_count = accept_count + 1	
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
				break
	#Reject
	else:													
		cn=0
		cn_previous=0
		reject_count = reject_count + 1
		#print('rejected')

	#Break based on number of unsuccessful movements
	if trials > max_trials:
		print('Maximum Number of unsuccessful movements reached')
		trials = 0
		break

realvalue = f(current_x,current_y,a,b)

#End of execution report
if report == True:
	print('  --------------------')
	print('  OSA RESULTS REPORT:')
	print('  --------------------\n')	

	print('\tSuccessful moves:         ', (accept_count) ,'/', count)
	print('\tRejected movements        ', reject_count)
	print('\tTimes resampled           ', resample_abs_count)
	print('\tAcceptance Ratio:         ', np.around((temp_count/count),decimals=5),'\n')

	print('\tFinal noisy f(x,y) is     ', current_f,'\twith (x,y)=(', current_x,',',current_y,')')
	print('\tFinal actual f(x,y) is    ', realvalue,'\n')

	print('\tNoisy min of f(x,y) was   ', min(f_noise_min_storage),'\twith (x,y)=(', 
		x_min_storage[f_noise_min_storage.index(min(f_noise_min_storage))],',',y_min_storage[f_noise_min_storage.index(min(f_noise_min_storage))],')')
	print('\tActual min of f(x,y) was  ', f_min_storage[f_noise_min_storage.index(min(f_noise_min_storage))],'\n')

	print('\tAbs min of the run was    ', min(f_real_storage),'\twith (x,y)=(',x_storage[f_real_storage.index(min(f_real_storage))],',',
		y_storage[f_real_storage.index(min(f_real_storage))],')')
	print('\tMeasurement was           ', f_noise_storage[f_real_storage.index(min(f_real_storage))])



	#print(x_storage)
export_all = []
export_all = [all_f_noise,all_f,all_x,all_y]
export_accepted = []
export_accepted = [f_noise_storage,f_real_storage,x_storage,y_storage]
export_max_accepted = []
export_max_accepted = [f_noise_min_storage,f_min_storage,x_min_storage,y_min_storage]


#End of execution plot
if plot == True:
	if plot_acceptance == True:
		plt.plot(range(len(A_storage)),A_storage, 'r--',label='Probability')
	plt.plot(range(len(f_real_storage)),f_noise_storage, 'g-', label='Noisy Value')
	plt.plot(range(len(f_real_storage)),f_real_storage, 'b-',label='Exact Value')
	plt.plot(range(len(f_real_storage)),f_noise_min_storage, 'k--',label='Minimum')
	plt.legend(bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure)
	#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
	xmin = 0
	xmax = 10+len(f_real_storage)
	ymin = min(0.8*min(min(f_real_storage),min(f_noise_storage)),-5)
	#ymax = 1.1*max(max(f_real_storage),max(f_noise_storage),10.1)
	ymax = 100
	plt.axis([ xmin, xmax, ymin , ymax])
	plt.xlabel('Accepted Solutions')
	plt.ylabel('Value of the Objective function')
	plt.title('OSA Results')
	plt.show()

if logtofile==True:
	version = 5
#	filename = 'Rosenbrock_{}sol_res{}_sd{}_t{}_drops{}_factor{}.csv'.format(max_scenarios, potential_solutions, sd,T_0,
#		temperature_drops,tempfactor)
#	with open(filename,"wb") as f:
#		writer = csv.writer(f)
#		writer.writerows(export)
#	filename2 ='Rosenbrock_MAX_{}sol_res{}_sd{}_t{}_drops{}_factor{}.csv'.format(max_scenarios, potential_solutions, sd,T_0,
#		temperature_drops,tempfactor)
	filename = 'Rosenbrock_{}sol_res{}_sd{}_t{}_drops{}_factor{}_rep{}.csv'.format(max_scenarios, potential_solutions, sd,T_0,
		temperature_drops,tempfactor,version)
	filename2 = 'Rosenbrock_MAX_{}sol_res{}_sd{}_t{}_drops{}_factor{}_rep{}.csv'.format(max_scenarios, potential_solutions, sd,T_0,
		temperature_drops,tempfactor,version)
	filename3 = 'Rosenbrock_all_{}sol_res{}_sd{}_t{}_drops{}_factor{}_rep{}.csv'.format(max_scenarios, potential_solutions, sd,T_0,
		temperature_drops,tempfactor,version)
		
	np.savetxt(filename, export_accepted, delimiter = ',')
	np.savetxt(filename2, export_max_accepted, delimiter = ',')
	np.savetxt(filename3,export_all,delimiter = ',')