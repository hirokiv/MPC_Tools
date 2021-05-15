import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import csv
from gekko import GEKKO
import time
import pysindy as ps

# create MPC with GEKKO
m = GEKKO()
m.time = [0,1,2,4,8,12,16,20]

# empirical constants
# Kp_h1 = 1.3
Kp_h1 = 0.5
tau_h1 = 18.4
# Kp_h2 = 1
Kp_h2 = 0.3
tau_h2 = 24.4

# manipulated variable
p = m.MV(value=0,lb=1e-5,ub=1)
# Determins whether p should be included into optimization : Default Off
p.STATUS = 1
# Adds a term to the objective function that gives minor penalty for changing the MV
p.DCOST = 0.01
# feedback status 
p.FSTATUS = 0

# unmeasured state
h1 = m.Var(value=0.0)

# controlled variable
h2 = m.CV(value=0.0)
# Determins whether p should be included into optimization : Default Off
h2.STATUS = 1
# feedback status 
h2.FSTATUS = 1
# Time constant for the setpoint trajectory
h2.TAU = 20
# Trajectory initialization
#    0 = dead-band
#    1 = re-center with coldstart/out-of-service
#    2 = re-center always
h2.TR_INIT = 2

# equations
#m.Equation(tau_h1*h1.dt()==-h1 + Kp_h1*p)
m.Equation(tau_h2*h2.dt()==-h2 + Kp_h2*h1)

# options
# IMODE 6 : MPC mode
m.options.IMODE = 6
# Control variable error model type : 1=linear, 2=squared, 3=ref_traj
m.options.CV_TYPE = 2

# simulated system (for measurements)
def tank(levels,t,pump,valve):
    h1 = max(1.0e-10,levels[0])
    h2 = max(1.0e-10,levels[1])
    c1 = 0.08 # inlet valve coefficient
    c2 = 0.04 # tank outlet coefficient
    dhdt1 = c1 * (1.0-valve) * pump - c2 * np.sqrt(h1)
    dhdt2 = c1 * valve * pump + c2 * np.sqrt(h1) - c2 * np.sqrt(h2)
    if h1>=1.0 and dhdt1>0.0:
        dhdt1 = 0
    if h2>=1.0 and dhdt2>0.0:
        dhdt2 = 0
    dhdt = [dhdt1,dhdt2]
    return dhdt

# Initial conditions (levels)
h0 = [0,0]

# Time points to report the solution
tf = 4000
t = np.linspace(0,tf,tf+1)

u_sin =  0.20*np.sin(2*np.pi/30*t) +  0.20*np.sin(2*np.pi/35*t) + 0.20*np.sin(2*np.pi/40*t) + 0.3

# Set point
sp = np.zeros(tf+1)
sp[5:50] = 0.2
sp[50:100] = 0.3
sp[100:150] = 0.5
sp[150:200] = 0.3
sp[200:250] = 0.5
sp[250:300] = 0.3
sp[300:350] = 0.5
sp[350:] = 0.3

# Inputs that can be adjusted
pump = np.zeros(tf+1)

# Disturbance
valve = 0.0

# Record the solution
y = np.zeros((tf+1,2))
y[0,:] = h0

# Create plot
plt.figure(figsize=(10,7))
plt.ion()
plt.show()

# Simulate the tank step test
for i in range(1,tf):
    #########################
    # MPC ###################
    #########################
    # measured height
    h2.MEAS = y[i,1]
    # set point deadband
    h2.SPHI = sp[i]+0.001
    h2.SPLO = sp[i]-0.001
    h2.SP = sp[i]
    # solve MPC
#    print('¥nLine 1 ' + str(time.time()))
#    m.solve(disp=True)
#    print('¥nLine 2 ' + str(time.time()))
    # retrieve 1st pump new value
#    pump[i] = p.NEWVAL
    pump[i] = u_sin[i]

    #########################
    # System ################
    #########################
    # Specify the pump and valve
    inputs = (pump[i],valve)
    # Integrate the model
    h = odeint(tank,h0,[0,1],inputs)
    # Record the result
    y[i+1,:] = h[-1,:]
    # Reset the initial condition
    h0 = h[-1,:]

    # update plot every 5 cycles
    if (i%10==3):
        plt.clf()
        plt.subplot(2,1,1)
        plt.plot(t[0:i],sp[0:i],'k-')    
        plt.plot(t[0:i],y[0:i,0],'b-')
        plt.plot(t[0:i],y[0:i,1],'r--')
        plt.ylabel('Height (m)')
        plt.legend(['Set point','Height 1','Height 2'])
        plt.subplot(2,1,2)
        plt.plot(t[0:i],pump[0:i],'k-')
        plt.ylabel('Pump')    
        plt.xlabel('Time (sec)')
        plt.draw()
        plt.pause(0.01)

plt.savefig('Output.png')
# Construct and save data file
data = np.vstack((t,pump))
data = np.hstack((np.transpose(data),y))
np.savetxt('data.txt',data,delimiter=',')


X = np.stack((y[:,1], pump), axis = -1) # First column is h2, second is pump (input u)
print(X)
optimizer = ps.STLSQ(threshold=0.02)
model = ps.SINDy(feature_names=["h2","u"], optimizer=optimizer)
model.fit(X[200:4000,:], t=t[200:4000])
model.print()
