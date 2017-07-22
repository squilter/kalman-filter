from math import sqrt

import matplotlib.pyplot as plt
import numpy as np

from sebekf import EKF

### tuning params ###
POSITION_NOISE_VARIANCE = 10
VELOCITY_NOISE_VARIANCE = 0.01
ACCELERATION_NOISE_VARIANCE = 0.00001
SAMPLES = 100
TIME = 100
TIMESTEP = TIME/SAMPLES

### set up data ###
t = np.linspace(0, TIME, SAMPLES)
pos =  100 *np.sin(t / 10.)
vel = np.diff(pos)
acc = np.diff(vel)
vel = np.insert(vel,0,0)
acc = np.insert(acc,0,0)
acc = np.insert(acc,0,0)

pos_noise = np.random.normal(0, sqrt(POSITION_NOISE_VARIANCE), SAMPLES) # mean, std.dev, length
vel_noise = np.random.normal(0, sqrt(VELOCITY_NOISE_VARIANCE), SAMPLES) # mean, std.dev, length
acc_noise = np.random.normal(0, sqrt(ACCELERATION_NOISE_VARIANCE), SAMPLES) # mean, std.dev, length

pos_observation = pos + pos_noise
vel_observation = vel + vel_noise
acc_observation = acc + acc_noise

### run the filter ###

pos_filtered = []
vel_filtered = []
acc_filtered = []

initial_state = np.zeros((3,1))
initial_error_cov = np.diag([0.1]*3)

Q = np.diag([1e-4]*3) # process noise
R = np.diag([POSITION_NOISE_VARIANCE, VELOCITY_NOISE_VARIANCE, ACCELERATION_NOISE_VARIANCE])# measurement noise
F = np.array([[1,TIMESTEP,0],[0,1,TIMESTEP],[0,0,1]])
F = np.eye(3)
B = None
H = np.eye(3)

# num_states, num_sensors, x_0, P_0, Q, R, F, B, H):
ekf = EKF(3, 3, initial_state, initial_error_cov, Q, R, F, B, H)

for observation in zip(pos_observation, vel_observation, acc_observation):
    filtered_state = ekf.step(observation)
    pos_filtered.append(filtered_state[0])
    vel_filtered.append(filtered_state[1])
    acc_filtered.append(filtered_state[2])

### plot data ###
f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=False)
ax1.set_title('position')
ax2.set_title('velocity')
ax3.set_title('acceleration')
ax1.plot(t, pos,linewidth=2)
ax2.plot(t, vel,linewidth=2)
ax3.plot(t, acc,linewidth=2)
# ax1.plot(t, pos_observation, color='r')
# ax2.plot(t, vel_observation, color='r')
# ax3.plot(t, acc_observation, color='r')
ax1.plot(t, pos_filtered, color='g')
ax2.plot(t, vel_filtered, color='g')
ax3.plot(t, acc_filtered, color='g')

# Fine-tune figure; make subplots close to each other and hide x ticks for
# all but bottom plot.
f.subplots_adjust(hspace=0)
plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)

plt.show()
