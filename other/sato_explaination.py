import numpy as np
import matplotlib.pyplot as plt

def gaussian(x, mu: float, sigma: float, A: float, noise: bool):

    y = np.zeros_like(x)
    for i, x_i in enumerate(x):
        y[i] = A*np.exp(-np.pow((x_i-mu)/sigma, 2))

    if noise == True:
        gaussian_noise = np.random.normal(0, 0.02, len(x))
    else:
        gaussian_noise = 0
    y = y + gaussian_noise

    return y

def derivative(y, step):
    
    y_prime = np.zeros(len(y)-1)
    for i in range(len(y)-1):
        if i == 0: continue
        y_prime[i] = (y[i+1] - y[i-1])/(2*step)
    print(len(y_prime))
    y_prime=y_prime[1:]
    print(len(y_prime))
    return y_prime

step = 0.1
x = np.arange(-50,50, step)

#2nd derivative on clean gaussian
y_clean = gaussian(x, 0, 5, 1, False)
y_clean_p = derivative(y_clean, step)
y_clean_pp = derivative(y_clean_p, step)
fig, ax = plt.subplots()
ax.plot([0, 0], [-0.5, 1], color='black', linestyle='--')
ax.plot([-25, 25], [0, 0], color='black', linestyle='--')
a, = ax.plot(x,y_clean, label="data")
b, = ax.plot(x[1:-1],y_clean_p, label="1st derivative")
c, = ax.plot(x[2:-2],y_clean_pp, label="2nd derivative")
ax.legend(handles=[a, b, c], fontsize=13)
ax.set_ylabel("Intensity", fontsize=13)
ax.set_xlabel("Position on X axis", fontsize=13)
ax.tick_params(labelsize=13)
plt.show()
y_all = [y_clean for i in range(100)]
plt.imshow(y_all, cmap="grey")
plt.show()

#2nd derivative on clean gaussian
y_clean = gaussian(x, 0, 5, 1, False)
y_noise = gaussian(x, 0, 5, 1, True)
y_clean_p = derivative(y_clean, step)
y_clean_pp = derivative(y_clean_p, step)
fig, ax = plt.subplots()
ax.plot([0, 0], [-0.5, 1], color='black', linestyle='--')
ax.plot([-25, 25], [0, 0], color='black', linestyle='--')
a, = ax.plot(x,y_clean, label="ideal data")
b, = ax.plot(x[1:-1],y_clean_p, label="1st derivative")
c, = ax.plot(x[2:-2],y_clean_pp, label="2nd derivative")
d, = ax.plot(x,y_noise, label='real data')
ax.legend(handles=[a, d, b, c], fontsize=13)
ax.set_ylabel("Intensity", fontsize=13)
ax.set_xlabel("Position on X axis", fontsize=13)
ax.tick_params(labelsize=13)
ax.set_xlim(-25,25)
plt.show()


#2nd derivative on noised gaussian
y_noise = gaussian(x, 0, 5, 1, True)
y_noise_p = derivative(y_noise, step)
y_noise_pp = derivative(y_noise_p, step)
fig, ax = plt.subplots()
ax.plot([0, 0], [-0.5, 1], color='black', linestyle='--')
ax.plot([-25, 25], [0, 0], color='black', linestyle='--')
c, = ax.plot(x[2:-2],y_noise_pp, label="2nd derivative")
b, = ax.plot(x[1:-1],y_noise_p, label="1st derivative")
a, = ax.plot(x,y_noise, label="data")
ax.legend(handles=[a, b, c], fontsize=13)
ax.set_ylabel("Intensity", fontsize=13)
ax.set_xlabel("Position on X axis", fontsize=13)
ax.tick_params(labelsize=13)
plt.show()
y_noised = [y_noise for i in range(100)]
plt.imshow(y_noised, cmap="grey")
plt.show()

#convolution then second derivative
conv = np.convolve(y_clean, y_noise, 'same')
conv_p = derivative(conv, step)
conv_pp = derivative(conv_p, step)
fig, ax = plt.subplots()
ax.plot([0, 0], [-68, 68], color='black', linestyle='--')
ax.plot([-25, 25], [0, 0], color='black', linestyle='--')
a, = ax.plot(x,conv, label="convolution")
b, = ax.plot(x[1:-1],5*conv_p, label="1st derivative")
c, = ax.plot(x[2:-2],25*conv_pp, label="2nd derivative")
ax.legend(handles=[d, e, a, b, c], fontsize=13)
ax.set_ylabel("Intensity", fontsize=13)
ax.set_xlabel("Position on X axis", fontsize=13)
ax.tick_params(labelsize=13)
ax.set_xlim(-25,25)
plt.show()

#3-D second derivative tube
y_noise2 = gaussian(x, 0, 3, 1, True)
conv2 = np.convolve(y_clean, y_noise2, 'same')
conv2_p = derivative(conv2, step)
conv2_pp = derivative(conv2_p, step)
y_noise3 = gaussian(x, 0, 1000, 1, True)
conv3 = np.convolve(y_clean, y_noise3, 'same')
conv3_p = derivative(conv3, step)
conv3_pp = derivative(conv3_p, step)
fig, ax = plt.subplots(ncols=3, sharex='all')
# ax[:].plot([0, 0], [-68, 68], color='black', linestyle='--')
# ax[:].plot([-25, 25], [0, 0], color='black', linestyle='--')
c, = ax[2].plot(x[2:-2],25*conv_pp, label="2nd derivative")
b, = ax[1].plot(x, conv, label="convolution")
a, = ax[0].plot(x, y_noise, label="x-axis")
c2, = ax[2].plot(x[2:-2],25*conv2_pp, label="2nd derivative")
b2, = ax[1].plot(x, conv2, label="convolution")
a2, = ax[0].plot(x, y_noise2, label="y-axis")
c3, = ax[2].plot(x[2:-2],25*conv3_pp, label="2nd derivative")
b3, = ax[1].plot(x, conv3, label="convolution")
a3, = ax[0].plot(x, y_noise3, label="z-axis")
ax[0].legend(handles=[a, a2, a3], fontsize=15)
ax[0].set_ylabel("Intensity", fontsize=15)
ax[1].set_xlabel("Position on X axis", fontsize=15)
ax[1].tick_params(labelsize=15)
ax[0].set_xlim(-25, 25)
ax[1].set_xlim(-25, 25)
ax[2].set_xlim(-25, 25)
plt.show()

#3-D second derivative sheet
y_noise2 = gaussian(x, 0, 100, 1, True)
conv2 = np.convolve(y_clean, y_noise2, 'same')
conv2_p = derivative(conv2, step)
conv2_pp = derivative(conv2_p, step)
y_noise3 = gaussian(x, 0, 1000, 1, True)
conv3 = np.convolve(y_clean, y_noise3, 'same')
conv3_p = derivative(conv3, step)
conv3_pp = derivative(conv3_p, step)
fig, ax = plt.subplots(ncols=3, sharex='all')
# ax[:].plot([0, 0], [-68, 68], color='black', linestyle='--')
# ax[:].plot([-25, 25], [0, 0], color='black', linestyle='--')
c, = ax[2].plot(x[2:-2],25*conv_pp, label="2nd derivative")
b, = ax[1].plot(x, conv, label="convolution")
a, = ax[0].plot(x, y_noise, label="x-axis")
c2, = ax[2].plot(x[2:-2],25*conv2_pp, label="2nd derivative")
b2, = ax[1].plot(x, conv2, label="convolution")
a2, = ax[0].plot(x, y_noise2, label="y-axis")
c3, = ax[2].plot(x[2:-2],25*conv3_pp, label="2nd derivative")
b3, = ax[1].plot(x, conv3, label="convolution")
a3, = ax[0].plot(x, y_noise3, label="z-axis")
ax[0].legend(handles=[a, a2, a3], fontsize=15)
ax[0].set_ylabel("Intensity", fontsize=15)
ax[1].set_xlabel("Position on X axis", fontsize=15)
ax[1].tick_params(labelsize=15)
ax[0].set_xlim(-25, 25)
ax[1].set_xlim(-25, 25)
ax[2].set_xlim(-25, 25)
plt.show()

#3-D second derivative blob
y_noise2 = gaussian(x, 0, 3, 1, True)
conv2 = np.convolve(y_clean, y_noise2, 'same')
conv2_p = derivative(conv2, step)
conv2_pp = derivative(conv2_p, step)
y_noise3 = gaussian(x, 0, 4, 1, True)
conv3 = np.convolve(y_clean, y_noise3, 'same')
conv3_p = derivative(conv3, step)
conv3_pp = derivative(conv3_p, step)
fig, ax = plt.subplots(ncols=3, sharex='all')
# ax[:].plot([0, 0], [-68, 68], color='black', linestyle='--')
# ax[:].plot([-25, 25], [0, 0], color='black', linestyle='--')
c, = ax[2].plot(x[2:-2],25*conv_pp, label="2nd derivative")
b, = ax[1].plot(x, conv, label="convolution")
a, = ax[0].plot(x, y_noise, label="x-axis")
c2, = ax[2].plot(x[2:-2],25*conv2_pp, label="2nd derivative")
b2, = ax[1].plot(x, conv2, label="convolution")
a2, = ax[0].plot(x, y_noise2, label="y-axis")
c3, = ax[2].plot(x[2:-2],25*conv3_pp, label="2nd derivative")
b3, = ax[1].plot(x, conv3, label="convolution")
a3, = ax[0].plot(x, y_noise3, label="z-axis")
ax[0].legend(handles=[a, a2, a3], fontsize=15)
ax[0].set_ylabel("Intensity", fontsize=15)
ax[1].set_xlabel("Position on X axis", fontsize=15)
ax[1].tick_params(labelsize=15)
ax[0].set_xlim(-25, 25)
ax[1].set_xlim(-25, 25)
ax[2].set_xlim(-25, 25)
plt.show()