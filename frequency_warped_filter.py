import scipy.signal as sp
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from zplane import zplane
from sound import *
from freqz import *

fs=44.1
fc=0.15*np.pi


# All pass co-efficient
a = 1.0674*(2/np.pi*np.arctan(0.6583*fs))**0.5 - 0.1916


def warpingphase(w, a):
#produces (outputs) phase wy for an allpass filter
#w: input vector of normlized frequencies (0..pi)
#a: allpass coefficient
#phase of allpass zero/pole :
    theta = np.angle(a)
#magnitude of allpass zero/pole :
    r = np.abs(a)
    wy = -w-2*np.arctan((r*np.sin(w-theta))/(1-
    r*np.cos(w-theta)))
    return wy



fcw=-warpingphase(fc,a)
fq=fcw/(2*np.pi)

c = sp.remez(6, [0, fq, fq+0.025, 0.5],[1, 0], weight=[1, 100])

zros=np.roots(c)

zplane(zros, 0, [-2,8,-2,2])

#The resulting Impulse Response:
plt.figure()
plt.plot(c)
plt.title('Filter Impulse Response')

m,p=sp.freqz(c)
plt.figure()
plt.plot(m, 20*np.log10(abs(p)))
plt.title('Filter Frequency Response')
plt.show()



#All pass filter
#Numerrator:
B = [-a.conjugate(), 1]
#Denominator:
A = [1, -a]


fs, x = wav.read('Track48.wav')
left = x[:, 0]
right = x[:, 1]


y=left
N=4
n=32
h_bpass=sp.remez(n, [0.0, 0.075, 0.125, 0.5] , [1.0, 0.0],weight=[1, 100])

h0 = h_bpass[0::N]
h1 = h_bpass[1::N]
h2 = h_bpass[2::N]
h3 = h_bpass[3::N]

y0 = sp.lfilter(h0,1,y)
y1 = sp.lfilter(h1,1,y)
y2 = sp.lfilter(h2,1,y)
y3 = sp.lfilter(h3,1,y)

L = max([len(y0), len(y1), len(y2), len(y3)])

yu = np.zeros(N*L)

yu[0::N] = y0
yu[1::N] = y1
yu[2::N] = y2
yu[3::N] = y3

# Y1(z)=A(z), Y2(z)=A^2(z),... 
# Warped delays:
y1 = sp.lfilter(B,A,x)
y2 = sp.lfilter(B,A,y1)
y3 = sp.lfilter(B,A,y2)
y4 = sp.lfilter(B,A,y3)
y5 = sp.lfilter(B,A,y4)

# Output of warped filter with impulse as input:
yout = c[0]*x+c[1]*y1+c[2]*y2+c[3]*y3+c[4]*y4+c[5]*y5
# frequency response:
#freqz(yout, 1)

"""#Impulse response:
plot(yout);
xlabel('Sample')
ylabel('value')
title('Impulse Response of Warped Lowpass Filter')
"""
#sound(x,fs)
#sound(yout,fs)

