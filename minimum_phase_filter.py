import scipy.io.wavfile as wav
import scipy.signal as sp
import numpy as np
import matplotlib.pyplot as plt
from zplane import zplane
from sound import *
from freqz import *

d=sp.remez(6, [0, 0.25, 0.35, 0.5],[1, 0],[1, 100]) 

rt=np.roots(d)
print("roots of filter", rt)
#zplane(np.roots(d), 0, [-3, 2, -1.1, 1.1])

[b, r] = sp.deconvolve(d, [1,-rt[0]])

rt=np.roots(b)
print("roots of filter after deconvolve", rt)

#zplane(np.roots(b), 0, [-3, 2, -1.1, 1.1])

dcomp=sp.convolve(b,[1,-1/rt[0].conjugate()])

rt=np.roots(dcomp)
print("roots of filter after deconvolve", rt)
e=np.roots(dcomp)                    #minimum phase filters root

zplane(np.roots(dcomp), 0, [-3, 2, -1.1, 1.1])

plt.figure()
plt.plot(dcomp)
plt.title('Filter Impulse Response')

x,v=sp.freqz(dcomp)

plt.figure()
plt.plot(x, 20*np.log10(abs(v)))
plt.title('Filter Frequency Response')
plt.show()


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

yout = sp.lfilter(dcomp,1,x)

#sound(x,fs)
#sound(yout,fs)

