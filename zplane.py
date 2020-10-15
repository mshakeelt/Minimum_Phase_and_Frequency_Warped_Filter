#Gerald Schuller, June 2016
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def zplane(nullstellen, pole, axis = None):
    
    plt.figure()
    plt.plot(np.real(pole),np.imag(pole),'x')
    plt.plot(np.real(nullstellen),np.imag(nullstellen),'o')
    plt.axis('equal')
    if axis is not None:
        plt.axis(axis)
    
    circlere=np.zeros(512)
    circleim=np.zeros(512)
    for k in range(512):
       circlere[k]=np.cos(2*np.pi/512*k)
       circleim[k]=np.sin(2*np.pi/512*k)
    
    plt.plot(circlere,circleim)
    plt.title('Complex z-Plane')
    plt.show()
    return()

