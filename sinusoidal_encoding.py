
import numpy as np
import matplotlib.pyplot as plt
 
def getSinusoidalPositionEncoding(seq_len, d, n=10000):
    P = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d/2)):
            denominator = np.power(n, 2*i/d)
            P[k, 2*i] = np.sin(k/denominator)
            P[k, 2*i+1] = np.cos(k/denominator)
    return P
 



def plotSinusoid(k, d=512, n=10000):
    x = np.arange(0, 100, 1)
    denominator = np.power(n, 2*x/d)
    y = np.sin(k/denominator)
    plt.plot(x, y)
    plt.title('k = ' + str(k))
 

    
if __name__ == "__main__":
    P = getSinusoidalPositionEncoding(seq_len=4, d=4, n=100)
    print(P)
    fig = plt.figure(figsize=(15, 4))    
    for i in range(4):
        plt.subplot(141 + i)
        plotSinusoid(i*4)
    