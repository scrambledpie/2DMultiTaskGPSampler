# This first block of code defines a test function creator\
# that takes a random seed and a correlation value and\
# returns a randomly generated function from [-1,1]*[-1,1] -> R\
import numpy as np\
from scipy.spatial import distance_matrix\
def createFunction(rho=-0.7,seed=1):\
    np.random.seed(seed)\
    Nx=10\
    lX=0.2\
    dx=1/Nx\
    ilX2 = -0.5/(lX*lX)\
    xT=np.linspace(-1+dx,1-dx,Nx)\
    XXT=np.array([np.repeat(xT,Nx),np.tile(xT,Nx)]).transpose()\
    Sigma=np.exp( ilX2* distance_matrix(XXT,XXT)**2 )\
\
    #rho=0.7 # correlation between objecgives\
    SigmaA = np.concatenate((Sigma,rho*Sigma))\
    SigmaB = np.concatenate((rho*Sigma,Sigma))\
    SigmaC = np.concatenate((SigmaA,SigmaB),1)\
    Ytrue  = np.random.multivariate_normal(np.zeros(2*Nx*Nx), SigmaC)\
    iK1=np.linalg.solve(Sigma, Ytrue[:(Nx*Nx)])\
    iK2=np.linalg.solve(Sigma, Ytrue[(Nx*Nx):])\
\
    def simout(x):\
        ks=np.zeros(Nx*Nx)\
        for i in range(Nx*Nx):\
            ks[i]=np.sum((x-XXT[i,:])**2)\
        ks=np.exp(ks*ilX2)\
        return[np.sum(ks*iK1),np.sum(ks*iK2)]\
    \
    return simout\
\
\
\
\
\
#############################################\
# an example where the objectvies are negatively correlated\
import matplotlib.pyplot as plt\
%matplotlib inline\
Sim1=createFunction(-0.9)\
x1=np.linspace(-1,1,100)\
XX1=np.array([np.repeat(x1,x1.size),np.tile(x1,x1.size)]).transpose()\
Y  =np.zeros(x1.size*x1.size)\
Y1 =np.zeros(x1.size*x1.size)\
for i in range(x1.size*x1.size):\
    A=Sim1(XX1[i,:])\
    Y[i]=A[0]\
    Y1[i]=A[1]\
    \
f, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(15,5))\
ax1.imshow(Y.reshape(x1.size,-1),cmap=plt.get_cmap("bwr"))\
ax2.imshow(Y1.reshape(x1.size,-1),cmap=plt.get_cmap("bwr"))\
ax3.plot(Y,Y1)\
for theta in np.linspace(0,0.5*np.pi,100):\
    ii=(Y*np.sin(theta) + Y1*np.cos(theta)).argmin()\
    ax3.plot(Y[ii],Y1[ii],"ro")\
\
\
\
\
\
#############################################\
# an example where the objectvies are positively correlated\
import matplotlib.pyplot as plt\
%matplotlib inline\
Sim1=createFunction(0.9)\
x1=np.linspace(-1,1,100)\
XX1=np.array([np.repeat(x1,x1.size),np.tile(x1,x1.size)]).transpose()\
Y  =np.zeros(x1.size*x1.size)\
Y1 =np.zeros(x1.size*x1.size)\
for i in range(x1.size*x1.size):\
    A=Sim1(XX1[i,:])\
    Y[i]=A[0]\
    Y1[i]=A[1]\
    \
f, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(15,5))\
ax1.imshow(Y.reshape(x1.size,-1),cmap=plt.get_cmap("bwr"))\
ax2.imshow(Y1.reshape(x1.size,-1),cmap=plt.get_cmap("bwr"))\
ax3.plot(Y,Y1)\
for theta in np.linspace(0,0.5*np.pi,100):\
    ii=(Y*np.sin(theta) + Y1*np.cos(theta)).argmin()\
    ax3.plot(Y[ii],Y1[ii],"ro")}