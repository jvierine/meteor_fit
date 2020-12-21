import numpy as n
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as so
import radiant_est


def get_v0(t,p):
    n_m=len(t)
    t_m = n.mean(t)
    t=t-t[0]
    A=n.zeros([n_m*3,6])
    A[0:n_m,0]=1.0
    A[0:n_m,3]=t
    A[n_m:(2*n_m),1]=1.0
    A[n_m:(2*n_m),4]=t
    A[(2*n_m):(3*n_m),2]=1.0
    A[(2*n_m):(3*n_m),5]=t
    m=n.concatenate((p[:,0],p[:,1],p[:,2]))
    xhat=n.linalg.lstsq(A,m)[0]
    p0=n.array([xhat[0],xhat[1],xhat[2]])
    v0=n.array([xhat[3],xhat[4],xhat[5]])    
    return(xhat,A,p0,v0)

def plot_linear_traj(ecef,t,xhat,A):
    n_m=len(t)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(ecef[:,0],ecef[:,1],ecef[:,2],".")
    ax.plot([xhat[0]],[xhat[1]],[xhat[2]],"x")
    traj=n.dot(A,xhat)
    ax.plot(traj[0:n_m],traj[n_m:(2*n_m)],traj[(2*n_m):(3*n_m)])
    plt.show()
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(ecef[:,0]-traj[0:n_m],ecef[:,1]-traj[n_m:(2*n_m)],ecef[:,2]-traj[(2*n_m):(3*n_m)],".")
    plt.show()
    
def est_var(ecef,xhat,A):
    n_m=ecef.shape[0]
    traj=n.dot(A,xhat)
    return(n.array([n.var(ecef[:,0]-traj[0:n_m]),n.var(ecef[:,1]-traj[n_m:(2*n_m)]),n.var(ecef[:,2]-traj[(2*n_m):(3*n_m)])]))





def fit_model(t,p,v_est,p_est,sigma2,hg):
    """ 
    nonlinear fit for trajectory 
    """
    t=t-t[0]
    def model(x):
        v0=10**x[0]
        v1=10**x[1]
        a1=10**x[2]
        u2=n.cos(x[3])
        u0=n.sin(x[3])*n.cos(x[4])
        u1=n.sin(x[3])*n.sin(x[4])
        p0=n.array([x[5],x[6],x[7]])

#        v1 = v0-a0*n.exp(a1*t)
        a0=(v1-v0)/n.exp(a1*n.max(t))
        v = v0-a0*n.exp(a1*t)
        modelf=n.zeros([len(t),3])
        dt=t[1]-t[0]
        modelf[0,:]=p_est+p0
        for i in range(1,len(t)):
            modelf[i,0]=modelf[i-1,0]+u0*v[i]*(t[i]-t[i-1])
            modelf[i,1]=modelf[i-1,1]+u1*v[i]*(t[i]-t[i-1])
            modelf[i,2]=modelf[i-1,2]+u2*v[i]*(t[i]-t[i-1])

        return(modelf)
    
    def ss(x):
        m=model(x)
        s=0.0
        for i in range(3):
            s+=(1.0/sigma2[i])*n.sum(n.abs(m[:,i]-p[:,i])**2.0)
#            s+=n.sum(n.abs(m[:,i]-p[:,i])**2.0)            
        print(s)
        return(s)
    
    vn0=n.linalg.norm(v_est)
    u0=v0/vn0
    # cos(theta)=z
    # sin(theta)*cos(phi)=x
    # sin(theta)*sin(phi)=y    
    theta=n.arccos(u0[2])
    phi=n.arccos(u0[0]/n.sin(theta))
    xhat=so.fmin(ss,[n.log10(vn0),n.log10(1.5*vn0),0,theta,phi,0,0,0])
    m=model(xhat)
    
    plt.plot(t,m[:,0]-p[:,0],".")
    plt.plot(t,m[:,1]-p[:,1],".")
    plt.plot(t,m[:,2]-p[:,2],".")
    plt.xlabel("Time (seconds since first observation)")
    plt.ylabel("Residual (m)")
    plt.show()

    xv0=10**xhat[0]
    xv1=10**xhat[1]
    xa1=10**xhat[2]
    xa0=(xv1-xv0)/n.exp(xa1*n.max(t))
    v = xv0-xa0*n.exp(xa1*t)
    
    plt.plot(t,v/1e3)
    plt.xlabel("Time (seconds since first observation)")
    plt.ylabel("Velocity (km/s)")
    plt.show()

    
    plt.plot(t[0:(len(t)-1)],n.diff(v)/n.diff(t)/1e3)
    plt.xlabel("Time (seconds since first observation)")
    plt.ylabel("Acceleration (km/s$^2$)")
    plt.show()
    
    print(xhat)
    return(10**xhat[0],xhat[3],xhat[4])


if __name__ == "__main__":
    h=h5py.File("data/2020-12-04-pajala.h5","r")
    print(h.keys())
    ecef=n.copy(h[("ecec_pos_m")])
    hg=n.copy(h[("h_km_wgs84")])
    t=n.copy(h[("t_unix")])
    
    
    xhat,A,p0,v0=get_v0(t,ecef)
    print(p0)
    print(t[0])
    print(v0/n.linalg.norm(v0))
    radiant_est.get_radiant(p0,t[0],v0/n.linalg.norm(v0))
    
    sigma2=est_var(ecef,xhat,A)
#    sigma2[2]=sigma2[2].0
    #plot_linear_traj(ecef,t,xhat,A)
    #print(n.sqrt(sigma2))
    
    
    

    v0,theta,phi=fit_model(t,ecef,v0,p0,sigma2,hg)
    u0=n.array([n.sin(theta)*n.cos(phi),n.sin(theta)*n.sin(phi),n.cos(theta)])
    print(radiant_est.get_radiant(p0,t[0],u0).icrs)
    
    print(n.linalg.norm(v0))
