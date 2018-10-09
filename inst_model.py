import numpy as np
import scipy.sparse as sps
from datetime import datetime
from scipy.sparse import linalg as lag

class inst_model2d:
  '''
    This model is a QG instability model based on Kontoyiannis (1997). It seeks a solution of the form:

          var*exp(ik*(x-ct))

     Input:
       u  => climatological zonal velocity [ms-1] (M,N)
       y  => meridian distance             [m]    (M,N)
       z  => depth                         [m]    (M,N)
       n2 => BV frequency                  [s-1]  (M,N)
       k  => zonal wavenumber              [m-1]  (S)

     Output:
       self.cr  => real phase speed
       self.sig => growth rate
       self.Pph => angle
       self.Pn  => Amplitude
  '''
  def __init__(self,u,y,z,n2,lat,k,dy=None,dz=None):
    self.u = u;
    self.y = y;
    self.z = z;
    self.n2 = n2;
    self.lat = lat;
    self.k = k;
    self.dy = dy;
    self.dz = dz;

  def func_model(self,Uz,Nz):
    f0   = self.f0;
    k    = self.k;
    n    = self.u.shape[0];
    Un   = self.u.T.flatten();
    Qyn  = self.Qy.T.flatten();
    N2n  = self.n2.T.flatten();
    dz   = self.dz.T.flatten();
    dy   = self.dy.T.flatten();
    Uzn  = Uz.T.flatten();
    Nzn  = Nz.T.flatten();

    Nn   = N2n**.5;                       # square-root of BV freq.

    cr = np.zeros([k.shape[0]]);
    sig = cr.copy();
    P = np.zeros([Un.shape[0],k.shape[0]],dtype=complex)
    for n2 in range(0,k.shape[0]):            # begin wavenumber loop
      print('%s (%1.2f)' % (n2,n2*1.0/k.shape[0]*100));
# Mounting Matrices
      L1=(f0/Nn)**2*(1./dz**2-(Nzn/Nn)/dz);
      L2=k[n2]**2+2*((f0/Nn)**2/dz**2+1./dy**2);
      L3=(f0/Nn)**2*(1./dz**2+(Nzn/Nn)/dz);
      L4=1./dy**2
# A-Matrix
      mdiag = Qyn-Un*L2;
      ddiag = Un*L1; ddiag[::n]    = 0;
      udiag = Un*L3; udiag[n-1::n] = 0;
      ndiag = Un*L4*np.ones(L3.shape);
# Boundary Conditions
      mdiag[::n] = (Qyn-Un*L2+2*L1*Uzn*dz)[::n];       # Surface
      udiag[::n] = (Un*(L3+L1))[::n];                  # Surface
      mdiag[n-1::n] = (Qyn-Un*L2-2*L3*Uzn*dz)[n-1::n]; # Bottom
      ddiag[n-1::n] = (Un*(L1+L3))[n-1::n];            # Bottom
#    mdiag = np.zeros(Un.shape[0]);
#    udiag = np.zeros(Un.shape[0]);
#    ddiag = np.zeros(Un.shape[0]);
      v1 = ddiag.tolist(); v1.remove(0.0); v1.append(0.0);
      v2 = udiag[:-1].tolist(); v2.insert(0,0.0);
      v3 = ndiag[n:].tolist();
      for i in range(n): v3.append(0);
      v4 = ndiag[:-n].tolist();
      for i in range(n): v4.insert(0,0.0);

      ddiag = np.array(v1); udiag = np.array(v2);

      pz0   = sps.spdiags(mdiag, 0, n**2, n**2);
      pz1   = sps.spdiags(ddiag, -1, n**2, n**2);
      pz2   = sps.spdiags(udiag, 1, n**2, n**2);

#      pymin = sps.spdiags(ndiag,-n, n**2, n**2);
#      pyman = sps.spdiags(ndiag,n, n**2, n**2);

      pymin = sps.spdiags(v3,-n, n**2, n**2);
      pyman = sps.spdiags(v4,n, n**2, n**2);


      pz    = pz0+pz1+pz2; py    = pymin+pyman;

      A = pz+py;

# B-Matrix
      mdiag = -1*L2;
      ndiag =  L4*np.ones(L3.shape);
      ddiag = L1.copy(); ddiag[::n]    = 0;
      udiag = L3.copy(); udiag[n-1::n] = 0;

# Boundary Conditions
      udiag[::n]    = (L3+L1)[::n];               # Surface
      ddiag[n-1::n] = (L1+L3)[n-1::n];            # Bottom

      v1 = ddiag.tolist(); v1.remove(0.0); v1.append(0.0);
      v2 = udiag[:-1].tolist(); v2.insert(0,0.0);
      ddiag = np.array(v1); udiag = np.array(v2);
      pz0   = sps.spdiags(mdiag, 0, n**2, n**2);
      pz1   = sps.spdiags(ddiag, -1, n**2, n**2);
      pz2   = sps.spdiags(udiag, 1, n**2, n**2);

      pymin = sps.spdiags(ndiag,-n, n**2, n**2);
      pyman = sps.spdiags(ndiag,n, n**2, n**2);
#      pymin = sps.spdiags(v3,-n, n**2, n**2);
#      pyman = sps.spdiags(v4,n, n**2, n**2);

      pz    = pz0+pz1+pz2; py    = pymin+pyman;
      B = pz+py;

      tini = datetime.now();
      print('->Solving B^(-1)A');
      C      = lag.spsolve(B.tocsc(),A.tocsc(),
                           permc_spec=None,use_umfpack=True);
      print('->Calc. eig');
      lamb,F = lag.eigs(C,k=3,which='LI',tol=1e-5);
      tfind = datetime.now();
      dt = tfind-tini;
      hour   = np.floor(dt.total_seconds()/60./60.);
      minute = np.floor((dt.total_seconds()/60./60.-hour)*60.);
      sec    = np.floor((dt.total_seconds()/60.-minute)*60.);
      print('    (%02.0f:%02.0f:%02.0f)' % (hour,minute,sec));
      print ''
# save most unstable mode and growth rates corresponding to k
      ci = np.nanmax(np.imag(lamb));
      jmax = np.where(np.imag(lamb) == ci)[0][0];
      if ci == 0:
        sig[n2]=0;
        cr[n2]=np.nan;
        P[:,n2]=np.nan*np.ones(Un.shape[0]);
      else:
        sig[n2]=k[n2]*ci*86400.               # growth rate in days^(-1)
#        cr[n]=np.real(lamb[jmax,jmax])*100;  # phase speed in cm/s
        cr[n2]=np.real(lamb[jmax]);       # phase speed in cm/s
    Pamp=abs(P);                         # amplitude
    Pphase=180./np.pi*np.angle(P);           # phase in degrees
    sigmax = np.nanmax(sig);
    imax = np.where(sig==sigmax)[0][0];
    Pmax=Pamp[:,imax];
#    Pmax=Pmax/np.nanmax(abs(Pmax));  # normalization by maximum value
#    Pmax=Pmax/(norm(Pmax));    # orthonormalization 
    Pphmax = Pphase[:,imax];

    Pn = np.zeros([n,n,P.shape[-1]],dtype=complex);
    for i in range(k.shape[-1]):
      Pn[:,:,i] = np.reshape(P[:,i],(n,n)).T;
    Pnmax = np.reshape(Pmax,(n,n)).T;
    Pnphmax = np.reshape(Pphmax,(n,n)).T;
    self.Pn = Pn;
    self.Pnmax = Pnmax;
    self.Pnphmax = Pnphmax;
    self.F = F;
    self.cr = cr;
    self.sig = sig;

  def func_run(self):
    y = self.y;
    z = self.z;
    U = self.u;
    lat = self.lat;
    N2 = self.n2;
    dy = self.dy;
    dz = self.dz;
    N = np.sqrt(N2);
#   Basic Evaluation of matrix accessibility
    if (y.ndim or U.ndim) != 2 or y.shape != z.shape or U.shape != (y.shape or z.shape):
      raise Exception('Dimensions of x,y,U not equal')
# Variable Declaration
    g = 9.8;                      # gravity (m.s^(-2))
#  rho0 = 1.025e3;               # Density (kg.m^(-3))
# Variable Processing
    n = y.shape[0];               # size of vector
    f0     = 2*(2*np.pi/86400)*np.sin(lat*np.pi/180.);       # Coriolis
    beta   = 2*(2*np.pi/86400)*np.cos(lat*np.pi/180)/6371e3; # Beta param
    if type(self.dy) == type(None):
      dy     = np.gradient(y,axis=1);
      Uy     = np.gradient(U,axis=1)/dy;        # variation of U in y
      Uyy    = np.gradient(Uy,axis=1)/dy;       # variation of Uy in y
    else:
      Uy     = np.gradient(U,dy,axis=1);        # variation of U in y
      Uyy    = np.gradient(Uy,dy,axis=1);       # variation of Uy in y
    if type(self.dz) == type(None):
      dz     = np.gradient(z,axis=0);
      Nz     = np.gradient(N,axis=0)/dz;           # variation of BV in z
      Uz     = np.gradient(U,axis=0)/dz;           # variation of U in z
      strain = np.gradient(f0*f0*Uz/N2,axis=0)/dz; # strain
    else:
      Nz     = np.gradient(N,dz,axis=0);       # variation of BV in z
      Uz     = np.gradient(U,dz,axis=0);        # variation of U in z
      strain = np.gradient(f0*f0*Uz/N2,dz,axis=0); # strain
    Qy     = beta - strain - Uyy;                # Potential vort

    self.Qy = Qy;
    self.dy = dy;
    self.dz = dz;
    self.f0 = f0;
    self.beta = beta;

    #Qy,Pn,Pmax,Pphmax,F,cr,sig  = run(self,Uz,Nz);
    self.func_model(Uz,Nz);


