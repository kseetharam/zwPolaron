function out = BECDensity(xDes, yDes, zDes, omegas, temperature, xTFinMuM)
% Modified from the DensityAndTemperatureConvolveNaK.m to calculate Na only
% peak global 3D peak density 190814

%Changes: convolve Na density with K Gaussian density to get the average
%overlap density for En
%Inputs: Bose and BEC bimodal fit structures from TOF, KData specifying
%K -9/2 gaussian waists defined as e^(-1/2), analysis box size in pixels
    %Chemical potential is given in kHz (BEC_X, BEC_Y, BEC_Z)
    %Temperature is given in Kelvin (Bose_XZ)
    %Statistical variance is computed from in-situ images of the BEC,
    %'xTFpixZcam'
%Outputs: local nBEC, nThermal, nTotal, local T/Tc computed for homogeneous
%gas, global T/Tc, En
%Method: takes chemical potential and computes Nc.  Takes chemical potential
%and temperature to compute in-situ thermal distribution, then computes
%Nthermal.  Absolute temperature is from fitting the wings of TOF to Bose
%function.

    
    omegaX = omegas(1);
    omegaY = omegas(2);
    omegaZ = omegas(3);
    
    aBohr=5.2917721067e-11;
    a=52*aBohr; %Na background scattering length
    PlanckConst=6.62607004*10^(-34);
    hbar=PlanckConst/2/pi;
    kB=1.38064852e-23;
    mElectron=9.1093837015e-31;
    mNeutron=1.674927471e-27;
    mProton=1.672621898e-27;
    mNa=12*mNeutron+11*mProton;
    mK=19*mProton+21*mNeutron;
    mReduced=mK*mNa/(mK+mNa);
%     muBohr=elementaryCharge*hbar/2/mElectron;
    
%     u = 1.661*10^-27;
%     mK = 39.96 * u;
%     mNa = 22.99 * u;
%     mReduced = (mNa * mK) / (mNa + mK);
%     PlanckConst =  1.0555*10^-34; hbar=PlanckConst;
%     aBohr = 5.29*10^-11;
%     kB = 1.380649*10^-23;
    
    xTF = xTFinMuM;
    yTF = xTF*omegaX/omegaY;
    zTF = xTF*omegaX/omegaZ;
    
    temp=temperature;
    chemicalPotential_kHz = 1/2*mNa*omegaX^2*(xTF/10^6)^2/(PlanckConst*1000);
    
    %10^6*sqrt(2*p(1)*10^3*PlanckConst./(mass*omega^2))
    
    chemicalPotential = chemicalPotential_kHz*1000*PlanckConst;
    
%%%%%%%%%%%%  Define constants for Na %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    omegabar = (omegaX*omegaY*omegaZ)^(1/3);
    a_s=52*aBohr; %Sodium F=1 scattering length, away from FB resonance
    prefactorEn=hbar^2/4/mReduced*(6*pi^2)^(2/3); %prefer REDUCED MASS, not Boson mass
    Nc = (2*chemicalPotential).^(2.5)/(15*hbar^2*sqrt(mNa)*omegabar^3*a_s);
    beta=1/kB/temp;
    lambdaDB= PlanckConst/sqrt(2*pi*mNa*kB*temp)*10^6;    %lambda is de Broglie wavelength in um
%     [MarqueeBoxesInMuMeter,numBoxes]=getMarqueeBoxes(boxWidths,boxHeigths,micronPerPixel);
    
%%%%%%%%%%%%% Get K parameters assuming Maxwell-Boltzmann distribution
% %The TiSa is circular, so the z and the y waists should be roughly equal
% %Assume the x-trap frequency is scaled from the measured Na x-trap freq and
% %only comes from the 1064 beam, not from the TiSa
%     xWaistK=KData(1); %in micron
%     yWaistK=KData(2);
%     zWaistK=KData(3); 
% 
%     fprintf('K x-y-z- Gaussian e^-(1/2) waists are: \n')
%     display([xWaistK yWaistK zWaistK])
%     %the Gaussian y-width divided by sqrt(2) is the real standard deviation (68% from -zWaistK to zWaistK) (micron)
%     
% %Input: micron,  Output: micron^-3
%     f_K = @(x,y,z) exp(-x.^2/2/xWaistK^2-y.^2/2/yWaistK^2-z.^2/2/zWaistK^2);
% % %     normK = 8*integral3(f_KU,0,10*xWaistK,0,10*yWaistK,0,10*zWaistK);
% % %     f_K = @(x,y,z) f_KU(x,y,z)/normK;
    
%%%%%%%%%%%%%% Define functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %The condensate probability density distribution function, normalized to 1 when
    %integrated over all space.  Units of um^-3.  Inputs in um.
    f_BEC=@(x,y,z) 15/(8*pi*xTF*yTF*zTF)*(1-x.^2/xTF^2-y.^2/yTF^2-z.^2/zTF^2).*heaviside(1-x.^2/xTF^2-y.^2/yTF^2-z.^2/zTF^2);
    
    %Thermal density bose distribution function, inputs in meter
    %use n3d_thermal Bose Gas (r) = g_1.5(z)/lambda^3
    %Equation 6 in Naraschewki+Stamper-Kurn (1998)    
    %Input micron, output micron^-3.  Not yet normalized.
    f_ThermU=@(x,y,z) real(PolyLog(1.5,exp(-beta*abs(chemicalPotential-1/2*mNa*10^(-12)*(omegaX^2*x.^2+omegaY^2*y.^2+omegaZ^2*z.^2)))));

    %Normalization of f_Therm so that f_Therm/normBose has integral across 3-space =1
    fprintf('calculating number of thermal Bosons \n')
    normBose=8*integral3(f_ThermU,0,xTF*10,0,yTF*10,0,zTF*10,'RelTol',1e-3);
    NThermalBosons=normBose/lambdaDB^3;

    f_Therm = @(x,y,z) f_ThermU(x,y,z)/normBose;
    
    %local 3D Na density, combined thermals + condensate, in um^-3
    %iintegral across 3-space is Nc+NThermalBosons
    n_Na = @(x,y,z) NThermalBosons*f_Therm(x,y,z)+Nc*f_BEC(x,y,z);
    
%%%%%%%%%%%%%%%%  Compute temp functions and properties %%%%%%%%%%%%%%%%%%%%%%
    %Global T/Tc for harmonic trap
    
    %ToverTcGlobal=(1-Nc/(Nc+NThermalBosons))^(1/3);
    Tc=.94*PlanckConst/2/pi*omegabar/kB*(Nc+NThermalBosons).^(1/3);
    ToverTcGlobal=temp/Tc;
    
    out = struct;
    out.ToverTc = ToverTcGlobal;
    out.nc0 = Nc*f_BEC(0,0,0)*(1e6)^3;%m^-3
    out.nthermal0 = NThermalBosons*f_Therm(0,0,0)*(1e6)^3;%m^-3
    out.chemicalPotential = chemicalPotential/PlanckConst*1e-3;%kHz
    out.densityArray = n_Na(xDes,yDes,zDes)*(1e6)^3;%m^-3
