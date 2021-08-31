clear;
load('oscdata/dataToExport.mat')
wB = dataToExport.omegasNa/(2*pi);
wI = dataToExport.omegaYK/(2*pi);
xTF = dataToExport.meanTFxArrayInMuM;
yTF = dataToExport.meanTFyArrayInMuM;
Npoints = 100000;
yMat = zeros(18, Npoints);
densityMat = zeros(18, Npoints);
n0Array = zeros(18, 1) 

for i = 1:18
yMat(i,:) = linspace(-4*yTF(i),4*yTF(i),Npoints);
out = BECDensity(0,yMat(i,:),0,2*pi*[12,75,100],80e-9,xTF(i));
densityMat(i,:) = out.densityArray;
n0Array(i) = out.nc0 + out.nthermal0;
end

save('yMat_InMuM.mat','yMat')
save('densityMat_InM-3.mat','densityMat')
save('n0Array_InM-3.mat','n0Array')

ind = 6;
figure(); 
plot(yMat(ind,:)./yTF(ind),densityMat(ind,:));
