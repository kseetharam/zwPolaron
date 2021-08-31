function Er = recoilEnergy(lambda,mass)
    PlanckConst=6.62607004*10^(-34);
    Er = 1/(2*mass)*(PlanckConst ./lambda).^2;
end

