function Er = recoilEnergy(lambda,mass)
    Er = 1/(2*mass)*(PlanckConst ./lambda).^2;
end

