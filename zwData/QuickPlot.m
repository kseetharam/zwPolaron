
load('dataToExport.mat')

for idx = 1:18
    figure(1),clf;
    subplot(3,1,1);
    e = errorbar(dataToExport.K_time(idx,:),dataToExport.K_vel(idx,:),dataToExport.K_vel_std(idx,:),'d','MarkerSize',1.2*8,'CapSize',0,'LineWidth',2 );
    
    title(['K velovity: a_{BF} = ' num2str(dataToExport.aBFs(idx)) ' a_{Bohr}'])
    ylabel('velocity (\mum/ms)');
    xlabel('time (ms)');
    set(gca, 'FontName', 'Arial')
    set(gca,'FontSize', 12);
    
    
    subplot(3,1,2);
    e = errorbar(dataToExport.Na_time(idx,:),dataToExport.Na_vel(idx,:),dataToExport.Na_vel_std(idx,:),'d','MarkerSize',1.2*8,'CapSize',0,'LineWidth',2 );
    
    title(['Na velovity: a_{BF} = ' num2str(dataToExport.aBFs(idx)) ' a_{Bohr}'])
    ylabel('velocity (\mum/ms)');
    xlabel('time (ms)');
    set(gca, 'FontName', 'Arial')
    set(gca,'FontSize', 12);
    
    subplot(3,1,3);
    e = errorbar(dataToExport.relVel_time(idx,:),dataToExport.relVel(idx,:),dataToExport.relVel_std(idx,:),'d','MarkerSize',1.2*8,'CapSize',0,'LineWidth',2 );
    hold on
    plot(xlim,[dataToExport.speedOfSound_array(idx),dataToExport.speedOfSound_array(idx)],'k--','LineWidth',2)
    plot(xlim,[-dataToExport.speedOfSound_array(idx),-dataToExport.speedOfSound_array(idx)],'k--','LineWidth',2)
    hold off
    
    
    title(['relative velocity: a_{BF} = ' num2str(dataToExport.aBFs(idx)) ' a_{Bohr}'])
    ylabel('velocity (\mum/ms)');
    xlabel('time (ms)');
    set(gca, 'FontName', 'Arial')
    set(gca,'FontSize', 12);
    
    waitforbuttonpress
end    