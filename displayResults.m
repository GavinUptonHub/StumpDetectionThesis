%% Display Thesis Results

% epochVar = [40, 50, 100, 200, 300];
% stump_topo_epochVar = [11.84, 23.81, 18.96, 17.01, 14.68];
epochVar = [30,50,100,150,200,300];
stump_topo_epochVar = [13.10,23.03,18.96,17.46,17.01,14.68];

% epochVar_stump = [50, 100,200];
% stump_epochVar = [18.10, 26.98, 23.81];
epochVar_stump = [40,50,100,150,200,250,400];
stump_epochVar = [14.68, 19.9,29.85,19.05,14.68,20.31,15.56];
epochsNum = [50,100,150,200];
alexAcc = [25.77, 30.83, 33.49,31.13];
figure(1);
plot(epochVar,stump_topo_epochVar,'b-o','MarkerFaceColor',[.1 .1 1],'MarkerSize',5);
hold on;
plot(epochVar_stump,stump_epochVar,'r-o','MarkerFaceColor',[1 .1 .1],'MarkerSize',5);
%plot(epochsNum,alexAcc,'-o','MarkerFaceColor',[.1 1 .1]);
%plot(50,27.10,'-o','MarkerFaceColor',[.1 1 .1]);
xlabel('Number of Epochs','FontSize',13);
xlim([20,450]);
ylim([0, 35]);
ylabel('Average Precision (\%)','FontSize',13);
%title('Average Precision of RGT & RGB models with Varied Epoch Number','FontSize',13);
legend('RGT', 'RGB');%,'FontSize',13);
grid minor
GraphGood();
%% Alexnet
alexAcc = [25.77, 30.83, 33.49,31.13];
epochs = categorical({'50','100','150','200'},{'50','100','150','200'});
epochsNum = [50,100,150,200];
epochVar_stump = [40,50,100,150,200];%,250,400];
stump_epochVar = [14.68, 19.9,29.85,19.05,14.68];%,20.31,15.56];
plot(epochsNum,alexAcc,'-o','Color',[.1 1 .1],'MarkerFaceColor',[.1 1 .1]);
hold on;
plot(epochVar_stump,stump_epochVar, '-o','Color', [1 .1 .1],'MarkerFaceColor',[1 .1 .1]);
%bar(epochs,alexAcc,0.6,'FaceColor',[0.90 0.5 0.1250]);
ylim([0,40]);
xlim([30,250]);
ylabel('Average Precision (\%)');
xlabel('Number of Epochs');
legend('AlexNet','Original');
%title('My network vs AlexNet Model Precision (RGB) versus Number of Epochs');
grid minor
%%
stumptopo_L2 = [14.63, 27.10, 24.92, 24.05, 23.02, 23.81, 13.68,15.82, 5.87];
L2_var = [5e-5, 7e-5, 8e-5,9e-5 1e-4,1.1e-4,1.2e-4, 1.3e-4, 1.5e-4];

plot(L2_var,stumptopo_L2, '-o','Color',[.1 .1 1],'MarkerFaceColor',[.1 .1 1]);
%hold on;
%plot(epochVar, stump_topo_epochVar,'-o','MarkerFaceColor',[.1 .1 1]);
xlabel('L2 Regularization Constant','FontSize',13);
xlim([4e-5 1.8e-4]);
ylim([0, 30]);
ylabel('Average Precision (\%)','FontSize',13);
%title('RGT Data (50 Epoch) with L2 Regularization variation','FontSize',13);
grid minor

%% Ground Classifier
features = categorical({'9Med','9Skew','27Skew','27Med','81Skew','81Med'},{'9Med','9Skew','27Skew','27Med','81Skew','81Med'});
barAcc = [77.9,86.2,87.9,86.4,90.5,91.5];
bar(features,barAcc,0.6,'FaceColor',[0.9290 0.6940 0.1250]);
ylim([70,100]);
xlabel('Feature Vector');
ylabel('Classification Accuracy (\%)');
grid minor
GraphGood();
%title('Classification Accuracy for Features Extracted');

%%
samples=[300,400,500,600,700,800,900,1000];
acc = [88.6,88.67,90.6,90.2,89.3,92.3,90.6];
new_acc = [88,88.33,89.87,90.23,91.8,93.2,93.46,93.1];%,92.6,92.75];

figure(2);
plot(samples,new_acc,'-o','MarkerFaceColor',[0 0.4470 0.7410]);
%hold on;
%plot(samples,acc,'g-o');
xlabel('Number of Training Samples per class');
ylabel('Classification Accuracy (\%)');
xlim([250,1050]);
ylim([80 100]);
%title('Accuracy of Ground Classifier versus Number of Training samples');
grid minor
%%
% dirt = [83.27,83.03,86,87.67,86.43,89.67,86.35];
% grass = [97.03,97.43,97.75,97.77,97.67,98.37,98.87];
% leaf = [85.47,85.6,85.7,85.17,83.87,88.63,86.8];
dirtPrec = [81.65, 82.3,85.3,86.1,88.3,90.2,91,90.4];
grassPrec = [98.2,98.7,97.9,98.5,98.2,99,98.5,98.9];
leafPrec = [81.7,83.1,86.4,88.8,88.9,90.3,90.9,89.75];

dirtRecall = [81.4,83.3,86.9,87.9,90.1,90,89.8,89.3];
grassRecall = [97.95,98.1,98.4,98.2,98.3,98.1,98.9,98.75];
leafRecall = [80.9,83.2,85.5,86.6,89.1,91.3,91.5,91.3];

%leaf = [85.47,85.6,
figure(1);
plot(samples,grassPrec,'g-o','MarkerFaceColor',[.1 1 .1]);
hold on;
plot(samples,dirtPrec,'r-o','MarkerFaceColor',[1 .1 .1]);
plot(samples,leafPrec,'b-o','MarkerFaceColor',[.1 .1 1]);
xlim([250 1050]);
ylim([80 100]);
ylabel('Precision (\%)');
xlabel('Number of Training Samples');
legend('Grass','Dirt','Leaf','Location','best');
%title('Recall of Classes versus Number of Training samples');
hold off;
grid minor
figure(2);
plot(samples,grassRecall,'g-o','MarkerFaceColor',[.1 1 .1]);
hold on;
plot(samples,dirtRecall,'r-o','MarkerFaceColor',[1 .1 .1]);
plot(samples,leafRecall,'b-o','MarkerFaceColor',[.1 .1 1]);
xlim([250 1050]);
ylim([80 100]);
ylabel('Recall (\%)');
xlabel('Number of Training Samples');
legend('Grass','Dirt','Leaf','Location','best');
%title('Recall of Classes versus Number of Training samples');
hold off;
grid minor

