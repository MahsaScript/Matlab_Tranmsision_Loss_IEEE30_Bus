%%--------------------------------------------------------------------------
% % This is the main program for training a Artificial Neural Network for predicting tranmsision loss for an IEEE 30 bus system 
% 
%--------------------------------------------------------------------------
clear 
db = xlsread('lossdb.xlsx');
Q=length(db(:,1));
trainRatio=0.95;
valRatio=0;
testRatio=0.05;

[trainInd,valInd,testInd] = dividerand(Q,trainRatio,valRatio,testRatio);
inputs=db(trainInd,1:6)';
targets=db(trainInd,7)';


net = feedforwardnet([7 5]);

net.divideParam.trainRatio=0.85;
net.divideParam.testRatio=0.0;
net.divideParam.valRatio=0.15;
net.layers{1}.transferFcn='tansig';
net.layers{2}.transferFcn='purelin';
net.layers{3}.transferFcn='purelin';
net.trainParam.min_grad=1e-8;
[net,tr] = train(net,inputs,targets);

predicted = net(db(testInd,1:6)');
actual=db(testInd,7)';
RMSE = sqrt(mean((actual - predicted).^2))
perf = perform(net,actual,predicted)

figure;
scatter(actual,predicted)
xlabel('Actual Loss in per unit')
ylabel('Predicted Loss in per unit')


figure;
plotperform(tr)







