% load data
% train, test split
% X,Y split
% initiate parameters
% activation function
% forward prop
% cost function
% one layer back propagation
% back prop
% training
% testing

close all;
ds = readtable("matlab_ml_dataset.csv","Delimiter",',');
res = ["M","B"];
ds.Var2 = categorical(ds.Var2,res);
ds= rmmissing(ds);
data_count = groupsummary(ds,"Var2");
cv = cvpartition(size(ds,1),'HoldOut',0.3);
idx = cv.test;
Train = ds(~idx,:);
Test  = ds(idx,:);

x_train = table2array(Train(:,3:end));
x_test = table2array(Test(:,3:end));
y_train = grp2idx(table2array(Train(:,2)));
y_test  = grp2idx(table2array(Test(:,2)));
y_train = y_train==2;
y_test = y_test==2;


% parameter initialisation
layer_dims = [size(x_train,2),64,32,16,8,4,2,1];
L = length(layer_dims);
params = init_params(layer_dims);

lr = 0.001;

[params,output_dict,cost_history] = train(x_train,y_train,layer_dims,lr,10);
plot(cost_history);
accuracyplot = [];
accuracy = test(x_test,y_test,params,thresh);


disp(accuracy);
%test
function accuracy = test(X,Y,params,thresh)
    accuracy=0;
    for i = 1:length(Y)
        forward = X(i,:);
        L = numEntries(params)/2;
    
        for j=1:L
            forward_prev = forward;
            Z = forward_prev*(cell2mat(params('W'+string(j))))+(cell2mat(params('B'+string(j))));
            forward = activation(Z);
        end
        if forward>=thresh
            out=1;
        else
            out=0;
        end        
        accuracy = accuracy+(Y(i)==out);
    end
    accuracy = accuracy/length(Y);
end

% train
function [params, output_dict, cost_history] =  train(X, Y, layer_dims, lr,epoch)
    cost_history  = [];
    for i=1:10:390
        cost = 0;
        for j=i+1:i+10
            x = X(j,:);
            y = Y(j);
            params = init_params(layer_dims);
            output_dict = forward_prop(x, params);
            Ypred = cell2mat(output_dict(string(length(layer_dims)-1)));
            cost = cost+ cost_func(Ypred,y);
            
            grads = backprop(Ypred, y, output_dict, params);
            
            params = update_parameters(params, grads, lr);
        end
        cost = cost/10;
        disp("epoch "+string(i)+" ------"+"cost: "+string(cost));
        cost_history = [cost_history cost];
    end
end

% update parameters
function params =  update_parameters(params, grads, learning_rate)
    L = numEntries(params)/2;
    for l =1:L
        params('W'+string(l)) = {cell2mat(params('W'+string(l)))-learning_rate*cell2mat(grads('dW'+string(l)))};
        params('B'+string(l)) = {cell2mat(params('B'+string(l)))-learning_rate*cell2mat(grads('dB'+string(l)))};
    end
end


% back propagation
function grads =  backprop(Ypred, Y,output_dict,params)
    grads = dictionary();
    L = numEntries(output_dict)-1;
    del_out = Ypred*(1-Ypred)*(Y-Ypred);
    dY = log(del_out./(1-del_out));
    [grads("dY"+string(L-1)), grads("dW"+string(L)), grads("dB"+string(L))] = one_layer_backward(dY,L,output_dict,params);
    
    for l = L-1:-1:1
        [grads("dY"+string(l-1)), grads("dW" + string(l)), grads("dB" + string(l))] = one_layer_backward(cell2mat(grads("dY"+string(l))),l,output_dict,params);
    end
end 

% one layer backward
function [dYpred_prev,dW,dB] = one_layer_backward(dYpred,l,output_dict,params)
    
    Ypred_prev = cell2mat(output_dict(string(l-1)));
    W = cell2mat(params('W'+string(l)));
    B = cell2mat(params('B'+string(l)));
    Z = Ypred_prev*W + B;
    dZ = dYpred.*activation(Z).*(1-activation(Z));
    m = length(Ypred_prev);
    dW = {(1/m)*(Ypred_prev'*dZ)};
    dB = {(1/m)*sum(dZ)};
    dYpred_prev = {dZ*cell2mat(dW)'};
end

%forward propagation
function output_dict = forward_prop(X, params)
    output_dict = dictionary();
    L = numEntries(params)/2;
    forward = X;
    for i=1:L
        forward_prev = forward;
        Z = forward_prev*(cell2mat(params('W'+string(i))))+(cell2mat(params('B'+string(i))));
        forward = activation(Z);
        output_dict(string(i)) = {forward};
    end
    output_dict("0") = {X};
end

% parameter initialisation
function params = init_params(layer_dims)
    params = dictionary();
    L = length(layer_dims);

    for i=2:L
        params('W'+string(i-1)) = {rand(layer_dims(i-1),layer_dims(i))};
        params('B'+string(i-1)) = {rand(1,layer_dims(i))};
    end
end

%sigmoid function
function res = activation(Z)
    res = sigmoid(dlarray(Z));
end

% cost function
function cost = cost_func(Ypred,Y)
    cost = (-1/length(Y))*((log(Ypred)*Y')+log(1-Ypred)*(1-Y)');
end
