%% 
close all;

ds = readtable("matlab_ml_dataset.csv","Delimiter",',');

res = ["M","B"];

ds.Var2 = categorical(ds.Var2,res);
%%
ds= rmmissing(ds);

data_count = groupsummary(ds,"Var2");

cv = cvpartition(size(ds,1),'HoldOut',0.3);
idx = cv.test;

train = ds(~idx,:);
test  = ds(idx,:);
%%
x_train = table2array(train(:,3:end));
x_test = table2array(test(:,3:end));
y_train = grp2idx(table2array(train(:,2)));
y_test  = grp2idx(table2array(test(:,2)));

%%
Wxl = rand(size(x_train,2),10);
b1 = rand(1,10);
Wly = rand(size(Wxl,2),1);
b2 = rand(1);

error=[];
totalacc=[];
for itr=1:10
    mse=[];
    accuracy=[];
    disp ('$$$$$$$$$$ iteration $$$$');
    disp(itr);
    for indx=1:399
        disp('##################index############');
        disp(indx);
        % FORWARD PROPAGATION
        l1_rawY = x_train(indx,:)*Wxl+b1;
        l1_Y = sigmoid(dlarray(l1_rawY));
        
        rawY = l1_Y*Wly +b2;
        y_pred = sigmoid(rawY);
        %%
        
        % output neuron DELTA
        del_out = outputLayer(y_train(indx),y_pred);
        
        %hidden layer
        del_hid = hiddenLayer(l1_rawY,del_out,Wly);
        
        %%
        %hidden weight updation
        learning_rate = 0.01;
        %disp(Wly);
        % OUTPUT LAYER WEIGHT UPDATION
        % NEW WEIGHT = OLD WEIGHT + LEARNING RATE * DELTA * OUTPUT OF THAT PATH
        q=size(Wly);
        for i=1:q(2)+1
            e = Wly(i)+ learning_rate*del_out(1)*l1_Y(i);
            if(e<0)
                Wly(i) = 0;
            else
                Wly(i)=e;
            end
        end
        % HIDDEN LAYER WEIGHT UPDATION
        q=size(del_hid);
        for j=1:q(2)
            for k=1:size(Wxl(:,1))
                e = Wxl(k,j)+learning_rate*del_hid(j)*x_train(indx,k)*Wxl(k,j);
    
                if(e<0)
                    Wxl(k,j)=0;
                else
                    Wxl(k,j) = e;
                end
        
            end
        end
        %%
        mse(end+1)= y_pred;
        
        if y_pred<0.5
            accuracy(end+1)= y_train(indx)==0;
        else
            accuracy(end+1)=y_train(indx)==1;
        end
    
    end
error = [error cost_func(mse,y_train)];
totalacc = [totalacc sum(accuracy)/399];
end

%%
testacc=[];
for te=1:170
    l1_rawY = x_test(te,:)*Wxl+b1;
        l1_Y = sigmoid(dlarray(l1_rawY));
        
        rawY = l1_Y*Wly +b2;
        y_pred = sigmoid(rawY);
        if y_pred<0.5
            testacc(end+1)= y_test(te)==0;
        else
            testacc(end+1)=y_test(te)==1;
        end
end
disp(sum(testacc)/170);
%%

function cost = cost_func(Ypred,Y)
    cost = (-1/length(Y))*((log(Ypred).*Y')+log(1-Ypred).*(1-Y)');
end
function del_hid = hiddenLayer(y_pred,del_nextL,w_nextL)
        del_hid=[];
        for ii=1:size(w_nextL(1))
            sum_his = 0;
            for jj=1:size(del_nextL)+1
                sum_his = sum_his+del_nextL(ii)*w_nextL(ii);
                del_hid = [del_hid y_pred(ii)*(1-y_pred(ii))*sum_his];
            end
        end
    end
    
function del_out = outputLayer(target,y_pred)
        del_out = y_pred*(1-y_pred)*(target-y_pred);
end







