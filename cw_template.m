%% Coursework 2022 -- Part 1 (Due on March 6)

%% Main Body -- Do NOT edit

close all; clear; clc;
load('dataset_wisconsin.mat'); % loads full data set: X and t

% Section 1
[X_tr,t_tr,X_te,t_te] = split_tr_te(X, t, 0.7); % tr stands for training, te for test

% Section 2
t_hat1_tr =             hard_predictor1(X_tr);
t_hat1_te =             hard_predictor1(X_te);
t_hat2_tr =             hard_predictor2(X_tr);
t_hat2_te =             hard_predictor2(X_te);

% Section 3
[sens1_te, spec1_te] =  sensitivity_and_specificity( t_hat1_te, t_te );
[sens2_te, spec2_te] =  sensitivity_and_specificity( t_hat2_te, t_te );

% Section 4
loss1_te =              detection_error_loss( t_hat1_te, t_te );
loss2_te =              detection_error_loss( t_hat2_te, t_te );

% Section 5
discussionA();

% Section 6
theta_ls3 =             LSsolver(               X3(X_tr) ,      t_tr    );
theta_ls4 =             LSsolver(               X4(X_tr) ,      t_tr    );

% Section 7
Ngrid =                 101; % number of ponts in grid
[mRadius,mTexture] =    meshgrid(linspace(5,30,Ngrid),linspace(8,40,Ngrid));
X_gr =                  [mRadius(:),mTexture(:)]; % gr for grid

t_hat_ls3_gr =          linear_combiner(        X3(X_gr) ,  theta_ls3 );
t_hat_ls4_gr =          linear_combiner(        X4(X_gr) ,  theta_ls4 );

figure; hold on;
contourf(mRadius,mTexture,max(0,min(1,reshape(t_hat_ls3_gr,[Ngrid,Ngrid]))),'ShowText','on','DisplayName','LS solution'); inc_vec = linspace(0,1,11).'; colormap([inc_vec,1-inc_vec,0*inc_vec]);
plot(X_te(t_te==0,1),X_te(t_te==0,2),'o','MarkerSize',6,'MarkerEdgeColor','k','MarkerFaceColor','c','DisplayName','t=0 test');
plot(X_te(t_te==1,1),X_te(t_te==1,2),'^','MarkerSize',6,'MarkerEdgeColor','k','MarkerFaceColor','m','DisplayName','t=1 test');
contour (mRadius,mTexture,max(0,min(1,reshape(t_hat_ls3_gr,[Ngrid,Ngrid]))),[0.5,0.5],'y--','LineWidth',3,'DisplayName','Decision line');
xlabel('$x^{(1)}$ radius mean','interpreter','latex'); ylabel('$x^{(2)}$ texture mean','interpreter','latex'); colorbar; title('$\hat{t}_3(X|\theta_3)$','interpreter','latex'); legend show;
figure; hold on;
contourf(mRadius,mTexture,max(0,min(1,reshape(t_hat_ls4_gr,[Ngrid,Ngrid]))),'ShowText','on','DisplayName','LS solution'); inc_vec = linspace(0,1,11).'; colormap([inc_vec,1-inc_vec,0*inc_vec]);
plot(X_te(t_te==0,1),X_te(t_te==0,2),'o','MarkerSize',6,'MarkerEdgeColor','k','MarkerFaceColor','c','DisplayName','t=0 test');
plot(X_te(t_te==1,1),X_te(t_te==1,2),'^','MarkerSize',6,'MarkerEdgeColor','k','MarkerFaceColor','m','DisplayName','t=1 test');
contour (mRadius,mTexture,max(0,min(1,reshape(t_hat_ls4_gr,[Ngrid,Ngrid]))),[0.5,0.5],'y--','LineWidth',3,'DisplayName','Decision line');
xlabel('$x^{(1)}$ radius mean','interpreter','latex'); ylabel('$x^{(2)}$ texture mean','interpreter','latex'); colorbar; title('$\hat{t}_4(X|\theta_4)$','interpreter','latex'); legend show;

% Section 8
t_hat_ls3_te =          linear_combiner(        X3(X_te) ,              theta_ls3   );
t_hat_ls4_te =          linear_combiner(        X4(X_te) ,              theta_ls4   );
mse_loss3_te =          mse_loss(               t_hat_ls3_te ,          t_te        );
mse_loss4_te =          mse_loss(               t_hat_ls4_te ,          t_te        );
det_loss3_te =          detection_error_loss(   (t_hat_ls3_te>0.5) ,    t_te        );
det_loss4_te =          detection_error_loss(   (t_hat_ls4_te>0.5) ,    t_te        );

% Section 9
discussionB();

% Section 10
theta_ls5 =             LSsolver(               X5(X_tr) ,              t_tr        );
t_hat_ls5_te =          linear_combiner(        X5(X_te) ,              theta_ls5   );
loss5_te =              detection_error_loss(   (t_hat_ls5_te>0.5) ,    t_te        );

% Section 11
v_ratio_tr =            (10:3:100)/100;
v_loss_LS =             loss_vs_training_size( X_tr, t_tr, X_te, t_te, v_ratio_tr );
figure; plot(v_ratio_tr,v_loss_LS);
xlabel('percentage of used first training samples'); ylabel('Test loss'); title('Detection Error test loss vs. training size');

function out = LSsolver(X,t) % Least Square solver
    out = ( X.' * X ) \ (X.' * t);
end


%% Functions -- Fill in the functions with your own code from this point

% Function 1
function [X_tr,t_tr,X_te,t_te] = split_tr_te(X, t, eta)
%     N_tr =          round(length(t)*eta);
% 	X_tr =          randn(N_tr,size(X,2));
%     t_tr =          (rand(N_tr,1)>0.5);
%     X_te =          randn(length(t)-N_tr,size(X,2));
%     t_te =          (rand(length(t)-N_tr,1)>0.5);
% DELETE ABOVE LINES AND THIS COMMENT, AND PLACE YOUR CODE INSTEAD
    N_tr=round(length(t)*eta);
    X_tr=X(1:N_tr,:);
    t_tr=t(1:N_tr);
    X_te=X(N_tr:end,:);
    t_te=t(N_tr:end);
    %
end

% Function 2
function t_hat1 = hard_predictor1(X)
%     t_hat1 =            (rand(size(X,1),1) > 0.5);
    % DELETE ABOVE LINE AND THIS COMMENT, AND PLACE YOUR CODE INSTEAD
    t_hat1=X(:,1)>14.0;
end

% Function 3
function t_hat2 = hard_predictor2(X)
%     t_hat2 =            (rand(size(X,1),1) > 0.5);
    % DELETE ABOVE LINE AND THIS COMMENT, AND PLACE YOUR CODE INSTEAD
    t_hat2=X(:,2)>20.0;
end

% Function 4
function [sens, spec] = sensitivity_and_specificity( t_hat, t )
%     sens =              rand(1);
%     spec =              rand(1);
    % DELETE ABOVE LINES AND THIS COMMENT, AND PLACE YOUR CODE INSTEAD
    sens=sum(t==1&t_hat==1)/sum(t==1);
    spec=sum(t==0&t_hat==0)/sum(t==0);
end

% Function 5
function loss = detection_error_loss( t_hat, t )
%     loss =              rand(1);
    % DELETE ABOVE LINE AND THIS COMMENT, AND PLACE YOUR CODE INSTEAD
    loss=sum(t_hat~=t)/length(t);
end

% Function 6
function discussionA()
    disp('discussion A:');
    disp('<<ADD DISCUSSION A HERE>>');
    disp('The loss of t1 is 0.1570')
    disp('The loss of t2 is 0.3256')
    disp('The sens and spec of t1 are 0.9231 and 0.8195');
    disp('The sens and spec of t2 are 0.7949 and 0.6391');
    disp('So t1 is better for having higher sensitivity and spectificity,lower loss');
%     corrcoef(X)
end

% Function 7
function out = X3(X)
%     out =               randn(size(X,1),3);
    % DELETE ABOVE LINE AND THIS COMMENT, AND PLACE YOUR CODE INSTEAD
    out=ones(length(X),3);
    out(:,2:3)=X(:,1:2);
    
end

% Function 8
function out = X4(X)
%     out =               randn(size(X,1),6);
    % DELETE ABOVE LINE AND THIS COMMENT, AND PLACE YOUR CODE INSTEAD
    out=ones(length(X),6);
    out(:,2:3)=X(:,1:2);
    out(:,4)=X(:,1).^2;
    out(:,5)=X(:,2).^2;
    out(:,6)=X(:,1).*X(:,2);
    
end

% Function 9
function out = linear_combiner( X ,  theta )
%     out =               randn(size(X,1),1);
    % DELETE ABOVE LINE AND THIS COMMENT, AND PLACE YOUR CODE INSTEAD
    out=X*theta;
end

% Function 10
function out = mse_loss( t_hat ,  t )
%     out =               rand(1);
    % DELETE ABOVE LINE AND THIS COMMENT, AND PLACE YOUR CODE INSTEAD
    n=length(t);
    out=1/n*sum((t_hat-t).^2);
end

% Function 11
function discussionB()
    disp('discussion B:');
    disp('<<ADD DISCUSSION B HERE>>');
    disp('The detection loss of t3 and t4 are almost same,and the mean square error loss of t4 is sightly lower than t4');
    disp('So,the higher complexity is sightly uesful')
    
end

% Function 12
function out = X5(X)
%     out =               rand([size(X,1),size(X,2)+1]);
    % DELETE ABOVE LINE AND THIS COMMENT, AND PLACE YOUR CODE INSTEAD
    out=ones(size(X,1),size(X,2)+1);
    out(:,2:end)=X;
    
end

% Function 13
function v_loss_LS = loss_vs_training_size( X_tr, t_tr, X_te, t_te, v_ratio_tr )
%     v_loss_LS =                 rand(size(v_ratio_tr));
    % DELETE ABOVE LINE AND THIS COMMENT, AND PLACE YOUR CODE INSTEAD
    for i=1:length(v_ratio_tr)
        n=round(v_ratio_tr(i)*size(X_tr,2));
        out1=ones(size(X_tr,1),n+1);
        out1(:,2:end)=X_tr(:,1:n);
        
        out2=ones(size(X_te,1),n+1);
        out2(:,2:end)=X_te(:,1:n);
        
        theta=LSsolver(out1,t_tr);
        t_hat_te=linear_combiner(out2,theta);
        v_loss_LS(i)=detection_error_loss((t_hat_te>0.5),t_te);
    end
    
    
    
end