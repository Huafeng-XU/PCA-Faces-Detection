clear; 
NumOfSamples = 4; 
str_Path = 'F:\PRLab\Lab1\Testing\';

for i = 1: NumOfSamples     
    str_Load = strcat(str_Path, num2str(i), '.bmp');
    Image = imread(str_Load);    
    TrainingImage(:,i) = double(reshape(Image, [ ], 1));
end 
MeanFace = 0;
for i = 1: NumOfSamples    
    MeanFace = MeanFace+TrainingImage(:,i); 
end
MeanFace = MeanFace/NumOfSamples;

for i = 1: NumOfSamples     
    DemeanFace(:,i) = TrainingImage(:,i)- MeanFace;
end 

ImageHeight = 72;
ImageWidth = 64;
for i = 1:NumOfSamples
    Display = DemeanFace(:,i);
    Display = reshape(Display,[ImageHeight ImageWidth]);
    figure(i),imagesc(Display),colorbar, colormap(gray),title('Demeaned Face');
end 

%covariance matrix CovFace1
TimeBegin = clock;
CovFace1 = zeros(4608,4608);
CovFace1 = 1/NumOfSamples * (DemeanFace * DemeanFace');
TimeOver = clock;
runtime = etime(TimeOver,TimeBegin);
% TimeBegin = clock;
% CovFace1 = zeros(20,20);
% CovFace1 = 1/NumOfSamples * (DemeanFace' * DemeanFace);
% TimeOver = clock;
% runtime = etime(TimeOver,TimeBegin);

%computes the eigenvectors and eigenvalues
[EV,ED] = eig(CovFace1);

%Save the order in decscending order
[DescendED,order] = sort(max(ED),'descend');

%Display the eigenvectors according to the eigenvalues in descending order
DescendEV = EV(:,order);
%disp(DescendEV);

%choose i = 1
%i = 1;
for i = 1:4
N = size(find(order>0),2);
for j = 1:N
    coef(i,j) = DemeanFace(:,i)' * EV(:,order(j));
end

%Reconstruct the training image using M = 4, 8, ?
M = 4;
ReconstImage(:,i) = zeros(4608,1);
for j = 1:M
    ReconstImage(:,i) = ReconstImage(:,i) + coef(i,j) * EV(:,order(j));
end
ReconstImage(:,i) = MeanFace + ReconstImage(:,i);

%compute the corresponding differences
Difference = TrainingImage(:,i) - ReconstImage(:,i);
SSE(i) = sum(sum(Difference.*Difference));

%Display the reconstructed images
Display_Re = ReconstImage(:,i);
Display_Re = reshape(Display_Re,[ImageHeight ImageWidth]);
figure(i+4),imagesc(Display_Re),colorbar, colormap(gray),title('ReconstImage Face');

end
%save the coefficients of projecting  testing  images are stored in TCoeff
TCoeff(1:2,:) = ReconstImage(:,1:2)';
save('TCoeff.mat','TCoeff');