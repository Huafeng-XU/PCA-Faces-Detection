clear; 
NumOfSamples = 20; 
%you must change the str_Path to your path
%str_Path = 'F:\PRLab\Lab1\Training\';
str_Path = 'Training\';

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

%show Mean Face
ImageHeight = 72;
ImageWidth = 64;

%show the MeanFace
Display_MeanFace = reshape(MeanFace,[ImageHeight ImageWidth]);
figure(21),imagesc(Display_MeanFace),colorbar, colormap(gray),title('Mean Face');

%show the demeaned faces
for i = 1: NumOfSamples     
    DemeanFace(:,i) = TrainingImage(:,i)- MeanFace;
end 


for i = 1:NumOfSamples
    Display = DemeanFace(:,i);
    Display = reshape(Display,[ImageHeight ImageWidth]);
    figure(i),imagesc(Display),colorbar, colormap(gray),title('Demeaned Face');
end 

%covariance matrix CovFace1
%find the runtime
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

for i = 1:NumOfSamples
    
N = size(find(order>0),2);
for j = 1:N
    coef(i,j) = DemeanFace(:,i)' * EV(:,order(j));
end

%Reconstruct the training image using M = 4, 8...
M = 8;
ReconstImage(:,i) = zeros(4608,1);
for j = 1:M
    ReconstImage(:,i) = ReconstImage(:,i) + coef(i,j) * EV(:,order(j));
end
ReconstImage(:,i) = MeanFace + ReconstImage(:,i);

%compute the corresponding differences
Difference = TrainingImage(:,i) - ReconstImage(:,i);
SSE(i) = sum(sum(Difference.*Difference));

%show the reconstructed images
Display_Re = ReconstImage(:,i);
Display_Re = reshape(Display_Re,[ImageHeight ImageWidth]);
figure(i+40),imagesc(Display_Re),colorbar, colormap(gray),title('ReconstImage Face');

end

%Face Recognition using Eigenfaces
%For next step,make sure run the TestingImage.m first
%load TCoeff.mat and tranform struct to matrix

% TCoeff = cell2mat(struct2cell(load('TCoeff.mat')));
% 
% for i = 1:2
%     for j = 1:20
%         TT_Difference = TrainingImage(:,j)' - TCoeff(i,:); 
%         TT_SSE(i,j) = sqrt(sum(TT_Difference.*TT_Difference));
%     end
% end
% 
% [UP_TT_SSE,index_min] = sort(TT_SSE,2);



 