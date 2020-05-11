%% Train Multi-Class SVM for Ground Classification - dirt, grass & leaf

% Notes from meeting: Have 9x9 image, but also take a 27x27 image and
% downsample it 3x, 81x81 image downsampled 9x, etc. Use the grayscale
% image of the height/topography information appended to feature vector.
% For stump detection, use R and G channel and then grayscale of height as
% the third channel. ALSO DETERMINE WAY TO MEASURE ACCURACY AND CONFUSION
% MATRIX SO THAT I CAN COMPARE DIFFERENT METHODS AND PROVE THAT ACCURACY IS
% INCREASING = PROVIDES QUANTITATIVE RESULTS WHICH CAN BE DISCUSSED.
% trainNum + testNum = iterations*3
% iterations = 300;
% trainNum = 720;
% testNum = 180;

% 78.5% Accuracy for GroundImages folders
% iterations = 400;
% trainNum = 1000;
% testNum = 200;
clear all;
clc;
iterations = 1250;
trainNum = 3000;
testNum = 750;
% iterations = 650;%1950
% trainNum = 1560;
% testNum = 390;


% 27 = median, 243 = raw rgb values, 54 = median+skew, 81 =
% median+skew+skewof27, 108 = med+skew+27med+27skew, 135 =
% med+skew+27med+27skew+81skew, 162 = 9MedSkew27MedSkew81MedSkew
%featuresPerimage = 162;
featuresPerimage = 162;

ground_features = zeros(iterations*3,featuresPerimage);
ground_labels = strings(iterations*3,1);
count = 1;

dirtNames = dir(fullfile('ground_dirtFINAL/9Dirt','*.png'));%'GroundImages_Dirt_New','*.png'));
grassNames = dir(fullfile('ground_grassFINAL/9Grass','*.png'));%'GroundImages_Grass_New_copy','*.png'));
leafNames = dir(fullfile('ground_leafFINAL/9Leaf','*.png'));%GroundImages_Leaf_New_copy','*.png'));
grassCount = 1;
%Grass to skip:46, 47,57-60,83-88,90-94,

for i = 1:iterations 
    
    % Dirt images
    if i<971
        image_9dirt = imread(sprintf('ground_dirtFINAL/9Dirt/%s',string(dirtNames(i).name)));
        dirtString = split(dirtNames(i).name,'.');
        dirtNumber = split(dirtString{1},'t');
    else
        image_9dirt = imread(sprintf('ground_dirtFINAL/9Dirt/%s',string(dirtNames(i-970).name)));
        dirtString = split(dirtNames(i-970).name,'.');
        dirtNumber = split(dirtString{1},'t');
    end    
    image_27dirt = imread(sprintf('ground_dirtFINAL/27Dirt/27dirt%s.png',dirtNumber{end}));
    image_81dirt = imread(sprintf('ground_dirtFINAL/81Dirt/81dirt%s.png',dirtNumber{end}));
    image_27dirtdown = image_27dirt(1:3:end,1:3:end,1:3);
    image_81dirtdown = image_81dirt(1:9:end,1:9:end,1:3);
    
    % Grass images
    if i <652  
        image_9grass = imread(sprintf('ground_grassFINAL/9Grass/%s',string(grassNames(i).name)));
        grassString = split(grassNames(i).name,'.');
        grassNumber = split(grassString{1},'s');
    else
        image_9grass = imread(sprintf('ground_grassFINAL/9Grass/%s',string(grassNames(i-651).name)));
        grassString = split(grassNames(i-651).name,'.');
        grassNumber = split(grassString{1},'s');
    end
    image_27grass = imread(sprintf('ground_grassFINAL/27Grass/27grass%s.png',grassNumber{end}));
    image_27grassdown = image_27grass(1:3:end,1:3:end,1:3);
    image_81grass = imread(sprintf('ground_grassFINAL/81Grass/81grass%s.png',grassNumber{end}));
    image_81grassdown = image_81grass(1:9:end,1:9:end,1:3);
    
    % Leaf images
    if i <752
        image_9leaf = imread(sprintf('ground_leafFINAL/9Leaf/%s',string(leafNames(i).name)));
        leafString = split(leafNames(i).name,'.');
        leafNumber = split(leafString{1},'f');
    else
        image_9leaf = imread(sprintf('ground_leafFINAL/9Leaf/%s',string(leafNames(i-751).name)));
        leafString = split(leafNames(i-751).name,'.');
        leafNumber = split(leafString{1},'f'); 
    end
    image_27leaf = imread(sprintf('ground_leafFINAL/27Leaf/27leaf%s.png',leafNumber{end}));
    image_27leafdown = image_27leaf(1:3:end,1:3:end,1:3);
    image_81leaf = imread(sprintf('ground_leafFINAL/81Leaf/81leaf%s.png',leafNumber{end}));
    image_81leafdown = image_81leaf(1:9:end,1:9:end,1:3);
   
    
%     image_9dirt = imread(sprintf('GroundImages_Dirt_New/%s',string(dirtNames(i).name)));
%     image_9leaf = imread(sprintf('GroundImages_Leaf_New/%s',string(leafNames(i).name)));
%     if i < 170
%         image_9grass = imread(sprintf('GroundImages_Grass_New/%s',string(grassNames(i).name)));
%     else
%         randInt = randi(169,1);
%         %randInt = grassCount;
%         image_9grass = imread(sprintf('GroundImages_Grass_new/%s',string(grassNames(randInt).name)));
%         grassCount = grassCount + 1;
%         %image_grass = imrotate(image_grass,90);
%     end
    
    % Features consist of median of each column
    dirtMed = reshape(double(median(image_9dirt)),1,27);
    grassMed = reshape(double(median(image_9grass)),1,27);
    leafMed = reshape(double(median(image_9leaf)),1,27);
    
    dirtSkew = reshape(skewness(double(image_9dirt)),1,27);
    grassSkew = reshape(skewness(double(image_9grass)),1,27);
    leafSkew = reshape(skewness(double(image_9leaf)),1,27);
    
    dirt27Med = reshape(double(median(image_27dirtdown)),1,27);
    grass27Med = reshape(double(median(image_27grassdown)),1,27);
    leaf27Med = reshape(double(median(image_27leafdown)),1,27);
    
    dirt27Skew = reshape(skewness(double(image_27dirtdown)),1,27);
    grass27Skew = reshape(skewness(double(image_27grassdown)),1,27);
    leaf27Skew = reshape(skewness(double(image_27leafdown)),1,27);
    
    dirt81Med = reshape(double(median(image_81dirtdown)),1,27);
    grass81Med = reshape(double(median(image_81grassdown)),1,27);
    leaf81Med = reshape(double(median(image_81leafdown)),1,27);
    
    dirt81Skew = reshape(skewness(double(image_81dirtdown)),1,27);
    grass81Skew = reshape(skewness(double(image_81grassdown)),1,27);
    leaf81Skew = reshape(skewness(double(image_81leafdown)),1,27);  
    
%     dirtKurt = reshape(kurtosis(double(image_dirt)),1,27);
%     grassKurt = reshape(kurtosis(double(image_grass)),1,27);
%     leafKurt = reshape(kurtosis(double(image_leaf)),1,27);
% Features consist of raw rgb values
%     dirtMed = reshape(double(image_dirt),1,243);
%     grassMed = reshape(double(image_grass),1,243);
%     leafMed = reshape(double(image_leaf),1,243);
    
    dirtFeat = [dirtMed, dirtSkew, dirt27Med,dirt27Skew,dirt81Med,dirt81Skew];%, dirtKurt];
    grassFeat = [grassMed, grassSkew,grass27Med,grass27Skew,grass81Med,grass81Skew];%, grassKurt];
    leafFeat = [leafMed, leafSkew,leaf27Med,leaf27Skew,leaf81Med,leaf81Skew];%, leafKurt];
    
    ground_features(count,:) = dirtFeat;
    ground_labels(count) = ["dirt"];
    ground_features(count+1,:) = grassFeat;
    ground_labels(count+1) = ["grass"];
    ground_features(count+2,:) = leafFeat;
    ground_labels(count+2) = ["leaf"];
    
    count = count+3;
    
 
end

combined_features = [ground_features, ground_labels];
shuffled_combined = combined_features(randperm(size(combined_features,1)),:);

shuffled_features = double(shuffled_combined(:,1:end-1));
shuffled_labels = cellstr(shuffled_combined(:,end));


mdl = fitcecoc(shuffled_features(1:trainNum,:),shuffled_labels(1:trainNum),'FitPosterior',true,'Verbose',2);
disp('Model training is complete.');

%% Loss function
predicted_labels = cell(testNum,1);
predicted_data = zeros(3,testNum);
test_features = shuffled_features(trainNum+1:trainNum+testNum,:);
test_labels = shuffled_labels(trainNum+1:trainNum+testNum);
test_data = zeros(3,testNum);
disp('Populating the confusion matrix...');
for i = 1:testNum
    if strcmpi(test_labels{i},'dirt')
        test_data(1,i) = 1;
    elseif strcmpi(test_labels{i},'grass')
        test_data(2,i) = 1;
    elseif strcmpi(test_labels{i},'leaf')
        test_data(3,i) = 1;
    else 
        print('Error');
        break;
    end
    [predicted_labels{i},~,~,Posterior] = predict(mdl,test_features(i,:));
    predicted_data(:,i) = Posterior';
end

plotconfusion(test_data,predicted_data);
%L = loss(mdl, shuffled_features(601:690,:),shuffled_labels(601:690));

%% 
input9 = imread('ground_grassFINAL/9Grass/9grass1.png');
input27 = imread('ground_grassFINAL/27Grass/27grass1.png');
input27 = input27(1:3:end,1:3:end,1:3);
input_features = reshape([double(median(input9)), skewness(double(input9)),skewness(double(input27))],1,81);

[label,NegLoss,PBScore,Posterior] = predict(mdl,input_features);

disp(Posterior);
%%
%input2 = imread('mosaics/mosaic_7_0_section.tif');
input2 = imread('stumps600/Train/stump117.png');
[rows,cols,chan] = size(input2);
ground_classification = zeros(rows,cols);

for r = 1:9:rows-mod(rows,9)
    disp(r);
    for c = 1:9:cols-mod(cols,9)
        window = input2(r:r+8,c:c+8,1:3);
        windowMed = reshape(double(median(window)),1,27);
        windowSkew = reshape(skewness(double(window)),1,27);
        window_features = [windowMed, windowSkew];
        [label] = predict(mdl,window_features);
        %if max(Posterior)>0.7
            if strcmpi(label{1},'dirt')
                ground_classification(r:r+8,c:c+8) = 1;
            elseif strcmpi(label{1},'grass')
                ground_classification(r:r+8,c:c+8) = 2;
            elseif strcmpi(label{1},'leaf')
                ground_classification(r:r+8,c:c+8) = 3;
            end
        %else
        %    ground_classification(r:r+8,c:c+8) = 0;
        %end
    end
end
figure(1);
imshow(input2(:,:,1:3));
figure(2);
imagesc(ground_classification);
colorbar;
se = strel('disk',6);
ground_opened = imopen(mat2gray(ground_classification),se);
ground_opened = ground_opened*3;
figure(3);
imagesc(ground_opened);
title("Dirt = 1, Grass = 2, Leaf = 3");
figure(4);
imshow(input2(:,:,1:3));
hold on;
imLabel = imagesc(ground_opened);
imLabel.AlphaData = 0.4;


%%
input3 = imread('topo_mosaics/mosaic_0_3.tif');
disp(max(max(input3)));
new_input3 = input3/919.4995;%max(max(input3));
other_input4 = input3/909.991;%max(max(input3));
maxVal = max(max(new_input3));%919.47
maxVal2 = max(max(other_input4));
minVal = min(min(new_input3(new_input3>0)));%906.98
minVal2 = min(min(other_input4(other_input4>0)));
realinput3 = imadjust(new_input3,[minVal 1]);
realinput4 = imadjust(other_input4,[minVal2,1]);
imshow(realinput3);
figure(2);
imshow(realinput4);
% imagesc(input3);
% cmap = jet(915);
% cmap(1:911,:) = zeros(911,3);
% colormap(cmap);
%colorbar