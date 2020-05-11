%% Messing around Thesis 

%image = imread('mosaics/mosaic_3_3.tif');
%image_new = image(1:2000,4000:7000,1:3);
%ROI_0_3 = load('MatlabStumpROIs/stumps_nonstumps_0_3.mat');
 %ROI_1_3 = load('MatlabStumpROIs/stumps_nonstumps_1_3.mat');
 %ROI_2_3 = load('MatlabStumpROIs/stumps_nonstumps_2_3.mat');
 %ROI_3_3 = load('MatlabStumpROIs/stumps_nonstumps_3_3.mat');
% ROI_4_2 = load('MatlabStumpROIs/stumps_nonstumps_4_2.mat');
 %ROI_5_2 = load('MatlabStumpROIs/stumps_nonstumps_5_2.mat');
% ROI_6_1 = load('MatlabStumpROIs/stumps_nonstumps_6_1.mat');
% ROI_7_0 = load('MatlabStumpROIs/stumps_nonstumps_7_0.mat');
 ROI_8_0 = load('MatlabStumpROIs/stumps_nonstumps_8_0.mat');
% 
%image_0_3 = imread('mosaics/mosaic_0_3.tif');
% image_1_3 = imread('mosaics/mosaic_1_3.tif');
 %image_2_3 = imread('mosaics/mosaic_2_3.tif');
 %image_3_3 = imread('mosaics/mosaic_3_3.tif');
 %image_4_2 = imread('mosaics/mosaic_4_2.tif');
% image_5_2 = imread('mosaics/mosaic_5_2.tif');
 image_6_1 = imread('mosaics/mosaic_6_1.tif');
% image_7_0 = imread('mosaics/mosaic_7_0.tif');
 image_8_0 = imread('mosaics/mosaic_8_0.tif');

%image_0_3_topo = imread('topo_mosaics/mosaic_0_3.tif');
% image_1_3_topo = imread('topo_mosaics/mosaic_1_3.tif');
% image_2_3_topo = imread('topo_mosaics/mosaic_2_3.tif');
 %image_3_3_topo = imread('topo_mosaics/mosaic_3_3.tif');
% image_4_2_topo = imread('topo_mosaics/mosaic_4_2.tif');
% image_5_2_topo = imread('topo_mosaics/mosaic_5_2.tif');
 image_6_1_topo = imread('topo_mosaics/mosaic_6_1.tif');
% image_7_0_topo = imread('topo_mosaics/mosaic_7_0.tif');
 image_8_0_topo = imread('topo_mosaics/mosaic_8_0.tif');
 
%% Extract Stump Images

for i = 1:length(ROI_8_0.nonStumps_8_0.Stump{1,1}(:,1))
    %figure(i);
    stumpCorner = ROI_8_0.nonStumps_8_0.Stump{1,1}(i,1:2);
    
    % 0_3: Max = 909.991, Min = 907.449, Min/Max = 0.9972
    % 1_3: Max = 912.481, Min = 909.599, Min/Max = 0.99684
    % 2_3: Max = 915.910, Min = 913.602, Min/Max = 0.99748
    % 3_3: Max = 915.693, Min = 912.228, Min/Max = 0.9962
    % 4_2: Max = 919.126, Min = 915.253, Min/Max = 0.9958
    % 5_2: Max = 918.562, Min = 913.775, Min/Max = 0.9948
    % 6_1: Max = 921.196, Min = 915.195, Min/Max = 0.9935
    % 7_0: Max = 920.708, Min = 916.619, Min/Max = 0.99556
    % 8_0: Max = 923.554, Min = 919.567, Min/Max = 0.99568
    
    image = image_6_1_topo;
    newInput = image/921.196; % Image/Max of mosaic
%     maxVal = max(max(newInput));
%     minVal = min(min(newInput(newInput>0)));
    finalInput = imadjust(newInput,[0.9935,1]); %[Min/Max, 1]
    %imshow(finalInput);
    % For 0_3 - images 1-9
%      if i > 6 || i == 1
%         stumpCorner = stumpCorner - 30;
%      end
    
    % For 1_3 - images 10-27
%      if stumpCorner(1) > 30 && stumpCorner(2) > 30
%          stumpCorner = stumpCorner - 230;
%      end
%      if i == 12 || i == 13 || i == 14
%          stumpCorner(2) = stumpCorner(2) -100;
%          if i == 13
%              stumpCorner(2) = stumpCorner(2) - 230;
%          end
%      end
         

    % For 2_3 - images 28-42
%     if i == 11 || i == 12
%         stumpCorner(2) = stumpCorner(2) - 120;
%     end
%     if i == 15
%         stumpCorner(1) = stumpCorner(1) + 200;
%     end
    
    % For 3_3 - images 43-60
%     if i == 15 || i == 2
%         ;
%     else
%         stumpCorner = stumpCorner - 20;
%     end
%     if i == 1 || i ==6
%         stumpCorner(1) = stumpCorner(1) + 100;
%         stumpCorner(2) = stumpCorner(2) + 150;
%     end
%     if i == 8
%         stumpCorner(1) = stumpCorner(1) - 180;
%     end
    
    
     % For 4_2 - images 61-75
%     if  i == 2 
%         ;
%     else
%         if i == 12
%             stumpCorner(1) = stumpCorner(1) + 15;
%         end
%         if i == 6 || i == 9
%             stumpCorner = stumpCorner + 10;
%             stumpCorner(1) = stumpCorner(1) + 10;
%         end
%         stumpCorner = stumpCorner + 30;
%     end
    
    % For 5_2 - images 76-83
%     if i == 5
%         stumpCorner(2) = stumpCorner(2)+273;
%     elseif i == 7
%         stumpCorner = stumpCorner + 20;
%     end

    % For 6_1 - images 84 - 100
%     if i == 9 || i == 5
%         ;
%     else
%         stumpCorner = stumpCorner - 20;
%     end
% %     
        
    % For 7_0 - images 101 - 110
%     if i == 10
%         stumpCorner(2) = stumpCorner(2) + 291;
%     end
    
    % For 8_0 - images 111 - 122
    if i == 12
        stumpCorner(1) = stumpCorner(1) + 100;
    end
    
% for 0_3, -200:+399,-200:+399
% for 1_3, 0:+599,0:+599
% for 2_3, -300:+299, -200:+399
% for 3_3, -300:+299, -200:+399
% for 4_2, -300:+299,-200:+399
% for 5_2, -300:+299,-200:+399
% for 6_1, -300:+299,-300:+299
% for 7_0, -300:+299,-300:+299
% for 8_0, -300:+299,-300:+299
    stumpImage = image_8_0(stumpCorner(2)-300:stumpCorner(2)+299,stumpCorner(1)-300:stumpCorner(1)+299);%,1:3);
    stumpImage2 = finalInput(stumpCorner(2)-300:stumpCorner(2)+299,stumpCorner(1)-300:stumpCorner(1)+299);
    %stumpImage = stumpImage(1:3:221,1:3:221,1:3);
    figure(1);
    imshow(stumpImage);
    figure(2);
    imshow(stumpImage2);
    %filename = strcat('stumps/stump',string(i),'.png');
    imwrite(stumpImage2,sprintf('topo_600/topo%s.png',string(i+110)));
    disp(sprintf('topo%s.png',string(i+110)));
    
    
end


%% Extract Non-Stump Images

% 0, 1, 2 = NonStump & 3 onwards = nonStump
% 0_3 = 1-8, 1_3 = 9-20 ,2_3 = 21-32, 3_3 = 33-43 , 4_2 = 44-56 , 5_2 =
% % 57-65, 6_1 = 66-80, 7_0 = 81-90 ,8_0 = 91-100
% for j = 1:length(ROI_8_0.nonStumps_8_0.nonStump{1,1}(:,1))
%     %figure(j);
%     nonstumpCorner = ROI_8_0.nonStumps_8_0.nonStump{1,1}(j,1:2);
%     
%     nonstumpImage = image_8_0(nonstumpCorner(2):nonstumpCorner(2)+220,nonstumpCorner(1):nonstumpCorner(1)+220,1:3);
%     %nonstumpImage = nonstumpImage(1:3:221,1:3:221,1:3);
%     %imshow(nonstumpImage);
%     imwrite(nonstumpImage,sprintf('nonstumps/nonstump%s.png',string(j+90)));
%     
% end





