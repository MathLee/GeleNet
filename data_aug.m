clear;
clc;
close all;

imPath = '/home/lgy/桌面/ORSI_SOD/dataset/EORSSD/train/image/';
GtPath = '/home/lgy/桌面/ORSI_SOD/dataset/EORSSD/train/GT/';

images = dir([GtPath '*.png']);
imagesNum = length(images);

for i = 1 : imagesNum
    im_name = images(i).name(1:end-4);

    gt = imread(fullfile(GtPath, [im_name '.png']));
    im = imread(fullfile(imPath, [im_name '.jpg']));
      
    im_1 = imrotate(im,90); 
    gt_1 = imrotate(gt,90); 
    imwrite(im_1, fullfile(imPath, [im_name '_90.jpg']));
    imwrite(gt_1, fullfile(GtPath, [im_name '_90.png']));

    im_2 = imrotate(im, 180); 
    gt_2 = imrotate(gt, 180); 
    imwrite(im_2, fullfile(imPath, [im_name '_180.jpg']));
    imwrite(gt_2, fullfile(GtPath, [im_name '_180.png']));

    im_3 = imrotate(im, 270); 
    gt_3 = imrotate(gt, 270); 
    imwrite(im_3, fullfile(imPath, [im_name '_270.jpg']));
    imwrite(gt_3, fullfile(GtPath, [im_name '_270.png']));
    

    fl_im = fliplr(im);
    fl_gt = fliplr(gt);
    imwrite(fl_im, fullfile(imPath, [im_name '_fl.jpg']));
    imwrite(fl_gt, fullfile(GtPath, [im_name '_fl.png'])); 
    
    im_1 = imrotate(fl_im,90); 
    gt_1 = imrotate(fl_gt,90);
    imwrite(im_1, fullfile(imPath, [im_name '_fl90.jpg']));
    imwrite(gt_1, fullfile(GtPath, [im_name '_fl90.png']));
    
    im_2 = imrotate(fl_im, 180); 
    gt_2 = imrotate(fl_gt, 180); 
    imwrite(im_2, fullfile(imPath, [im_name '_fl180.jpg']));
    imwrite(gt_2, fullfile(GtPath, [im_name '_fl180.png']));

    im_3 = imrotate(fl_im, 270); 
    gt_3 = imrotate(fl_gt, 270); 
    imwrite(im_3, fullfile(imPath, [im_name '_fl270.jpg']));
    imwrite(gt_3, fullfile(GtPath, [im_name '_fl270.png']));

end
