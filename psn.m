data = "D:\SungRung\mnist_SEG(Noise)\images\0\resultDog\combined";

imds = imageDatastore(data,'IncludeSubfolders',true,'LabelSource','foldernames');
% unet = readimage(imds,1);
% 
combined = readimage(imds,2);
% 
% % 
noise  = readimage(imds,5);
% 
orignal = readimage(imds,6);
% % % 
% unet = readimage(imds,5);
% 


net = denoisingNetwork("DnCNN");
% 
dncnn= denoiseImage(noise, net);
% 
% subplot(1,5,1);
% imshow(unet)
% subplot(1,5,2);
% imshow(dncnn)
% subplot(1,5,3);
% imshow(combined)
% subplot(1,5,4);
% imshow(noise)
% subplot(1,5,5);
% imshow(dncnn)
% 
% pause;
combined = cast(combined, 'uint8');
combined = combined * 255;

subplot(1,2,1);
imshow(noise);
title("noise");
% 
% pause;
J = imadjust(noise,[0.4 0.6],[]);
% J = J + 0.2;
subplot(1,2,2);
imshow(J);
title("imadjust");
% pause;
% imwrite(dncnn, fullfile(data,"dncnn.png"));
% imshow(combined)
[peaksnr, snr] = psnr(orignal, noise);
% %   
fprintf('\n The Peak-SNR value is %0.4f', peaksnr);

