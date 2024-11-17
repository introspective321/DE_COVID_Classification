cd '.\Front\'

load('termalImages.mat')
figure(1)
subplot(2,3,1)
imshow(averageSignal,[])
subplot(2,3,4)
imshow(varSignal,[])
subplot(2,3,2)
imshow(averageXMotion,[])
subplot(2,3,5)
imshow(varXMotion,[])
subplot(2,3,3)
imshow(averageYMotion,[])
subplot(2,3,6)
imshow(varYMotion,[])
figure(2)
subplot(1,3,1)
imshow(firstTermalImage,[23,33])
subplot(1,3,2)
imshow(integratedTemp,[23,33])
subplot(1,3,3)
imshow(termalImage,[23,33])

skinTempthr = 22;

atResTermal = imresize(termalImage,size(averageSignal));
skinROI = atResTermal > skinTempthr;
figure(3);
subplot(1,2,1)
imshow(atResTermal,[23,33])
subplot(1,2,2)
imshow(skinROI,[])

imwrite(uint8(5*atResTermal),'TermalTemplate.png');
imshow(atResTermal,[]);
cd '..\'

