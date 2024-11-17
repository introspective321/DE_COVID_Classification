%%%%%%%%%%%%%%%%%%%%%%%%%%Video Preprocessing Parameters

outSize = [384 288];
insideROI_i = 15:384;
insideROI_j = 1:(384-15);
vout.Quality = 100;
%PSF = power(conv2(fspecial('disk',4.0),fspecial('gaussian',9,0.35)),0.1);
PSF = fspecial('disk',9.5);
PSF = power(conv2(PSF,fspecial('gaussian',13,4.0)),2.00);

PSF = PSF/sum(sum(PSF));
wienerNoise = 0.025;
grayScaleThr = 100;

%%%%%%%%%%%%%%%%%%%%%% Generate Video

open(vout)

figure(1)
subplot(1,3,1)
mesh(PSF)
nframes = v.NumFrames;

skipFactor = round(v.NumFrames/v.Duration/5);
if (skipFactor < 1)
    skipFactor = 1;
end
nf = 0;
while hasFrame(v)
    frame = readFrame(v);
    if ((nframes < 100) || (mod(nf,skipFactor) == 0))
        nf
        h=figure(1);
        grayscale_1 = double(frame(:,:,1));
        if (flipping)
            grayscale_1 = flip(grayscale_1,2)';
        end
        grayScaleThr = 255*graythresh(grayscale_1)
        fl0 = grayscale_1 < 0.9*grayScaleThr;
        grayScaleThr = 0.5*(median(reshape(grayscale_1(fl0),1,[])) + grayScaleThr);

        fg0 = (grayscale_1 >= grayScaleThr) & (grayScaleThr < 255);
        subplot(1,4,2)
        imshow(fg0,[])
        
        fl0 = grayscale_1 < 0.9*grayScaleThr;
        
        bodytemp1 = 0.5*median(reshape(grayscale_1(fg0),1,[])) + 0.5*mean(reshape(grayscale_1(fg0),1,[])) 
        backtemp1 = 0.85*median(reshape(grayscale_1(fl0),1,[])) + 0.25*mean(reshape(grayscale_1(fl0),1,[]))
        if (nf == 0)
            subplot(1,4,1)
            imshow(grayscale_1,[50,220])
            startbacklevel = backtemp1
            startbodylevel = bodytemp1
        end
        subplot(1,4,3)
        imshow(grayscale_1,[50,220])
        grayscale_1 = (grayscale_1 - backtemp1)*(startbodylevel-startbacklevel)/(bodytemp1-backtemp1)+startbacklevel;
        grayscale_1(grayscale_1 < 0) = 0; 
        bodytemp1 = 0.5*(median(reshape(grayscale_1(fg0),1,[])) + mean(reshape(grayscale_1(fg0),1,[])))
        backtemp1 = 0.85*median(reshape(grayscale_1(fl0),1,[])) + 0.25*mean(reshape(grayscale_1(fl0),1,[]))
        if (Debluring)        
            grayscale_1 = grayscale_1/255.0;
            J = deconvwnr(grayscale_1,PSF,wienerNoise);
            meantemp2 = 0.5*(median(reshape(J(fg0),1,[])) + mean(reshape(J(fg0),1,[])));
            grayscale_1 = J*(bodytemp1/meantemp2);
            grayscale_1 = medfilt2(grayscale_1);
        end
        if (flipping)
            grayscale_1 = imresize(grayscale_1 ,outSize,'box');
            grayscale_1 = grayscale_1(insideROI_i,:);
        else
            grayscale_1 = imresize(grayscale_1 ,flip(outSize),'box');
            grayscale_1 = grayscale_1(:,insideROI_j);
        end

        grayscale_1(grayscale_1 < 0) = 0;
        grayscale_1(grayscale_1 >= 255) = 255;
        J8B = uint8(round(grayscale_1));
        subplot(1,4,4)
        imshow(J8B,[50,220])
        writeVideo(vout,J8B);
    end
    nf = nf + 1;
end
close(vout)


