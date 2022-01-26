clear all;close all;%clc;
addpath(genpath('ompbox10'),genpath('ksvdbox13'));
imdir='./images';
imgname='barbara.png';
% imgname='lena.png';
sigma=15;
blocksize=8;
trainnum=62001;
sizeDimg=[5 5]*15;
gain=1.1;
iternum=10;


rng('default');
im=double(imread(fullfile(imdir,imgname)));
imnoise=im+sigma*randn(size(im));
imnoise=double(uint8(imnoise));
alldata=im2col(imnoise,blocksize*[1,1],'sliding');
meandata=mean(alldata,1);
alldata=alldata-repmat(meandata,blocksize*blocksize,1);

if 1
    %%%% create training data %%%
    p=2;
    ids = cell(p,1);
    if (p==1)
        ids{1} = reggrid(length(imnoise)-ones(1,p)*blocksize+1, trainnum, 'eqdist');
    else
        [ids{:}] = reggrid(size(imnoise)-ones(1,p)*blocksize+1, trainnum, 'eqdist');
    end
    data = sampgrid(imnoise,blocksize*[1 1],ids{:});
    
    % remove dc in blocks to conserve memory %
    batchsize = 2000;
    for i = 1:batchsize:size(data,2)
        blockids = i : min(i+batchsize-1,size(data,2));
        data(:,blockids) = remove_dc(data(:,blockids),'columns');
    end
else
    data=im2col(imnoise,[blocksize,blocksize],'sliding');
    data=data-repmat(mean(data,1),blocksize*blocksize,1);
end

options.initDimg=randn(sizeDimg);
options.codemode='error';options.Edata=blocksize*sigma*gain;
options.codemode='sparsity';options.Tdata=2;
options.miu0=2e-5;
options.iternum=iternum;
options.memusage='high';
options.batchsize=sqrt(trainnum);
[Dimg,D]=signature_dictionary_learning(data,options);
options.dict=D;
options.x=imnoise;
options.blocksize=blocksize;
options.sigma=sigma;

options.codemode='error';options.Edata=blocksize*sigma*gain;
G = D'*D;
ompparams = {'checkdict','off'};
% Gamma = omp2(D'*alldata,XtX,G,options.Edata,ompparams{:});
    % remove dc in blocks to conserve memory %
    batchsize = 4000;
    for i = 1:batchsize:size(alldata,2)
        blockids = i : min(i+batchsize-1,size(alldata,2));        
        XtX = colnorms_squared(alldata(:,blockids));
        Gamma(:,blockids) = omp2(D'*alldata(:,blockids),XtX,G,options.Edata,ompparams{:});
    end
recdata=D*Gamma+repmat(meandata,blocksize*blocksize,1);
imout=zeros(size(im));
imout=double(uint8(imout));
weight=zeros(size(im));
[nr,nc]=size(im);
for c=1:nc-blocksize+1
    for r=1:nr-blocksize+1
        imout(r:r+blocksize-1,c:c+blocksize-1)=imout(r:r+blocksize-1,c:c+blocksize-1)+reshape(recdata(:,(c-1)*(nr-blocksize+1)+r),blocksize,blocksize);
        weight(r:r+blocksize-1,c:c+blocksize-1)=weight(r:r+blocksize-1,c:c+blocksize-1)+1;
    end
end
imout=imout./weight;
psnrin=psnr(im,imnoise,255)
psnrout=psnr(im,imout,255)
% psnrout=psnr(im(1:nr-blocksize+1,1:nc-blocksize+1),imout(1:nr-blocksize+1,1:nc-blocksize+1),255)

figure;imshow([im,imnoise,imout],[]);
figure;imshow(Dimg,[]);
return;
%==================================================================================%
for ct_thresh=0:0.5:2.5
idx=std(data)<ct_thresh*sigma;

% idx=kmeans(data',2);
% data=data(:,idx==2);

options.initDimg=randn(sizeDimg);
options.codemode='error';options.Edata=blocksize*sigma*gain;
options.codemode='sparsity';options.Tdata=2;
options.miu0=2e-6;
[Dimg1,D1]=signature_dictionary_learning(data(:,idx),options);



options.initDimg=randn(sizeDimg);
options.codemode='error';options.Edata=blocksize*sigma*gain;
options.codemode='sparsity';options.Tdata=2;
options.miu0=2e-6;
[Dimg2,D2]=signature_dictionary_learning(data(:,~idx),options);

figure;imshow([Dimg1,Dimg2],[]);drawnow;

recdata=zeros(size(alldata));

allcIdx=std(alldata)<ct_thresh*sigma;
cdata=alldata(:,allcIdx);
tdata=alldata(:,~allcIdx);
ompparams = {'checkdict','off'};
G = D1'*D1;XtX = colnorms_squared(cdata);
Gamma1 = omp2(D1'*cdata,XtX,G,options.Edata,ompparams{:});
recdata(:,allcIdx)=D1*Gamma1;

G = D2'*D2;XtX = colnorms_squared(tdata);
Gamma2 = omp2(D2'*tdata,XtX,G,options.Edata,ompparams{:});
recdata(:,~allcIdx)=D2*Gamma2;

imout=col2imstep(recdata,size(im),blocksize*[1,1]);
psnrout=psnr(im,imout);
fprintf('ct_thresh=%.2f, psnrout=%.2f\n',ct_thresh,psnrout);
end