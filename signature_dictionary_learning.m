function [Dimg,D]=signature_dictionary_learning(data,options)
%signature_dictionary_learning 此处显示有关此函数的摘要
%   此处显示详细说明


global CODE_SPARSITY CODE_ERROR codemode
global MEM_LOW MEM_NORMAL MEM_HIGH memusage
global ompfunc ompparams 

CODE_SPARSITY = 1;
CODE_ERROR = 2;

MEM_LOW = 1;
MEM_NORMAL = 2;
MEM_HIGH = 3;

% coding mode %

if (isfield(options,'codemode'))
  switch lower(options.codemode)
    case 'sparsity'
      codemode = CODE_SPARSITY;
      thresh = options.Tdata;
    case 'error'
      codemode = CODE_ERROR;
      thresh = options.Edata;
    otherwise
      error('Invalid coding mode specified');
  end
elseif (isfield(options,'Tdata'))
  codemode = CODE_SPARSITY;
  thresh = options.Tdata;
elseif (isfield(options,'Edata'))
  codemode = CODE_ERROR;
  thresh = options.Edata;

else
  error('Data sparse-coding target not specified');
end


ompparams = {'checkdict','off'};
% max number of atoms %

if (codemode==CODE_ERROR && isfield(options,'maxatoms'))
  ompparams{end+1} = 'maxatoms';
  ompparams{end+1} = options.maxatoms;
end

% memory usage %
if (isfield(options,'memusage'))
  switch lower(options.memusage)
    case 'low'
      memusage = MEM_LOW;
    case 'normal'
      memusage = MEM_NORMAL;
    case 'high'
      memusage = MEM_HIGH;
    otherwise
      error('Invalid memory usage mode');
  end
else
  memusage = MEM_NORMAL;
end

% omp function %
if (codemode == CODE_SPARSITY)
  ompfunc = @omp;
else
  ompfunc = @omp2;
end

if isfield(options,'blocksize')
    blocksize=options.blocksize;
else
    blocksize=sqrt(size(data,1));
end

if isfield(options,'initDimg')
    Dimg=options.initDimg;
elseif isfield(options,'sizeDimg')
    Dimg=randn(options.sizeDimg);
end
Dimg=Dimg/norm(Dimg,'fro');
sizeDimg=size(Dimg);


if isfield(options,'iternum')
    iternum=options.iternum;
else
    iternum=10;
end
if isfield(options,'miu0')
    miu0=options.miu0;
else
    miu0=2e-7;
end
if isfield(options,'batchsize')
    batchsize=options.batchsize;
else
    batchsize=ceil(sqrt(size(data,2)));
end

% data norms %
XtX = []; %XtXg = [];
if (codemode==CODE_ERROR && memusage==MEM_HIGH)
  XtX = colnorms_squared(data);
%   if (testgen)
%     XtXg = colnorms_squared(testdata);
%   end
end

trainnum=size(data,2);
[D,normAtoms]=ISD2regularD(Dimg,blocksize);

CIndexMat=calcCindexMat(sizeDimg,blocksize);

err = zeros(1,iternum);
recerr = zeros(1,iternum);
for iter=1:iternum
    
    G = [];
    if (memusage >= MEM_NORMAL)
        G = D'*D;
    end
    
    tempIdx=randperm(trainnum);
    tempdata=data(:,tempIdx);
    XtX = colnorms_squared(tempdata);
    tempGamma=sparsecode(tempdata,D,XtX,G,thresh);

    ErrMat=D*tempGamma-tempdata;
    
    err(iter)=norm(ErrMat,'fro');
    % recerr(iter)=norm(ErrMat,'fro');
    % fprintf('before %2d, err=%6.2f, recerr=%6.2f\n',iter,err(iter),recerr(iter));
    % fprintf('before %2d, err=%6.2f\n',iter,err(iter));
    

    % calculate deltaDimg
    deltaDimg=zeros(sizeDimg);
    for i=1:batchsize:trainnum
        tIdx=i:min(i+batchsize,trainnum);
        tdata=tempdata(:,tIdx);
        tGamma=tempGamma(:,tIdx);
        for k=1:numel(tIdx)
            [row,~,v]=find(abs(tGamma(:,k))>eps);
            for j=1:numel(v)
                Cindex=CIndexMat(:,row(j));
                deltaDimg(Cindex)=deltaDimg(Cindex)+v(j)/max(eps,normAtoms(row(j)))*ErrMat(:,tIdx(k));
            end
        end        
    end
    % total variation
    % divDimg=imdivergence2(Dimg);Dimg=Dimg-miu0/iter*(deltaDimg)+0.001*divDimg;
    
    % update Dimg
    
    Dimg=Dimg-miu0/iter*deltaDimg;
    
    [D,normAtoms]=ISD2regularD(Dimg,blocksize);
    
    figure(1);
    subplot(121);imshow(mat2gray(Dimg));title(num2str(iter));drawnow;
    % subplot(122);imshow(mat2gray(divDimg));title(num2str(iter));drawnow;
end
 figure(1);subplot(122);plot(err);drawnow;
 
end

function [D,normAtoms]=ISD2regularD(Dimg,blocksize)
    DimgExt=[Dimg,Dimg(:,1:blocksize-1)];
    DimgExt=[DimgExt;DimgExt(1:blocksize-1,:)];
    D=im2col(DimgExt,[blocksize,blocksize],'sliding');
    D=D-repmat(mean(D,1),blocksize*blocksize,1);
    normAtoms=sqrt(sum(D.^2,1));
    idx=normAtoms~=0;
    D(:,idx)=D(:,idx)./repmat(normAtoms(:,idx),blocksize*blocksize,1);
end

function CindexMat=calcCindexMat(sizeDimg,blocksize)
    index=reshape(1:prod(sizeDimg),sizeDimg);
    
    indexExt=[index,index(:,1:blocksize-1)];
    indexExt=[indexExt;indexExt(1:blocksize-1,:)];
    
    CindexMat=im2col(indexExt,[blocksize,blocksize],'sliding');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             sparsecode               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function Gamma = sparsecode(data,D,XtX,G,thresh)

global CODE_SPARSITY codemode
global MEM_HIGH memusage
global ompfunc ompparams

if (memusage < MEM_HIGH)
  Gamma = ompfunc(D,data,G,thresh,ompparams{:});
  
else  % memusage is high
  
  if (codemode == CODE_SPARSITY)
    Gamma = ompfunc(D'*data,G,thresh,ompparams{:});
    
  else
    Gamma = ompfunc(D'*data,XtX,G,thresh,ompparams{:});
  end
  
end

end


function div=imdivergence(I)
    ep2=0.01;
    [ny,nx]=size(I);
    I_x = (I(:,[2:nx 1])-I(:,[nx 1:nx-1]))/2;
    I_y = (I([2:ny 1],:)-I([ny 1:ny-1],:))/2;
    I_xx = I(:,[2:nx 1])+I(:,[nx 1:nx-1])-2*I;
    I_yy = I([2:ny 1],:)+I([ny 1:ny-1],:)-2*I;

    Dp = I([2:ny 1],[2:nx 1])+I([ny 1:ny-1],[nx 1:nx-1]);

    Dm = I([ny 1:ny-1],[2:nx 1])+I([2:ny 1],[nx 1:nx-1]);

    I_xy = (Dp-Dm)/4;

   % compute flow

   Num = I_xx.*(ep2+I_y.^2)-2*I_x.*I_y.*I_xy+I_yy.*(ep2+I_x.^2);

   Den = (ep2+I_x.^2+I_y.^2).^(3/2);

   div = Num./Den;
end
function div=imdivergence2(I)
    [ny,nx]=size(I);
    Ix=0.5*(I(:,[2:nx,1])-I(:,[nx,1:nx-1]));
    Iy=0.5*(I([2:ny,1],:)-I([ny,1:ny-1],:));
    gradient=Ix.^2+Iy.^2+eps;
    
%     Ix_back=I-I(:,[nx,1:nx-1]);
%     Ix_forward=I(:,[2:nx,1])-I;
%     Iy_back=I-I([ny,1:ny-1],:);
%     Iy_forward=I([2:ny,1],:)-I;
    
    Ixx=(I(:,[2:nx,1])-I)-(I-I(:,[nx,1:nx-1]));
    Ixy=0.25*((I([2:ny,1],[2:nx,1])-I([2:ny,1],[nx,1:nx-1]))...
        -(I([ny,1:ny-1],[2:nx,1])-I([ny,1:ny-1],[nx,1:nx-1])));
    Iyy=(I([2:ny,1],:)-I)-(I-I([ny,1:ny-1],:));
    
    %%中心差分
    div=(Iy.^2.*Ixx-2*Ix.*Iy.*Ixy+Ix.^2.*Iyy)./gradient;
end