%this matlab code implements the MCFS model for infrared small target 
% detection.
%
% Reference:
% Method of Infrared Small Moving Target Detection Based on
%Coarse-to-Fine Structure in Complex Scenes
%
% Written by Yapeng Ma 
% 2023-5-30

clear all
addpath '.\functions'
%% 图像和结果文件
datafile_path='img\';%文件
b=['res\'];
%% 参数设置
num=100;
K=2;
m=2;
ss=[2,3];
D=zeros(256,256,num);
%% 数据处理
img_path_list = dir(fullfile(datafile_path,'*.bmp'));%获取该文件夹中所有bmp格式的图像
files_name =sort_nat({img_path_list.name});%重新排序
  for i = 1:num
    newname=fullfile(datafile_path, files_name{i});
    img1 = imread(newname);
    nn = ndims(img1);
    if nn==3
      img1= rgb2gray(img1);
    end
      D(:,:,i) = double(img1);
  end
[row, col, dm]=size(D);
litter=length(ss);
M_out=ones(row,col,litter);
%% 实验分析
for q=1:num
img=D(:,:,q);
% Smooth filtering 
K_G5=[-1,-1,-1,-1,-1;-1,0,0,0,-1;-1,0,16,0,-1;-1,0,0,0,-1;-1,-1,-1,-1,-1];
E1=conv2(img,K_G5,'same');
[lambda1, lambda2] = structure_tensor_lambda(img, 3);
cornerStrength = (((lambda1.*lambda2)./(lambda1 + lambda2)));
cornerStrength=mat2gray(cornerStrength)*10;
EValue = (lambda1-lambda2);
priorWeight3=((m+1).*cornerStrength.*EValue)./(m.*EValue+cornerStrength);
priorWeight3=mat2gray(priorWeight3);%加权先验
% tidu image
img_tidu=priorWeight3.*E1;
EX=uint8( mat2gray(E1)*256 );
E = uint8( mat2gray(img_tidu)*256 );%方便处理
%imshow(img_tidu,[]);
%% 局部对比度计算
for kk=1:litter
    scs=ss(kk);%窗口大小
    %% 滤波模板
    [mask1,mask2,mask3,mask4,mask5,mask6,mask7,mask8,...
     omask1,omask2,omask3,omask4,omask5,omask6,omask7,...
     omask8,omask9,omask10,omask11,omask12,omask13,...
     omask14,omask15,omask16] = create_mask16(scs);
    [mask_row,mask_col]=size(mask1);
    mask=zeros(mask_row,mask_col,24);
    mask(:,:,1)=mask1;mask(:,:,2)=mask2;mask(:,:,3)=mask3;mask(:,:,4)=mask4;
    mask(:,:,5)=mask5;mask(:,:,6)=mask6;mask(:,:,7)=mask7;mask(:,:,8)=mask8;
    mask(:,:,9)=omask1;mask(:,:,10)=omask2;mask(:,:,11)=omask3;mask(:,:,12)=omask4;
    mask(:,:,13)=omask5;mask(:,:,14)=omask6;mask(:,:,15)=omask7;mask(:,:,16)=omask8;
    mask(:,:,17)=omask9;mask(:,:,18)=omask10;mask(:,:,19)=omask11;mask(:,:,20)=omask12;
    mask(:,:,21)=omask13;mask(:,:,22)=omask14;mask(:,:,23)=omask15;mask(:,:,24)=omask16;
    %这个E1要是最后效果不好，就修改
    meanT0=imfilter(img_tidu,ones(scs),'replicate')/(scs^2); %目标区域均值
    MeanT0=(mat2gray(meanT0)*255);%所有的归一化处理均为了方便观察数据
    oriT0=imfilter(img,ones(scs),'replicate')/(scs^2); %原始图像目标区域均值
    oriT1=imfilter(img,ones(scs),'replicate')/(scs^2);
    oriT2=ordfilt2(img,round(scs^2/2), ones(scs));%取中值
    oriT3=min(oriT1,oriT2);%取两者小值
    STDT0=((meanT0-img_tidu).^2)/2;
    Stdt0=uint8(mat2gray(STDT0)*255);
    oriSTD=((oriT0-img).^2)/2;%原始图像目标域的方差
    IB_mean=zeros(row,col,8); %内邻域均值和方差
    %% 预留内存
    STD_mean=zeros(row,col,8);
    IB_nril=zeros(row,col,8);
    IB_two=zeros(row,col,8);
    STDtwo=zeros(row,col,8);
    IB_j=zeros(row,col,8);
    medianIB_j=zeros(row,col,8);
    for i=1:8
    IB_mean(:,:,i)=imfilter(img_tidu,mask(:,:,i),'replicate')/(scs^2);
    IB_two(:,:,i)=imfilter(img,mask(:,:,i),'replicate')/(scs^2);
    %中值计算
    medianIB_j(:,:,i)=ordfilt2(img,round(scs^2/2), mask(:,:,i));%内邻域中值
    IB_j(:,:,i)=min(IB_two(:,:,i),medianIB_j(:,:,i));
    end
    IB_temp=max(IB_mean,[],3);%内邻域均值最大
    IB_temp=reshape(IB_temp,row,col);
    IB_out=zeros(row,col);
    IB_LC=(meanT0>IB_temp).*((meanT0-IB_temp));%DNGM中的对比度处理
    A3=uint8( mat2gray(IB_LC)*256 );
    meanOB=zeros(row,col,16);
    STD_meanOB=zeros(row,col,16);
    OB_nril=zeros(row,col,16);
    medianB_j=zeros(row,col,16);
    OB_j=zeros(row,col,16);
    STDtwo_O=zeros(row,col,16);
    for i=1:16
    % 计算第一部分时用到
    meanOB(:,:,i)=imfilter(img_tidu,mask(:,:,i+8),'replicate')/(scs^2);
    %计算第二部分时用到
    OB_nril(:,:,i)=imfilter(img,mask(:,:,i+8),'replicate')/(scs^2);
    medianB_j(:,:,i)=ordfilt2(img,round(scs^2/2), mask(:,:,i+8));
    %% 改成最大的试一试（测试）
    %STDtwo_O(:,:,i)=((OB_nril(:,:,i)-img).^2)/2;
    %% 改成最大的试一试
    OB_j(:,:,i)=min(OB_nril(:,:,i),medianB_j(:,:,i));
    end
    OB_temp=max(meanOB,[],3);
    OB_temp=reshape(OB_temp,row,col);
    OB_out=zeros(row,col);
    matmeant0=(mat2gray(meanT0)*255);
     matOB_temp=(mat2gray(OB_temp)*255);
    OB_LC=(matmeant0>matOB_temp).*((matmeant0-matOB_temp));
    B3=uint8( mat2gray(OB_LC)*256 );
    I_fccha=double(B3).*double(A3).*(double(STDT0));%潜在目标区域提取
    %I_fccha=double(B3).*double(A3).*(double(STDT0));%暂时用这个
    %% 目标聚合，可有可无，在本数据中几乎不受影响
    [row_max,col_max] = find(I_fccha==max(max(I_fccha)));
    [L1,~] = bwlabel(I_fccha,8);
    j=L1(row_max,col_max);
    j=max(max(j));
    [r,c] = find(L1 == j);
    lengthr=length(r);
    for k=1:lengthr
        I_fccha(r(k),c(k))=max(max(I_fccha));
    end
    % fangcha=double(B1).*double(A1);
    %% 计算第二部分    
    m0=zeros(size(img,1),size(img,2),K);
    m0_min=zeros(size(img,1),size(img,2),K);
    om16=zeros(size(img,1),size(img,2),K,16);
    oi16=zeros(size(img,1),size(img,2),K,16);
    im_8=zeros(size(img,1),size(img,2),K,8);
    ii_8=zeros(size(img,1),size(img,2),K,8);
    for j =1:K
        m0(:,:,j)=ordfilt2(img,scs^2+1-j, ones(scs));
        m0_min(:,:,j)=ordfilt2(img,j, ones(scs));
        for vec=1:16
            om16(:,:,j,vec)=ordfilt2(img,scs^2+1-j, mask(:,:,j+8));%外邻域前K最大值
            oi16(:,:,j,vec)=ordfilt2(img,j, mask(:,:,j+8));
        end
        for vec =1:8
            im_8(:,:,j,vec)=ordfilt2(img,scs^2+1-j, mask(:,:,j)); 
            ii_8(:,:,j,vec)=ordfilt2(img,j, mask(:,:,j));
        end
    end
    M0=mean(m0,3);
    M0_min=mean(m0_min,3);
    OM=zeros(row,col,16);
    OM_min=zeros(row,col,16);
    IM=zeros(row,col,8);
    IM_min=zeros(row,col,8);
    NRILB1=zeros(row,col,16);
    NRILB2=zeros(row,col,16);
    RILB=zeros(row,col,16);
    NRILB_I1=zeros(row,col,8);
   NRILB_I2= zeros(row,col,8);
   RILB_I=zeros(row,col,8);
    for vec =1:16
        OM(:,:,vec)=mean(om16(:,:,:,vec),3);%M_k(obj)
        OM_min(:,:,vec)=mean(oi16(:,:,:,vec),3);%M_k_min(obj)
    end
    for vec = 1:8
        IM(:,:,vec)=mean(im_8(:,:,:,vec),3);%M_k(ibj))
        IM_min(:,:,vec)=mean(ii_8(:,:,:,vec),3);%M_k_min(ibj))
    end
    for i =1:16
        NRILB1(:,:,i)=OM(:,:,i)-OB_j(:,:,i);%这个OB_j看看能修改吗
        NRILB2(:,:,i)=OM(:,:,i)-OM_min(:,:,i);
        RILB(:,:,i)=NRILB1(:,:,i).*NRILB2(:,:,i);
    end
    for i =1:8
        NRILB_I1(:,:,i)=IM(:,:,i)-IB_j(:,:,i);
        NRILB_I2(:,:,i)=IM(:,:,i)- IM_min(:,:,i);
        RILB_I(:,:,i)=NRILB_I1(:,:,i).*NRILB_I2(:,:,i);
    end
    NRILB=max(RILB,[],3);
    matNRILB=mat2gray(NRILB)*255;
    NRILBI=max(RILB_I,[],3);
    matNRILBI=mat2gray(NRILBI)*255;
    M2=(M0-oriT3).*(M0-M0_min);
    M1=mat2gray(M2)*255;
    NRIL0=max((M1-matNRILB),(M1-matNRILBI));
    %NSTDtwo=mean(STDtwo,3);%方差用均值有待考虑，STDtwo是内邻域方差
    %NSTDtwo=mat2gray(NSTDtwo).*255;
    %NSTDtwo_O=mean(STDtwo_O,3);
    %NSTDtwo_O=mat2gray(NSTDtwo_O).*255;
    %NRILT=(oriSTD./NSTDtwo).*(NRIL0);
    %(NSTDtwo_O<oriSTD).*(oriSTD-NSTDtwo_O)).*
    oriSTD=mat2gray(oriSTD).*255;
    NRILT=(NRIL0).*oriSTD;%空域加权的权重
    %NRILT=(NRIL0);
    matNRILT=(mat2gray(NRILT)*255);
    I_twopart=matNRILT;
    I_onepart=(mat2gray(I_fccha)*255);
    %% 加权计算
    I_final=I_twopart.*I_onepart;
    %finalNRIL=((matNRILT./NRILBJ).*matNRILT)-matNRILT;%要是抑制不明显可以加平方
    %I_twopart1=uint8(mat2gray(finalNRIL)*255);
M_out(:,:,kk)=I_final;
end
%subplot(121),imshow(img,[]);
 maxM_out=max(M_out,[],3);
 matmaxM_out=mat2gray(maxM_out)*255;
 %% 时域计算 帧数取前后8帧
 d=8;
if q>8&q<num-d
    D1=D(:,:,q-d:q+d);
    Z=zeros(256,256,d-4);
    D2=cat(3,Z,D1,Z);
    ent =kurtosis(D2,1,3);%峰度
    d1=1.2;
    ent1=((3-d1<ent)&(ent<3+d1)).*ent;%%
    ent11=(ent1>0).*(abs(3-abs(3-ent1)));

%ent12=mat2gray(ent12)*255;
    [L,n] = bwlabel(ent11,8);
    lengthL=n;
    for j=1:lengthL
        [r,c] = find(L == j);
        lengthr=length(r);
        r_min=min(r);
        r_max=max(r);
        c_min=min(c);
        c_max=max(c);
        if (r_min>2&&c_min>2&& r_max<254&&c_max<254)
            for o=1:lengthr
                maxjuzhen=ones(3,3);
                ent11(r(o)-1:r(o)+1,c(o)-1:c(o)+1)= ent11(r(o),c(o))*maxjuzhen;
            end
        else
        continue
        end 
    end
    ent12=(ent11>0);%只确定目标的时域位置
    ent11=mat2gray(ent11)*255;%目标时域位置位置和大小
    I_thr=matmaxM_out.*ent11;
    %I_thr=matmaxM_out.*ent11;
    matI_thr=uint8(mat2gray(I_thr)*255);
 %subplot(122),imshow(maxM_out,[]); 
 %title('多尺度空域检测的结果');
else
    matI_thr=uint8(mat2gray(maxM_out)*255);
end
%% 可视化
%subplot(221),imshow(img,[]);title('原始图像');
%subplot(222),imshow(maxM_out,[]);title('空域');
%subplot(223),imshow(ent11,[]);title('时域');
%subplot(224),imshow(matI_thr,[]);title('空时');
 %matI_thr=uint8(mat2gray(maxM_out)*255);
 q
 %% 写入图像
imwrite(matI_thr,fullfile(b,files_name{q}));
end

