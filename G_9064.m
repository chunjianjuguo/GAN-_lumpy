clear all
close all
%get measurements using the analytic way.
stepsize=1/32;
x1 = -1+1/64:stepsize:1-1/64;%find 64*64 point to evaluate the object(lumpy backgournd)
x2 = 1-1/64:-stepsize:-1+1/64;
l=length(x1);
[X1,X2] = meshgrid(x1,x2);
X = [X1(:) X2(:)];
L=length(X);

kba=30;
K=poissrnd(kba);%number of lumps
Angle=90;%angle
for s=1:10%get 10 files of measurements 
    disp(s);
for o=1:10000%10000 measurements per file
    if mod(o,1000)==0
        disp(o);
    end
for m=1:K
    b0(m)=rand(1)*10;% weight for each lump
    cen1(m)=rand(1)*2-1;%center for x 
    cen2(m)=rand(1)*2-1;%center for y
    d1(m)=0.05*rand(1);%variance for x
    d2(m)=0.05*rand(1);%variance for y
    
    for i=1:L
        ob(m,i)=exp(-1/2*((X(i,1)-cen1(m))^2/d1(m)+(X(i,2)-cen2(m))^2/d2(m)));%get image
    end
    
    for n=1:Angle%calculate the measurements data using the analytic way.number of angles
        for j=1:64%number of detectors
             if n<=Angle/2
             sita(n,j)=pi/Angle*(n-1);
             a(n,j)=tan(sita(n,j));
             r(n,j)=-1+1/64+1/32*(j-1);
             bp(n,j)=r(n,j)/cos(sita(n,j));
             xs(n,j)=max(-1,(-1-bp(n,j))/tan(sita(n,j)));
             xe(n,j)=min(1,(1-bp(n,j))/tan(sita(n,j)));
             x1(n,j)=xs(n,j)-cen1(m);
             x2(n,j)=xe(n,j)-cen1(m);
             b(n,j)=a(n,j)*cen1(m)+bp(n,j)-cen2(m);
             c1(n,j)=(d2(m)+a(n,j)^2*d1(m))/2*d1(m)*d2(m);
             c2(n,j)=a(n,j)*b(n,j)*d1(m)/(d2(m)+a(n,j)^2*d1(m));
             f0(n,j)=sqrt(a(n,j)^2+1);
             in2(n,j)=b(n,j)^2/(2*(d2(m)+d1(m)*a(n,j)^2));
             f1(n,j)=exp(-in2(n,j));
             t0(n,j)=(x1(n,j)+c2(n,j))*sqrt(c1(n,j));
             t1(n,j)=(x2(n,j)+c2(n,j))*sqrt(c1(n,j));
             cof(n,j)=f0(n,j)*f1(n,j)/sqrt(c1(n,j));
             dif(n,j)=t1(n,j)-t0(n,j);
             ga(n,j)=cof(n,j)*(sign(t1(n,j))*erf(abs(t1(n,j)))/2*sqrt(pi)-sign(t0(n,j))*erf(abs(t0(n,j)))/2*sqrt(pi));
             else
             sita(n,j)=pi/Angle*(n-1);
             a(n,j)=cot(sita(n,j));
             r(n,j)=1-1/64-1/32*(j-1);
             bp(n,j)=r(n,j)/sin(pi-sita(n,j));
             xs(n,j)=max(-1,(1-bp(n,j))*tan(sita(n,j)));
                 if xs(n,j)>=exp(5)
                      xs(n,j)=-1;
                 end
             xe(n,j)=min(1,(-1-bp(n,j))*tan(sita(n,j)));
                  if xe(n,j)<=-exp(5)
                      xe(n,j)=1;
                 end
             x1(n,j)=xs(n,j)-cen2(m);
             x2(n,j)=xe(n,j)-cen2(m);
             b(n,j)=a(n,j)*cen2(m)+bp(n,j)-cen1(m);
             c1(n,j)=(d1(m)+a(n,j)^2*d2(m))/2*d1(m)*d2(m);
             c2(n,j)=a(n,j)*b(n,j)*d2(m)/(d1(m)+a(n,j)^2*d2(m));
             f0(n,j)=sqrt(a(n,j)^2+1);
             in2(n,j)=b(n,j)^2/(2*(d1(m)+d2(m)*a(n,j)^2));
             f1(n,j)=exp(-in2(n,j));
             t0(n,j)=(x1(n,j)+c2(n,j))*sqrt(c1(n,j));
             t1(n,j)=(x2(n,j)+c2(n,j))*sqrt(c1(n,j));
             cof(n,j)=f0(n,j)*f1(n,j)/sqrt(c1(n,j));
             dif(n,j)=t1(n,j)-t0(n,j);
             ga(n,j)=cof(n,j)*(sign(t1(n,j))*erf(abs(t1(n,j)))/2*sqrt(pi)-sign(t0(n,j))*erf(abs(t0(n,j)))/2*sqrt(pi));
             end
             
end
end
gam(m,:)=ga(:); 
end

obf(o,:)=ob'*b0';%get objects
gamf(o,:)=gam'*b0';%get measurements
end

figure(1)

obff=reshape(obf(2,:),64,64);%show the image of one object
imagesc(obff)
colorbar;
colormap(gray);

figure(2)

gamff=reshape(gamf(2,:),Angle,64);%show the image of one sinogram
imagesc(gamff)
colorbar;
colormap(gray);
str1 = sprintf('gamf9064_1000_%d.csv', s);
writematrix(gamf, str1)%write the csv file for measurements
str2 = sprintf('obf6464_1000_%d.csv', s);
writematrix(obf, str2)%write the csv file for obf

end
