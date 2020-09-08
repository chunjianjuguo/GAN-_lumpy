clear all
close all
%get the H matrix using pixle method
% meshgrid for pixel
stepsize=1/32;
x1 = -1:stepsize:1;
x2 = -1:stepsize:1;
[X1,X2] = meshgrid(x1,x2);
X = [X1(:) X2(:)];
L=length(X);

Angle=90;

%find middle point
for n=1:Angle
    disp(n)
for i=1:65
    r(i)=-1+(i-1)/32;
    theta(n,i)=(sign(r(i)))*pi/2+((n-1)*pi/Angle);
    thetaline(n)=(-1)*pi/2+((n-1)*pi/Angle);
    xc(n,i)=cos(theta(n,i))*abs(r(i));% find the middle point of each line
    yc(n,i)=sin(theta(n,i))*abs(r(i));% find the middle point of each line
end
end
%scatter (xc(:),yc(:));

totalpixels=(2*ceil(1/stepsize))^2;
row=length(x1)-1;
col=length(x2)-1;

p=1;q=1;
n=1;
j=1;
H=zeros(Angle*64,64*64);
for n=1:Angle 
disp(n)
for j=1:64 %detector
for p=1:64 % row
for q=1:64 %column(fix row first , from up to bottom,from left to right)
 rs=-1+(p-1)*1/32;% row start
 re=-1+p*1/32;
 cs=1-(q-1)*1/32;% col start
 ce=1-q*1/32;
 ro = (rand(1,2))*1/32+rs;
 co = -(rand(1,2))*1/32+cs;
 [RO,CO] = meshgrid(ro,co);

Y = [RO(:) CO(:)];% all pixel in one pixel [p,q]


LM=length(Y);
K1=zeros(2,LM);
K2=zeros(2,LM);
 for i=1:LM
    M=[cos(thetaline(n)),sin(thetaline(n)); cos(thetaline(n)-pi/2),sin(thetaline(n)-pi/2)];
    C1=xc(n,j)*cos(thetaline(n))+yc(n,j)*sin(thetaline(n));
    C2=xc(n,j+1)*cos(thetaline(n))+yc(n,j+1)*sin(thetaline(n));    
    CD=cos(thetaline(n)-pi/2)*Y(i,1)+sin(thetaline(n)-pi/2)*Y(i,2);
    mc1=[C1;CD];
    mc2=[C2;CD];
   K1(:,i)=M\mc1;
   K2(:,i)=M\mc2;
  
 end
  K1=K1';
  K2=K2';
 for d=1:LM
        if((sqrt((Y(d,1)-K1(d,1))^2+(Y(d,2)-K1(d,2))^2)+sqrt((Y(d,1)-K2(d,1))^2+(Y(d,2)-K2(d,2))^2))<=1/31.999999)
        h(d)=1;
        else
        h(d)=0;    
        end
 end
 H((n-1)*64+j,(p-1)*64+q)=(sum(h(:)))/LM;
end
end
end
end


writematrix(H, 'Hmatrix9064.csv')





 
 
 