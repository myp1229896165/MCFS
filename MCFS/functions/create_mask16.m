function [mask1,mask2,mask3,mask4,mask5,mask6,mask7,mask8,omask1,omask2,omask3,omask4,omask5,omask6,omask7,omask8,...
    omask9,omask10,omask11,omask12,omask13,omask14,omask15,omask16] = create_mask16(c_size)

t_mask=zeros(5*c_size,5*c_size);

mask1=t_mask;
mask1(c_size+1:2*c_size,c_size+1:2*c_size)=ones(c_size,c_size);

mask2=t_mask;
mask2(c_size+1:2*c_size,2*c_size+1:3*c_size)=ones(c_size,c_size);

mask3=t_mask;
mask3(c_size+1:2*c_size,3*c_size+1:4*c_size)=ones(c_size,c_size);

mask4=t_mask;
mask4(2*c_size+1:3*c_size,3*c_size+1:4*c_size)=ones(c_size,c_size);

mask5=t_mask;
mask5(3*c_size+1:4*c_size,3*c_size+1:4*c_size)=ones(c_size,c_size);

mask6=t_mask;
mask6(3*c_size+1:4*c_size,2*c_size+1:3*c_size)=ones(c_size,c_size);

mask7=t_mask;
mask7(3*c_size+1:4*c_size,c_size+1:2*c_size)=ones(c_size,c_size);

mask8=t_mask;
mask8(2*c_size+1:3*c_size,c_size+1:2*c_size)=ones(c_size,c_size);
%%omask
omask1=t_mask;
omask1(1:c_size,1:c_size)=ones(c_size,c_size);

omask2=t_mask;
omask2(1:c_size,c_size+1:2*c_size)=ones(c_size,c_size);

omask3=t_mask;
omask3(1:c_size,2*c_size+1:3*c_size)=ones(c_size,c_size);

omask4=t_mask;
omask4(1:c_size,3*c_size+1:4*c_size)=ones(c_size,c_size);

omask5=t_mask;
omask5(1:c_size,4*c_size+1:5*c_size)=ones(c_size,c_size);

omask6=t_mask;
omask6(c_size+1:2*c_size,4*c_size+1:5*c_size)=ones(c_size,c_size);

omask7=t_mask;
omask7(2*c_size+1:3*c_size,4*c_size+1:5*c_size)=ones(c_size,c_size);

omask8=t_mask;
omask8(3*c_size+1:4*c_size,4*c_size+1:5*c_size)=ones(c_size,c_size);

omask9=t_mask;
omask9(4*c_size+1:5*c_size,4*c_size+1:5*c_size)=ones(c_size,c_size);

omask10=t_mask;
omask10(4*c_size+1:5*c_size,3*c_size+1:4*c_size)=ones(c_size,c_size);

omask11=t_mask;
omask11(4*c_size+1:5*c_size,2*c_size+1:3*c_size)=ones(c_size,c_size);

omask12=t_mask;
omask12(4*c_size+1:5*c_size,c_size+1:2*c_size)=ones(c_size,c_size);

omask13=t_mask;
omask13(4*c_size+1:5*c_size,1:c_size)=ones(c_size,c_size);

omask14=t_mask;
omask14(3*c_size+1:4*c_size,1:c_size)=ones(c_size,c_size);

omask15=t_mask;
omask15(2*c_size+1:3*c_size,1:c_size)=ones(c_size,c_size);

omask16=t_mask;
omask16(c_size+1:2*c_size,1:c_size)=ones(c_size,c_size);
end