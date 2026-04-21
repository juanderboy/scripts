function f=perfil(k);
% calcula el perfil de reactivo y producto en un decaimiento
% monoexponencial

global R;
global t;
global c0;
global W;
global Q;
global Wcalc;

aux(1,:)=c0*exp(-k*t);
aux(2,:)=c0-aux(1,:);


% devuelve el perfil, que esta en vecores fila

f=aux;