
% explorando el experimento de cinética
% +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%                          IMPORTANTE:
% los espectros estan verticales, lambda es un vector columna,
% y cada columna de A es un tiempo distinto (t es un vector fila)
% +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

clear all

% se declaran las globales que se van a usar en el fiteo
global R;
global t;
global c0;
global W;
global Q;
global Wcalc;

% c0 es la concentracion analitica 

c0=1;


% cargar los files

aux=dlmread('117.txt',';');

%saco los tiempos en un vector fila (primera fila tiene los lambdas)

t=aux(1,2:end);   %vector fila
lambda=aux(2:end,1);   %vector columna

%Extraigo todos los espectros en una variable

A=aux(2:end,2:end);  % cada columna es un espectro a un determinado tiempo

% recortamos tiempos a t> 5 minutos


% q=find(t>5*60);
% t=t(q);
% t=t-t(1);
% 
% A=A(:,q);

% correccion de linea de base, usando el promedio de los ultimos 20 puntos

for zz=1:1:size(A,2);
    auxz=mean(A(end-20:end,zz));
    A(:,zz)=A(:,zz)-auxz;
end;

% acotar el rango de lambdas, tienen que ser numeros pares!!!!!!!!!!!!!!!!

lambda_min=320;
lambda_max=820;

i=find(lambda==lambda_min);
j=find(lambda==lambda_max);

lambda=lambda(i:j,1);
A=A(i:j,:);



% factor de analisis
[U,S,V]=svd(A);


% Retener la informacion de solo dos componentes
% Q contiene los espectros, y W las concentraciones!!

Q=U(:,1:3);
W=S*V';
W=W(1:3,:);

pause
% para poder hacert el ajuste, se necesita una funcion que calcule la
% diferencia entre el perfil calculado y el experimental (en el espacio del
% factor analisis).



% % +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% % primero se explora el error en un rango grande de valores de constante,
% % para acotar lo mejor posible
% 
% raux=[6:-.1:-6];
% rango=10.^raux;
% 
% for zz=1:1:size(rango,2);
%     EE(1,zz)=terror(rango(1,zz));
% end;
% 
% i=find(EE==min(EE));
% k0=rango(i);
% % +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

pause

k0=7e-5;

% se usa este valor para largar el fiteo


options=optimset('Display','off');
options=optimset('TolFun',1e-8);

k1=fminsearch('terror',k0);

% recalcular espectro de las especies y perfil ed concentraciones

C=perfil(k1);
E=Q*inv(R);


Acalc=E*C;

figure(1);hold on
plot(lambda,E);

figure(2);clf;
plot(lambda,A-Acalc);



