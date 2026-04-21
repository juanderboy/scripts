function f=terror(k);
% dado un valor de k, devuelve la norma de la diferencia entre el perfil
% calculado y el medido, en el espacio de W.


global R;
global t;
global c0;
global W;
global Q;
global Wcalc;


% calcular el perfil en espacio de concentraciones

C=perfil(k);

% calcular R y Wcalc

R=C*pinv(W);
Wcalc=inv(R)*C;

% evaluar la diferencia

aux=(W-Wcalc)';
f=norm(aux(:,1))+norm(aux(:,2));

