# Oscilador de Rayleigh estocástico

Simulación de la ecuación de Rayleigh con forzamiento de ruido blanco
gaussiano. El programa integra una trayectoria mediante Euler–Maruyama,
Euler–Cromer o Heun estocástico; los scripts permiten graficar la trayectoria y
comparar ensambles de realizaciones.

## Modelo y teoría

La ecuación diferencial del oscilador es

$$\ddot{x} + \epsilon \Bigg(\frac{\dot{x}^2}{3} - 1 \Bigg)\dot{x} + x = \sigma \xi(t)$$

El término de amortiguamiento no lineal permite que el sistema gane energía a
bajas velocidades y la disipe a velocidades altas. $\epsilon$ controla la intensidad
de ese amortiguamiento. $\xi(t)$ representa ruido blanco gaussiano y $\sigma$ controla
su intensidad. Cuando $\sigma = 0$, se recupera el oscilador de Rayleigh
determinista, cuyas trayectorias no triviales convergen a un ciclo límite
estable; con ruido, las trayectorias fluctúan alrededor de ese ciclo.

Definiendo $v = \dot{x}$ y representando el ruido como el incremento de un proceso de
Wiener $dW_t$, se obtiene el sistema de ecuaciones estocásticas:

$$dx = v dt$$

$$dv = [-\epsilon ((v² / 3) - 1) v - x] dt + \sigma dW_t$$

En una discretización de paso $\Delta t$, el incremento de Wiener se aproxima como
$\Delta W_n \approx \sqrt{\Delta t} \,\eta_n$, donde $\eta_n$ son muestras independientes de una normal estándar.
Así, Euler–Maruyama actualiza la velocidad mediante

$$v_(n+1) = v_n + [-\epsilon ((v_n² / 3) - 1) v_n - x_n] \Delta t + \sigma \sqrt{\Delta t \eta_n}$$

$$x_(n+1) = x_n + v_n \Delta t$$

La implementación también ofrece Euler–Cromer y Heun estocástico. Las
trayectorias se generan con el motor pseudoaleatorio `mt19937` y una semilla
que se puede especificar para reproducir la simulación. El informe teórico está
en [`docs/Rayleigh_Equation.pdf`](docs/Rayleigh_Equation.pdf).
