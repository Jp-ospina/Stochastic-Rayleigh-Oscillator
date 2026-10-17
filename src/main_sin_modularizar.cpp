#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include "../include/rayleigh.h"

class BrownianMotion {
private:
    double x;
    double vx;
    int n;
    double epsilon;
    double sigma;
    double dt;
    int metodo;
    unsigned int semilla = 123456;
    std::mt19937 generador{semilla};
    std::normal_distribution<double> distribucionNormal{0.0, 1.0};

    void pedirDatos() {
        std::cout << "Ingrese la posicion inicial x(0): ";
        if (!(std::cin >> x)) {
            throw std::invalid_argument("La posicion inicial debe ser numérica.");
        }

        std::cout << "Ingrese la velocidad inicial vx(0): ";
        if (!(std::cin >> vx)) {
            throw std::invalid_argument("La velocidad inicial debe ser numerica.");
        }

        std::cout << "Ingrese el numero de pasos N: ";
        if (!(std::cin >> n)) {
            throw std::invalid_argument("El numero de pasos debe ser un entero.");
        }

        std::cout << "Ingrese el valor de epsilon: ";
        if (!(std::cin >> epsilon)) {
            throw std::invalid_argument("Epsilon debe ser numerico.");
        }

        std::cout << "Ingrese el valor de sigma: ";
        if (!(std::cin >> sigma)) {
            throw std::invalid_argument("Sigma debe ser numerico.");
        }

        std::cout << "Ingrese el paso de tiempo dt: ";
        if (!(std::cin >> dt)) {
            throw std::invalid_argument("dt debe ser numerico.");
        }

        std::cout << "Seleccione el metodo: 1) Euler-Maruyama, "
                     "2) Euler-Cromer, 3) Heun estocastico: ";
        if (!(std::cin >> metodo) || metodo < 1 || metodo > 3) {
            throw std::invalid_argument("Seleccione el metodo 1, 2 o 3.");
        }

        std::cout << "Ingrese la semilla del generador: ";
        if (!(std::cin >> semilla)) {
            throw std::invalid_argument("La semilla debe ser un entero.");
        }
        generador.seed(semilla);
    }

    void verificarDatos() const {
        if (!std::isfinite(x) || !std::isfinite(vx)
            || !std::isfinite(epsilon) || !std::isfinite(sigma)
            || !std::isfinite(dt)) {
            throw std::invalid_argument("Todos los parametros deben ser finitos.");
        }
        if (n <= 0 || epsilon < 0.0 || sigma < 0.0 || dt <= 0.0) {
            throw std::invalid_argument(
                "N y dt deben ser positivos; epsilon y sigma no pueden ser negativos.");
        }
    }

    double aceleracionDeterminista(double posicion, double velocidad) const {
        return -epsilon * (velocidad * velocidad / 3.0 - 1.0) * velocidad
               - posicion;
    }

    double siguienteVelocidad(double posicion, double velocidad) {
        const double eta = distribucionNormal(generador);
        return velocidad
               + aceleracionDeterminista(posicion, velocidad) * dt
               + sigma * std::sqrt(dt) * eta;
    }

    void escribirEstado(std::ofstream& archivo, int paso, double tiempo) const {
        const double amplitud = std::hypot(x, vx);
        const double fase = std::atan2(-vx, x);
        archivo << paso << ',' << tiempo << ',' << x << ',' << vx << ','
                << amplitud << ',' << fase << ',' << epsilon << ',' << sigma
                << ',' << dt << ',' << metodo << ',' << semilla << '\n';
    }

public:
    BrownianMotion() {
        pedirDatos();
        verificarDatos();
    }

    void simular() {
        const std::string rutaArchivo = "data/simulacion.csv";
        std::ofstream archivo(rutaArchivo.c_str());
        if (!archivo) {
            throw std::runtime_error(
                "No se pudo abrir " + rutaArchivo
                + ". Verifique que la carpeta data exista.");
        }

        archivo << "paso,t,x,vx,amplitud,fase,epsilon,sigma,dt,metodo,semilla\n"
                << std::setprecision(17);
        escribirEstado(archivo, 0, 0.0);

        for (int paso = 0; paso < n; ++paso) {
            if (metodo == 1) {
                const double posicionAnterior = x;
                const double velocidadAnterior = vx;
                const double posicionNueva = posicionAnterior + velocidadAnterior * dt;
                const double velocidadNueva = siguienteVelocidad(posicionAnterior, velocidadAnterior);
                x = posicionNueva;
                vx = velocidadNueva;
            } else if (metodo == 2) {
                vx = siguienteVelocidad(x, vx);
                x += vx * dt;
            } else {
                const double ruido =
                    sigma * std::sqrt(dt) * distribucionNormal(generador);
                const double aceleracionInicial =
                    aceleracionDeterminista(x, vx);
                const double posicionPredicha = x + vx * dt;
                const double velocidadPredicha =
                    vx + aceleracionInicial * dt + ruido;
                const double aceleracionPredicha =
                    aceleracionDeterminista(posicionPredicha, velocidadPredicha);
                x += 0.5 * (vx + velocidadPredicha) * dt;
                vx += 0.5 * (aceleracionInicial + aceleracionPredicha) * dt
                      + ruido;
            }

            if (!std::isfinite(x) || !std::isfinite(vx)) {
                throw std::runtime_error(
                    "El estado del oscilador dejó de ser finito; reduzca dt.");
            }

            escribirEstado(archivo, paso + 1, (paso + 1) * dt);
        }

        archivo.close();
        if (!archivo) {
            throw std::runtime_error(
                "Ocurrio un problema al escribir " + rutaArchivo + ".");
        }

        std::cout << "Se guardaron " << static_cast<long long>(n) + 1
                  << " estados. Tiempo final: " << std::setprecision(17) << n * dt
                  << "\nMetodo: "
                  << (metodo == 1 ? "Euler-Maruyama"
                      : metodo == 2 ? "Euler-Cromer" : "Heun estocastico")
                  << "\nSemilla: " << semilla << std::endl;
    }
};

int ejecutarRayleigh() {
    BrownianMotion simulacion;
    simulacion.simular();
    return 0;
}
