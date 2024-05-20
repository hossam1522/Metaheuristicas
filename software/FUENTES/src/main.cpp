#include <iostream>
#include "p3.h"

using namespace std;

int main(int argc, char const **argv) {
  if (argc != 2 && argc != 3) {
    cout << "Uso: " << argv[0] << " <semilla> [<openmp>]" << endl;
    cout << "Donde:" << endl;
    cout << "  <semilla> es la semilla para el generador de números aleatorios" << endl;
    cout << "  <openmp> es un valor booleano opcional para activar/desactivar OpenMP (por defecto, es true)" << endl;
    cout << "     true: activa OpenMP " << endl;
    cout << "     false: desactiva OpenMP " << endl;
    cout << "     Si se indica un valor distinto de true o false, se mantendrá el valor por defecto" << endl;
    cout << "Ejemplo: " << argv[0] << " 97435 false" << endl;
    return 1;
  }

  if (argc == 3) {
    if (string(argv[2]) == "true") {
      openmp = true;
    }
    else if (string(argv[2]) == "false") {
      openmp = false;
    }
  }
  else {
    openmp = true;
  }

  long int semilla = strtol(argv[1], nullptr, 10);
  Random::seed(semilla);

  // Para algoritmo 1NN sin ponderación
  // printResultados(0);

  // Para algoritmo Greedy Relief
  // printResultados(1);

  // Para algoritmo Búsqueda Local
  // printResultados(2);

  // Para algoritmo AGG-BLX
  // printResultados(3);

  // Para algoritmo AGG-CA
  // printResultados(4);

  // Para algoritmo AGE-BLX
  // printResultados(5);

  // Para algoritmo AGE-CA
  // printResultados(6);

  // Para algoritmo AM-(10,1.0)
  // printResultados(7);

  // Para algoritmo AM-(10,0.1)
  // printResultados(8);

  // Para algoritmo AM-(10,0.1mej)
  // printResultados(9);

  return 0;
}