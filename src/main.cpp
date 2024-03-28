#include <iostream>
#include "p1.h"

using namespace std;

int main(int argc, char const **argv) {
  if (argc != 2) {
    cout << "Uso: " << argv[0] << " <semilla>" << endl;
    return 1;
  }

  long int semilla = strtol(argv[1], nullptr, 10);
  Random::seed(semilla);

  // Para algoritmo 1NN sin ponderación
  //printResultados(0);

  // Para algoritmo 1NN usando Greedy Relief
  printResultados(1);

  // Para algoritmo 1NN usando Búsqueda Local
  //printResultados(2);

  return 0;
}