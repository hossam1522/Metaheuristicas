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

  printResultados1NN();
  //printResultadosGreedy();
  //printResultadosBL();

  return 0;
}