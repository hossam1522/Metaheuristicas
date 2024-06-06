#include <iostream>
#include "p3.h"

using namespace std;

/************************************************************
************************************************************
FUNCIONES AUXILIARES
************************************************************
************************************************************/

Solucion solucion_aleatoria(const Dataset &datos){
  uniform_real_distribution<double> distribucion(0.0, 1.0);
  Solucion sol;

  sol.pesos.set_size(datos.data.n_cols);

  for (int i = 0; i < datos.data.n_cols; ++i)
    sol.pesos(i) = Random::get(distribucion);

  double tasa_clasificacion = tasa_clas(datos, sol.pesos);
  double tasa_reduccion = tasa_red(sol.pesos);
  sol.fitness = fitness(tasa_clasificacion, tasa_reduccion);

  return sol;
}

/************************************************************
************************************************************
BÚSQUEDA LOCAL MULTIARRANQUE BÁSICA (BMB)
************************************************************
************************************************************/

arma::rowvec BMB (const Dataset &datos, const int &num_ejecuciones, const int &maxIter){
  Solucion mejor_solucion;

  for (int i = 0; i < num_ejecuciones; ++i){
    Solucion sol = solucion_aleatoria(datos);
    
    int iter = 0;
    Solucion mutacion = busquedaLocal(datos, sol.pesos, iter, CONST_MAX_VECINOS, maxIter);

    if (mejor_solucion < mutacion)
      mejor_solucion = mutacion;

  }

  return mejor_solucion.pesos;
}

/************************************************************
************************************************************
ENFRIAMIENTO SIMULADO (ES)
************************************************************
************************************************************/

Solucion ES(const Dataset &datos, const Solucion &solucion_pasada, const int &maxIter){
  uniform_int_distribution<int> distribucion_componente(0, datos.data.n_cols-1);
  normal_distribution<double> distribucion_normal(0.0, SIGMA);
  uniform_real_distribution<double> distribucion_uniforme(0.0, 1.0);

  Solucion solucion;

  if (solucion_pasada.pesos.n_cols == 0)
    solucion = solucion_aleatoria(datos);
  else 
    solucion = solucion_pasada;

  Solucion mejor_solucion = solucion;
  double T0 = MU * solucion.fitness / -log(PHI);

  double Tfinal = Tf;

  while (Tfinal >= T0)
    Tfinal = Tfinal / 10.0;

  int num_vecinos = 0;
  int num_exitos = 1;
  int num_iter = 0;
  const double max_vecinos = MAX_VECINOS_ES*datos.data.n_cols;
  const double max_exitos = MAX_EXITOS * max_vecinos;
  const double M = maxIter / max_vecinos;
  const double BETA = (T0 - Tfinal) / (M * T0 * Tfinal);

  while (num_iter < maxIter && num_exitos > 0){
    
    num_vecinos = 0;
    num_exitos = 0;

    while (num_vecinos < max_vecinos && num_exitos < max_exitos && num_iter < maxIter){
      arma::rowvec pesos_mutados = solucion.pesos;
      int componente = Random::get(distribucion_componente);
      pesos_mutados(componente) += Random::get(distribucion_normal);

      if (pesos_mutados(componente) < 0.0) pesos_mutados(componente) = 0.0;
      if (pesos_mutados(componente) > 1.0) pesos_mutados(componente) = 1.0;

      double tasa_clasif = tasa_clas(datos, pesos_mutados);
      double tasa_reducc = tasa_red(pesos_mutados);
      double fit = fitness(tasa_clasif, tasa_reducc);

      double delta = solucion.fitness - fit;

      if (delta < 0 || Random::get(distribucion_uniforme) <= exp(-delta / T0)){
        solucion.pesos = pesos_mutados;
        solucion.fitness = fit;
        num_exitos++;

        if (mejor_solucion < solucion)
          mejor_solucion = solucion;
      }

      num_vecinos++;
      num_iter++;
    }

    T0 = T0 / (1.0 + BETA * T0);

  }

  return mejor_solucion;

}

/************************************************************
************************************************************
BÚSQUEDA LOCAL REITERADA (ILS)
************************************************************
************************************************************/

Solucion mutacion_ILS(const Dataset &datos, const Solucion &solucion, const double &operadorMutacion){
  uniform_int_distribution<int> distribucion_componente(0, solucion.pesos.n_cols-1);
  uniform_real_distribution<double> distribucion_uniforme(-0.25, 0.25);

  Solucion sol = solucion;
  
  int num_mutaciones = sol.pesos.n_cols * operadorMutacion;
  num_mutaciones = max(num_mutaciones, 3);

  for (int i = 0; i < num_mutaciones; ++i){
    int componente = Random::get(distribucion_componente);
    sol.pesos(componente) += Random::get(distribucion_uniforme);

    if (sol.pesos(componente) < 0.0) sol.pesos(componente) = 0.0;
    if (sol.pesos(componente) > 1.0) sol.pesos(componente) = 1.0;
  }

  double tasa_clasificacion = tasa_clas(datos, sol.pesos);
  double tasa_reduccion = tasa_red(sol.pesos);
  sol.fitness = fitness(tasa_clasificacion, tasa_reduccion);

  return sol;
}


arma::rowvec ILS(const Dataset &datos, const int &num_ejecuciones, const int &maxIter){
  Solucion sol = solucion_aleatoria(datos);
  int iter = 0;
  sol = busquedaLocal(datos, sol.pesos, iter, CONST_MAX_VECINOS, maxIter);
  Solucion mejor_solucion = sol;

  for (int i = 0; i < num_ejecuciones-1; ++i){
    Solucion mutacion = mutacion_ILS(datos, mejor_solucion, OPERADOR_MUTACION);
    int iter = 0;
    mutacion = busquedaLocal(datos, mutacion.pesos, iter, CONST_MAX_VECINOS, maxIter);

    if (mejor_solucion < mutacion)
      mejor_solucion = mutacion;

  }

  return mejor_solucion.pesos;
}


arma::rowvec ILS_ES(const Dataset &datos, const int &num_ejecuciones, const int &maxIter){
  Solucion sol = solucion_aleatoria(datos);
  sol = ES(datos, sol, maxIter);
  Solucion mejor_solucion = sol;

  for (int i = 0; i < num_ejecuciones-1; ++i){
    Solucion mutacion = mutacion_ILS(datos, mejor_solucion, OPERADOR_MUTACION);
    mutacion = ES(datos, mutacion, maxIter);

    if (mejor_solucion < mutacion)
      mejor_solucion = mutacion;

  }

  return mejor_solucion.pesos;
}
