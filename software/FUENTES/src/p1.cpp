#include <iostream>
#include "p1.h"

using namespace std;


/************************************************************
************************************************************
GREEDY RELIEF
************************************************************
************************************************************/

int enemigoMasCercano(const Dataset &ejemplo, const Dataset &datos){
  double minDist = numeric_limits<double>::max();
  int enemigo = -1;

  for (size_t i = 0; i < datos.data.n_rows; ++i) {
    if (ejemplo.categoria[0] != datos.categoria[i]) {
      double dist = distanciaEuclidea(ejemplo.data.row(0), datos.data.row(i));
      if (dist < minDist) {
        minDist = dist;
        enemigo = i;
      }
    }
  }

  return enemigo;
}


int amigoMasCercano(const Dataset &ejemplo, const Dataset &datos){
  double minDist = numeric_limits<double>::max();
  int amigo = -1;

  for (size_t i = 0; i < datos.data.n_rows; ++i) {
    if (ejemplo.categoria[0] == datos.categoria[i] && any(datos.data.row(i) != ejemplo.data.row(0))){
      double dist = distanciaEuclidea(ejemplo.data.row(0), datos.data.row(i));
      if (dist < minDist) {
        minDist = dist;
        amigo = i;
      }
    }
  }

  return amigo;

}

arma::rowvec greedy(const Dataset &datos){
  arma::rowvec pesos(datos.data.n_cols, arma::fill::zeros);

  for (size_t i = 0; i < datos.data.n_rows; ++i) {
    Dataset ejemplo;
    ejemplo.data = datos.data.row(i);
    ejemplo.categoria.push_back(datos.categoria[i]);
    int pos_enemigo = enemigoMasCercano(ejemplo, datos);
    int pos_amigo = amigoMasCercano(ejemplo, datos);

    if (pos_enemigo != -1 && pos_amigo != -1){
      arma::rowvec enemigo = datos.data.row(pos_enemigo);
      arma::rowvec amigo = datos.data.row(pos_amigo);
      for (size_t j = 0; j < datos.data.n_cols; ++j) {
        pesos(j) += abs(datos.data(i,j) - enemigo(j)) - abs(datos.data(i, j) - amigo(j));
      }
    }
  }

  double max = pesos.max();

  for (size_t i = 0; i < pesos.n_cols; ++i) {
    if (pesos(i) < 0) {
      pesos(i) = 0;
    } else {
      pesos(i) /= max;
    }
  }


  return pesos;

}

/************************************************************
************************************************************
BUSQUEDA LOCAL
************************************************************
************************************************************/

arma::rowvec busquedaLocal(const Dataset &datos){
  normal_distribution<double> distribucion_normal(0.0, SIGMA);
  uniform_real_distribution<double> distribucion_uniforme(0.0, 1.0);

  vector<int> indices(datos.data.n_cols);
  arma::rowvec pesos(datos.data.n_cols);

  // La solución inicial se generará de forma aleatoria utilizando una distribución uniforme en [0, 1]
  for (size_t i = 0; i < pesos.size(); ++i) 
    pesos(i) = Random::get(distribucion_uniforme);

  for (size_t i = 0; i < pesos.size(); ++i) 
    indices[i] = i;

  Random::shuffle(indices);

  double tasa_clasif = tasa_clas(datos, pesos);
  double tasa_reducc = tasa_red(pesos);
  double mejor_fit = fitness(tasa_clasif, tasa_reducc);
  
  int num_iteraciones = 0;
  size_t num_vecinos = 0;
  bool mejora = false;
  while ( num_iteraciones < MAX_ITER && num_vecinos < (pesos.n_cols+1)*CONST_MAX_VECINOS){
    int componente = indices[num_iteraciones % pesos.n_cols];

    // Mutamos la solución actual
    arma::rowvec pesos_mutados = pesos;
    pesos_mutados(componente) += Random::get(distribucion_normal);

    // Truncamos el valor de la componente mutada
    if (pesos_mutados(componente) < 0.0) pesos_mutados(componente) = 0.0;
    if (pesos_mutados(componente) > 1.0) pesos_mutados(componente) = 1.0;

    // Calculamos el fitness de la solución mutada
    tasa_clasif = tasa_clas(datos, pesos_mutados);
    tasa_reducc = tasa_red(pesos_mutados);
    double fit = fitness(tasa_clasif, tasa_reducc);

    // Si la solución mutada es mejor que la actual, la actualizamos
    if (fit > mejor_fit) {
      pesos = pesos_mutados;
      mejor_fit = fit;
      mejora = true;
      num_vecinos = 0;
    }
    else {
      ++num_vecinos;
    }

    ++num_iteraciones;

    // Actualizamos los índices para la siguiente iteración si ha habido mejora
    // o si hemos llegado al final de la lista de índices
    if (mejora || num_iteraciones % pesos.n_cols == 0) {
      mejora = false;
      Random::shuffle(indices);
    }
    
  }

  return pesos;

}


