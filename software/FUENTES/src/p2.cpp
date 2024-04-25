#include <iostream>
#include "p2.h"

using namespace std;

/************************************************************
************************************************************
FUNCIONES AUXILIARES
************************************************************
************************************************************/

Poblacion poblacion_inicial(const Dataset &datos) {
  uniform_real_distribution<double> distribucion_uniforme(0.0, 1.0);

  Poblacion poblacion;
  for (int i = 0; i < NUM_INDIVIDUOS_AGG; ++i) {
    Cromosoma cromosoma;
    for (int j = 0; j < datos.data.n_cols; ++j) {
      cromosoma.caracteristicas.insert_cols(j, Random::get(distribucion_uniforme));
    }
    double tasa_clasif = tasa_clas(datos, cromosoma.caracteristicas);
    double tasa_reducc = tasa_red(cromosoma.caracteristicas);
    cromosoma.fitness = fitness(tasa_clasif, tasa_reducc);
    poblacion.insert(cromosoma);
  }
  return poblacion;
}

/************************************************************
************************************************************
ALGORITMO GÉNETICO GENERACIONAL (AGG)
************************************************************
************************************************************/

void cruceBLX(const Cromosoma &padre1, const Cromosoma &padre2, Cromosoma &hijo1, Cromosoma &hijo2) {
  
  for (int i = 0; i < padre1.caracteristicas.n_cols; ++i) {
    double cmin = min(padre1.caracteristicas(i), padre2.caracteristicas(i));
    double cmax = max(padre1.caracteristicas(i), padre2.caracteristicas(i));
    double I = cmax - cmin;

    double minimo = cmin - I * ALPHA_BLX;
    double maximo = cmax + I * ALPHA_BLX;

    uniform_real_distribution<double> distribucion_uniforme(minimo, maximo);
    hijo1.caracteristicas(i) = Random::get(distribucion_uniforme);
    hijo2.caracteristicas(i) = Random::get(distribucion_uniforme);

    if (hijo1.caracteristicas(i) < 0) {
      hijo1.caracteristicas(i) = 0;
    } else if (hijo1.caracteristicas(i) > 1) {
      hijo1.caracteristicas(i) = 1;
    }

    if (hijo2.caracteristicas(i) < 0) {
      hijo2.caracteristicas(i) = 0;
    } else if (hijo2.caracteristicas(i) > 1) {
      hijo2.caracteristicas(i) = 1;
    }
  }

  hijo1.fitness = -1;
  hijo2.fitness = -1;
}

void cruceAritmetico(const Cromosoma &padre1, const Cromosoma &padre2, Cromosoma &hijo) {
  for (int i=0; i<padre1.caracteristicas.n_cols; ++i)
    hijo.caracteristicas[i] = (padre1.caracteristicas[i] + padre2.caracteristicas[i])/2;

  hijo.fitness = -1;
}

void mutacion(Cromosoma &cromosoma) {
  normal_distribution<double> distribucion_normal(0.0, SIGMA);
  uniform_real_distribution<double> distribucion_uniforme_mutacion(0.0, 1.0);

  for (int i = 0; i < cromosoma.caracteristicas.n_cols; ++i) {
    if (Random::get(distribucion_uniforme_mutacion) < 1/cromosoma.caracteristicas.n_cols) {
      cromosoma.caracteristicas(i) += Random::get(distribucion_normal);
      if (cromosoma.caracteristicas(i) < 0) {
        cromosoma.caracteristicas(i) = 0;
      } else if (cromosoma.caracteristicas(i) > 1) {
        cromosoma.caracteristicas(i) = 1;
      }
    }
  }
}

arma::rowvec AGG (const Dataset &datos, int tipoCruce){
  // Inicialización de la población
  Poblacion poblacion = poblacion_inicial(datos);
  Poblacion nueva_poblacion;
  Cromosoma mejor;

  int num_iter = 0;

  while (num_iter < MAX_ITER) {
    // Guardamos el mejor individuo
    //mejor = *poblacion.begin();

    // Seleccionamos la población intermedia (no necesitamos ordenar la población)
    vector<Cromosoma> poblacion_intermedia;

    // Elitismo (guardamos el mejor individuo)
    poblacion_intermedia.push_back(*poblacion.begin());

    // Reservamos espacio para la población intermedia
    if (tipoCruce == 0)
      poblacion_intermedia.resize(NUM_INDIVIDUOS_AGG);
    else if (tipoCruce == 1)
      poblacion_intermedia.resize(NUM_INDIVIDUOS_AGG*2);

    for (int i = 0; i < poblacion_intermedia.size()-1; ++i) {
      uniform_int_distribution<int> distribucion_uniforme(0, NUM_INDIVIDUOS_AGG-1);

      // Selección de los padres
      int padre1_idx = Random::get(distribucion_uniforme);
      int padre2_idx = Random::get(distribucion_uniforme);

      Cromosoma padre1 = *(next(poblacion.begin(), padre1_idx));
      Cromosoma padre2 = *(next(poblacion.begin(), padre2_idx));

      // Cruce (con probabilidad PROB_CRUCE_AGG)
      uniform_real_distribution<double> distribucion_uniforme_cruce(0.0, 1.0);
      Cromosoma hijo1, hijo2;
      if (Random::get(distribucion_uniforme_cruce) < PROB_CRUCE_AGG) {
        if (tipoCruce == 0)
          cruceBLX(padre1, padre2, hijo1, hijo2);
        else if (tipoCruce == 1)
          cruceAritmetico(padre1, padre2, hijo1);
      } else {
        hijo1 = padre1;
        hijo2 = padre2;
      }

      // Mutación (con probabilidad PROB_MUTACION)
      uniform_real_distribution<double> distribucion_uniforme_mutacion(0.0, 1.0);
      if (Random::get(distribucion_uniforme_mutacion) < PROB_MUTACION)
        mutacion(hijo1);
      if (tipoCruce == 0 && Random::get(distribucion_uniforme_mutacion) < PROB_MUTACION)
        mutacion(hijo2);

      // Evaluación de los hijos
      double tasa_clasif_hijo1 = tasa_clas(datos, hijo1.caracteristicas);
      double tasa_red_hijo1 = tasa_red(hijo1.caracteristicas);
      hijo1.fitness = fitness(tasa_clasif_hijo1, tasa_red_hijo1);

      if (tipoCruce == 0){
        double tasa_clasif_hijo2 = tasa_clas(datos, hijo2.caracteristicas);
        double tasa_red_hijo2 = tasa_red(hijo2.caracteristicas);
        hijo2.fitness = fitness(tasa_clasif_hijo2, tasa_red_hijo2);
      }

      // Inserción de los hijos en la población intermedia
      poblacion_intermedia.push_back(hijo1);
      if (tipoCruce == 0)
        poblacion_intermedia.push_back(hijo2);

      // Actualizamos el numero de iteraciones
      ++num_iter;
      if (tipoCruce == 0){
        ++num_iter;
        ++i;
      }
    }

    // Sustituimos la población por la población intermedia
    for (int i = 0; i < poblacion_intermedia.size(); ++i) {
      nueva_poblacion.insert(poblacion_intermedia[i]);
    }
    poblacion = nueva_poblacion;
    
  }

  // Cogemos y devolvemos el mejor individuo
  mejor = *poblacion.begin();
  return mejor.caracteristicas;
  
}

