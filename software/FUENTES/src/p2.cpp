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
    poblacion.push_back(cromosoma);
  }
  return poblacion;
}

Poblacion_ordenada ordenar_poblacion(const Poblacion &poblacion) {
  Poblacion_ordenada poblacion_ordenada;
  for (const Cromosoma &cromosoma : poblacion) {
    poblacion_ordenada.insert(cromosoma);
  }
  return poblacion_ordenada;
}

/************************************************************
************************************************************
ALGORITMO GÉNETICO GENERACIONAL (AGG)
************************************************************
************************************************************/

Cromosoma seleccion(const Poblacion &poblacion) {
  uniform_int_distribution<int> distribucion_uniforme(0, poblacion.size()-1);
  Cromosoma mejor;
  mejor.fitness = -1;

  vector<int> indices;
  for (int i = 0; i < 3; ++i) {
    int idx = Random::get(distribucion_uniforme);
    while (find(indices.begin(), indices.end(), idx) != indices.end()) {
      idx = Random::get(distribucion_uniforme);
    }
    indices.push_back(idx);
    Cromosoma cromosoma = poblacion[idx];
    if (cromosoma.fitness > mejor.fitness) {
      mejor = cromosoma;
    }
  } 

  return mejor;
}

void cruceBLX(const Dataset &datos,const Cromosoma &padre1, const Cromosoma &padre2, Cromosoma &hijo1, Cromosoma &hijo2) {
  
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

  // Evaluación de los hijos
  double tasa_clasif_hijo1 = tasa_clas(datos, hijo1.caracteristicas);
  double tasa_red_hijo1 = tasa_red(hijo1.caracteristicas);
  hijo1.fitness = fitness(tasa_clasif_hijo1, tasa_red_hijo1);

  double tasa_clasif_hijo2 = tasa_clas(datos, hijo2.caracteristicas);
  double tasa_red_hijo2 = tasa_red(hijo2.caracteristicas);
  hijo2.fitness = fitness(tasa_clasif_hijo2, tasa_red_hijo2);
}

void cruceAritmetico(const Dataset &datos, const Cromosoma &padre1, const Cromosoma &padre2, Cromosoma &hijo1, Cromosoma &hijo2) {
  uniform_real_distribution<double> distribucion_uniforme(0.0, 1.0);

  for (int i = 0; i < padre1.caracteristicas.n_cols; ++i) {
    double alpha = Random::get(distribucion_uniforme);
    hijo1.caracteristicas(i) = alpha * padre1.caracteristicas(i) + (1 - alpha) * padre2.caracteristicas(i);
    hijo2.caracteristicas(i) = alpha * padre2.caracteristicas(i) + (1 - alpha) * padre1.caracteristicas(i);
  }

  // Evaluación de los hijos
  double tasa_clasif_hijo1 = tasa_clas(datos, hijo1.caracteristicas);
  double tasa_red_hijo1 = tasa_red(hijo1.caracteristicas);
  hijo1.fitness = fitness(tasa_clasif_hijo1, tasa_red_hijo1);

  double tasa_clasif_hijo2 = tasa_clas(datos, hijo2.caracteristicas);
  double tasa_red_hijo2 = tasa_red(hijo2.caracteristicas);
  hijo2.fitness = fitness(tasa_clasif_hijo2, tasa_red_hijo2);
}

void mutacion(Cromosoma &cromosoma, const int gen) {
  normal_distribution<double> distribucion_normal(0.0, SIGMA);
  
  cromosoma.caracteristicas(gen) += Random::get(distribucion_normal);
  if (cromosoma.caracteristicas(gen) < 0) {
    cromosoma.caracteristicas(gen) = 0;
  } else if (cromosoma.caracteristicas(gen) > 1) {
    cromosoma.caracteristicas(gen) = 1;
  }
}

arma::rowvec AGG (const Dataset &datos, int tipoCruce){
  // Inicialización de la población
  Poblacion poblacion = poblacion_inicial(datos);

  int num_iter = poblacion.size();

  while (num_iter < MAX_ITER) {
    // Seleccionamos la población intermedia (no necesitamos ordenar la población)
    Poblacion poblacion_intermedia(NUM_INDIVIDUOS_AGG);

    // Generamos la población intermedia
    for (int i = 0; i < poblacion_intermedia.size(); ++i) {
      poblacion_intermedia[i] = seleccion(poblacion);
    }

    // Cruce (con probabilidad PROB_CRUCE_AGG)
    Cromosoma hijo1, hijo2;
    size_t num_esperado_cruces = NUM_INDIVIDUOS_AGG/2 * PROB_CRUCE_AGG;
    for (size_t i = 0; i < num_esperado_cruces*2; i+=2) {
      if (tipoCruce == 0)
        cruceBLX(datos, poblacion_intermedia[i], poblacion_intermedia[i+1], hijo1, hijo2);
      else if (tipoCruce == 1)
        cruceAritmetico(datos, poblacion_intermedia[i], poblacion_intermedia[i+1], hijo1, hijo2);
    
      poblacion_intermedia[i] = hijo1;
      poblacion_intermedia[i+1] = hijo2;
    }

    // Mutación (con probabilidad PROB_MUTACION)
    size_t num_esperado_mutaciones = NUM_INDIVIDUOS_AGG * poblacion_intermedia[0].caracteristicas.n_cols * PROB_MUTACION;
    uniform_int_distribution<int> distribucion_cromosoma(0, NUM_INDIVIDUOS_AGG-1);
    uniform_int_distribution<int> distribucion_gen(0, poblacion_intermedia[0].caracteristicas.n_cols-1);
    for (size_t i = 0; i < num_esperado_mutaciones; ++i) {
      size_t idx = Random::get(distribucion_cromosoma);
      size_t gen = Random::get(distribucion_gen);
      mutacion(poblacion_intermedia[idx], gen);
    }
  
    // Ordenamos la población original y la de hijos
    Poblacion_ordenada poblacion_original = ordenar_poblacion(poblacion_intermedia);
    Poblacion_ordenada hijos = ordenar_poblacion(poblacion_intermedia);
    Cromosoma mejor_solucion_original = *poblacion_original.begin();

    // Si la mejor solucion original no se encuentra en la poblacion de hijos
    if (find(hijos.begin(), hijos.end(), mejor_solucion_original) == hijos.end()) {
      // Si la mejor solucion de los hijos es peor que la mejor solucion original
      if (mejor_solucion_original.fitness > hijos.begin()->fitness) {
        // Buscamos la peor solucion de los hijos en la poblacion intermedia
        auto it = find (poblacion_intermedia.begin(), poblacion_intermedia.end(), *prev(hijos.end()));
        // Sustituimos la peor solucion de los hijos por la mejor solucion original
        int idx = distance(poblacion_intermedia.begin(), it);
        poblacion_intermedia[idx] = mejor_solucion_original;
      }
    }

    // Actualizamos la población
    poblacion = poblacion_intermedia;
    
    // Actualizamos el numero de iteraciones
    num_iter += poblacion_intermedia.size();
  }

  // Cogemos y devolvemos el mejor individuo
  Poblacion_ordenada poblacion_ordenada = ordenar_poblacion(poblacion);
  return poblacion_ordenada.begin()->caracteristicas;
}


/************************************************************
************************************************************
ALGORITMO GENÉTICO ESTACIONARIO (AGE)
************************************************************
************************************************************/

arma::rowvec AGE (const Dataset &datos, int tipoCruce){
  // Inicialización de la población
  Poblacion poblacion = poblacion_inicial(datos);

  int num_iter = poblacion.size();

  while (num_iter < MAX_ITER) {
    // Seleccionamos la población intermedia (no necesitamos ordenar la población)
    Poblacion poblacion_intermedia(NUM_INDIVIDUOS_AGE);

    // Generamos la población intermedia
    for (int i = 0; i < poblacion_intermedia.size(); ++i) {
      poblacion_intermedia[i] = seleccion(poblacion);
    }

    // Cruce (con probabilidad PROB_CRUCE_AGE)
    Cromosoma hijo1, hijo2;
    size_t num_esperado_cruces = NUM_INDIVIDUOS_AGE/2 * PROB_CRUCE_AGE;
    for (size_t i = 0; i < num_esperado_cruces*2; i+=2) {
      if (tipoCruce == 0)
        cruceBLX(datos, poblacion_intermedia[i], poblacion_intermedia[i+1], hijo1, hijo2);
      else if (tipoCruce == 1)
        cruceAritmetico(datos, poblacion_intermedia[i], poblacion_intermedia[i+1], hijo1, hijo2);
    
      poblacion_intermedia[i] = hijo1;
      poblacion_intermedia[i+1] = hijo2;
    }

    // Mutación (con probabilidad PROB_MUTACION)
    size_t num_esperado_mutaciones = NUM_INDIVIDUOS_AGE * poblacion_intermedia[0].caracteristicas.n_cols * PROB_MUTACION;
    uniform_int_distribution<int> distribucion_cromosoma(0, NUM_INDIVIDUOS_AGE-1);
    uniform_int_distribution<int> distribucion_gen(0, poblacion_intermedia[0].caracteristicas.n_cols-1);
    for (size_t i = 0; i < num_esperado_mutaciones; ++i) {
      size_t idx = Random::get(distribucion_cromosoma);
      size_t gen = Random::get(distribucion_gen);
      mutacion(poblacion_intermedia[idx], gen);
    }
  
    // Ordenamos la población original y cogemos a los 2 peores individuos
    Poblacion_ordenada poblacion_original = ordenar_poblacion(poblacion_intermedia);
    Cromosoma peor_solucion_original = *prev(poblacion_original.end());
    Cromosoma peor_solucion_original2 = *prev(prev(poblacion_original.end()));
    
    // De los 2 hijos y los 2 peores de la poblacion original, cogemos los 2 mejores
    Poblacion_ordenada mejores = ordenar_poblacion(poblacion_intermedia);
    mejores.insert(peor_solucion_original);
    mejores.insert(peor_solucion_original2);

    // Sacar índices de los 2 sustituidos en la población
    auto it1 = find(poblacion.begin(), poblacion.end(), peor_solucion_original);
    auto it2 = find(poblacion.begin(), poblacion.end(), peor_solucion_original2);

    // Sustituir los 2 peores de la población por los 2 mejores de la población intermedia
    int idx1 = distance(poblacion.begin(), it1);
    int idx2 = distance(poblacion.begin(), it2);
    poblacion[idx1] = *mejores.begin();
    poblacion[idx2] = *next(mejores.begin());

    // Actualizamos el numero de iteraciones
    num_iter += poblacion_intermedia.size();

  }

  // Cogemos y devolvemos el mejor individuo
  Poblacion_ordenada poblacion_ordenada = ordenar_poblacion(poblacion);
  return poblacion_ordenada.begin()->caracteristicas;
}

/************************************************************
************************************************************
ALGORTIMOS MEMÉTICOS (AMs)
************************************************************
************************************************************/

Cromosoma BL_BI(const Dataset &datos, const Cromosoma &cromosoma, int &iteraciones){
  normal_distribution<double> distribucion_normal(0.0, SIGMA);
  
  vector<int> indices(cromosoma.caracteristicas.n_cols);
  arma::rowvec pesos = cromosoma.caracteristicas;

  for (size_t i = 0; i < pesos.size(); ++i) 
    indices[i] = i;

  Random::shuffle(indices);

  double tasa_clasif = tasa_clas(datos, pesos);
  double tasa_reducc = tasa_red(pesos);
  double mejor_fit = fitness(tasa_clasif, tasa_reducc);

  int num_iteraciones = 0;
  size_t num_vecinos = 0;
  bool mejora = false;
  while ( num_iteraciones + iteraciones < MAX_ITER && num_vecinos < (pesos.n_cols+1)*CONST_MAX_VECINOS_P2){
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

  iteraciones += num_iteraciones;

  return Cromosoma{pesos, mejor_fit};
}

// Probar a sustituir toda la AGG por la misma función
arma::rowvec AM (const Dataset &datos, int tipoAlg){
  // Inicialización de la población
  Poblacion poblacion = poblacion_inicial(datos);

  int num_iter = 0;

  while (num_iter < MAX_ITER) {

    // Primero aplicamos el AGG
    for (int i=0; i<FREQ_BUSQUEDA_LOCAL && num_iter < MAX_ITER; ++i) {

      // Seleccionamos la población intermedia (no necesitamos ordenar la población)
      Poblacion poblacion_intermedia(NUM_INDIVIDUOS_AM);

      // Generamos la población intermedia
      for (int i = 0; i < poblacion_intermedia.size(); ++i) {
        poblacion_intermedia[i] = seleccion(poblacion);
      }

      // Cruce (con probabilidad PROB_CRUCE_AM)
      Cromosoma hijo1, hijo2;
      size_t num_esperado_cruces = NUM_INDIVIDUOS_AM/2 * PROB_CRUCE_AM;
      for (size_t i = 0; i < num_esperado_cruces*2; i+=2) {

        // Mirar cuál proporciona mejores resultados, por ahora usar BLX
        //if (tipoCruce == 0)
        //  cruceBLX(datos, poblacion_intermedia[i], poblacion_intermedia[i+1], hijo1, hijo2);
        //else if (tipoCruce == 1)
        //  cruceAritmetico(datos, poblacion_intermedia[i], poblacion_intermedia[i+1], hijo1, hijo2);
        cruceBLX(datos, poblacion_intermedia[i], poblacion_intermedia[i+1], hijo1, hijo2);

        poblacion_intermedia[i] = hijo1;
        poblacion_intermedia[i+1] = hijo2;
      }

      // Mutación (con probabilidad PROB_MUTACION)
      size_t num_esperado_mutaciones = NUM_INDIVIDUOS_AM * poblacion_intermedia[0].caracteristicas.n_cols * PROB_MUTACION;
      uniform_int_distribution<int> distribucion_cromosoma(0, NUM_INDIVIDUOS_AM-1);
      uniform_int_distribution<int> distribucion_gen(0, poblacion_intermedia[0].caracteristicas.n_cols-1);
      for (size_t i = 0; i < num_esperado_mutaciones; ++i) {
        size_t idx = Random::get(distribucion_cromosoma);
        size_t gen = Random::get(distribucion_gen);
        mutacion(poblacion_intermedia[idx], gen);
      }
    
      // Orden
      Poblacion_ordenada poblacion_original = ordenar_poblacion(poblacion_intermedia);
      Cromosoma mejor_solucion_original = *poblacion_original.begin();

      // Si la mejor solucion original no se encuentra en la poblacion de hijos
      if (find(poblacion_intermedia.begin(), poblacion_intermedia.end(), mejor_solucion_original) == poblacion_intermedia.end()) {
        // Si la mejor solucion de los hijos es peor que la mejor solucion original
        if (mejor_solucion_original.fitness > poblacion_intermedia.begin()->fitness) {
          // Buscamos la peor solucion de los hijos en la poblacion intermedia
          auto it = find (poblacion_intermedia.begin(), poblacion_intermedia.end(), *prev(poblacion_intermedia.end()));
          // Sustituimos la peor solucion de los hijos por la mejor solucion original
          int idx = distance(poblacion_intermedia.begin(), it);
          poblacion_intermedia[idx] = mejor_solucion_original;
        }
      }

      // Actualizamos la población
      poblacion = poblacion_intermedia;

      // Actualizamos el numero de iteraciones
      num_iter += poblacion_intermedia.size();

    }

    // Aplicamos la búsqueda local
    if (num_iter < MAX_ITER) {
      
      if (tipoAlg == 0) {
        for (int i = 0; i < poblacion.size(); ++i) {
          Cromosoma cromosoma = BL_BI(datos, poblacion[i], num_iter);
          poblacion[i] = cromosoma;
        }
      }

      else {
        int num_cromosomas_bl = poblacion.size() * PROB_LS;

        if (tipoAlg == 1){

          uniform_int_distribution<int> distribucion_uniforme(0, poblacion.size()-1);
          vector<int> indices;

          for (int i = 0; i < num_cromosomas_bl; ++i) {
            int idx = Random::get(distribucion_uniforme);
            while (find(indices.begin(), indices.end(), idx) != indices.end()) {
              idx = Random::get(distribucion_uniforme);
            }
            indices.push_back(idx);
            
            Cromosoma cromosoma = BL_BI(datos, poblacion[idx], num_iter);
            poblacion[idx] = cromosoma;
          } 


        }

        else if (tipoAlg == 2) {
          Poblacion_ordenada poblacion_ordenada = ordenar_poblacion(poblacion);
          for (int i = 0; i < num_cromosomas_bl; ++i) {
            Cromosoma cromosoma = BL_BI(datos, *poblacion_ordenada.begin(), num_iter);
            // Buscar indice del cromosoma en la poblacion
            auto it = find(poblacion.begin(), poblacion.end(), *poblacion_ordenada.begin());
            int idx = distance(poblacion.begin(), it);
            poblacion[idx] = cromosoma;
            poblacion_ordenada.erase(poblacion_ordenada.begin());
          }
        }


      }

    }
  }

  // Cogemos y devolvemos el mejor individuo
  Poblacion_ordenada poblacion_ordenada = ordenar_poblacion(poblacion);
  return poblacion_ordenada.begin()->caracteristicas;
}
