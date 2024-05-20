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
  for (size_t i = 0; i < NUM_INDIVIDUOS_AGG; ++i) {
    Cromosoma cromosoma;
    cromosoma.caracteristicas.set_size(datos.data.n_cols);
    for (int j = 0; j < datos.data.n_cols; ++j) {
      cromosoma.caracteristicas(j) = Random::get(distribucion_uniforme);
    }
    double tasa_clasif = tasa_clas(datos, cromosoma.caracteristicas);
    double tasa_reducc = tasa_red(cromosoma.caracteristicas);
    cromosoma.fitness = fitness(tasa_clasif, tasa_reducc);
    poblacion.push_back(cromosoma);
  }

  return poblacion;
}

Cromosoma ordenar_poblacion(const Poblacion &poblacion, int pos) {
  Poblacion aux = poblacion;
  nth_element(aux.begin(), aux.begin()+pos, aux.end(), CompareCromosoma());
  return aux[pos];
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
  for (size_t i = 0; i < 3; ++i) {
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
  
  hijo1.caracteristicas.set_size(padre1.caracteristicas.n_cols);
  hijo2.caracteristicas.set_size(padre1.caracteristicas.n_cols);

  for (size_t i = 0; i < padre1.caracteristicas.n_cols; ++i) {
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

  hijo1.caracteristicas.set_size(padre1.caracteristicas.n_cols);
  hijo2.caracteristicas.set_size(padre1.caracteristicas.n_cols);

  for (size_t i = 0; i < padre1.caracteristicas.n_cols; ++i) {
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

arma::rowvec AGG (const Dataset &datos, int tipoCruce, Poblacion &poblacion, const int max_iter, 
                          const int num_individuos, const double prob_cruce, const double prob_mutacion){
  // Inicialización de la población
  if (poblacion.size() == 0)
    poblacion = poblacion_inicial(datos);

  int num_iter = 0;

  while (num_iter < max_iter) {
    // Seleccionamos la población intermedia (no necesitamos ordenar la población)
    Poblacion poblacion_intermedia(num_individuos);

    // Generamos la población intermedia
    for (size_t i = 0; i < poblacion_intermedia.size(); ++i) {
      poblacion_intermedia[i] = seleccion(poblacion);
    }

    // Cruce (con probabilidad PROB_CRUCE_AGG)
    Cromosoma hijo1, hijo2;
    size_t num_esperado_cruces = ((double)num_individuos/2) * prob_cruce;
    for (size_t i = 0; i < num_esperado_cruces*2; i+=2) {
      if (tipoCruce == 0)
        cruceBLX(datos, poblacion_intermedia[i], poblacion_intermedia[i+1], hijo1, hijo2);
      else if (tipoCruce == 1)
        cruceAritmetico(datos, poblacion_intermedia[i], poblacion_intermedia[i+1], hijo1, hijo2);
    
      poblacion_intermedia[i] = hijo1;
      poblacion_intermedia[i+1] = hijo2;
    }

    // Mutación (con probabilidad PROB_MUTACION)
    size_t num_esperado_mutaciones = num_individuos * poblacion_intermedia[0].caracteristicas.n_cols * prob_mutacion;
    uniform_int_distribution<int> distribucion_cromosoma(0, num_individuos-1);
    uniform_int_distribution<int> distribucion_gen(0, poblacion_intermedia[0].caracteristicas.n_cols-1);
    for (size_t i = 0; i < num_esperado_mutaciones; ++i) {
      size_t idx = Random::get(distribucion_cromosoma);
      size_t gen = Random::get(distribucion_gen);
      mutacion(poblacion_intermedia[idx], gen);
    }
  
    // Para mantener el elitismo, comprobamos si el mejor individuo de la población original
    // (el mejor padre) es mejor que el mejor individuo de la población intermedia (el mejor hijo)
    // En caso afirmativo, significa que el padre no se encuentra en la población intermedia,
    // y como no hay ningún individuo mejor que él, lo sustituimos por el peor hijo
    Cromosoma mejor_padre = ordenar_poblacion(poblacion, 0);
    Cromosoma mejor_hijo = ordenar_poblacion(poblacion_intermedia, 0);
    if (mejor_padre.fitness > mejor_hijo.fitness) {
      Cromosoma peor_hijo = ordenar_poblacion(poblacion_intermedia, num_individuos-1);
      int idx = distance(poblacion_intermedia.begin(), 
                        find(poblacion_intermedia.begin(), poblacion_intermedia.end(), peor_hijo));
      poblacion_intermedia[idx] = mejor_padre;
    }

    poblacion = poblacion_intermedia;
    
    // Actualizamos el numero de iteraciones
    num_iter += poblacion_intermedia.size();
  }

  // Cogemos y devolvemos el mejor individuo
  return ordenar_poblacion(poblacion, 0).caracteristicas;
}


/************************************************************
************************************************************
ALGORITMO GENÉTICO ESTACIONARIO (AGE)
************************************************************
************************************************************/

arma::rowvec AGE (const Dataset &datos, int tipoCruce){
  // Inicialización de la población
  Poblacion poblacion = poblacion_inicial(datos);

  int num_iter = 0;

  while (num_iter < MAX_ITER) {
    // Seleccionamos la población intermedia (no necesitamos ordenar la población)
    Poblacion poblacion_intermedia(NUM_INDIVIDUOS_AGE);

    // Generamos la población intermedia
    for (size_t i = 0; i < poblacion_intermedia.size(); ++i) {
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
    Cromosoma peor_solucion_original = ordenar_poblacion(poblacion, poblacion.size()-1);
    Cromosoma peor_solucion_original2 = ordenar_poblacion(poblacion, poblacion.size()-2);
    
    Poblacion mejores = {poblacion_intermedia[0], poblacion_intermedia[1],
                         peor_solucion_original, peor_solucion_original2};

    // Sacar índices de los 2 sustituidos en la población
    auto it1 = find(poblacion.begin(), poblacion.end(), peor_solucion_original);
    auto it2 = find(poblacion.begin(), poblacion.end(), peor_solucion_original2);

    // Sustituir los 2 peores de la población por los 2 mejores de la población intermedia
    int idx1 = distance(poblacion.begin(), it1);
    int idx2 = distance(poblacion.begin(), it2);

    partial_sort(mejores.begin(), mejores.begin()+2, mejores.end(), CompareCromosoma());
    poblacion[idx1] = mejores[0];
    poblacion[idx2] = mejores[1];

    // Actualizamos el numero de iteraciones
    num_iter += poblacion_intermedia.size();

  }

  // Cogemos y devolvemos el mejor individuo
  return ordenar_poblacion(poblacion, 0).caracteristicas;
}

/************************************************************
************************************************************
ALGORTIMOS MEMÉTICOS (AMs)
************************************************************
************************************************************/

arma::rowvec AM (const Dataset &datos, int tipoAlg){
  // Inicialización de la población
  Poblacion poblacion = poblacion_inicial(datos);

  int num_iter = 0;

  while (num_iter < MAX_ITER) {

    arma::rowvec pesos;

    if (num_iter + FREQ_BUSQUEDA_LOCAL < MAX_ITER) {
      pesos = AGG(datos, 0, poblacion, FREQ_BUSQUEDA_LOCAL, NUM_INDIVIDUOS_AM, PROB_CRUCE_AM, PROB_MUTACION);
      num_iter += FREQ_BUSQUEDA_LOCAL;
    }
    else {
      pesos = AGG(datos, 0, poblacion, MAX_ITER - num_iter, NUM_INDIVIDUOS_AM, PROB_CRUCE_AM, PROB_MUTACION);
      num_iter += MAX_ITER - num_iter;
    }

    // Aplicamos la búsqueda local
    if (num_iter < MAX_ITER) {
      
      if (tipoAlg == 0) {
        for (size_t i = 0; i < poblacion.size(); ++i) {
          Cromosoma cromosoma = busquedaLocal(datos, poblacion[i].caracteristicas, num_iter, CONST_MAX_VECINOS_P2, MAX_ITER);
          poblacion[i] = cromosoma;
        }
      }

      else {
        int num_cromosomas_bl = poblacion.size() * PROB_LS;

        if (tipoAlg == 1){

          uniform_int_distribution<int> distribucion_uniforme(0, poblacion.size()-1);
          vector<int> indices;

          for (size_t i = 0; i < num_cromosomas_bl; ++i) {
            int idx = Random::get(distribucion_uniforme);
            while (find(indices.begin(), indices.end(), idx) != indices.end()) {
              idx = Random::get(distribucion_uniforme);
            }
            indices.push_back(idx);
            
            Cromosoma cromosoma = busquedaLocal(datos, poblacion[idx].caracteristicas, num_iter, CONST_MAX_VECINOS_P2, MAX_ITER);
            poblacion[idx] = cromosoma;
          } 


        }

        else if (tipoAlg == 2) {
          for (size_t i = 0; i < num_cromosomas_bl; ++i) {
            //Cromosoma a aplicar la BL
            Cromosoma cromosoma_orig = ordenar_poblacion(poblacion, i);
            // Aplicar la BL
            Cromosoma cromosoma = busquedaLocal(datos, cromosoma_orig.caracteristicas, num_iter, CONST_MAX_VECINOS_P2, MAX_ITER);
            // Buscar indice del cromosoma en la poblacion
            auto it = find(poblacion.begin(), poblacion.end(), cromosoma_orig);
            int idx = distance(poblacion.begin(), it);
            poblacion[idx] = cromosoma;
          }
        }


      }

    }
  }

  // Cogemos y devolvemos el mejor individuo
  return ordenar_poblacion(poblacion, 0).caracteristicas;
}

arma::rowvec AM_All(const Dataset &datos){
  return AM(datos, 0);
}

arma::rowvec AM_Rand(const Dataset &datos){
  return AM(datos, 1);
}

arma::rowvec AM_Best(const Dataset &datos){
  return AM(datos, 2);
}



