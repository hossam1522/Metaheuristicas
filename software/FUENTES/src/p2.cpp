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

arma::rowvec AGG (const Dataset &datos, int tipoCruce){
  // Inicialización de la población
  Poblacion poblacion = poblacion_inicial(datos);

  int num_iter = poblacion.size();

  while (num_iter < MAX_ITER) {
    // Seleccionamos la población intermedia (no necesitamos ordenar la población)
    Poblacion poblacion_intermedia(NUM_INDIVIDUOS_AGG);

    // Generamos la población intermedia
    for (size_t i = 0; i < poblacion_intermedia.size(); ++i) {
      poblacion_intermedia[i] = seleccion(poblacion);
    }

    // Cruce (con probabilidad PROB_CRUCE_AGG)
    Cromosoma hijo1, hijo2;
    size_t num_esperado_cruces = ((double)NUM_INDIVIDUOS_AGG/2) * PROB_CRUCE_AGG;
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
  
    // Cogemos al mejor padre
    Cromosoma mejor_padre = ordenar_poblacion(poblacion, 0);
    // Si el mejor padre no se encuentra en la poblacion de hijos
    if (find (poblacion_intermedia.begin(), poblacion_intermedia.end(), mejor_padre) == poblacion_intermedia.end()) {
      Cromosoma mejor_hijo = ordenar_poblacion(poblacion_intermedia, 0);
      // Si el mejor hijo es peor que el mejor padre
      if (mejor_padre.fitness > mejor_hijo.fitness) {
        // Sustituimos al peor hijo por el mejor padre 
        Cromosoma peor_hijo = ordenar_poblacion(poblacion_intermedia, NUM_INDIVIDUOS_AGG-1);
        int idx = distance(poblacion_intermedia.begin(), 
                           find(poblacion_intermedia.begin(), poblacion_intermedia.end(), peor_hijo));
        poblacion_intermedia[idx] = mejor_padre;
      }
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

  int num_iter = poblacion.size();

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
    /* Poblacion_ordenada poblacion_original = ordenar_poblacion(poblacion);
    Cromosoma peor_solucion_original = *prev(poblacion_original.end());
    Cromosoma peor_solucion_original2 = *prev(prev(poblacion_original.end())); */
    Cromosoma peor_solucion_original = ordenar_poblacion(poblacion, poblacion.size()-1);
    Cromosoma peor_solucion_original2 = ordenar_poblacion(poblacion, poblacion.size()-2);
    
    // De los 2 hijos y los 2 peores de la poblacion original, cogemos los 2 mejores
    /* Poblacion_ordenada mejores = ordenar_poblacion(poblacion_intermedia);
    mejores.insert(peor_solucion_original);
    mejores.insert(peor_solucion_original2); */
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
  /* Poblacion_ordenada poblacion_ordenada = ordenar_poblacion(poblacion);
  return poblacion_ordenada.begin()->caracteristicas; */
  return ordenar_poblacion(poblacion, 0).caracteristicas;
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
      for (size_t i = 0; i < poblacion_intermedia.size(); ++i) {
        poblacion_intermedia[i] = seleccion(poblacion);
      }

      // Cruce (con probabilidad PROB_CRUCE_AM)
      Cromosoma hijo1, hijo2;
      size_t num_esperado_cruces = NUM_INDIVIDUOS_AM/2 * PROB_CRUCE_AM;
      for (size_t i = 0; i < num_esperado_cruces*2; i+=2) {
        // BLX porque es el que mejores resultados da
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
    
      // Cogemos al mejor padre
      Cromosoma mejor_padre = ordenar_poblacion(poblacion, 0);
      // Si el mejor padre no se encuentra en la poblacion de hijos
      if (find (poblacion_intermedia.begin(), poblacion_intermedia.end(), mejor_padre) == poblacion_intermedia.end()) {
        Cromosoma mejor_hijo = ordenar_poblacion(poblacion_intermedia, 0);
        // Si el mejor hijo es peor que el mejor padre
        if (mejor_padre.fitness > mejor_hijo.fitness) {
          // Sustituimos al peor hijo por el mejor padre 
          Cromosoma peor_hijo = ordenar_poblacion(poblacion_intermedia, NUM_INDIVIDUOS_AGG-1);
          int idx = distance(poblacion_intermedia.begin(), 
                            find(poblacion_intermedia.begin(), poblacion_intermedia.end(), peor_hijo));
          poblacion_intermedia[idx] = mejor_padre;
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
        for (size_t i = 0; i < poblacion.size(); ++i) {
          Cromosoma cromosoma = BL_BI(datos, poblacion[i], num_iter);
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
            
            Cromosoma cromosoma = BL_BI(datos, poblacion[idx], num_iter);
            poblacion[idx] = cromosoma;
          } 


        }

        else if (tipoAlg == 2) {
          for (size_t i = 0; i < num_cromosomas_bl; ++i) {
            //Cromosoma a aplicar la BL
            Cromosoma cromosoma_orig = ordenar_poblacion(poblacion, i);
            // Aplicar la BL
            Cromosoma cromosoma = BL_BI(datos, cromosoma_orig, num_iter);
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

/************************************************************
************************************************************
FUNCIONES PARA MOSTRAR RESULTADOS
************************************************************
************************************************************/

void printResultados(int algoritmo) {

  string nombre_archivo;
  vector<arma::rowvec> total_pesos;

  cout << "******************************************************************************" << endl;
  cout << "******************************************************************************" << endl;
  if (algoritmo == 0)
    cout << "*********************** 1-NN sin ponderaciones *******************************" << endl;
  else if (algoritmo == 1)
    cout << "*********************** Greedy Relief ****************************************" << endl;
  else if (algoritmo == 2)
    cout << "*********************** Búsqueda Local ***************************************" << endl;
  else if (algoritmo == 3)
    cout << "*********************** AGG-BLX **********************************************" << endl;
  else if (algoritmo == 4)
    cout << "*********************** AGG-CA ***********************************************" << endl;
  else if (algoritmo == 5)
    cout << "*********************** AGE-BLX **********************************************" << endl;
  else if (algoritmo == 6)
    cout << "*********************** AGE-CA ***********************************************" << endl;
  else if (algoritmo == 7)
    cout << "*********************** AM-(10,1.0) *****************************************" << endl;
  else if (algoritmo == 8)
    cout << "*********************** AM-(10,0.1) *****************************************" << endl;
  else if (algoritmo == 9)
    cout << "*********************** AM-(10,0.1mej) **************************************" << endl;
  cout << "******************************************************************************" << endl;
  cout << "******************************************************************************" << endl;


  for(int l = 0; l < NUM_DATASETS; ++l) {

    switch(l) {
      case 0:
        nombre_archivo = "ecoli";
      break;

      case 1:
        nombre_archivo = "parkinsons";
      break;

      case 2:
        nombre_archivo = "breast-cancer";
      break;
    }


    Dataset dataset1 = leerDatos("../BIN/Instancias_APC/" + nombre_archivo + "_1.arff");
    Dataset dataset2 = leerDatos("../BIN/Instancias_APC/" + nombre_archivo + "_2.arff");
    Dataset dataset3 = leerDatos("../BIN/Instancias_APC/" + nombre_archivo + "_3.arff");
    Dataset dataset4 = leerDatos("../BIN/Instancias_APC/" + nombre_archivo + "_4.arff");
    Dataset dataset5 = leerDatos("../BIN/Instancias_APC/" + nombre_archivo + "_5.arff");

    vector<Dataset> dataset;
    dataset.push_back(dataset1);
    dataset.push_back(dataset2);
    dataset.push_back(dataset3);
    dataset.push_back(dataset4);
    dataset.push_back(dataset5);

    // Normalizar los datos
    normalizarDatos(dataset);

    cout << endl << endl;
    if (algoritmo == 0)
      cout << "************************************ " << nombre_archivo << " (1-NN) ************************************" << endl;
    else if (algoritmo == 1)
      cout << "************************************ " << nombre_archivo << " (Greedy Relief) ************************************" << endl;
    else if (algoritmo == 2)
      cout << "************************************ " << nombre_archivo << " (Búsqueda Local) **********************************" << endl;
    else if (algoritmo == 3)
      cout << "************************************ " << nombre_archivo << " (AGG-BLX) ************************************" << endl;
    else if (algoritmo == 4)
      cout << "************************************ " << nombre_archivo << " (AGG-CA) *************************************" << endl;
    else if (algoritmo == 5)
      cout << "************************************ " << nombre_archivo << " (AGE-BLX) ************************************" << endl;
    else if (algoritmo == 6)
      cout << "************************************ " << nombre_archivo << " (AGE-CA) *************************************" << endl;
    else if (algoritmo == 7)
      cout << "************************************ " << nombre_archivo << " (AM-(10,1.0)) ********************************" << endl;
    else if (algoritmo == 8)
      cout << "************************************ " << nombre_archivo << " (AM-(10,0.1)) ********************************" << endl;
    else if (algoritmo == 9)
      cout << "************************************ " << nombre_archivo << " (AM-(10,0.1mej)) *****************************" << endl;

    cout << endl << "....................................................................................................." << endl;
    cout << "::: Particion ::: Tasa de Clasificacion (%) ::: Tasa de Reduccion (%) ::: Fitness ::: Tiempo (s)  :::" << endl;
    cout << "....................................................................................................." << endl;


    // Declaración de los resultados que vamos a acumular para mostrar finalmente un resultado medio
    double tasa_clas_acum = 0.0;
    double tasa_red_acum = 0.0;
    double fit_acum = 0.0;
    double tiempo_acum = 0.0;

    // Ejecución del algoritmo 1-NN en las diferentes particiones
    for(size_t i = 0; i < NUM_PARTICIONES; ++i) {
      // Elegimos en la iteración i como test al archivo l
      Dataset test = dataset[i];

      // El resto de archivos serán para entrenamiento
      vector<Dataset> entrenam;
      for(size_t j = 0; j < NUM_PARTICIONES; ++j)
        if (j != i) {
          Dataset ejemplos_entrenamiento = dataset[j];
          entrenam.push_back(ejemplos_entrenamiento);
        }

      // Juntamos los datasets de entrenamiento en un único dataset
      Dataset entrenamiento;
      for(size_t j = 0; j < NUM_PARTICIONES-1; ++j) {
        entrenamiento.data = arma::join_vert(entrenamiento.data, entrenam[j].data);
        entrenamiento.categoria.insert(entrenamiento.categoria.end(), entrenam[j].categoria.begin(), entrenam[j].categoria.end());
      }

      arma::rowvec w(test.data.n_cols);
      
      tiempo_punto momentoInicio, momentoFin;

      if (algoritmo == 0){
        // Vector de pesos para el algoritmo 1-NN
        for(size_t j = 0; j < w.size(); ++j)
          w[j] = 1.0;
      }
      else if (algoritmo == 1){
        momentoInicio = chrono::high_resolution_clock::now();
        // Vector de pesos para el algoritmo Greedy Relief
        w = greedy(entrenamiento);
        momentoFin = chrono::high_resolution_clock::now();
      }
      else if (algoritmo == 2){
        momentoInicio = chrono::high_resolution_clock::now();
        // Vector de pesos para el algoritmo Búsqueda Local
        w = busquedaLocal(entrenamiento);
        momentoFin = chrono::high_resolution_clock::now();
      }
      else if (algoritmo == 3){
        momentoInicio = chrono::high_resolution_clock::now();
        // Vector de pesos para el algoritmo AGG-BLX
        w = AGG(entrenamiento, 0);
        momentoFin = chrono::high_resolution_clock::now();
      }
      else if (algoritmo == 4){
        momentoInicio = chrono::high_resolution_clock::now();
        // Vector de pesos para el algoritmo AGG-CA
        w = AGG(entrenamiento, 1);
        momentoFin = chrono::high_resolution_clock::now();
      }
      else if (algoritmo == 5){
        momentoInicio = chrono::high_resolution_clock::now();
        // Vector de pesos para el algoritmo AGE-BLX
        w = AGE(entrenamiento, 0);
        momentoFin = chrono::high_resolution_clock::now();
      }
      else if (algoritmo == 6){
        momentoInicio = chrono::high_resolution_clock::now();
        // Vector de pesos para el algoritmo AGE-CA
        w = AGE(entrenamiento, 1);
        momentoFin = chrono::high_resolution_clock::now();
      }
      else if (algoritmo == 7){
        momentoInicio = chrono::high_resolution_clock::now();
        // Vector de pesos para el algoritmo AM-(10,1.0)
        w = AM_All(entrenamiento);
        momentoFin = chrono::high_resolution_clock::now();
      }
      else if (algoritmo == 8){
        momentoInicio = chrono::high_resolution_clock::now();
        // Vector de pesos para el algoritmo AM-(10,0.1)
        w = AM_Rand(entrenamiento);
        momentoFin = chrono::high_resolution_clock::now();
      }
      else if (algoritmo == 9){
        momentoInicio = chrono::high_resolution_clock::now();
        // Vector de pesos para el algoritmo AM-(10,0.1mej)
        w = AM_Best(entrenamiento);
        momentoFin = chrono::high_resolution_clock::now();
      }

      total_pesos.push_back(w);

      if (algoritmo==0)
        momentoInicio = chrono::high_resolution_clock::now();

      // Calculo los valores de las tasas y del fitness, donde se ejecuta el algoritmo 1-NN, y los sumo a las variables acumuladas
      double tasa_clasificacion = tasa_clas(test, entrenamiento, w);
      double tasa_reduccion = tasa_red(w);
      double fit = fitness(tasa_clasificacion, tasa_reduccion);

      if (algoritmo==0)
        momentoFin = chrono::high_resolution_clock::now();

      // Calculo el tiempo que le ha tomado al algoritmo ejecutarse
      // y lo muestro en segundos usando notacion cientifica
      chrono::duration<double> tiempo_transcurrido = momentoFin - momentoInicio;
      double tiempo = tiempo_transcurrido.count();

      tasa_clas_acum += tasa_clasificacion;
      tasa_red_acum += tasa_reduccion;
      fit_acum += fit;
      tiempo_acum += tiempo;

      // Muestro los resultados específicos de cada iteración por pantalla
      cout << fixed << setprecision(2);
      cout << ":::" << setw(6) << (i+1) << setw(8) << ":::" << setw(15) << tasa_clasificacion << setw(15) << ":::" << setw(13) << tasa_reduccion;
      cout << setw(13) << ":::" << setw(7) << fit << setw(5) << "::: " << setw(9) << scientific << tiempo << fixed << setw(7) << ":::" << endl;
    }

    cout << ":::" << setw(8) << "MEDIA" << setw(6) << ":::" << setw(15) << (tasa_clas_acum/NUM_PARTICIONES) << setw(15) << ":::" << setw(13) << (tasa_red_acum/NUM_PARTICIONES);
    cout << setw(13) << ":::" << setw(7) << (fit_acum/NUM_PARTICIONES) << setw(5) << "::: " << setw(9) << scientific << (tiempo_acum/NUM_PARTICIONES) << fixed << setw(7) << ":::" << endl;
    cout << "....................................................................................................." << endl << endl;
  
    // Mostrar los pesos de cada particion separados por comas
    /* if (algoritmo == 2){
      cout << "Pesos obtenidos en cada partición:" << endl;
      for (size_t i = 0; i < total_pesos.size(); ++i) {
        cout << "Partición " << i+1 << ": ";
        for (size_t j = 0; j < total_pesos[i].size(); ++j) {
          cout << scientific << total_pesos[i](j) << fixed;
          if (j != total_pesos[i].size()-1) {
            cout << ", ";
          }
        }
        cout << endl;
      }
    } */

    total_pesos.clear();
  }

}

