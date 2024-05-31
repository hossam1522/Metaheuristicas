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
  num_mutaciones = min(num_mutaciones, 3);

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
  else if (algoritmo == 10)
    cout << "*********************** BMB **************************************************" << endl;
  else if (algoritmo == 11)
    cout << "*********************** ES ***************************************************" << endl;
  else if (algoritmo == 12)
    cout << "*********************** ILS **************************************************" << endl;
  else if (algoritmo == 13)
    cout << "*********************** ILS-ES ************************************************" << endl;
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
    else if (algoritmo == 10)
      cout << "************************************ " << nombre_archivo << " (BMB) ***************************************" << endl;
    else if (algoritmo == 11)
      cout << "************************************ " << nombre_archivo << " (ES) ****************************************" << endl;
    else if (algoritmo == 12)
      cout << "************************************ " << nombre_archivo << " (ILS) ***************************************" << endl;
    else if (algoritmo == 13)
      cout << "************************************ " << nombre_archivo << " (ILS-ES) ************************************" << endl;

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

      arma::rowvec w;
      
      tiempo_punto momentoInicio, momentoFin;

      if (algoritmo == 0){
        w.resize(entrenamiento.data.n_cols);
        w.fill(1.0);
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
        //w = busquedaLocal(entrenamiento).first;
        int num = 0;
        w = busquedaLocal(entrenamiento, w, num, CONST_MAX_VECINOS, MAX_ITER).first;
        momentoFin = chrono::high_resolution_clock::now();
      }
      else if (algoritmo == 3){
        momentoInicio = chrono::high_resolution_clock::now();
        Poblacion poblacion;
        // Vector de pesos para el algoritmo AGG-BLX
        w = AGG(entrenamiento, 0, poblacion, MAX_ITER, NUM_INDIVIDUOS_AGG, PROB_CRUCE_AGG, PROB_MUTACION);
        momentoFin = chrono::high_resolution_clock::now();
      }
      else if (algoritmo == 4){
        momentoInicio = chrono::high_resolution_clock::now();
        Poblacion poblacion;
        // Vector de pesos para el algoritmo AGG-CA
        w = AGG(entrenamiento, 1, poblacion, MAX_ITER, NUM_INDIVIDUOS_AGG, PROB_CRUCE_AGG, PROB_MUTACION);
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
      else if (algoritmo == 10){
        momentoInicio = chrono::high_resolution_clock::now();
        // Vector de pesos para el algoritmo BMB
        w = BMB(entrenamiento, SOL_INICIAL, MAX_ITER_BMB);
        momentoFin = chrono::high_resolution_clock::now();
      }
      else if (algoritmo == 11){
        momentoInicio = chrono::high_resolution_clock::now();
        // Vector de pesos para el algoritmo ES
        w = ES(entrenamiento, Solucion(), MAX_ITER_ES).pesos;
        momentoFin = chrono::high_resolution_clock::now();
      }
      else if (algoritmo == 12){
        momentoInicio = chrono::high_resolution_clock::now();
        // Vector de pesos para el algoritmo ILS
        w = ILS(entrenamiento, ITER_ILS, MAX_ITER_ILS);
        momentoFin = chrono::high_resolution_clock::now();
      }
      else if (algoritmo == 13){
        momentoInicio = chrono::high_resolution_clock::now();
        // Vector de pesos para el algoritmo ILS-ES
        w = ILS_ES(entrenamiento, ITER_ILS, MAX_ITER_ILS);
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