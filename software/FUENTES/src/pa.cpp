#include <iostream>
#include "pa.h"

using namespace std;

/************************************************************
************************************************************
FUNCIONES AUXILIARES
************************************************************
************************************************************/

Solucion ordenar_poblacion(const Population &poblacion, int pos){
  Population aux = poblacion;
  nth_element(aux.begin(), aux.begin() + pos, aux.end(), CompararSoluciones());
  return aux[pos];
}

/************************************************************
************************************************************
HUNGER GAMES SEARCH (HGS)
************************************************************
************************************************************/

// Función sech
double sech(double x) {
    return 2.0 / (exp(x) + exp(-x));
}

// N es tamaño de la poblacion
// D es el tamaño de la solucion (numero de pesos) (dimensiones del problema de optimizacion)
// L es VC2
// W1 es weight4
// W2 es weight3
// R es vb
// LH es 100 
// E es VC1[]
// Poblacion[] es X[]

arma::rowvec HGS(const Dataset &datos, const int &tam_pob, const int &maxIter){
  uniform_real_distribution<double> distribucion(0.0, 1.0);
  normal_distribution<double> distribucion_normal(0.0, 1.0);
  
  Solucion mejor_solucion;
  mejor_solucion.fitness = numeric_limits<double>::min();
  Solucion peor_solucion;
  peor_solucion.fitness = numeric_limits<double>::max();
  Population pos_poblacion_temp(tam_pob);
  Population poblacion(tam_pob);
  vector<double> hungry(tam_pob, 0.0);
  vector<double> AllFitness(tam_pob);
  vector<double> E(tam_pob, 1.0);

  //arma::mat W1 = arma::ones<arma::mat>(tam_pob, datos.data.n_cols);
  //arma::mat W2 = arma::ones<arma::mat>(tam_pob, datos.data.n_cols);
  arma::mat W1 = arma::mat(tam_pob, datos.data.n_cols, arma::fill::ones);
  arma::mat W2 = arma::mat(tam_pob, datos.data.n_cols, arma::fill::ones);
  //vector<vector<double>> W1(tam_pob, vector<double>(datos.data.n_cols, 1.0));
  //vector<vector<double>> W2(tam_pob, vector<double>(datos.data.n_cols, 1.0));

  // Inicializar la población
  for (int i = 0; i < tam_pob; ++i) {
    poblacion[i] = solucion_aleatoria(datos);
  }

  int count = 0;

  // Bucle principal
  for (int iter = 1; iter <= maxIter; iter+=tam_pob) {

    double suma_hungry = 0.0;

    sort(poblacion.begin(), poblacion.end(), CompararSoluciones());

    double bestFitness = poblacion[0].fitness;
    double worstFitness = poblacion[tam_pob-1].fitness;

    if (bestFitness > mejor_solucion.fitness) {
      mejor_solucion = poblacion[0];
      count = 0;
    }

    if (worstFitness < peor_solucion.fitness) {
      peor_solucion = poblacion[tam_pob-1];
    }

    for (int i = 0; i < tam_pob; ++i) {
      E[i] = sech(abs(poblacion[i].fitness - mejor_solucion.fitness));
      if (mejor_solucion.fitness == poblacion[i].fitness) {
        hungry[i] = 0;
        if (count < tam_pob) {
          pos_poblacion_temp[count] = poblacion[i];
          ++count;
        }
      } else {
        double temprand = Random::get(distribucion);
        double c = (poblacion[i].fitness - mejor_solucion.fitness) / (peor_solucion.fitness - mejor_solucion.fitness) * temprand * 2.0;
        double b = (c < LH) ? LH * (1 + temprand) : c;
        hungry[i] += max(b, 0.0);
        suma_hungry += hungry[i];
      }
    }

    for (int i = 0; i < tam_pob; ++i) {
      for (int j = 0; j < datos.data.n_cols; ++j) {
        W1(i, j) = (Random::get(distribucion) < L) ? hungry[i] * tam_pob / suma_hungry * Random::get(distribucion) : 1.0;
        W2(i, j) = (1.0 - exp(-abs(hungry[i] - suma_hungry))) * Random::get(distribucion) * 2;
      }
    }

    double shrink = 2 * (1 - iter / (double)maxIter);
    for (int i = 0; i < tam_pob; ++i) {
      if (Random::get(distribucion) < L) {
        for (int j = 1; j < datos.data.n_cols; ++j) {
          poblacion[i].pesos[j] *= (1.0 + Random::get(distribucion_normal));
        }
      } else {
        uniform_int_distribution<int> distribucion_int(0, count-1);
        int A = Random::get(distribucion_int);
        for (int j = 0; j < datos.data.n_cols; ++j) {
          double r = Random::get(distribucion);
          double R = 2 * shrink * r - shrink;
          if (r > E[i]) {
            poblacion[i].pesos[j] = W1(i, j) * pos_poblacion_temp[A].pesos[j] + R * W2(i, j) * abs(pos_poblacion_temp[A].pesos[j] - poblacion[i].pesos[j]);
          } else {
            poblacion[i].pesos[j] = W1(i, j) * pos_poblacion_temp[A].pesos[j] - R * W2(i, j) * abs(pos_poblacion_temp[A].pesos[j] - poblacion[i].pesos[j]);
          }

        }
      }

      poblacion[i].fitness = fitness(tasa_clas(datos, poblacion[i].pesos), tasa_red(poblacion[i].pesos));
    }
  }

  return mejor_solucion.pesos;
}

/************************************************************
************************************************************
TRIBUTE SELECTION ALGORITHM (TSA) - Heurística propia
************************************************************
************************************************************/


Solucion seleccion(const Population& distrito) {
  uniform_int_distribution<int> distribucion_uniforme(0, distrito.size() - 1);
  
  Solucion mejor;
  mejor.fitness = -1;

  vector<int> indices;
  for (size_t i = 0; i < 3; ++i) {
    int idx = Random::get(distribucion_uniforme);
    while (find(indices.begin(), indices.end(), idx) != indices.end()) {
      idx = Random::get(distribucion_uniforme);
    }
    indices.push_back(idx);
    Solucion solucion = distrito[idx];
    if (solucion.fitness > mejor.fitness) {
      mejor = solucion;
    }
  } 

  return mejor;
}

// Cruzamiento de dos individuos
Solucion cruzamiento(const Dataset &datos, Solucion& padre1, const Solucion& padre2) {
    arma::rowvec hijo(padre1.pesos.size());
    for (size_t i = 0; i < padre1.pesos.size(); ++i) {
        double Cmin = std::min(padre1.pesos(i), padre2.pesos(i));
        double Cmax = std::max(padre1.pesos(i), padre2.pesos(i));
        double I = Cmax - Cmin;
        double lower_bound = Cmin - ALPHA_BLX * I;
        double upper_bound = Cmax + ALPHA_BLX * I;
        std::uniform_real_distribution<double> dis(lower_bound, upper_bound);
        hijo(i) = Random::get(dis);

        if (hijo(i) < 0.0) hijo(i) = 0.0;
        if (hijo(i) > 1.0) hijo(i) = 1.0;
    }
    return Solucion(hijo, fitness(tasa_clas(datos, hijo), tasa_red(hijo)));
}

// Mutación de un individuo
void mutacion(Solucion& individuo) {
    normal_distribution<double> distribucion(0.0, SIGMA);
    uniform_int_distribution<int> distribucion_posicion(0, individuo.pesos.size() - 1);

    for (int i = 0; i < individuo.pesos.size(); ++i) {
        if (Random::get(distribucion) < PROB_MUTACION) {
            individuo.pesos(i) += Random::get(distribucion);
            if (individuo.pesos(i) < 0.0) individuo.pesos(i) = 0.0;
            if (individuo.pesos(i) > 1.0) individuo.pesos(i) = 1.0;
        }
    }
}


arma::rowvec TSA(const Dataset &datos, const int &tam_pob, const int &maxIter) {
  uniform_real_distribution<double> distribucion(0.0, 1.0);

  vector<Population> poblacion(NUM_DISTRITOS);
  for (int i = 0; i < NUM_DISTRITOS; ++i) {
    Population distrito;
    for (int j = 0; j < tam_pob; ++j) {
      distrito.push_back(solucion_aleatoria(datos));
    }
    poblacion[i] = distrito;
  }

  for (int iter = 0; iter < maxIter; iter+=tam_pob*NUM_DISTRITOS) {
    vector<Population> nueva_poblacion(NUM_DISTRITOS);
    vector<pair<int, int>> alianzas;
    
    for (int distrito = 0; distrito < NUM_DISTRITOS; ++distrito) {
      for (int i = 0; i < tam_pob; ++i) {
        auto sobreviviente = seleccion(poblacion[distrito]);
        nueva_poblacion[distrito].push_back(sobreviviente);
      }

      if (Random::get(distribucion) < PROB_ALIANZA) {
        uniform_int_distribution<int> dis(0, NUM_DISTRITOS - 1);
        int aliado_distrito = Random::get(dis);
        alianzas.push_back({distrito, aliado_distrito});
      }
    }

    for (const auto& alianza : alianzas) {
      if (Random::get(distribucion) < PROB_TRAICION) {
        int traidor = Random::get(distribucion) < 0.5 ? 0 : 1;
        if (traidor == 0) {
          nueva_poblacion[alianza.first].pop_back();
        } else {
          nueva_poblacion[alianza.second].pop_back();
        }
      }
    }

    Population hijos;
    for (int distrito = 0; distrito < NUM_DISTRITOS; ++distrito) {
      for (int i = 0; i < tam_pob / 2; ++i) {
        uniform_int_distribution<int> indices(0, nueva_poblacion[distrito].size() - 1);
        int idx1 = Random::get(indices);
        int idx2 = Random::get(indices);
        Solucion hijo = cruzamiento(datos, nueva_poblacion[distrito][idx1], nueva_poblacion[distrito][idx2]);
        mutacion(hijo);
        hijos.push_back(hijo);
      }
    }

    uniform_int_distribution<int> eventos(0, 3);
    int evento = Random::get(eventos);
    uniform_int_distribution<int> distritos(0, NUM_DISTRITOS - 1);
    if (evento == 0) {
      int distrito_afectado = Random::get(distritos);
      nueva_poblacion[distrito_afectado].resize(nueva_poblacion[distrito_afectado].size() / 2);
    } else if (evento == 1) {
      int distrito_afectado = Random::get(distritos);
      nueva_poblacion[distrito_afectado].erase(nueva_poblacion[distrito_afectado].begin(), nueva_poblacion[distrito_afectado].begin() + nueva_poblacion[distrito_afectado].size() / 2);
    } else if (evento == 2) {
      int distrito_beneficiado = Random::get(distritos);
      nueva_poblacion[distrito_beneficiado].insert(nueva_poblacion[distrito_beneficiado].end(), hijos.begin(), hijos.begin() + hijos.size() / NUM_DISTRITOS);
    }

    for (int distrito = 0; distrito < NUM_DISTRITOS; ++distrito) {
      while (nueva_poblacion[distrito].size() < tam_pob) {
        uniform_int_distribution<int> indice(0, hijos.size() - 1);
        int hijo_idx = Random::get(indice);
        nueva_poblacion[distrito].push_back(hijos[hijo_idx]);
      }
      poblacion[distrito] = nueva_poblacion[distrito];
    }
  }

  Solucion mejor_solucion;
  mejor_solucion.fitness = numeric_limits<double>::min();
  for (int distrito = 0; distrito < NUM_DISTRITOS; ++distrito) {
    for (const auto& individuo : poblacion[distrito]) {
      if (individuo.fitness > mejor_solucion.fitness) {
        mejor_solucion = individuo;
      }
    }
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
  else if (algoritmo == 14)
    cout << "*********************** HGS **************************************************" << endl;
  else if (algoritmo == 15)
    cout << "*********************** TSA **************************************************" << endl;
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
    else if (algoritmo == 14)
      cout << "************************************ " << nombre_archivo << " (HGS) ***************************************" << endl;
    else if (algoritmo == 15)
      cout << "************************************ " << nombre_archivo << " (TSA) ***************************************" << endl;

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
      else if (algoritmo == 14){
        momentoInicio = chrono::high_resolution_clock::now();
        // Vector de pesos para el algoritmo HGS
        w = HGS(entrenamiento, NUM_INDIVIDUOS_HGS, MAX_ITER_HGS);
        momentoFin = chrono::high_resolution_clock::now();
      }
      else if (algoritmo == 15){
        momentoInicio = chrono::high_resolution_clock::now();
        // Vector de pesos para el algoritmo TSA
        w = TSA(entrenamiento, NUM_INDIVIDUOS_TSA, MAX_ITER_TSA);
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