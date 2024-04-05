#include <iostream>
#include "p1.h"

using namespace std;


/************************************************************
************************************************************
CLASIFICADOR 1-NN
************************************************************
************************************************************/

string clasificador1NN(const arma::rowvec &ejemplo, const Dataset &datos,
                            const arma::rowvec &pesos){
  double minDist = numeric_limits<double>::max();
  string categoria;

  for (size_t i = 0; i < datos.data.n_rows; ++i) {
    double dist = distanciaEuclidea(ejemplo, datos.data.row(i), pesos);
    if (dist < minDist) {
      minDist = dist;
      categoria = datos.categoria[i];
    }
  }

  return categoria;
}

string clasificador1NN(const int ejemplo, const Dataset &datos,
                            const arma::rowvec &pesos){
  double minDist = numeric_limits<double>::max();
  string categoria;

  for (size_t i = 0; i < datos.data.n_rows; ++i) {
    if ((size_t)ejemplo != i){
      double dist = distanciaEuclidea(datos.data.row(ejemplo), datos.data.row(i), pesos);
      if (dist < minDist) {
        minDist = dist;
        categoria = datos.categoria[i];
      }
    }
  }

  return categoria;
}

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
  while ( num_iteraciones < MAX_ITER && num_vecinos < pesos.n_cols*CONST_MAX_VECINOS){
    //int componente = indices[num_iteraciones % pesos.n_cols];
    int componente = indices[num_vecinos % pesos.n_cols];

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
    cout << "*********************** Greedy Relief (1-NN) *********************************" << endl;
  else if (algoritmo == 2)
    cout << "*********************** Búsqueda Local (1-NN) ********************************" << endl;
  cout << "******************************************************************************" << endl;
  cout << "******************************************************************************" << endl;


  for(int l = 0; l < NUM_DATASETS; ++l) {

    switch(l) {
      case 0:
        nombre_archivo = "breast-cancer";
      break;

      case 1:
        nombre_archivo = "ecoli";
      break;

      case 2:
        nombre_archivo = "parkinsons";
      break;
    }

    Dataset dataset1 = leerDatos("Instancias_APC/" + nombre_archivo + "_1.arff");
    Dataset dataset2 = leerDatos("Instancias_APC/" + nombre_archivo + "_2.arff");
    Dataset dataset3 = leerDatos("Instancias_APC/" + nombre_archivo + "_3.arff");
    Dataset dataset4 = leerDatos("Instancias_APC/" + nombre_archivo + "_4.arff");
    Dataset dataset5 = leerDatos("Instancias_APC/" + nombre_archivo + "_5.arff");

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
      
      clock_t momentoInicio, momentoFin;

      if (algoritmo == 0){
        // Vector de pesos para el algoritmo 1-NN
        for(size_t j = 0; j < w.size(); ++j)
          w[j] = 1.0;
      }
      else if (algoritmo == 1){
        momentoInicio = clock();
        // Vector de pesos para el algoritmo Greedy Relief
        w = greedy(entrenamiento);
        momentoFin = clock();
      }
      else if (algoritmo == 2){
        momentoInicio = clock();
        // Vector de pesos para el algoritmo Búsqueda Local
        w = busquedaLocal(entrenamiento);
        momentoFin = clock();
      }

      total_pesos.push_back(w);

      if (algoritmo==0)
        momentoInicio = clock();

      // Calculo los valores de las tasas y del fitness, donde se ejecuta el algoritmo 1-NN, y los sumo a las variables acumuladas
      double tasa_clasificacion = tasa_clas(test, entrenamiento, w);
      double tasa_reduccion = tasa_red(w);
      double fit = fitness(tasa_clasificacion, tasa_reduccion);

      if (algoritmo==0)
        momentoFin = clock();

      // Calculo el tiempo que le ha tomado al algoritmo ejecutarse
      // y lo muestro en segundos usando notacion cientifica
      double tiempo = (momentoFin - momentoInicio) / (double)CLOCKS_PER_SEC;

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
    if (algoritmo == 2){
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
    }

    total_pesos.clear();
  }

}

/************************************************************
************************************************************
FUNCIONES DE EVALUACIÓN
************************************************************
************************************************************/

double tasa_clas(const Dataset &test, const Dataset &entrenamiento, const arma::rowvec &pesos){
  size_t aciertos = 0;

  for (size_t i = 0; i < test.data.n_rows; ++i) {
    string categoria = clasificador1NN(test.data.row(i), entrenamiento, pesos);
    if (categoria == test.categoria[i]) {
      ++aciertos;
    }
  }

  return static_cast<double>(aciertos) / test.data.n_rows * 100.0;
}

// Usando Leave-One-Out
double tasa_clas(const Dataset &entrenamiento, const arma::rowvec &pesos){
  double aciertos = 0;

  for (size_t i = 0; i < entrenamiento.data.n_rows; ++i) {
    string categoria = clasificador1NN(i, entrenamiento, pesos);
    if (categoria == entrenamiento.categoria[i]) {
      ++aciertos;
    }
  }

  return aciertos / entrenamiento.data.n_rows * 100.0;
}


double tasa_red(const arma::rowvec &pesos){
  double descartados = 0;

  for (size_t i = 0; i < pesos.size(); ++i) {
    if (pesos(i) <= 0.1) {
      ++descartados;
    }
  }

  return descartados / pesos.size()* 100.0;
}


double fitness(const double tasa_clas, const double tasa_red){
  return ALPHA*tasa_clas + (1-ALPHA)*tasa_red;
}