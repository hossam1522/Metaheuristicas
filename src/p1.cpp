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
        double dist = distanciaEuclideaPonderada(ejemplo, datos.data.row(i), pesos);
        if (dist < minDist) {
            minDist = dist;
            categoria = datos.categoria[i];
        }
    }

    return categoria;
}

/************************************************************
************************************************************
FUNCIONES PARA MOSTRAR RESULTADOS
************************************************************
************************************************************/

void printResultados1NN() {

  string nombre_archivo;

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

    // Leer y normalizar dataset
    /* vector< vector<Ejemplo> > dataset;
    vector<Ejemplo> particion_1 = leerFicheroARFF("../Instancias_APC/" + nombre_archivo + "_1.arff");
    vector<Ejemplo> particion_2 = leerFicheroARFF("../Instancias_APC/" + nombre_archivo + "_2.arff");
    vector<Ejemplo> particion_3 = leerFicheroARFF("../Instancias_APC/" + nombre_archivo + "_3.arff");
    vector<Ejemplo> particion_4 = leerFicheroARFF("../Instancias_APC/" + nombre_archivo + "_4.arff");
    vector<Ejemplo> particion_5 = leerFicheroARFF("../Instancias_APC/" + nombre_archivo + "_5.arff");

    dataset.push_back(particion_1);
    dataset.push_back(particion_2);
    dataset.push_back(particion_3);
    dataset.push_back(particion_4);
    dataset.push_back(particion_5); */

    Dataset dataset1 = leerDatos("../Instancias_APC/" + nombre_archivo + "_1.arff");
    Dataset dataset2 = leerDatos("../Instancias_APC/" + nombre_archivo + "_2.arff");
    Dataset dataset3 = leerDatos("../Instancias_APC/" + nombre_archivo + "_3.arff");
    Dataset dataset4 = leerDatos("../Instancias_APC/" + nombre_archivo + "_4.arff");
    Dataset dataset5 = leerDatos("../Instancias_APC/" + nombre_archivo + "_5.arff");

    // Juntar los 5 datasets anteriores en uno solo
    /* Dataset dataset;
    dataset.data = arma::join_cols(dataset1.data, dataset2.data);
    dataset.data = arma::join_cols(dataset.data, dataset3.data);
    dataset.data = arma::join_cols(dataset.data, dataset4.data);
    dataset.data = arma::join_cols(dataset.data, dataset5.data);

    dataset.categoria = arma::join_cols(dataset1.categoria, dataset2.categoria);
    dataset.categoria = arma::join_cols(dataset.categoria, dataset3.categoria);
    dataset.categoria = arma::join_cols(dataset.categoria, dataset4.categoria);
    dataset.categoria = arma::join_cols(dataset.categoria, dataset5.categoria); */

    vector<Dataset> dataset;
    dataset.push_back(dataset1);
    dataset.push_back(dataset2);
    dataset.push_back(dataset3);
    dataset.push_back(dataset4);
    dataset.push_back(dataset5);

    // Normalizar los datos
    //normalizarDatos(dataset);
    for (size_t i = 0; i < dataset.size(); ++i) {
      normalizarDatos(dataset[i]);
    }

    cout << endl << endl;
    cout << "************************************ " << nombre_archivo << " (1-NN) *******************............." << endl;

    cout << endl << "............................................................................................." << endl;
    cout << "::: Particion ::: Tasa de Clasificacion (%) ::: Tasa de Reduccion (%) ::: Fitness ::: Tiempo (ms) :::" << endl;
    cout << "....................................................................................................." << endl;

    // Declaración de los resultados que vamos a acumular para mostrar finalmente un resultado medio
    double tasa_clas_acum = 0.0;
    double tasa_red_acum = 0.0;
    double fit_acum = 0.0;
    double tiempo_acum = 0.0;

    // Ejecución del algoritmo 1-NN en las diferentes particiones
    for(size_t i = 0; i < NUM_PARTICIONES; ++i) {
      // Elegimos en la iteración i como test al archivo i
      //vector<Ejemplo> test = dataset[i];
      Dataset test = dataset[i];

      // El resto de archivos serán para entrenamiento
      vector<Dataset> entrenam;
      for(size_t j = 0; j < NUM_PARTICIONES; ++j)
        if (j != i) {
            Dataset ejemplos_entrenamiento = dataset[j];
            //entrenamiento.insert(entrenamiento.end(), ejemplos_entrenamiento.begin(), ejemplos_entrenamiento.end());
            entrenam.push_back(ejemplos_entrenamiento);
        }

      // Juntamos los datasets de entrenamiento en un único dataset
      Dataset entrenamiento = entrenam[0];
      for(size_t j = 1; j < NUM_PARTICIONES-1; ++j) {
        entrenamiento.data = arma::join_cols(entrenamiento.data, entrenam[j].data);
        //entrenamiento.categoria = arma::join_cols(entrenamiento.categoria, entrenam[j].categoria);
        entrenamiento.categoria.insert(entrenamiento.categoria.end(), entrenam[j].categoria.begin(), entrenam[j].categoria.end());
      }

      //vector<double> w(test[0].num_caracts);
      arma::rowvec w(test.data.n_cols);

      // Vector de pesos para el algoritmo 1-NN
      for(size_t j = 0; j < w.size(); ++j)
        w[j] = 1.0;

      auto momentoInicio = clock();

      // Calculo los valores de las tasas y del fitness, donde se ejecuta el algoritmo 1-NN, y los sumo a las variables acumuladas
      double tasa_clasificacion = tasa_clas(test, entrenamiento, w);
      double tasa_reduccion = tasa_red(w);
      double fit = fitness(tasa_clasificacion, tasa_reduccion);

      auto momentoFin = clock();

      // Calculo el tiempo que le ha tomado al algoritmo ejecutarse
      double tiempo = 1000.0*(momentoFin - momentoInicio)/CLOCKS_PER_SEC;

      tasa_clas_acum += tasa_clasificacion;
      tasa_red_acum += tasa_reduccion;
      fit_acum += fit;
      tiempo_acum += tiempo;

      // Muestro los resultados específicos de cada iteración por pantalla
      cout << fixed << setprecision(2);
      cout << ":::" << setw(6) << (i+1) << setw(8) << ":::" << setw(15) << tasa_clasificacion << setw(15) << ":::" << setw(13) << tasa_reduccion;
      cout << setw(13) << ":::" << setw(7) << fit << setw(5) << "::: " << setw(9) << tiempo << setw(7) << ":::" << endl;
    }

    cout << ":::" << setw(8) << "MEDIA" << setw(6) << ":::" << setw(15) << (tasa_clas_acum/NUM_PARTICIONES) << setw(15) << ":::" << setw(13) << (tasa_red_acum/NUM_PARTICIONES);
    cout << setw(13) << ":::" << setw(7) << (fit_acum/NUM_PARTICIONES) << setw(5) << "::: " << setw(9) << (tiempo_acum/NUM_PARTICIONES) << setw(7) << ":::" << endl;
    cout << "......................................................................................................." << endl << endl;
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

  return (static_cast<double>(aciertos) / test.data.n_rows)*100.0;
}

// Usando Leave-One-Out
double tasa_clas(const Dataset &entrenamiento, const arma::rowvec &pesos){
  size_t aciertos = 0;

  for (size_t i = 0; i < entrenamiento.data.n_rows; ++i) {
    Dataset test;
    test.data = entrenamiento.data.row(i);
    test.categoria = entrenamiento.categoria;

    Dataset entrenar;
    entrenar.data = entrenamiento.data;
    entrenar.categoria = entrenamiento.categoria;

    entrenar.data.shed_row(i);
    //entrenar.categoria.shed_row(i);
    entrenar.categoria.erase(entrenar.categoria.begin() + i);

    string categoria = clasificador1NN(test.data.row(0), entrenar, pesos);
    if (categoria == test.categoria[0]) {
      ++aciertos;
    }
  }

  return (static_cast<double>(aciertos) / entrenamiento.data.n_rows)*100.0;
}


double tasa_red(const arma::rowvec &pesos){
  size_t descartados = 0;

  for (size_t i = 0; i < pesos.size(); ++i) {
    if (pesos(i) < 0.1) {
      ++descartados;
    }
  }

  return (static_cast<double>(descartados) / pesos.size())*100.0;
}


double fitness(const double &tasa_clas, const double &tasa_red){
  return ALPHA*tasa_clas + (1-ALPHA)*tasa_red;
}