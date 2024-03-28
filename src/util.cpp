/**
 * @file util.cpp
 * @author Hossam El Amraoui Leghzali
 * @brief 
 * Asignatura: Metahuristicas
 * Archivo con funciones de utilidad
 * 
 */

#include "util.h"

using namespace std;

/************************************************************
************************************************************
FUNCIONES DE LECTURA Y NORMALIZACION DE DATOS
************************************************************
************************************************************/

Dataset leerDatos(string nombre_archivo) {
  Dataset dataset;
  ifstream archivo(nombre_archivo);
  string linea;

  // Variables para almacenar los datos leídos
  vector<vector<string>> datos;
  vector<string> categorias;

  // Flags para indicar en qué parte del archivo nos encontramos
  bool leyendoDatos = false;

  while (getline(archivo, linea)) {
    // Saltar líneas vacías y comentarios
    if (linea.empty() || linea[0] == '%')
      continue;

    // Buscar la cabecera de datos
    if (linea.find("@data") != string::npos) {
      leyendoDatos = true;
      continue;
    }

    if (leyendoDatos) {
      // Almacenar los datos
      istringstream ss(linea);
      vector<string> fila;
      string valor;
      while (getline(ss, valor, ',')) {
        fila.push_back(valor);
      }
      if (!fila.empty()) {
        // Guardar el último valor como categoría
        categorias.push_back(fila.back());
        fila.pop_back();
        datos.push_back(fila);
      }
    }

  }
  
  // Convertir los datos a una matriz de Armadillo
  size_t numFilas = datos.size();
  size_t numColumnas = datos[0].size();
  dataset.data.set_size(numFilas, numColumnas);
  for (size_t i = 0; i < numFilas; ++i) {
    for (size_t j = 0; j < numColumnas; ++j) {
      // Convertir los valores numéricos a double
      dataset.data(i, j) = stod(datos[i][j]);
    }
  }

  dataset.categoria = categorias;

  return dataset;
}

void normalizarDatos(vector<Dataset> &datasets) {
  // Hacerlo con min y max de cada columna pero teniendo en cuenta todos los datasets
  for (size_t i=0; i<datasets[0].data.n_cols; ++i) {
    double min = datasets[0].data.col(i).min();
    double max = datasets[0].data.col(i).max();
    for (size_t j=1; j<datasets.size(); ++j) {
      double minAux = datasets[j].data.col(i).min();
      double maxAux = datasets[j].data.col(i).max();
      if (minAux < min) min = minAux;
      if (maxAux > max) max = maxAux;
    }
    for (size_t j=0; j<datasets.size(); ++j) {
      datasets[j].data.col(i) = (datasets[j].data.col(i) - min) / (max - min);
    }
  }
}

/************************************************************
************************************************************
FUNCIONES PARA CALCUAR DISTANCIAS
************************************************************
************************************************************/

double distanciaEuclidea(const arma::rowvec &x, const arma::rowvec &y) {
  return sqrt(arma::accu(arma::pow(x - y, 2)));
}

double distanciaEuclideaPonderada(const arma::rowvec &x, const arma::rowvec &y, const arma::rowvec &pesos) {
  /*
  En este caso específico, arma::pow(x - y, 2) calcula la diferencia entre los elementos de x e y 
  y luego eleva al cuadrado cada elemento. Después, % pesos multiplica cada elemento elevado al 
  cuadrado por el peso correspondiente en el vector de pesos.

  Cuando se utiliza en el contexto de Armadillo, el operador % representa la multiplicación
  elemento por elemento
  */
  // Hacerlo solo en caso de que el peso sea mayor que 0.1
  arma::rowvec pesosFiltrados = pesos;
  pesosFiltrados.for_each([](double &valor) { valor = valor > 0.1 ? valor : 0; });
  return sqrt(arma::accu(arma::pow(x - y, 2) % pesosFiltrados));
}