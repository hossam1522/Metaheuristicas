/**
 * @file util.h
 * @author Hossam El Amraoui Leghzali
 * @brief 
 * Asignatura: Metahuristicas
 * Archivo de cabecera con funciones de utilidad
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <armadillo>
#include <vector>
#include <string>

using namespace std;

/************************************************************
************************************************************
CONSTANTES GLOBALES
************************************************************
************************************************************/

// Constante para el número de conjuntos de datos
const int NUM_DATASETS = 3;

// Constante para el número de particiones de cada conjunto de datos
const int NUM_PARTICIONES = 5;

// Porcentaje del uso de total de datos disponibles
const float ALPHA = 0.8;


/************************************************************
************************************************************
ESTRUCTURAS DE DATOS
************************************************************
************************************************************/

/**
 * @brief 
 * Estructura para almacenar los datos de un conjunto
 * 
 */
struct Dataset {
    arma::mat data;
    arma::Row<string> categoria;
};


/************************************************************
************************************************************
FUNCIONES DE LECTURA Y NORMALIZACION DE DATOS
************************************************************
************************************************************/

/**
 * @brief 
 * Función para leer los datos de un fichero
 * 
 * @param nombre_archivo Nombre del fichero
 * @return Dataset Estructura con los datos leídos
 */
Dataset leerDatos(string nombre_archivo);

/**
 * @brief 
 * Función para normalizar los datos de un conjunto
 * 
 * @param dataset Conjunto de datos a normalizar
 * 
 */
void normalizarDatos(Dataset &dataset);


/************************************************************
************************************************************
FUNCIONES PARA CALCUAR DISTANCIAS
************************************************************
************************************************************/

/**
 * @brief 
 * Función para calcular la distancia euclídea entre dos puntos
 * 
 * @param x Punto 1
 * @param y Punto 2
 * @return double Distancia euclídea
 */
double distanciaEuclidea(const arma::rowvec &x, const arma::rowvec &y);