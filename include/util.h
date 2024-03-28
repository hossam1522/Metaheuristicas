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

/************************************************************
************************************************************
CONSTANTES GLOBALES
************************************************************
************************************************************/

// Constante para el número de conjuntos de datos
const int NUM_DATASETS = 3;

// Constante para el número de particiones de cada conjunto de datos
const int NUM_PARTICIONES = 5;

// Ponderación de la importancia entre el acierto y la reducción de la solución encontrada
const float ALPHA = 0.75;


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
    std::vector<std::string> categoria;
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
Dataset leerDatos(std::string nombre_archivo);

/**
 * @brief 
 * Función para normalizar los datos de un conjunto
 * 
 * @param dataset Conjunto de datos a normalizar
 * 
 */
void normalizarDatos(std::vector<Dataset> &datasets);


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

/**
 * @brief 
 * Función para calcular la distancia euclídea entre dos puntos con pesos
 * 
 * @param x Punto 1
 * @param y Punto 2
 * @param pesos Pesos para las características
 * @return double Distancia euclídea
 */
double distanciaEuclideaPonderada(const arma::rowvec &x, const arma::rowvec &y, const arma::rowvec &pesos);