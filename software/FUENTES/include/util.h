/**
 * @file util.h
 * @author Hossam El Amraoui Leghzali
 * @brief 
 * Asignatura: Metahuristicas
 * Archivo de cabecera con funciones de utilidad
 */

#ifndef OPENMP_VARIABLE
#define OPENMP_VARIABLE

extern bool openmp;

#endif

#ifndef UTIL_H
#define UTIL_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <armadillo>
#include <vector>
#include <string>
#include <chrono>

// Definición de alias para el tipo de dato de los puntos
typedef std::chrono::high_resolution_clock::time_point tiempo_punto;

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
const double ALPHA = 0.75;


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
 * Función para calcular la distancia euclídea entre dos puntos de
 * forma ponderada usando los pesos de las características
 * 
 * @param x Punto 1
 * @param y Punto 2
 * @param pesos Pesos para las características
 * @return double Distancia euclídea
 */
double distanciaEuclidea(const arma::rowvec &x, const arma::rowvec &y, const arma::rowvec &pesos);

/************************************************************
************************************************************
CLASIFICADOR 1-NN
************************************************************
************************************************************/

/**
 * @brief 
 * Función para clasificar un ejemplo con el clasificador 1-NN
 * 
 * @param ejemplo Ejemplo a clasificar
 * @param datos Conjunto de datos de entrenamiento
 * @param pesos Pesos para las características
 * @return std::string  Categoría a la que pertenece el ejemplo
 */
std::string clasificador1NN(const arma::rowvec &ejemplo, const Dataset &datos,
                            const arma::rowvec &pesos);

/**
 * @brief
 * Función para clasificar un ejemplo con el clasificador 1-NN usando la técnica de Leave-One-Out
 * 
 * @param ejemplo Ejemplo a clasificar
 * @param datos Conjunto de datos de entrenamiento
 * @param pesos Pesos para las características
 * @return std::string Categoría a la que pertenece el ejemplo
 */
std::string clasificador1NN(const int ejemplo, const Dataset &datos,
                            const arma::rowvec &pesos);


/************************************************************
************************************************************
FUNCIONES DE EVALUACIÓN
************************************************************
************************************************************/

/**
 * @brief 
 * Función para calcular la tasa de clasificación de un conjunto de datos
 * 
 * @param test Conjunto de datos de test
 * @param entrenamiento Conjunto de datos de entrenamiento
 * @param pesos Pesos para las características
 * @return double Tasa de clasificación
 */
double tasa_clas(const Dataset &test, const Dataset &entrenamiento, const arma::rowvec &pesos);

/**
 * @brief 
 * Función para calcular la tasa de clasificación de un conjunto de datos
 * usando la técnica de Leave-One-Out
 * 
 * @param entrenamiento Conjunto de datos de entrenamiento
 * @param pesos Pesos para las características
 * @return double Tasa de clasificación
 */
double tasa_clas(const Dataset &entrenamiento, const arma::rowvec &pesos);


/**
 * @brief 
 * Función para calcular la tasa de reducción de un conjunto de datos
 * 
 * @param pesos Pesos para las características
 * @return double Tasa de reducción
 */
double tasa_red(const arma::rowvec &pesos);

/**
 * @brief 
 * Función para calcular la función objetivo
 * 
 * @param tasa_clas Tasa de clasificación
 * @param tasa_red Tasa de reducción
 * @return double Valor de la función objetivo
 */
double fitness(const double tasa_clas, const double tasa_red);


#endif