/**
 * @file p2.h
 * @author Hossam El Amraoui Leghzali
 * @brief 
 * Asignatura: Metahuristicas
 * 
 */

#ifndef P2_H
#define P2_H

#include <algorithm>
#include "p1.h"

/************************************************************
************************************************************
CONSTANTES GLOBALES
************************************************************
************************************************************/

// Constante a multiplicar por el número de características
// para obtener el número máximo de vecinos a considerar en la búsqueda local
const int CONST_MAX_VECINOS_P2 = 2;

// Parámetro alpha para el operador BLX
const double ALPHA_BLX = 0.3;

// Probabilidad de cruce en el AGG
const double PROB_CRUCE_AGG = 0.68;

// Probabilidad de cruce en el AGE
const double PROB_CRUCE_AGE = 1.0;

// Probabilidad de mutación tanto en el AGG como en el AM por individuo
const double PROB_MUTACION = 0.08;

// Número de individuos en la población para el AGG
const int NUM_INDIVIDUOS_AGG = 50;

// Número de individuos en la población para el AGE
const int NUM_INDIVIDUOS_AGE = 2;

// Número de individuos en la población para el AM
const int NUM_INDIVIDUOS_AM = 50;

// Probabilidad de cruce en el AM
const double PROB_CRUCE_AM = 0.7;

// Probabilidad de mutación por cromosoma en el AM
const double PROB_MUTACION_AM = 0.7;

// Frecuencia de aplicacion de la búsqueda local en el AM
const int FREQ_BUSQUEDA_LOCAL = 10;

// Probabilidad de selección de un subconjunto de cromosomas para la búsqueda local en el AM
const double PROB_LS = 0.1;

/************************************************************
************************************************************
ESTRUCTURAS DE DATOS
************************************************************
************************************************************/

/**
 * @brief 
 * Estructura para almacenar los datos de un cromosoma
 * 
 */
/* struct Cromosoma {
    arma::rowvec caracteristicas;
    double fitness;

    bool operator == (const Cromosoma &c) const {
        return arma::approx_equal(caracteristicas, c.caracteristicas, "absdiff", 1e-5);
    }
}; */
struct Cromosoma {
    arma::rowvec caracteristicas;
    double fitness;

    // Constructor de copia
    //Cromosoma(const Cromosoma& otro) : caracteristicas(otro.caracteristicas), fitness(otro.fitness) {}

    // Operador de asignación
    /* Cromosoma& operator=(const Cromosoma& otro) {
        if (this != &otro) {
            caracteristicas = otro.caracteristicas;
            fitness = otro.fitness;
        }
        return *this;
    } */

    bool operator==(const Cromosoma &c) const {
        return arma::approx_equal(caracteristicas, c.caracteristicas, "absdiff", 1e-5);
    }
};

/**
 * @brief 
 * Estructura para comparar dos cromosomas y que se ordenen los sets de mayor
 * a menor fitness
 * 
 */
struct CompareCromosoma {
    bool operator()(const Cromosoma &c1, const Cromosoma &c2) {
        return c1.fitness > c2.fitness;
    }
};

/**
 * @brief 
 * Vector que almacena la población de cromosomas
 * 
 */
typedef std::vector<Cromosoma> Poblacion;

/**
 * @brief 
 * Multiset para almacenar los cromosomas y que se ordenen por fitness
 * 
 */
//typedef std::multiset<Cromosoma, CompareCromosoma> Poblacion_ordenada;

/************************************************************
************************************************************
FUNCIONES AUXILIARES
************************************************************
************************************************************/

/**
 * @brief 
 * Función para obtener la población inicial de cromosomas
 * 
 * @param datos Conjunto de datos de entrenamiento
 * @return std::vector<Cromosoma> Población inicial
 */
Poblacion poblacion_inicial(const Dataset &datos);

/**
 * @brief
 * Función para ordenar la población de cromosomas por fitness y devolver el 
 * individuo en la posición pos
 * 
 * @param poblacion Población de cromosomas
 * @param pos Posición del individuo a devolver
 * @return Cromosoma Cromosoma en la posición pos
 */
Cromosoma ordenar_poblacion(const Poblacion &poblacion, int pos);

/************************************************************
************************************************************
ALGORITMO GÉNETICO GENERACIONAL (AGG)
************************************************************
************************************************************/

/**
 * @brief 
 * Función que elige tres individuos aleatorios de la población y develve el mejor
 * de ellos (es decir, usando torneo)
 * 
 * @param poblacion Población de cromosomas
 * @return Cromosoma Mejor cromosoma
 */
Cromosoma seleccion(const Poblacion &poblacion);

/**
 * @brief 
 * Función para calcular el cruce BLX entre dos cromosomas
 * 
 * @param datos Conjunto de datos de entrenamiento
 * @param padre1 Cromosoma 1
 * @param padre2 Cromosoma 2
 * @param hijo1 Cromosoma hijo 1
 * @param hijo2 Cromosoma hijo 2
 * @return void
 */
void cruceBLX(const Dataset &datos, const Cromosoma &padre1, const Cromosoma &padre2, Cromosoma &hijo1, Cromosoma &hijo2);

/**
 * @brief 
 * Función para calcular el cruce aritmético entre dos cromosomas
 * 
 * @param datos Conjunto de datos de entrenamiento
 * @param padre1 Cromosoma 1
 * @param padre2 Cromosoma 2
 * @param hijo1 Cromosoma hijo 1
 * @param hijo2 Cromosoma hijo 2
 * @return void
 */
void cruceAritmetico(const Dataset &datos, const Cromosoma &padre1, const Cromosoma &padre2, Cromosoma &hijo1, Cromosoma &hijo2);

/**
 * @brief 
 * Operador de mutación para un cromosoma
 * 
 * @param cromosoma Cromosoma a mutar
 * @param gen Gen a mutar
 * @return void
 */
void mutacion(Cromosoma &cromosoma, const int gen);

/**
 * @brief 
 * Función para aplicar el algoritmo genético generacional tanto con cruce BLX como con cruce aritmético
 * 
 * @param datos Conjunto de datos de entrenamiento
 * @param tipoCruce Tipo de cruce a aplicar (0 para BLX y 1 para aritmético)
 * @return arma::rowvec Pesos de las características
 */

arma::rowvec AGG (const Dataset &datos, int tipoCruce);

/************************************************************
************************************************************
ALGORITMO GENÉTICO ESTACIONARIO (AGE)
************************************************************
************************************************************/

/**
 * @brief 
 * Función para aplicar el algoritmo genético estacionario tanto con cruce BLX como con cruce aritmético
 * 
 * @param datos Conjunto de datos de entrenamiento
 * @param tipoCruce Tipo de cruce a aplicar (0 para BLX y 1 para aritmético)
 * @return arma::rowvec Pesos de las características
 */

arma::rowvec AGE (const Dataset &datos, int tipoCruce);

/************************************************************
************************************************************
ALGORTIMOS MEMÉTICOS (AMs)
************************************************************
************************************************************/

/**
 * @brief 
 * Función para aplicar la búsqueda local de baja intensidad
 * 
 * @param datos Conjunto de datos de entrenamiento
 * @param cromosoma Cromosoma a aplicar la búsqueda local
 * @param iteraciones Número de iteraciones que se han realizado
 * @return Cromosoma Cromosoma con la búsqueda local aplicada
 */
Cromosoma BL_BI(const Dataset &datos, const Cromosoma &cromosoma, int &iteraciones);

/**
 * @brief
 * Función para aplicar el algoritmo memético según
 * el tipo que se le pase como parámetro
 * 
 * @param datos Conjunto de datos de entrenamiento 
 * @param tipoAlg Tipo de algoritmo memético a aplicar (0 para AM_All, 1 para AM_Rand y 2 para AM_Best)
 * @return arma::rowvec Pesos de las características
 */
arma::rowvec AM (const Dataset &datos, int tipoAlg);

/**
 * @brief 
 * Función para aplicar el algoritmo memético donde
 * cada 10 generaciones, se aplica la BL sobre todos los cromosomas de la población
 * 
 * @param datos Conjunto de datos de entrenamiento
 * @return arma::rowvec Pesos de las características
 */

arma::rowvec AM_All (const Dataset &datos);

/**
 * @brief 
 * Función para aplicar el algoritmo memético donde
 * Cada 10 generaciones, se aplica la BL sobre un subconjunto de
 * cromosomas de la población seleccionado aleatoriamente con probabilidad pLS
 * igual a 0.1 para cada cromosoma.
 * 
 * @param datos Conjunto de datos de entrenamiento
 * @return arma::rowvec Pesos de las características
 */

arma::rowvec AM_Rand (const Dataset &datos);

/**
 * @brief 
 * Función para aplicar el algoritmo memético donde
 * Cada 10 generaciones, aplicar la BL sobre los 0.1·N mejores
 * cromosomas de la población actual (N es el tamaño de ésta).
 * 
 * @param datos Conjunto de datos de entrenamiento
 * @return arma::rowvec Pesos de las características
 */

arma::rowvec AM_Best (const Dataset &datos);

/************************************************************
************************************************************
FUNCIONES PARA MOSTRAR RESULTADOS
************************************************************
************************************************************/

/**
 * @brief 
 * Función para mostrar los resultados de la tasa de clasificación y reducción
 * sin ponderaciones, con Greedy Relief, con Búsqueda Local, AGG-BLX, AGG-CA,
 * AGE-BLX, AGE-CA, AM-(10,1.0), AM-(10,0.1) y AM-(10,0.1mej)
 * 
 * @param algoritmo Algoritmo a mostrar, 0 para sin ponderaciones, 
 *                  1 para Greedy Relief, 2 para Búsqueda Local,
 *                  3 para AGG-BLX, 4 para AGG-CA, 5 para AGE-BLX,
 *                  6 para AGE-CA, 7 para AM-(10,1.0), 8 para AM-(10,0.1)
 *                  y 9 para AM-(10,0.1mej)
 */
void printResultados(int algoritmo);


#endif