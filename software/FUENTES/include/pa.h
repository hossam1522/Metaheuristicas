/**
 * @file pa.h
 * @author Hossam El Amraoui Leghzali
 * @brief 
 * Asignatura: Metahuristicas
 * Práctica alternativa
 * 
 */

#ifndef PA_H
#define PA_H

#include <algorithm>
#include "p3.h"

/************************************************************
************************************************************
CONSTANTES GLOBALES
************************************************************
************************************************************/

const double L = 0.08;
const int LH = 10000;

/************************************************************
************************************************************
ESTRUCTURAS DE DATOS
************************************************************
************************************************************/

typedef std::vector<Solucion> Population;

struct CompararSoluciones {
  bool operator()(const Solucion &sol1, const Solucion &sol2) {
    return sol1.fitness < sol2.fitness;
  }
};

/************************************************************
************************************************************
FUNCIONES AUXILIARES
************************************************************
************************************************************/

/**
 * @brief 
 * Función para ordenar la población de soluciones por fitness y devolver
 * el individuo en la posición indicada
 * 
 * @param poblacion Población de soluciones
 * @param pos Posición del individuo a devolver
 * @return Solucion Individuo en la posición pos
 */
Solucion ordenar_poblacion(const Population &poblacion, int pos);

/************************************************************
************************************************************
FUNCIONES PARA MOSTRAR RESULTADOS
************************************************************
************************************************************/

/**
 * @brief 
 * Función para mostrar los resultados de la tasa de clasificación y reducción
 * sin ponderaciones, con Greedy Relief, con Búsqueda Local, AGG-BLX, AGG-CA,
 * AGE-BLX, AGE-CA, AM-(10,1.0), AM-(10,0.1) y AM-(10,0.1mej), BMB, ES, ILS y
 * ILS con Enfriamiento Simulado
 * 
 * @param algoritmo Algoritmo a mostrar, 0 para sin ponderaciones, 
 *                  1 para Greedy Relief, 2 para Búsqueda Local,
 *                  3 para AGG-BLX, 4 para AGG-CA, 5 para AGE-BLX,
 *                  6 para AGE-CA, 7 para AM-(10,1.0), 8 para AM-(10,0.1),
 *                  9 para AM-(10,0.1mej), 10 para BMB, 11 para ES, 12 para ILS y
 *                  13 para ILS con Enfriamiento Simulado
 */
void printResultados(int algoritmo);



#endif
