/*
 * NewtonSolverData.h
 *
 *  Created on: Nov 9, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_NEWTONSOLVERDATA_H_
#define INCLUDE_NEWTONSOLVERDATA_H_


struct NewtonSolverData
{
  NewtonSolverData()
    :
    abs_tol(1.e-20),
    rel_tol(1.e-12),
    max_iter(100)
  {}

  double abs_tol;
  double rel_tol;
  unsigned int max_iter;
};


#endif /* INCLUDE_NEWTONSOLVERDATA_H_ */
