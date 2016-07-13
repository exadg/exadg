/*
 * FE_Parameters.h
 *
 *  Created on: May 10, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_FE_PARAMETERS_H_
#define INCLUDE_FE_PARAMETERS_H_

#include "InputParameters.h"

class FEParameters
{
public:
  FEParameters()
    :
    viscosity(1.0),
    cs(1.0),
    ml(1.0),
    variabletauw(false),
    dtauw(1.0),
    max_wdist_xwall(1.0)
//    wdist(nullptr),
//    tauw(nullptr)
  {
  }

  FEParameters(InputParameters const & param)
    :
    viscosity(param.viscosity),
    cs(param.cs),
    ml(param.ml),
    variabletauw(param.variabletauw),
    dtauw(param.dtauw),
    max_wdist_xwall(param.max_wdist_xwall)
//    wdist(nullptr),
//    tauw(nullptr)
  {
  }

  double const viscosity;
  double const cs;
  double const ml;
  bool const variabletauw;
  double const dtauw;
  double const max_wdist_xwall;
  parallel::distributed::Vector<double> wdist;
  parallel::distributed::Vector<double> tauw;
};

#endif /* INCLUDE_FE_PARAMETERS_H_ */
