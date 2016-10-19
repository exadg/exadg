/*
 * FE_Parameters.h
 *
 *  Created on: May 10, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_FE_PARAMETERS_H_
#define INCLUDE_FE_PARAMETERS_H_

#include <deal.II/base/vectorization.h>
#include <deal.II/base/aligned_vector.h>
#include "InputParametersNavierStokes.h"

using namespace dealii;

template<int dim>
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
    max_wdist_xwall(1.0),
    wdist(nullptr),
    tauw(nullptr),
    enrichment(nullptr),
    enrichment_gradient(nullptr)
  {
  }

  FEParameters(InputParametersNavierStokes<dim> const & param)
    :
    viscosity(param.viscosity),
    cs(param.cs),
    ml(param.ml),
    variabletauw(param.variabletauw),
    dtauw(param.dtauw),
    max_wdist_xwall(param.max_wdist_xwall),
    wdist(nullptr),
    tauw(nullptr),
    enrichment(nullptr),
    enrichment_gradient(nullptr)
  {
  }

    void setup(parallel::distributed::Vector<double> * wd,
             parallel::distributed::Vector<double> * tw)
  {
    wdist = wd;
    tauw = tw;
  }

  void setup(parallel::distributed::Vector<double> * wd,
             parallel::distributed::Vector<double> * tw,
             AlignedVector<AlignedVector<VectorizedArray<double> > > * enr,
             AlignedVector<AlignedVector<Tensor<1,dim,VectorizedArray<double> > > > * enr_grad)
  {
    wdist = wd;
    tauw = tw;
    enrichment = enr;
    enrichment_gradient = enr_grad;
  }

  double const viscosity;
  double const cs;
  double const ml;
  bool const variabletauw;
  double const dtauw;
  double const max_wdist_xwall;
  parallel::distributed::Vector<double> * wdist;
  parallel::distributed::Vector<double> * tauw;
  AlignedVector<AlignedVector<VectorizedArray<double> > > * enrichment;
  AlignedVector<AlignedVector<Tensor<1,dim,VectorizedArray<double> > > > * enrichment_gradient;
  std_cxx11::shared_ptr<Function<dim,double> > enrichment_is_within;
};

#endif /* INCLUDE_FE_PARAMETERS_H_ */
