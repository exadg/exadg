/*
 * FE_Parameters.h
 *
 *  Created on: May 10, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_INFRASTRUCTURE_FE_PARAMETERS_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_INFRASTRUCTURE_FE_PARAMETERS_H_

#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/vectorization.h>

#include "../../incompressible_navier_stokes/user_interface/input_parameters.h"

using namespace dealii;

namespace IncNS
{
template<int dim>
class FEParameters
{
public:
  typedef LinearAlgebra::distributed::Vector<double> VectorType;

  FEParameters()
    : viscosity(1.0),
      variabletauw(false),
      dtauw(1.0),
      max_wdist_xwall(1.0),
      wdist(nullptr),
      tauw(nullptr),
      enrichment(nullptr),
      enrichment_gradient(nullptr)
  {
  }

  FEParameters(IncNS::InputParameters<dim> const & param)
    : viscosity(param.viscosity),
      variabletauw(param.variabletauw),
      dtauw(param.dtauw),
      max_wdist_xwall(param.max_wdist_xwall),
      wdist(nullptr),
      tauw(nullptr),
      enrichment(nullptr),
      enrichment_gradient(nullptr)
  {
  }

  void
  setup(VectorType * wd, VectorType * tw)
  {
    wdist = wd;
    tauw  = tw;
  }

  void
  setup(VectorType *                                                            wd,
        VectorType *                                                            tw,
        AlignedVector<AlignedVector<VectorizedArray<double>>> *                 enr,
        AlignedVector<AlignedVector<Tensor<1, dim, VectorizedArray<double>>>> * enr_grad)
  {
    wdist               = wd;
    tauw                = tw;
    enrichment          = enr;
    enrichment_gradient = enr_grad;
  }

  double const                                                            viscosity;
  bool const                                                              variabletauw;
  double const                                                            dtauw;
  double const                                                            max_wdist_xwall;
  VectorType *                                                            wdist;
  VectorType *                                                            tauw;
  AlignedVector<AlignedVector<VectorizedArray<double>>> *                 enrichment;
  AlignedVector<AlignedVector<Tensor<1, dim, VectorizedArray<double>>>> * enrichment_gradient;
  std::shared_ptr<Function<dim, double>>                                  enrichment_is_within;
};


} // namespace IncNS

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_INFRASTRUCTURE_FE_PARAMETERS_H_ */
