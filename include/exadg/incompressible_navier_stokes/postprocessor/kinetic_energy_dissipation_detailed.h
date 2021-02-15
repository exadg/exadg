/*
 * kinetic_energy_dissipation_detailed.h
 *
 *  Created on: May 17, 2019
 *      Author: fehn
 */

#ifndef INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_KINETIC_ENERGY_DISSIPATION_DETAILED_H_
#define INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_KINETIC_ENERGY_DISSIPATION_DETAILED_H_

#include <exadg/incompressible_navier_stokes/spatial_discretization/spatial_operator_base.h>
#include <exadg/postprocessor/kinetic_energy_calculation.h>

namespace ExaDG
{
namespace IncNS
{
using namespace dealii;

template<int dim, typename Number>
class KineticEnergyCalculatorDetailed : public KineticEnergyCalculator<dim, Number>
{
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef KineticEnergyCalculator<dim, Number> Base;

  typedef SpatialOperatorBase<dim, Number> NavierStokesOperator;

public:
  KineticEnergyCalculatorDetailed(MPI_Comm const & comm);

  void
  setup(NavierStokesOperator const &    navier_stokes_operator_in,
        MatrixFree<dim, Number> const & matrix_free_in,
        unsigned int const              dof_index_in,
        unsigned int const              quad_index_in,
        KineticEnergyData const &       kinetic_energy_data_in);

  void
  evaluate(VectorType const & velocity, double const & time, int const & time_step_number);

private:
  void
  calculate_detailed(VectorType const & velocity,
                     double const       time,
                     unsigned int const time_step_number);

  SmartPointer<NavierStokesOperator const> navier_stokes_operator;
};

} // namespace IncNS
} // namespace ExaDG


#endif /* INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_KINETIC_ENERGY_DISSIPATION_DETAILED_H_ \
        */
