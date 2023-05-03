/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2021 by the ExaDG authors
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_ORR_SOMMERFELD_POSTPROCESSOR_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_ORR_SOMMERFELD_POSTPROCESSOR_H_

// ExaDG
#include "perturbation_energy.h"

namespace ExaDG
{
namespace IncNS
{
template<int dim>
struct MyPostProcessorData
{
  PostProcessorData<dim> pp_data;
  PerturbationEnergyData energy_data;
};

template<int dim, typename Number>
class MyPostProcessor : public PostProcessor<dim, Number>
{
public:
  typedef PostProcessor<dim, Number> Base;

  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef typename Base::Operator Operator;

  MyPostProcessor(MyPostProcessorData<dim> const & pp_data_os, MPI_Comm const & mpi_comm)
    : Base(pp_data_os.pp_data, mpi_comm),
      energy_data(pp_data_os.energy_data),
      energy_calculator(mpi_comm)
  {
  }

  void
  setup(Operator const & pde_operator) final
  {
    // call setup function of base class
    Base::setup(pde_operator);

    energy_calculator.setup(pde_operator.get_matrix_free(),
                            pde_operator.get_dof_index_velocity(),
                            pde_operator.get_quad_index_velocity_linear(),
                            energy_data);
  }

  void
  do_postprocessing(VectorType const &     velocity,
                    VectorType const &     pressure,
                    double const           time,
                    types::time_step const time_step_number) final
  {
    Base::do_postprocessing(velocity, pressure, time, time_step_number);

    energy_calculator.evaluate(velocity, time, time_step_number);
  }

  PerturbationEnergyData                    energy_data;
  PerturbationEnergyCalculator<dim, Number> energy_calculator;
};

} // namespace IncNS
} // namespace ExaDG


#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_ORR_SOMMERFELD_POSTPROCESSOR_H_ */
