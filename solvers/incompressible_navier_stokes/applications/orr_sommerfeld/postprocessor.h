/*
 * postprocessor.h
 *
 *  Created on: 29.03.2020
 *      Author: fehn
 */

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_ORR_SOMMERFELD_POSTPROCESSOR_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_ORR_SOMMERFELD_POSTPROCESSOR_H_

// ExaDG
#include <exadg/incompressible_navier_stokes/postprocessor/perturbation_energy_orr_sommerfeld.h>

namespace ExaDG
{
namespace IncNS
{
namespace OrrSommerfeld
{
using namespace dealii;

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

  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef typename Base::Operator Operator;

  MyPostProcessor(MyPostProcessorData<dim> const & pp_data_os, MPI_Comm const & mpi_comm)
    : Base(pp_data_os.pp_data, mpi_comm),
      energy_data(pp_data_os.energy_data),
      energy_calculator(mpi_comm)
  {
  }

  void
  setup(Operator const & pde_operator)
  {
    // call setup function of base class
    Base::setup(pde_operator);

    energy_calculator.setup(pde_operator.get_matrix_free(),
                            pde_operator.get_dof_index_velocity(),
                            pde_operator.get_quad_index_velocity_linear(),
                            energy_data);
  }

  void
  do_postprocessing(VectorType const & velocity,
                    VectorType const & pressure,
                    double const       time,
                    int const          time_step_number)
  {
    Base::do_postprocessing(velocity, pressure, time, time_step_number);

    energy_calculator.evaluate(velocity, time, time_step_number);
  }

  PerturbationEnergyData                    energy_data;
  PerturbationEnergyCalculator<dim, Number> energy_calculator;
};

} // namespace OrrSommerfeld
} // namespace IncNS
} // namespace ExaDG


#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_ORR_SOMMERFELD_POSTPROCESSOR_H_ */
