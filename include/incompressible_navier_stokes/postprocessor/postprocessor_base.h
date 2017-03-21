/*
 * PostProcessorBase.h
 *
 *  Created on: Oct 25, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_POSTPROCESSOR_BASE_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_POSTPROCESSOR_BASE_H_

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/lac/parallel_vector.h>

#include "../../incompressible_navier_stokes/user_interface/analytical_solution.h"

/*
 *  This struct contains information about
 *  indices of dof_handlers and quadrature formulas
 *  that is needed in order to perform integrals
 */
struct DofQuadIndexData
{
  DofQuadIndexData()
   :
   dof_index_velocity(0),
   dof_index_pressure(1),
   quad_index_velocity(0)
  {}

  unsigned int dof_index_velocity;
  unsigned int dof_index_pressure;
  unsigned int quad_index_velocity;
};

/*
 *  Interface class for postprocessor of the
 *  incompressible Navier-Stokes equation.
 *
 */
template<int dim>
class PostProcessorBase
{
public:
  PostProcessorBase(){}

  virtual ~PostProcessorBase(){}

  /*
   * Setup function.
   */
  virtual void setup(DoFHandler<dim> const                                        &dof_handler_velocity,
                     DoFHandler<dim> const                                        &dof_handler_pressure,
                     Mapping<dim> const                                           &mapping,
                     MatrixFree<dim,double> const                                 &matrix_free_data,
                     DofQuadIndexData const                                       &dof_quad_index_data,
                     std_cxx11::shared_ptr<AnalyticalSolutionNavierStokes<dim> >  analytical_solution) = 0;


  /*
   * This function has to be called to apply the postprocessing tools.
   * It is currently used for unsteady solvers, but the plan is to use this
   * function also for the steady solver and that the individual postprocessing
   * tools decide whether to apply the steady or unsteady postprocessing functions.
   */
  virtual void do_postprocessing(parallel::distributed::Vector<double> const &velocity,
                                 parallel::distributed::Vector<double> const &intermediate_velocity,
                                 parallel::distributed::Vector<double> const &pressure,
                                 parallel::distributed::Vector<double> const &vorticity,
                                 parallel::distributed::Vector<double> const &divergence,
                                 double const                                time = 0.0,
                                 int const                                   time_step_number = -1) = 0;
};


#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_POSTPROCESSOR_BASE_H_ */
