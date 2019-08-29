/*
 * lift_and_drag_calculation.cpp
 *
 *  Created on: May 17, 2019
 *      Author: fehn
 */

#include <deal.II/matrix_free/fe_evaluation_notemplate.h>

#include "lift_and_drag_calculation.h"

template<int dim, typename Number>
void
calculate_lift_and_drag_force(MatrixFree<dim, Number> const &      matrix_free,
                              unsigned int const &                 dof_index_velocity,
                              unsigned int const &                 quad_index_velocity,
                              unsigned int const &                 dof_index_pressure,
                              std::set<types::boundary_id> const & boundary_IDs,
                              LinearAlgebra::distributed::Vector<Number> const & velocity,
                              LinearAlgebra::distributed::Vector<Number> const & pressure,
                              double const &                                     viscosity,
                              Tensor<1, dim, Number> &                           Force)
{
  FaceIntegrator<dim, dim, Number> integrator_velocity(matrix_free,
                                                       true,
                                                       dof_index_velocity,
                                                       quad_index_velocity);
  FaceIntegrator<dim, 1, Number>   integrator_pressure(matrix_free,
                                                     true,
                                                     dof_index_pressure,
                                                     quad_index_velocity);

  for(unsigned int d = 0; d < dim; ++d)
    Force[d] = 0.0;

  for(unsigned int face = matrix_free.n_inner_face_batches();
      face < (matrix_free.n_inner_face_batches() + matrix_free.n_boundary_face_batches());
      face++)
  {
    integrator_velocity.reinit(face);
    integrator_velocity.read_dof_values(velocity);
    integrator_velocity.evaluate(false, true);

    integrator_pressure.reinit(face);
    integrator_pressure.read_dof_values(pressure);
    integrator_pressure.evaluate(true, false);

    types::boundary_id boundary_id = matrix_free.get_boundary_id(face);

    typename std::set<types::boundary_id>::iterator it = boundary_IDs.find(boundary_id);
    if(it != boundary_IDs.end())
    {
      for(unsigned int q = 0; q < integrator_velocity.n_q_points; ++q)
      {
        VectorizedArray<Number> pressure = integrator_pressure.get_value(q);

        Tensor<1, dim, VectorizedArray<Number>> normal = integrator_velocity.get_normal_vector(q);
        Tensor<2, dim, VectorizedArray<Number>> velocity_gradient =
          integrator_velocity.get_gradient(q);

        Tensor<1, dim, VectorizedArray<Number>> tau =
          pressure * normal -
          viscosity * (velocity_gradient + transpose(velocity_gradient)) * normal;

        integrator_velocity.submit_value(tau, q);
      }

      Tensor<1, dim, VectorizedArray<Number>> Force_local = integrator_velocity.integrate_value();

      // sum over all entries of VectorizedArray
      for(unsigned int d = 0; d < dim; ++d)
      {
        for(unsigned int n = 0; n < matrix_free.n_active_entries_per_face_batch(face); ++n)
          Force[d] += Force_local[d][n];
      }
    }
  }
  Force = Utilities::MPI::sum(Force, MPI_COMM_WORLD);
}

template<int dim, typename Number>
LiftAndDragCalculator<dim, Number>::LiftAndDragCalculator()
  : clear_files_lift_and_drag(true),
    matrix_free(nullptr),
    dof_index_velocity(0),
    dof_index_pressure(1),
    quad_index(0),
    c_L_max(-std::numeric_limits<double>::max()),
    c_D_max(-std::numeric_limits<double>::max())
{
}

template<int dim, typename Number>
void
LiftAndDragCalculator<dim, Number>::setup(DoFHandler<dim> const &         dof_handler_velocity_in,
                                          MatrixFree<dim, Number> const & matrix_free_in,
                                          unsigned int const              dof_index_velocity_in,
                                          unsigned int const              dof_index_pressure_in,
                                          unsigned int const              quad_index_in,
                                          LiftAndDragData const &         lift_and_drag_data_in)
{
  dof_handler_velocity = &dof_handler_velocity_in;
  matrix_free          = &matrix_free_in;
  dof_index_velocity   = dof_index_velocity_in;
  dof_index_pressure   = dof_index_pressure_in;
  quad_index           = quad_index_in;
  lift_and_drag_data   = lift_and_drag_data_in;
}

template<int dim, typename Number>
void
LiftAndDragCalculator<dim, Number>::evaluate(VectorType const & velocity,
                                             VectorType const & pressure,
                                             Number const &     time) const
{
  if(lift_and_drag_data.calculate_lift_and_drag == true)
  {
    Tensor<1, dim, Number> Force;

    calculate_lift_and_drag_force<dim, Number>(*matrix_free,
                                               dof_index_velocity,
                                               quad_index,
                                               dof_index_pressure,
                                               lift_and_drag_data.boundary_IDs,
                                               velocity,
                                               pressure,
                                               lift_and_drag_data.viscosity,
                                               Force);

    // compute lift and drag coefficients (c = (F/rho)/(1/2 U² A)
    double const reference_value = lift_and_drag_data.reference_value;
    Force /= reference_value;

    double const drag = Force[0], lift = Force[1];
    c_D_max = std::max(c_D_max, drag);
    c_L_max = std::max(c_L_max, lift);

    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      std::string filename_drag, filename_lift;
      filename_drag = lift_and_drag_data.filename_drag;
      filename_lift = lift_and_drag_data.filename_lift;

      unsigned int precision = 12;

      std::ofstream f_drag, f_lift;
      if(clear_files_lift_and_drag)
      {
        f_drag.open(filename_drag.c_str(), std::ios::trunc);
        f_lift.open(filename_lift.c_str(), std::ios::trunc);

        // clang-format off
        f_drag << std::setw(precision+8) << std::left << "time t"
               << std::setw(precision+8) << std::left << "c_D(t)"
               << std::setw(precision+8) << std::left << "c_D_max"
               << std::endl;

        f_lift << std::setw(precision+8) << std::left << "time t"
               << std::setw(precision+8) << std::left << "c_L(t)"
               << std::setw(precision+8) << std::left << "c_L_max"
               << std::endl;
        // clang-format on

        clear_files_lift_and_drag = false;
      }
      else
      {
        f_drag.open(filename_drag.c_str(), std::ios::app);
        f_lift.open(filename_lift.c_str(), std::ios::app);
      }

      // clang-format off
      f_drag << std::scientific << std::setprecision(precision)
             << std::setw(precision+8) << std::left << time
             << std::setw(precision+8) << std::left << drag
             << std::setw(precision+8) << std::left << c_D_max
             << std::endl;

      f_drag.close();

      f_lift << std::scientific << std::setprecision(precision)
             << std::setw(precision+8) << std::left << time
             << std::setw(precision+8) << std::left << lift
             << std::setw(precision+8) << std::left << c_L_max
             << std::endl;

      f_lift.close();
      // clang-format on
    }
  }
}

template class LiftAndDragCalculator<2, float>;
template class LiftAndDragCalculator<2, double>;

template class LiftAndDragCalculator<3, float>;
template class LiftAndDragCalculator<3, double>;
