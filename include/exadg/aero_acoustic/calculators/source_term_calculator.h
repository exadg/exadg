#ifndef INCLUDE_EXADG_AERO_ACOUSTIC_SOURCE_TERM_CALCULATOR_H_
#define INCLUDE_EXADG_AERO_ACOUSTIC_SOURCE_TERM_CALCULATOR_H_

#include <exadg/matrix_free/integrators.h>
#include <exadg/utilities/lazy_ptr.h>

namespace ExaDG
{
namespace AeroAcoustic
{
struct SourceTermCalculatorData
{
  unsigned int dof_index_pressure;
  unsigned int dof_index_velocity;
  unsigned int quad_index;

  // density of the underlying fluid.
  double density;

  // use material or partial temporal derivative of pressure as source term.
  bool consider_convection;
};

/**
 * A class that knows how to compute the aeroacoustic source term on the fluid mesh.
 * evaluate_integrate() computes and integrates the source term on the fluid mesh.
 *
 * The aeroacoustic source term f is definded as:
 * f = - rho * (dp/dt + u * grad(p)).
 * The scaling factor rho has to be used since the pressure of the incompressible
 * module is a kinematic pressure. Using consider_convection=false
 * f = -rho * (dp/dt).
 */
template<int dim, typename Number>
class SourceTermCalculator
{
  using This       = SourceTermCalculator<dim, Number>;
  using VectorType = dealii::LinearAlgebra::distributed::Vector<Number>;

  using CellIntegratorScalar = CellIntegrator<dim, 1, Number>;
  using CellIntegratorVector = CellIntegrator<dim, dim, Number>;

public:
  SourceTermCalculator() : matrix_free(nullptr)
  {
  }

  void
  setup(dealii::MatrixFree<dim, Number> const & matrix_free_in,
        SourceTermCalculatorData const &        data_in)
  {
    matrix_free = &matrix_free_in;
    data        = data_in;
  }

  void
  evaluate_integrate(VectorType &       dst,
                     VectorType const & velocity_cfd_in,
                     VectorType const & pressure_cfd_in,
                     VectorType const & pressure_cfd_time_derivative_in)
  {
    dst.zero_out_ghost_values();

    if(data.consider_convection)
    {
      velocity_cfd.reset(velocity_cfd_in);
      velocity_cfd->update_ghost_values();

      pressure_cfd.reset(pressure_cfd_in);
      pressure_cfd->update_ghost_values();
    }

    matrix_free->cell_loop(
      &This::compute_integrated_source_term, this, dst, pressure_cfd_time_derivative_in, true);
  }

private:
  void
  compute_integrated_source_term(dealii::MatrixFree<dim, Number> const &       matrix_free_in,
                                 VectorType &                                  dst,
                                 VectorType const &                            dp_cfd_dt,
                                 std::pair<unsigned int, unsigned int> const & cell_range) const
  {
    CellIntegratorScalar dpdt(matrix_free_in, data.dof_index_pressure, data.quad_index);
    CellIntegratorScalar p(matrix_free_in, data.dof_index_pressure, data.quad_index);
    CellIntegratorVector u(matrix_free_in, data.dof_index_velocity, data.quad_index);

    Number rho = static_cast<Number>(data.density);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      dpdt.reinit(cell);
      dpdt.gather_evaluate(dp_cfd_dt, dealii::EvaluationFlags::values);

      if(data.consider_convection)
      {
        p.reinit(cell);
        p.gather_evaluate(*pressure_cfd, dealii::EvaluationFlags::gradients);
        u.reinit(cell);
        u.gather_evaluate(*velocity_cfd, dealii::EvaluationFlags::values);

        for(unsigned int q = 0; q < dpdt.n_q_points; ++q)
          dpdt.submit_value(-rho * (dpdt.get_value(q) + u.get_value(q) * p.get_gradient(q)), q);
      }
      else
      {
        for(unsigned int q = 0; q < dpdt.n_q_points; ++q)
          dpdt.submit_value(-rho * dpdt.get_value(q), q);
      }

      dpdt.integrate_scatter(dealii::EvaluationFlags::values, dst);
    }
  }

  dealii::MatrixFree<dim, Number> const * matrix_free;

  SourceTermCalculatorData data;

  lazy_ptr<VectorType> velocity_cfd;
  lazy_ptr<VectorType> pressure_cfd;
};
} // namespace AeroAcoustic
} // namespace ExaDG

#endif /*INCLUDE_EXADG_AERO_ACOUSTIC_SOURCE_TERM_CALCULATOR_H_*/
