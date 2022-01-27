/*
 * divergence_penalty_operator.h
 *
 *  Created on: Jun 25, 2019
 *      Author: fehn
 */

#ifndef INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_DIVERGENCE_PENALTY_OPERATOR_H_
#define INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_DIVERGENCE_PENALTY_OPERATOR_H_

#include <exadg/incompressible_navier_stokes/user_interface/parameters.h>
#include <exadg/matrix_free/integrators.h>
#include <exadg/operators/integrator_flags.h>
#include <exadg/operators/mapping_flags.h>

namespace ExaDG
{
namespace IncNS
{
/*
 *  Divergence penalty operator:
 *
 *    ( div(v_h) , tau_div * div(u_h) )_Omega^e
 *
 *  where
 *
 *   v_h : test function
 *   u_h : solution
 *   tau_div: divergence penalty factor
 *
 *            use convective term:  tau_div_conv = K * ||U||_mean * h_eff
 *
 *                                  where h_eff = h / (k_u+1) and
 *                                  h = V_e^{1/3} with the element volume V_e
 *
 *            use viscous term:     tau_div_viscous = K * nu
 *
 *            use both terms:       tau_div = tau_div_conv + tau_div_viscous
 *
 */

namespace Operators
{
struct DivergencePenaltyKernelData
{
  DivergencePenaltyKernelData()
    : type_penalty_parameter(TypePenaltyParameter::ConvectiveTerm),
      viscosity(0.0),
      degree(1),
      penalty_factor(1.0)
  {
  }

  // type of penalty parameter (viscous and/or convective terms)
  TypePenaltyParameter type_penalty_parameter;

  // viscosity, needed for computation of penalty factor
  double viscosity;

  // degree of finite element shape functions
  unsigned int degree;

  // the penalty term can be scaled by 'penalty_factor'
  double penalty_factor;
};

template<int dim, typename Number>
class DivergencePenaltyKernel
{
private:
  typedef CellIntegrator<dim, dim, Number> IntegratorCell;

  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef dealii::VectorizedArray<Number> scalar;

public:
  DivergencePenaltyKernel()
    : matrix_free(nullptr), dof_index(0), quad_index(0), array_penalty_parameter(0)
  {
  }

  void
  reinit(dealii::MatrixFree<dim, Number> const & matrix_free,
         unsigned int const                      dof_index,
         unsigned int const                      quad_index,
         DivergencePenaltyKernelData const &     data)
  {
    this->matrix_free = &matrix_free;

    this->dof_index  = dof_index;
    this->quad_index = quad_index;

    this->data = data;

    unsigned int n_cells = matrix_free.n_cell_batches() + matrix_free.n_ghost_cell_batches();
    array_penalty_parameter.resize(n_cells);
  }

  DivergencePenaltyKernelData
  get_data()
  {
    return this->data;
  }

  IntegratorFlags
  get_integrator_flags() const
  {
    IntegratorFlags flags;

    flags.cell_evaluate  = CellFlags(false, true, false);
    flags.cell_integrate = CellFlags(false, true, false);

    // no face integrals

    return flags;
  }

  static MappingFlags
  get_mapping_flags()
  {
    MappingFlags flags;

    flags.cells = dealii::update_JxW_values | dealii::update_gradients;

    // no face integrals

    return flags;
  }

  void
  calculate_penalty_parameter(VectorType const & velocity)
  {
    velocity.update_ghost_values();

    IntegratorCell integrator(*matrix_free, dof_index, quad_index);

    dealii::AlignedVector<scalar> JxW_values(integrator.n_q_points);

    unsigned int n_cells = matrix_free->n_cell_batches() + matrix_free->n_ghost_cell_batches();
    for(unsigned int cell = 0; cell < n_cells; ++cell)
    {
      scalar tau_convective = dealii::make_vectorized_array<Number>(0.0);
      scalar tau_viscous    = dealii::make_vectorized_array<Number>(data.viscosity);

      if(data.type_penalty_parameter == TypePenaltyParameter::ConvectiveTerm ||
         data.type_penalty_parameter == TypePenaltyParameter::ViscousAndConvectiveTerms)
      {
        integrator.reinit(cell);
        integrator.read_dof_values(velocity);
        integrator.evaluate(true, false);

        scalar volume      = dealii::make_vectorized_array<Number>(0.0);
        scalar norm_U_mean = dealii::make_vectorized_array<Number>(0.0);
        for(unsigned int q = 0; q < integrator.n_q_points; ++q)
        {
          volume += integrator.JxW(q);
          norm_U_mean += integrator.JxW(q) * integrator.get_value(q).norm();
        }
        norm_U_mean /= volume;

        tau_convective =
          norm_U_mean * std::exp(std::log(volume) / (double)dim) / (double)(data.degree + 1);
      }

      if(data.type_penalty_parameter == TypePenaltyParameter::ConvectiveTerm)
      {
        array_penalty_parameter[cell] = data.penalty_factor * tau_convective;
      }
      else if(data.type_penalty_parameter == TypePenaltyParameter::ViscousTerm)
      {
        array_penalty_parameter[cell] = data.penalty_factor * tau_viscous;
      }
      else if(data.type_penalty_parameter == TypePenaltyParameter::ViscousAndConvectiveTerms)
      {
        array_penalty_parameter[cell] = data.penalty_factor * (tau_convective + tau_viscous);
      }
    }
  }

  void
  reinit_cell(IntegratorCell & integrator) const
  {
    tau = integrator.read_cell_data(array_penalty_parameter);
  }

  /*
   * Volume flux, i.e., the term occurring in the volume integral
   */
  inline DEAL_II_ALWAYS_INLINE //
    scalar
    get_volume_flux(IntegratorCell const & integrator, unsigned int const q) const
  {
    return tau * integrator.get_divergence(q);
  }

private:
  dealii::MatrixFree<dim, Number> const * matrix_free;

  unsigned int dof_index;
  unsigned int quad_index;

  DivergencePenaltyKernelData data;

  dealii::AlignedVector<scalar> array_penalty_parameter;

  mutable scalar tau;
};

} // namespace Operators

struct DivergencePenaltyData
{
  DivergencePenaltyData() : dof_index(0), quad_index(0)
  {
  }

  unsigned int dof_index;
  unsigned int quad_index;
};

template<int dim, typename Number>
class DivergencePenaltyOperator
{
private:
  typedef DivergencePenaltyOperator<dim, Number> This;

  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef std::pair<unsigned int, unsigned int> Range;

  typedef CellIntegrator<dim, dim, Number> IntegratorCell;

  typedef Operators::DivergencePenaltyKernel<dim, Number> Kernel;

public:
  DivergencePenaltyOperator();

  void
  initialize(dealii::MatrixFree<dim, Number> const & matrix_free,
             DivergencePenaltyData const &           data,
             std::shared_ptr<Kernel> const           kernel);

  void
  update(VectorType const & velocity);

  void
  apply(VectorType & dst, VectorType const & src) const;

  void
  apply_add(VectorType & dst, VectorType const & src) const;

private:
  void
  cell_loop(dealii::MatrixFree<dim, Number> const & matrix_free,
            VectorType &                            dst,
            VectorType const &                      src,
            Range const &                           range) const;

  void
  do_cell_integral(IntegratorCell & integrator) const;

  dealii::MatrixFree<dim, Number> const * matrix_free;

  DivergencePenaltyData data;

  std::shared_ptr<Operators::DivergencePenaltyKernel<dim, Number>> kernel;
};

} // namespace IncNS
} // namespace ExaDG

#endif /* INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_DIVERGENCE_PENALTY_OPERATOR_H_ \
        */
