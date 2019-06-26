/*
 * continuity_penalty_operator.h
 *
 *  Created on: Jun 25, 2019
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_CONTINUITY_PENALTY_OPERATOR_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_CONTINUITY_PENALTY_OPERATOR_H_

#include <deal.II/matrix_free/fe_evaluation_notemplate.h>

#include "../../../operators/integrator_flags.h"
#include "../../../operators/mapping_flags.h"
#include "../../user_interface/input_parameters.h"

using namespace dealii;

namespace IncNS
{
/*
 *  Continuity penalty operator:
 *
 *    ( v_h , tau_conti * jump(u_h) )_dOmega^e
 *
 *  where
 *
 *   v_h : test function
 *   u_h : solution
 *
 *   jump(u_h) = u_h^{-} - u_h^{+} or ( (u_h^{-} - u_h^{+})*normal ) * normal
 *
 *     where "-" denotes interior information and "+" exterior information
 *
 *   tau_conti: continuity penalty factor
 *
 *            use convective term:  tau_conti_conv = K * ||U||_mean
 *
 *            use viscous term:     tau_conti_viscous = K * nu / h
 *
 *                                  where h_eff = h / (k_u+1) and
 *                                  h = V_e^{1/3} with the element volume V_e
 *
 *            use both terms:       tau_conti = tau_conti_conv + tau_conti_viscous
 */

namespace Operators
{
struct ContinuityPenaltyKernelData
{
  ContinuityPenaltyKernelData()
    : type_penalty_parameter(TypePenaltyParameter::ConvectiveTerm),
      which_components(ContinuityPenaltyComponents::Normal),
      viscosity(0.0),
      degree(1),
      penalty_factor(1.0)
  {
  }

  // type of penalty parameter (viscous and/or convective terms)
  TypePenaltyParameter type_penalty_parameter;

  // the continuity penalty term can be applied to all velocity components or to the normal
  // component only
  ContinuityPenaltyComponents which_components;

  // viscosity, needed for computation of penalty factor
  double viscosity;

  // degree of finite element shape functions
  unsigned int degree;

  // the penalty term can be scaled by 'penalty_factor'
  double penalty_factor;
};

template<int dim, typename Number>
class ContinuityPenaltyKernel
{
private:
  typedef CellIntegrator<dim, dim, Number> IntegratorCell;
  typedef FaceIntegrator<dim, dim, Number> IntegratorFace;

  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef VectorizedArray<Number>                 scalar;
  typedef Tensor<1, dim, VectorizedArray<Number>> vector;

public:
  ContinuityPenaltyKernel()
    : matrix_free(nullptr), dof_index(0), quad_index(0), array_penalty_parameter(0)
  {
  }

  void
  reinit(MatrixFree<dim, Number> const &     matrix_free,
         unsigned int const                  dof_index,
         unsigned int const                  quad_index,
         ContinuityPenaltyKernelData const & data)
  {
    this->matrix_free = &matrix_free;

    this->dof_index  = dof_index;
    this->quad_index = quad_index;

    this->data = data;

    unsigned int n_cells = matrix_free.n_cell_batches() + matrix_free.n_ghost_cell_batches();
    array_penalty_parameter.resize(n_cells);
  }

  IntegratorFlags
  get_integrator_flags() const
  {
    IntegratorFlags flags;

    // no cell integrals

    flags.face_evaluate  = FaceFlags(true, false);
    flags.face_integrate = FaceFlags(true, false);

    return flags;
  }

  static MappingFlags
  get_mapping_flags()
  {
    MappingFlags flags;

    // no cell integrals

    flags.inner_faces = update_JxW_values | update_normal_vectors;

    // no boundary face integrals

    return flags;
  }

  void
  calculate_penalty_parameter(VectorType const & velocity)
  {
    velocity.update_ghost_values();

    IntegratorCell integrator(*matrix_free, dof_index, quad_index);

    AlignedVector<scalar> JxW_values(integrator.n_q_points);

    unsigned int n_cells = matrix_free->n_cell_batches() + matrix_free->n_ghost_cell_batches();
    for(unsigned int cell = 0; cell < n_cells; ++cell)
    {
      integrator.reinit(cell);
      integrator.read_dof_values(velocity);
      integrator.evaluate(true, false);
      scalar volume      = make_vectorized_array<Number>(0.0);
      scalar norm_U_mean = make_vectorized_array<Number>(0.0);

      for(unsigned int q = 0; q < integrator.n_q_points; ++q)
      {
        volume += integrator.JxW(q);
        norm_U_mean += integrator.JxW(q) * integrator.get_value(q).norm();
      }

      norm_U_mean /= volume;

      scalar tau_convective = norm_U_mean;
      scalar h_eff          = std::exp(std::log(volume) / (double)dim) / (double)(data.degree + 1);
      scalar tau_viscous    = make_vectorized_array<Number>(data.viscosity) / h_eff;

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
  reinit_face(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const
  {
    tau = 0.5 * (integrator_m.read_cell_data(array_penalty_parameter) +
                 integrator_p.read_cell_data(array_penalty_parameter));
  }

  void
  reinit_face_cell_based(types::boundary_id const boundary_id,
                         IntegratorFace &         integrator_m,
                         IntegratorFace &         integrator_p) const
  {
    if(boundary_id == numbers::internal_face_boundary_id) // internal face
    {
      tau = 0.5 * (integrator_m.read_cell_data(array_penalty_parameter) +
                   integrator_p.read_cell_data(array_penalty_parameter));
    }
    else // boundary face
    {
      // will not be used since the continuity penalty operator is zero on boundary faces
      tau = integrator_m.read_cell_data(array_penalty_parameter);
    }
  }

  inline DEAL_II_ALWAYS_INLINE //
    vector
    calculate_flux(vector const & u_m, vector const & u_p, vector const & normal_m) const
  {
    vector jump_value = u_m - u_p;

    vector flux;

    if(data.which_components == ContinuityPenaltyComponents::All)
    {
      // penalize all velocity components
      flux = tau * jump_value;
    }
    else if(data.which_components == ContinuityPenaltyComponents::Normal)
    {
      flux = tau * (jump_value * normal_m) * normal_m;
    }
    else
    {
      AssertThrow(false, ExcMessage("not implemented."));
    }

    return flux;
  }


private:
  MatrixFree<dim, Number> const * matrix_free;

  unsigned int dof_index;
  unsigned int quad_index;

  ContinuityPenaltyKernelData data;

  AlignedVector<scalar> array_penalty_parameter;

  mutable scalar tau;
};

} // namespace Operators

struct ContinuityPenaltyData
{
  ContinuityPenaltyData() : dof_index(0), quad_index(0)
  {
  }

  unsigned int dof_index;

  unsigned int quad_index;
};

template<int dim, typename Number>
class ContinuityPenaltyOperator
{
private:
  typedef ContinuityPenaltyOperator<dim, Number> This;

  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef Tensor<1, dim, VectorizedArray<Number>> vector;

  typedef std::pair<unsigned int, unsigned int> Range;

  typedef FaceIntegrator<dim, dim, Number> IntegratorFace;

  typedef Operators::ContinuityPenaltyKernel<dim, Number> Kernel;

public:
  ContinuityPenaltyOperator();

  void
  reinit(MatrixFree<dim, Number> const & matrix_free,
         ContinuityPenaltyData const &   data,
         std::shared_ptr<Kernel> const   kernel);

  void
  update(VectorType const & velocity);

  void
  apply(VectorType & dst, VectorType const & src) const;

  void
  apply_add(VectorType & dst, VectorType const & src) const;

private:
  void
  cell_loop_empty(MatrixFree<dim, Number> const & matrix_free,
                  VectorType &                    dst,
                  VectorType const &              src,
                  Range const &                   range) const;

  void
  face_loop(MatrixFree<dim, Number> const & matrix_free,
            VectorType &                    dst,
            VectorType const &              src,
            Range const &                   face_range) const;

  void
  boundary_face_loop_empty(MatrixFree<dim, Number> const & matrix_free,
                           VectorType &                    dst,
                           VectorType const &              src,
                           Range const &                   range) const;

  void
  do_face_integral(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const;

  MatrixFree<dim, Number> const * matrix_free;

  ContinuityPenaltyData data;

  std::shared_ptr<Operators::ContinuityPenaltyKernel<dim, Number>> kernel;
};

} // namespace IncNS



#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_CONTINUITY_PENALTY_OPERATOR_H_ \
        */
