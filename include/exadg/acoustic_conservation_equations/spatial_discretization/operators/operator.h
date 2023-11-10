/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2023 by the ExaDG authors
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

#ifndef EXADG_ACOUSTIC_CONSERVATION_EQUATIONS_SPATIAL_DISCRETIZATION_OPERATORS_OPERATOR_H_
#define EXADG_ACOUSTIC_CONSERVATION_EQUATIONS_SPATIAL_DISCRETIZATION_OPERATORS_OPERATOR_H_

#include <deal.II/lac/la_parallel_block_vector.h>

#include <exadg/acoustic_conservation_equations/spatial_discretization/operators/weak_boundary_conditions.h>
#include <exadg/acoustic_conservation_equations/user_interface/boundary_descriptor.h>
#include <exadg/acoustic_conservation_equations/user_interface/enum_types.h>
#include <exadg/matrix_free/integrators.h>
#include <exadg/operators/integrator_flags.h>

namespace ExaDG
{
namespace Acoustics
{
namespace Operators
{
template<int dim, typename Number>
class Kernel
{
  using scalar = dealii::VectorizedArray<Number>;
  using vector = dealii::Tensor<1, dim, dealii::VectorizedArray<Number>>;

  using CellIntegratorP = CellIntegrator<dim, 1, Number>;
  using CellIntegratorU = CellIntegrator<dim, dim, Number>;

public:
  /*
   * Volume flux for the momentum equation, i.e., the term occurring in the volume integral for
   * weak formulation (performing integration-by-parts)
   */
  inline DEAL_II_ALWAYS_INLINE //
    vector
    get_volume_flux_weak_momentum(CellIntegratorU const & velocity, unsigned int const q) const
  {
    // minus sign due to integration by parts
    return -velocity.get_value(q);
  }

  /*
   * Volume flux for the momentum equation, i.e., the term occurring in the volume integral for
   * strong formulation (integration-by-parts performed twice)
   */
  inline DEAL_II_ALWAYS_INLINE //
    scalar
    get_volume_flux_strong_momentum(CellIntegratorU const & velocity, unsigned int const q) const
  {
    return velocity.get_divergence(q);
  }


  /*
   * Volume flux for the mass equation, i.e., the term occurring in the volume integral for
   * weak formulation (performing integration-by-parts)
   */
  inline DEAL_II_ALWAYS_INLINE //
    scalar
    get_volume_flux_weak_mass(CellIntegratorP const & pressure, unsigned int const q) const
  {
    // minus sign due to integration by parts
    return -pressure.get_value(q);
  }

  /*
   * Volume flux for the mass equation, i.e., the term occurring in the volume integral for
   * strong formulation (integration-by-parts performed twice)
   */
  inline DEAL_II_ALWAYS_INLINE //
    vector
    get_volume_flux_strong_mass(CellIntegratorP const & pressure, unsigned int const q) const
  {
    return pressure.get_gradient(q);
  }

  /*
   * Lax Friedrichs flux for the momentum equation.
   */
  inline DEAL_II_ALWAYS_INLINE //
    vector
    calculate_lax_friedrichs_flux_momentum(vector const & um,
                                           vector const & up,
                                           Number const & gamma,
                                           scalar const & pm,
                                           scalar const & pp,
                                           vector const & n) const
  {
    return Number{0.5} * (um + up) + gamma * (pm - pp) * n;
  }

  /*
   * Lax Friedrichs flux for the mass equation.
   */
  inline DEAL_II_ALWAYS_INLINE //
    scalar
    calculate_lax_friedrichs_flux_mass(scalar const & pm,
                                       scalar const & pp,
                                       Number const & tau,
                                       vector const & um,
                                       vector const & up,
                                       vector const & n) const
  {
    return Number{0.5} * (pm + pp) + tau * (um - up) * n;
  }
};

} // namespace Operators

template<int dim>
struct OperatorData
{
  OperatorData()
    : dof_index_pressure(0),
      dof_index_velocity(1),
      quad_index(0),
      formulation(Formulation::SkewSymmetric),
      bc(nullptr),
      density(-1.0),
      speed_of_sound(-1.0)
  {
  }

  unsigned int dof_index_pressure;
  unsigned int dof_index_velocity;

  unsigned int quad_index;

  Formulation formulation;

  std::shared_ptr<BoundaryDescriptor<dim> const> bc;

  double density;
  double speed_of_sound;
};

template<int dim, typename Number>
class Operator
{
  using This = Operator<dim, Number>;

  using BlockVectorType = dealii::LinearAlgebra::distributed::BlockVector<Number>;

  using scalar = dealii::VectorizedArray<Number>;
  using vector = dealii::Tensor<1, dim, dealii::VectorizedArray<Number>>;

  using Range = std::pair<unsigned int, unsigned int>;

  using CellIntegratorU = CellIntegrator<dim, dim, Number>;
  using CellIntegratorP = CellIntegrator<dim, 1, Number>;

  using FaceIntegratorU = FaceIntegrator<dim, dim, Number>;
  using FaceIntegratorP = FaceIntegrator<dim, 1, Number>;

public:
  Operator()
    : evaluation_time(Number{0.0}),
      rhocc(Number{0.0}),
      rho_inv(Number{0.0}),
      tau(Number{0.0}),
      gamma(Number{0.0})
  {
  }

  void
  initialize(dealii::MatrixFree<dim, Number> const & matrix_free_in,
             OperatorData<dim> const &               data_in)
  {
    // Set Matrixfree and OperatorData.
    matrix_free = &matrix_free_in;
    data        = data_in;

    // Precompute numbers that are needed in kernels.
    rhocc   = (Number)(data_in.density * data_in.speed_of_sound * data_in.speed_of_sound);
    rho_inv = (Number)(1.0 / data_in.density);
    tau     = (Number)(0.5 * data_in.speed_of_sound * data_in.density);
    gamma   = (Number)(0.5 / (data_in.speed_of_sound * data_in.density));

    // Set integration flags.
    if(data_in.formulation == Formulation::Weak)
    {
      integrator_flags_p.cell_evaluate  = dealii::EvaluationFlags::values;
      integrator_flags_p.cell_integrate = dealii::EvaluationFlags::gradients;
      integrator_flags_p.face_evaluate  = dealii::EvaluationFlags::values;
      integrator_flags_p.face_integrate = dealii::EvaluationFlags::values;

      integrator_flags_u.cell_evaluate  = dealii::EvaluationFlags::values;
      integrator_flags_u.cell_integrate = dealii::EvaluationFlags::gradients;
      integrator_flags_u.face_evaluate  = dealii::EvaluationFlags::values;
      integrator_flags_u.face_integrate = dealii::EvaluationFlags::values;
    }
    else if(data_in.formulation == Formulation::Strong)
    {
      integrator_flags_p.cell_evaluate  = dealii::EvaluationFlags::gradients;
      integrator_flags_p.cell_integrate = dealii::EvaluationFlags::values;
      integrator_flags_p.face_evaluate  = dealii::EvaluationFlags::values;
      integrator_flags_p.face_integrate = dealii::EvaluationFlags::values;

      integrator_flags_u.cell_evaluate  = dealii::EvaluationFlags::gradients;
      integrator_flags_u.cell_integrate = dealii::EvaluationFlags::values;
      integrator_flags_u.face_evaluate  = dealii::EvaluationFlags::values;
      integrator_flags_u.face_integrate = dealii::EvaluationFlags::values;
    }
    else if(data_in.formulation == Formulation::SkewSymmetric)
    {
      integrator_flags_p.cell_evaluate  = dealii::EvaluationFlags::gradients;
      integrator_flags_p.cell_integrate = dealii::EvaluationFlags::gradients;
      integrator_flags_p.face_evaluate  = dealii::EvaluationFlags::values;
      integrator_flags_p.face_integrate = dealii::EvaluationFlags::values;

      integrator_flags_u.cell_evaluate  = dealii::EvaluationFlags::values;
      integrator_flags_u.cell_integrate = dealii::EvaluationFlags::values;
      integrator_flags_u.face_evaluate  = dealii::EvaluationFlags::values;
      integrator_flags_u.face_integrate = dealii::EvaluationFlags::values;
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
    }
  }

  void
  evaluate(BlockVectorType & dst, BlockVectorType const & src, double const time) const
  {
    do_evaluate(dst, src, true, time);
  }

  void
  evaluate_add(BlockVectorType & dst, BlockVectorType const & src, double const time) const
  {
    do_evaluate(dst, src, false, time);
  }

private:
  void
  do_evaluate(BlockVectorType &       dst,
              BlockVectorType const & src,
              double const            time,
              bool const              zero_dst_vector) const
  {
    evaluation_time = (Number)time;

    matrix_free->loop(&This::cell_loop,
                      &This::face_loop,
                      &This::boundary_face_loop,
                      this,
                      dst,
                      src,
                      zero_dst_vector,
                      dealii::MatrixFree<dim, Number>::DataAccessOnFaces::values,
                      dealii::MatrixFree<dim, Number>::DataAccessOnFaces::values);
  }

  template<bool weight_neighbor>
  inline DEAL_II_ALWAYS_INLINE //
    void
    face_kernel(FaceIntegratorP &  pressure_m,
                FaceIntegratorP &  pressure_p,
                FaceIntegratorU &  velocity_m,
                FaceIntegratorU &  velocity_p,
                scalar const &     pm,
                scalar const &     pp,
                vector const &     um,
                vector const &     up,
                vector const &     n,
                unsigned int const q) const
  {
    vector const flux_momentum =
      kernel.calculate_lax_friedrichs_flux_momentum(um, up, gamma, pm, pp, n);

    scalar const flux_mass = kernel.calculate_lax_friedrichs_flux_mass(pm, pp, tau, um, up, n);

    if(data.formulation == Formulation::Weak)
    {
      scalar const flux_momentum_weak = rhocc * flux_momentum * n;
      vector const flux_mass_weak     = rho_inv * flux_mass * n;

      pressure_m.submit_value(flux_momentum_weak, q);
      velocity_m.submit_value(flux_mass_weak, q);

      if constexpr(weight_neighbor)
      {
        // minus signs since n⁺ = - n⁻
        pressure_p.submit_value(-flux_momentum_weak, q);
        velocity_p.submit_value(-flux_mass_weak, q);
      }
    }
    else if(data.formulation == Formulation::Strong)
    {
      pressure_m.submit_value(rhocc * (flux_momentum - um) * n, q);
      velocity_m.submit_value(rho_inv * (flux_mass - pm) * n, q);

      if constexpr(weight_neighbor)
      {
        // minus signs since n⁺ = - n⁻
        pressure_p.submit_value(-rhocc * (flux_momentum - up) * n, q);
        velocity_p.submit_value(-rho_inv * (flux_mass - pp) * n, q);
      }
    }
    else if(data.formulation == Formulation::SkewSymmetric)
    {
      scalar const flux_momentum_weak = rhocc * flux_momentum * n;

      pressure_m.submit_value(flux_momentum_weak, q);
      velocity_m.submit_value(rho_inv * (flux_mass - pm) * n, q);

      if constexpr(weight_neighbor)
      {
        // minus signs since n⁺ = - n⁻
        pressure_p.submit_value(-flux_momentum_weak, q);
        velocity_p.submit_value(-rho_inv * (flux_mass - pp) * n, q);
      }
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
    }
  }

  void
  cell_loop(dealii::MatrixFree<dim, Number> const & matrix_free_in,
            BlockVectorType &                       dst,
            BlockVectorType const &                 src,
            Range const &                           cell_range) const
  {
    CellIntegratorP pressure(matrix_free_in, data.dof_index_pressure, data.quad_index);
    CellIntegratorU velocity(matrix_free_in, data.dof_index_velocity, data.quad_index);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      pressure.reinit(cell);
      pressure.gather_evaluate(src.block(0), integrator_flags_p.cell_evaluate);

      velocity.reinit(cell);
      velocity.gather_evaluate(src.block(1), integrator_flags_u.cell_evaluate);

      if(data.formulation == Formulation::Weak)
      {
        for(unsigned int q : pressure.quadrature_point_indices())
        {
          vector const flux_momentum = kernel.get_volume_flux_weak_momentum(velocity, q);
          scalar const flux_mass     = kernel.get_volume_flux_weak_mass(pressure, q);

          pressure.submit_gradient(rhocc * flux_momentum, q);
          velocity.submit_divergence(rho_inv * flux_mass, q);
        }
      }
      else if(data.formulation == Formulation::Strong)
      {
        for(unsigned int q : pressure.quadrature_point_indices())
        {
          scalar const flux_momentum = kernel.get_volume_flux_strong_momentum(velocity, q);
          vector const flux_mass     = kernel.get_volume_flux_strong_mass(pressure, q);

          pressure.submit_value(rhocc * flux_momentum, q);
          velocity.submit_value(rho_inv * flux_mass, q);
        }
      }
      else if(data.formulation == Formulation::SkewSymmetric)
      {
        for(unsigned int q : pressure.quadrature_point_indices())
        {
          vector const flux_momentum = kernel.get_volume_flux_weak_momentum(velocity, q);
          vector const flux_mass     = kernel.get_volume_flux_strong_mass(pressure, q);

          pressure.submit_gradient(rhocc * flux_momentum, q);
          velocity.submit_value(rho_inv * flux_mass, q);
        }
      }
      else
      {
        AssertThrow(false, dealii::ExcMessage("Not implemented."));
      }

      pressure.integrate_scatter(integrator_flags_p.cell_integrate, dst.block(0));
      velocity.integrate_scatter(integrator_flags_u.cell_integrate, dst.block(1));
    }
  }

  void
  face_loop(dealii::MatrixFree<dim, Number> const & matrix_free_in,
            BlockVectorType &                       dst,
            BlockVectorType const &                 src,
            Range const &                           face_range) const
  {
    FaceIntegratorP pressure_m(matrix_free_in, true, data.dof_index_pressure, data.quad_index);
    FaceIntegratorP pressure_p(matrix_free_in, false, data.dof_index_pressure, data.quad_index);

    FaceIntegratorU velocity_m(matrix_free_in, true, data.dof_index_velocity, data.quad_index);
    FaceIntegratorU velocity_p(matrix_free_in, false, data.dof_index_velocity, data.quad_index);

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      pressure_m.reinit(face);
      pressure_m.gather_evaluate(src.block(0), integrator_flags_p.face_evaluate);
      pressure_p.reinit(face);
      pressure_p.gather_evaluate(src.block(0), integrator_flags_p.face_evaluate);

      velocity_m.reinit(face);
      velocity_m.gather_evaluate(src.block(1), integrator_flags_u.face_evaluate);
      velocity_p.reinit(face);
      velocity_p.gather_evaluate(src.block(1), integrator_flags_u.face_evaluate);

      for(unsigned int q : pressure_m.quadrature_point_indices())
      {
        scalar const pm = pressure_m.get_value(q);
        scalar const pp = pressure_p.get_value(q);
        vector const um = velocity_m.get_value(q);
        vector const up = velocity_p.get_value(q);
        vector const n  = pressure_m.normal_vector(q);

        face_kernel<true>(pressure_m, pressure_p, velocity_m, velocity_p, pm, pp, um, up, n, q);
      }

      pressure_m.integrate_scatter(integrator_flags_p.face_integrate, dst.block(0));
      pressure_p.integrate_scatter(integrator_flags_p.face_integrate, dst.block(0));

      velocity_m.integrate_scatter(integrator_flags_u.face_integrate, dst.block(1));
      velocity_p.integrate_scatter(integrator_flags_u.face_integrate, dst.block(1));
    }
  }

  void
  boundary_face_loop(dealii::MatrixFree<dim, Number> const & matrix_free_in,
                     BlockVectorType &                       dst,
                     BlockVectorType const &                 src,
                     Range const &                           face_range) const
  {
    FaceIntegratorP pressure_m(matrix_free_in, true, data.dof_index_pressure, data.quad_index);
    BoundaryFaceIntegratorP<dim, Number> pressure_p(pressure_m, *data.bc->pressure);

    FaceIntegratorU velocity_m(matrix_free_in, true, data.dof_index_velocity, data.quad_index);
    BoundaryFaceIntegratorU<dim, Number> velocity_p(velocity_m, *data.bc->velocity);

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      pressure_m.reinit(face);
      pressure_m.gather_evaluate(src.block(0), integrator_flags_p.face_evaluate);
      pressure_p.reinit(face, evaluation_time);

      velocity_m.reinit(face);
      velocity_m.gather_evaluate(src.block(1), integrator_flags_u.face_evaluate);
      velocity_p.reinit(face, evaluation_time);

      for(unsigned int q : pressure_m.quadrature_point_indices())
      {
        scalar const pm = pressure_m.get_value(q);
        scalar const pp = pressure_p.get_value(q);
        vector const um = velocity_m.get_value(q);
        vector const up = velocity_p.get_value(q);
        vector const n  = pressure_m.normal_vector(q);

        face_kernel<false>(pressure_m,
                           pressure_m /* unused */,
                           velocity_m,
                           velocity_m /* unused */,
                           pm,
                           pp,
                           um,
                           up,
                           n,
                           q);
      }

      pressure_m.integrate_scatter(integrator_flags_p.face_integrate, dst.block(0));
      velocity_m.integrate_scatter(integrator_flags_u.face_integrate, dst.block(1));
    }
  }

  Operators::Kernel<dim, Number> kernel;

  mutable Number evaluation_time;

  dealii::SmartPointer<dealii::MatrixFree<dim, Number> const> matrix_free;
  OperatorData<dim>                                           data;

  Number rhocc;
  Number rho_inv;
  Number tau;
  Number gamma;

  IntegratorFlags integrator_flags_p;
  IntegratorFlags integrator_flags_u;
};

} // namespace Acoustics
} // namespace ExaDG


#endif /*EXADG_ACOUSTIC_CONSERVATION_EQUATIONS_SPATIAL_DISCRETIZATION_OPERATORS_OPERATOR_H_*/
