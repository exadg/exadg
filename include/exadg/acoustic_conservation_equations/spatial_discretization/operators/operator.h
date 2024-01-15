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
#include <exadg/operators/mapping_flags.h>

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
  static MappingFlags
  get_mapping_flags()
  {
    MappingFlags flags;

    flags.cells       = dealii::update_JxW_values | dealii::update_gradients;
    flags.inner_faces = dealii::update_JxW_values | dealii::update_normal_vectors;
    flags.boundary_faces =
      dealii::update_JxW_values | dealii::update_quadrature_points | dealii::update_normal_vectors;

    return flags;
  }

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
    calculate_lax_friedrichs_flux_momentum(vector const & rho_um,
                                           vector const & rho_up,
                                           Number const & gamma,
                                           scalar const & pm,
                                           scalar const & pp,
                                           vector const & n) const
  {
    return Number{0.5} * (rho_um + rho_up) + gamma * (pm - pp) * n;
  }

  /*
   * Lax Friedrichs flux for the mass equation.
   */
  inline DEAL_II_ALWAYS_INLINE //
    scalar
    calculate_lax_friedrichs_flux_mass(scalar const & pm,
                                       scalar const & pp,
                                       Number const & tau,
                                       vector const & rho_um,
                                       vector const & rho_up,
                                       vector const & n) const
  {
    return Number{0.5} * (pm + pp) + tau * (rho_um - rho_up) * n;
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
      block_index_pressure(0),
      block_index_velocity(1),
      formulation(Formulation::SkewSymmetric),
      bc(nullptr),
      speed_of_sound(-1.0)
  {
  }

  unsigned int dof_index_pressure;
  unsigned int dof_index_velocity;

  unsigned int quad_index;

  unsigned int block_index_pressure;
  unsigned int block_index_velocity;

  Formulation formulation;

  std::shared_ptr<BoundaryDescriptor<dim> const> bc;

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
  Operator() : evaluation_time(Number{0.0}), tau(Number{0.0}), gamma(Number{0.0})
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
    tau   = (Number)(0.5 * data_in.speed_of_sound);
    gamma = (Number)(0.5 / data_in.speed_of_sound);

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

  void
  do_cell_integral(CellIntegratorP & pressure, CellIntegratorU & velocity) const
  {
    if(data.formulation == Formulation::Weak)
    {
      for(unsigned int q : pressure.quadrature_point_indices())
      {
        vector const flux_momentum = kernel.get_volume_flux_weak_momentum(velocity, q);
        scalar const flux_mass     = kernel.get_volume_flux_weak_mass(pressure, q);

        pressure.submit_gradient(flux_momentum, q);
        velocity.submit_divergence(flux_mass, q);
      }
    }
    else if(data.formulation == Formulation::Strong)
    {
      for(unsigned int q : pressure.quadrature_point_indices())
      {
        scalar const flux_momentum = kernel.get_volume_flux_strong_momentum(velocity, q);
        vector const flux_mass     = kernel.get_volume_flux_strong_mass(pressure, q);

        pressure.submit_value(flux_momentum, q);
        velocity.submit_value(flux_mass, q);
      }
    }
    else if(data.formulation == Formulation::SkewSymmetric)
    {
      for(unsigned int q : pressure.quadrature_point_indices())
      {
        vector const flux_momentum = kernel.get_volume_flux_weak_momentum(velocity, q);
        vector const flux_mass     = kernel.get_volume_flux_strong_mass(pressure, q);

        pressure.submit_gradient(flux_momentum, q);
        velocity.submit_value(flux_mass, q);
      }
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
    }
  }


  template<bool weight_neighbor, // = false for cell centric loops and boundary loops
           typename ExteriorFaceIntegratorP,
           typename ExteriorFaceIntegratorU>
  void
  do_face_integral(FaceIntegratorP &         pressure_m,
                   ExteriorFaceIntegratorP & pressure_p,
                   FaceIntegratorU &         velocity_m,
                   ExteriorFaceIntegratorU & velocity_p) const
  {
    for(unsigned int q : pressure_m.quadrature_point_indices())
    {
      scalar const pm     = pressure_m.get_value(q);
      scalar const pp     = pressure_p.get_value(q);
      vector const rho_um = velocity_m.get_value(q);
      vector const rho_up = velocity_p.get_value(q);
      vector const n      = pressure_m.normal_vector(q);

      vector const flux_momentum =
        kernel.calculate_lax_friedrichs_flux_momentum(rho_um, rho_up, gamma, pm, pp, n);

      scalar const flux_mass =
        kernel.calculate_lax_friedrichs_flux_mass(pm, pp, tau, rho_um, rho_up, n);

      if(data.formulation == Formulation::Weak)
      {
        scalar const flux_momentum_weak = flux_momentum * n;
        vector const flux_mass_weak     = flux_mass * n;

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
        pressure_m.submit_value((flux_momentum - rho_um) * n, q);
        velocity_m.submit_value((flux_mass - pm) * n, q);

        if constexpr(weight_neighbor)
        {
          // minus signs since n⁺ = - n⁻
          pressure_p.submit_value((rho_up - flux_momentum) * n, q);
          velocity_p.submit_value((pp - flux_mass) * n, q);
        }
      }
      else if(data.formulation == Formulation::SkewSymmetric)
      {
        scalar const flux_momentum_weak = flux_momentum * n;

        pressure_m.submit_value(flux_momentum_weak, q);
        velocity_m.submit_value((flux_mass - pm) * n, q);

        if constexpr(weight_neighbor)
        {
          // minus signs since n⁺ = - n⁻
          pressure_p.submit_value(-flux_momentum_weak, q);
          velocity_p.submit_value((pp - flux_mass) * n, q);
        }
      }
      else
      {
        AssertThrow(false, dealii::ExcMessage("Not implemented."));
      }
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
      pressure.gather_evaluate(src.block(data.block_index_pressure),
                               integrator_flags_p.cell_evaluate);

      velocity.reinit(cell);
      velocity.gather_evaluate(src.block(data.block_index_velocity),
                               integrator_flags_u.cell_evaluate);

      do_cell_integral(pressure, velocity);

      pressure.integrate_scatter(integrator_flags_p.cell_integrate,
                                 dst.block(data.block_index_pressure));
      velocity.integrate_scatter(integrator_flags_u.cell_integrate,
                                 dst.block(data.block_index_velocity));
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
      pressure_m.gather_evaluate(src.block(data.block_index_pressure),
                                 integrator_flags_p.face_evaluate);
      pressure_p.reinit(face);
      pressure_p.gather_evaluate(src.block(data.block_index_pressure),
                                 integrator_flags_p.face_evaluate);

      velocity_m.reinit(face);
      velocity_m.gather_evaluate(src.block(data.block_index_velocity),
                                 integrator_flags_u.face_evaluate);
      velocity_p.reinit(face);
      velocity_p.gather_evaluate(src.block(data.block_index_velocity),
                                 integrator_flags_u.face_evaluate);

      do_face_integral<true>(pressure_m, pressure_p, velocity_m, velocity_p);

      pressure_m.integrate_scatter(integrator_flags_p.face_integrate,
                                   dst.block(data.block_index_pressure));
      pressure_p.integrate_scatter(integrator_flags_p.face_integrate,
                                   dst.block(data.block_index_pressure));

      velocity_m.integrate_scatter(integrator_flags_u.face_integrate,
                                   dst.block(data.block_index_velocity));
      velocity_p.integrate_scatter(integrator_flags_u.face_integrate,
                                   dst.block(data.block_index_velocity));
    }
  }

  void
  boundary_face_loop(dealii::MatrixFree<dim, Number> const & matrix_free_in,
                     BlockVectorType &                       dst,
                     BlockVectorType const &                 src,
                     Range const &                           face_range) const
  {
    FaceIntegratorP pressure_m(matrix_free_in, true, data.dof_index_pressure, data.quad_index);
    BoundaryFaceIntegratorP<dim, Number> pressure_p(pressure_m, *data.bc);

    FaceIntegratorU velocity_m(matrix_free_in, true, data.dof_index_velocity, data.quad_index);
    BoundaryFaceIntegratorU<dim, Number> velocity_p(velocity_m,
                                                    pressure_m,
                                                    data.speed_of_sound,
                                                    *data.bc);

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      pressure_m.reinit(face);
      pressure_m.gather_evaluate(src.block(data.block_index_pressure),
                                 integrator_flags_p.face_evaluate);
      pressure_p.reinit(face, evaluation_time);

      velocity_m.reinit(face);
      velocity_m.gather_evaluate(src.block(data.block_index_velocity),
                                 integrator_flags_u.face_evaluate);
      velocity_p.reinit(face, evaluation_time);

      do_face_integral<false>(pressure_m, pressure_p, velocity_m, velocity_p);

      pressure_m.integrate_scatter(integrator_flags_p.face_integrate,
                                   dst.block(data.block_index_pressure));
      velocity_m.integrate_scatter(integrator_flags_u.face_integrate,
                                   dst.block(data.block_index_velocity));
    }
  }

  Operators::Kernel<dim, Number> kernel;

  mutable Number evaluation_time;

  dealii::SmartPointer<dealii::MatrixFree<dim, Number> const> matrix_free;
  OperatorData<dim>                                           data;

  Number tau;
  Number gamma;

  IntegratorFlags integrator_flags_p;
  IntegratorFlags integrator_flags_u;
};

} // namespace Acoustics
} // namespace ExaDG


#endif /*EXADG_ACOUSTIC_CONSERVATION_EQUATIONS_SPATIAL_DISCRETIZATION_OPERATORS_OPERATOR_H_*/
