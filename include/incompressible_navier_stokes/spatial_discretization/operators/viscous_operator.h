/*
 * viscous_operator.h
 *
 *  Created on: Nov 5, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_VISCOUS_OPERATOR_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_VISCOUS_OPERATOR_H_

#include <deal.II/matrix_free/fe_evaluation_notemplate.h>

#include "../../../operators/interior_penalty_parameter.h"
#include "../../user_interface/input_parameters.h"

#include "weak_boundary_conditions.h"

using namespace dealii;

namespace IncNS
{
template<int dim>
struct ViscousOperatorData
{
  ViscousOperatorData()
    : formulation_viscous_term(FormulationViscousTerm::DivergenceFormulation),
      penalty_term_div_formulation(PenaltyTermDivergenceFormulation::Symmetrized),
      IP_formulation(InteriorPenaltyFormulation::SIPG),
      degree(1),
      IP_factor(1.0),
      dof_index(0),
      quad_index(0),
      viscosity(1.0),
      use_cell_based_loops(false)
  {
  }

  FormulationViscousTerm           formulation_viscous_term;
  PenaltyTermDivergenceFormulation penalty_term_div_formulation;
  InteriorPenaltyFormulation       IP_formulation;

  unsigned int degree;
  double       IP_factor;
  unsigned int dof_index;
  unsigned int quad_index;

  std::shared_ptr<BoundaryDescriptorU<dim>> bc;

  /*
   * This variable 'viscosity' is only used when initializing the ViscousOperator.
   * In order to change/update this coefficient during the simulation (e.g., varying
   * viscosity/turbulence) use the element variable 'const_viscosity' of ViscousOperator and the
   * corresponding setter set_constant_viscosity().
   */
  double viscosity;

  // use cell based loops
  bool use_cell_based_loops;
};

template<int dim, typename Number>
class ViscousOperator
{
public:
  typedef ViscousOperator<dim, Number> This;

  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef VectorizedArray<Number>                 scalar;
  typedef Tensor<1, dim, VectorizedArray<Number>> vector;
  typedef Tensor<2, dim, VectorizedArray<Number>> tensor;

  typedef std::pair<unsigned int, unsigned int> Range;

  typedef Number value_type;

  typedef CellIntegrator<dim, dim, Number> CellIntegratorU;
  typedef FaceIntegrator<dim, dim, Number> FaceIntegratorU;

  ViscousOperator()
    : matrix_free(nullptr),
      const_viscosity(-1.0),
      eval_time(0.0),
      n_mpi_processes(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
  {
  }

  void
  initialize(Mapping<dim> const &             mapping,
             MatrixFree<dim, Number> const &  matrix_free_in,
             ViscousOperatorData<dim> const & operator_data_in)
  {
    this->matrix_free   = &matrix_free_in;
    this->operator_data = operator_data_in;

    IP::calculate_penalty_parameter<dim, Number>(array_penalty_parameter,
                                                 *matrix_free,
                                                 mapping,
                                                 operator_data.degree,
                                                 operator_data.dof_index);

    const_viscosity = operator_data.viscosity;

    // Block Jacobi elementwise
    integrator.reset(
      new CellIntegratorU(*matrix_free, operator_data.dof_index, operator_data.quad_index));
    integrator_m.reset(
      new FaceIntegratorU(*matrix_free, true, operator_data.dof_index, operator_data.quad_index));
    integrator_p.reset(
      new FaceIntegratorU(*matrix_free, false, operator_data.dof_index, operator_data.quad_index));
  }

  ViscousOperatorData<dim> const &
  get_operator_data() const
  {
    return operator_data;
  }

  void
  set_constant_viscosity(Number const viscosity_in)
  {
    const_viscosity = viscosity_in;
  }

  Number
  get_const_viscosity() const
  {
    return const_viscosity;
  }

  scalar
  get_viscosity(unsigned int const face, unsigned int const q) const
  {
    scalar viscosity = make_vectorized_array<Number>(const_viscosity);

    if(viscosity_is_variable())
      viscosity = viscous_coefficient_face[face][q];

    return viscosity;
  }

  /*
   *  This function returns true if viscous_coefficient table has been filled
   *  with spatially varying viscosity values.
   */
  bool
  viscosity_is_variable() const
  {
    return viscous_coefficient_cell.n_elements() > 0;
  }

  void
  initialize_viscous_coefficients()
  {
    viscous_coefficient_cell.reinit(matrix_free->n_cell_batches(),
                                    Utilities::pow(operator_data.degree + 1, dim));
    viscous_coefficient_cell.fill(make_vectorized_array<Number>(const_viscosity));

    viscous_coefficient_face.reinit(matrix_free->n_inner_face_batches() +
                                      matrix_free->n_boundary_face_batches(),
                                    Utilities::pow(operator_data.degree + 1, dim - 1));
    viscous_coefficient_face.fill(make_vectorized_array<Number>(const_viscosity));

    viscous_coefficient_face_neighbor.reinit(matrix_free->n_inner_face_batches(),
                                             Utilities::pow(operator_data.degree + 1, dim - 1));
    viscous_coefficient_face_neighbor.fill(make_vectorized_array<Number>(const_viscosity));

    // TODO
    //    viscous_coefficient_face_cell_based.reset(new
    //    Table<3,VectorizedArray<Number>>(matrix_free->n_cell_batches(),
    //                                                                                         2*dim,
    //                                                                                         Utilities::pow(n_actual_q_points_vel_linear,
    //                                                                                         dim -
    //                                                                                         1)));
    //    viscous_coefficient_face_cell_based->fill(make_vectorized_array<Number>(const_viscosity));
  }

  void
  set_viscous_coefficient_cell(unsigned int const cell, unsigned int const q, scalar const & value)
  {
    viscous_coefficient_cell[cell][q] = value;
  }

  void
  set_viscous_coefficient_face(unsigned int const face, unsigned int const q, scalar const & value)
  {
    viscous_coefficient_face[face][q] = value;
  }

  void
  set_viscous_coefficient_face_neighbor(unsigned int const face,
                                        unsigned int const q,
                                        scalar const &     value)
  {
    viscous_coefficient_face_neighbor[face][q] = value;
  }

  Table<2, scalar> const &
  get_viscous_coefficient_face() const
  {
    return viscous_coefficient_face;
  }

  Table<2, scalar> const &
  get_viscous_coefficient_cell() const
  {
    return viscous_coefficient_cell;
  }

  // apply matrix vector multiplication
  void
  vmult(VectorType & dst, VectorType const & src) const
  {
    apply(dst, src);
  }

  // apply matrix vector multiplication
  void
  apply(VectorType & dst, VectorType const & src) const
  {
    matrix_free->loop(&This::cell_loop,
                      &This::face_loop,
                      &This::boundary_face_loop_hom_operator,
                      this,
                      dst,
                      src,
                      true /*zero_dst_vector = true*/);
  }

  void
  apply_add(VectorType & dst, VectorType const & src) const
  {
    matrix_free->loop(&This::cell_loop,
                      &This::face_loop,
                      &This::boundary_face_loop_hom_operator,
                      this,
                      dst,
                      src,
                      false /*zero_dst_vector = false*/);
  }

  // apply matrix vector multiplication for block Jacobi operator
  void
  apply_block_diagonal(VectorType & dst, VectorType const & src) const
  {
    matrix_free->loop(&This::cell_loop,
                      &This::face_loop_block_jacobi,
                      &This::boundary_face_loop_hom_operator,
                      this,
                      dst,
                      src,
                      true /*zero_dst_vector = true*/);
  }

  void
  apply_block_diagonal_add(VectorType & dst, VectorType const & src) const
  {
    matrix_free->loop(&This::cell_loop,
                      &This::face_loop_block_jacobi,
                      &This::boundary_face_loop_hom_operator,
                      this,
                      dst,
                      src,
                      false /*zero_dst_vector = false*/);
  }

  void
  rhs(VectorType & dst, Number const evaluation_time) const
  {
    dst = 0.0;
    rhs_add(dst, evaluation_time);
  }

  void
  rhs_add(VectorType & dst, Number const evaluation_time) const
  {
    this->eval_time = evaluation_time;

    VectorType tmp;
    tmp.reinit(dst, false /* init with 0 */);

    matrix_free->loop(&This::cell_loop_empty,
                      &This::face_loop_empty,
                      &This::boundary_face_loop_inhom_operator,
                      this,
                      tmp,
                      tmp,
                      false /*zero_dst_vector = false*/);

    // multiply by -1.0 since the boundary face integrals have to be shifted to the right hand side
    dst.add(-1.0, tmp);
  }

  void
  evaluate(VectorType & dst, VectorType const & src, Number const evaluation_time) const
  {
    this->eval_time = evaluation_time;

    matrix_free->loop(&This::cell_loop,
                      &This::face_loop,
                      &This::boundary_face_loop_full_operator,
                      this,
                      dst,
                      src,
                      true /*zero_dst_vector = true*/);
  }

  void
  evaluate_add(VectorType & dst, VectorType const & src, Number const evaluation_time) const
  {
    this->eval_time = evaluation_time;

    matrix_free->loop(&This::cell_loop,
                      &This::face_loop,
                      &This::boundary_face_loop_full_operator,
                      this,
                      dst,
                      src,
                      false /*zero_dst_vector = false*/);
  }

  void
  calculate_diagonal(VectorType & diagonal) const
  {
    VectorType src;

    matrix_free->loop(&This::cell_loop_diagonal,
                      &This::face_loop_diagonal,
                      &This::boundary_face_loop_diagonal,
                      this,
                      diagonal,
                      src,
                      true /*zero_dst_vector = true*/);
  }

  void
  add_diagonal(VectorType & diagonal) const
  {
    VectorType src;

    matrix_free->loop(&This::cell_loop_diagonal,
                      &This::face_loop_diagonal,
                      &This::boundary_face_loop_diagonal,
                      this,
                      diagonal,
                      src,
                      false /*zero_dst_vector = false*/);
  }

  void
  add_block_diagonal_matrices(std::vector<LAPACKFullMatrix<Number>> & matrices) const
  {
    VectorType src;

    if(operator_data.use_cell_based_loops)
    {
      matrix_free->cell_loop(&This::cell_based_loop_calculate_block_diagonal, this, matrices, src);
    }
    else
    {
      AssertThrow(
        n_mpi_processes == 1,
        ExcMessage(
          "Block diagonal calculation with separate loops over cells and faces only works in serial. "
          "Use cell based loops for parallel computations."));

      matrix_free->loop(&This::cell_loop_calculate_block_diagonal,
                        &This::face_loop_calculate_block_diagonal,
                        &This::boundary_face_loop_calculate_block_diagonal,
                        this,
                        matrices,
                        src);
    }
  }

  void
  apply_add_block_diagonal_elementwise(unsigned int const   cell,
                                       scalar * const       dst,
                                       scalar const * const src,
                                       unsigned int const   problem_size = 1) const
  {
    (void)problem_size;

    unsigned int dofs_per_cell = integrator->dofs_per_cell;

    integrator->reinit(cell);

    for(unsigned int i = 0; i < dofs_per_cell; ++i)
      integrator->begin_dof_values()[i] = src[i];

    integrator->evaluate(false, true, false);

    do_cell_integral(*integrator, cell);

    integrator->integrate(false, true);

    for(unsigned int i = 0; i < dofs_per_cell; ++i)
      dst[i] += integrator->begin_dof_values()[i];

    // loop over all faces
    unsigned int const n_faces = GeometryInfo<dim>::faces_per_cell;
    for(unsigned int face = 0; face < n_faces; ++face)
    {
      integrator_m->reinit(cell, face);
      integrator_p->reinit(cell, face);

      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        integrator_m->begin_dof_values()[i] = src[i];

      // do not need to read dof values for integrator_p (already initialized with 0)

      integrator_m->evaluate(true, true);

      auto bids = matrix_free->get_faces_by_cells_boundary_id(cell, face);
      auto bid  = bids[0];

      if(bid == numbers::internal_face_boundary_id) // internal face
      {
        // TODO specify the correct cell, face indices to obtain the correct, variable viscosity
        do_face_int_integral(*integrator_m, *integrator_p, 0 /* cell, face */);
      }
      else // boundary face
      {
        // TODO specify the correct cell, face indices to obtain the correct, variable viscosity
        do_boundary_integral(*integrator_m, OperatorType::homogeneous, bid, 0 /* cell, face */);
      }

      integrator_m->integrate(true, true);

      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        dst[i] += integrator_m->begin_dof_values()[i];
    }
  }

private:
  template<typename Integrator>
  inline void
  do_cell_integral(Integrator & integrator, unsigned int const cell) const
  {
    AssertThrow(const_viscosity >= 0.0, ExcMessage("Constant viscosity has not been set!"));

    for(unsigned int q = 0; q < integrator.n_q_points; ++q)
    {
      scalar viscosity = make_vectorized_array<Number>(const_viscosity);
      if(viscosity_is_variable())
        viscosity = viscous_coefficient_cell[cell][q];

      if(operator_data.formulation_viscous_term == FormulationViscousTerm::DivergenceFormulation)
      {
        integrator.submit_gradient(viscosity * make_vectorized_array<Number>(2.) *
                                     integrator.get_symmetric_gradient(q),
                                   q);
      }
      else if(operator_data.formulation_viscous_term == FormulationViscousTerm::LaplaceFormulation)
      {
        integrator.submit_gradient(viscosity * integrator.get_gradient(q), q);
      }
      else
      {
        AssertThrow(operator_data.formulation_viscous_term ==
                        FormulationViscousTerm::DivergenceFormulation ||
                      operator_data.formulation_viscous_term ==
                        FormulationViscousTerm::LaplaceFormulation,
                    ExcMessage("Specified formulation of viscous term is not implemented."));
      }
    }
  }

  template<typename Integrator>
  void
  do_face_integral(Integrator &       integrator_m,
                   Integrator &       integrator_p,
                   unsigned int const face) const
  {
    scalar penalty_parameter =
      IP::get_penalty_factor<Number>(operator_data.degree, operator_data.IP_factor) *
      std::max(integrator_m.read_cell_data(array_penalty_parameter),
               integrator_p.read_cell_data(array_penalty_parameter));

    for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
    {
      scalar average_viscosity = make_vectorized_array<Number>(const_viscosity);
      if(viscosity_is_variable())
        average_viscosity = calculate_average_viscosity(face, q);

      vector value_m = integrator_m.get_value(q);
      vector value_p = integrator_p.get_value(q);
      vector normal  = integrator_m.get_normal_vector(q);

      tensor value_flux = calculate_value_flux(value_m, value_p, normal, average_viscosity);

      vector normal_gradient_m = calculate_normal_gradient(q, integrator_m);
      vector normal_gradient_p = calculate_normal_gradient(q, integrator_p);

      vector gradient_flux = calculate_gradient_flux(normal_gradient_m,
                                                     normal_gradient_p,
                                                     value_m,
                                                     value_p,
                                                     normal,
                                                     average_viscosity,
                                                     penalty_parameter);

      integrator_m.submit_gradient(value_flux, q);
      integrator_p.submit_gradient(value_flux, q);

      integrator_m.submit_value(-gradient_flux, q);
      integrator_p.submit_value(gradient_flux, q); // + sign since n⁺ = -n⁻
    }
  }

  template<typename Integrator>
  void
  do_face_int_integral(Integrator &       integrator_m,
                       Integrator &       integrator_p,
                       unsigned int const face) const
  {
    scalar penalty_parameter =
      IP::get_penalty_factor<Number>(operator_data.degree, operator_data.IP_factor) *
      std::max(integrator_m.read_cell_data(array_penalty_parameter),
               integrator_p.read_cell_data(array_penalty_parameter));

    for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
    {
      scalar average_viscosity = make_vectorized_array<Number>(const_viscosity);
      if(viscosity_is_variable())
        average_viscosity = calculate_average_viscosity(face, q);

      // set exterior values to zero
      vector value_m = integrator_m.get_value(q);
      vector value_p;

      vector normal_m = integrator_m.get_normal_vector(q);

      tensor value_flux = calculate_value_flux(value_m, value_p, normal_m, average_viscosity);

      vector normal_gradient_m = calculate_normal_gradient(q, integrator_m);
      vector normal_gradient_p; // set exterior gradient to zero

      vector gradient_flux = calculate_gradient_flux(normal_gradient_m,
                                                     normal_gradient_p,
                                                     value_m,
                                                     value_p,
                                                     normal_m,
                                                     average_viscosity,
                                                     penalty_parameter);

      integrator_m.submit_gradient(value_flux, q);
      integrator_m.submit_value(-gradient_flux, q);
    }
  }


  template<typename Integrator>
  void
  do_face_ext_integral(Integrator &       integrator_m,
                       Integrator &       integrator_p,
                       unsigned int const face) const
  {
    scalar penalty_parameter =
      IP::get_penalty_factor<Number>(operator_data.degree, operator_data.IP_factor) *
      std::max(integrator_m.read_cell_data(array_penalty_parameter),
               integrator_p.read_cell_data(array_penalty_parameter));

    for(unsigned int q = 0; q < integrator_p.n_q_points; ++q)
    {
      scalar average_viscosity = make_vectorized_array<Number>(const_viscosity);
      if(viscosity_is_variable())
        average_viscosity = calculate_average_viscosity(face, q);

      // set exterior values to zero
      vector value_m;
      vector value_p = integrator_p.get_value(q);
      // multiply by -1.0 to get the correct normal vector !!!
      vector normal_p = -integrator_p.get_normal_vector(q);

      tensor value_flux = calculate_value_flux(value_p, value_m, normal_p, average_viscosity);

      // set exterior gradient to zero
      vector normal_gradient_m;
      // multiply by -1.0 since normal vector n⁺ = -n⁻ !!!
      vector normal_gradient_p = -calculate_normal_gradient(q, integrator_p);

      vector gradient_flux = calculate_gradient_flux(normal_gradient_p,
                                                     normal_gradient_m,
                                                     value_p,
                                                     value_m,
                                                     normal_p,
                                                     average_viscosity,
                                                     penalty_parameter);

      integrator_p.submit_gradient(value_flux, q);
      integrator_p.submit_value(-gradient_flux, q);
    }
  }

  template<typename Integrator>
  void
  do_boundary_integral(Integrator &               integrator,
                       OperatorType const &       operator_type,
                       types::boundary_id const & boundary_id,
                       unsigned int const         face) const
  {
    BoundaryTypeU boundary_type = operator_data.bc->get_boundary_type(boundary_id);

    scalar penalty_parameter =
      IP::get_penalty_factor<Number>(operator_data.degree, operator_data.IP_factor) *
      integrator.read_cell_data(array_penalty_parameter);

    for(unsigned int q = 0; q < integrator.n_q_points; ++q)
    {
      scalar viscosity = make_vectorized_array<Number>(const_viscosity);
      if(viscosity_is_variable())
        viscosity = viscous_coefficient_face[face][q];

      vector value_m = calculate_interior_value(q, integrator, operator_type);
      vector value_p = calculate_exterior_value(value_m,
                                                q,
                                                integrator,
                                                operator_type,
                                                boundary_type,
                                                boundary_id,
                                                operator_data.bc,
                                                this->eval_time);

      vector normal = integrator.get_normal_vector(q);

      tensor value_flux = calculate_value_flux(value_m, value_p, normal, viscosity);

      vector normal_gradient_m = calculate_interior_normal_gradient(q, integrator, operator_type);
      vector normal_gradient_p = calculate_exterior_normal_gradient(normal_gradient_m,
                                                                    q,
                                                                    integrator,
                                                                    operator_type,
                                                                    boundary_type,
                                                                    boundary_id,
                                                                    operator_data.bc,
                                                                    this->eval_time);

      vector gradient_flux = calculate_gradient_flux(normal_gradient_m,
                                                     normal_gradient_p,
                                                     value_m,
                                                     value_p,
                                                     normal,
                                                     viscosity,
                                                     penalty_parameter);

      integrator.submit_gradient(value_flux, q);
      integrator.submit_value(-gradient_flux, q);
    }
  }

  /*
   *  This function calculates the average viscosity for interior faces.
   */
  inline DEAL_II_ALWAYS_INLINE //
    scalar
    calculate_average_viscosity(unsigned int const face, unsigned int const q) const
  {
    scalar average_viscosity = make_vectorized_array<Number>(0.0);

    // harmonic mean (harmonic weighting according to Schott and Rasthofer et al. (2015))
    average_viscosity =
      2.0 * viscous_coefficient_face[face][q] * viscous_coefficient_face_neighbor[face][q] /
      (viscous_coefficient_face[face][q] + viscous_coefficient_face_neighbor[face][q]);

    // arithmetic mean
    //    average_viscosity = 0.5 * (viscous_coefficient_face[face][q] +
    //    viscous_coefficient_face_neighbor[face][q]);

    // maximum value
    //    average_viscosity = std::max(viscous_coefficient_face[face][q],
    //    viscous_coefficient_face_neighbor[face][q]);

    return average_viscosity;
  }


  /*
   *  Calculation of "value_flux".
   */
  inline DEAL_II_ALWAYS_INLINE //
    tensor
    calculate_value_flux(vector const & value_m,
                         vector const & value_p,
                         vector const & normal,
                         scalar const & viscosity) const
  {
    tensor value_flux;

    vector jump_value  = value_m - value_p;
    tensor jump_tensor = outer_product(jump_value, normal);

    if(operator_data.formulation_viscous_term == FormulationViscousTerm::LaplaceFormulation)
    {
      if(operator_data.IP_formulation == InteriorPenaltyFormulation::NIPG)
      {
        value_flux = 0.5 * viscosity * jump_tensor;
      }
      else if(operator_data.IP_formulation == InteriorPenaltyFormulation::SIPG)
      {
        value_flux = -0.5 * viscosity * jump_tensor;
      }
      else
      {
        AssertThrow(false,
                    ExcMessage("Specified interior penalty formulation is not implemented."));
      }
    }
    else if(operator_data.formulation_viscous_term == FormulationViscousTerm::DivergenceFormulation)
    {
      if(operator_data.IP_formulation == InteriorPenaltyFormulation::NIPG)
      {
        value_flux = 0.5 * viscosity * (jump_tensor + transpose(jump_tensor));
      }
      else if(operator_data.IP_formulation == InteriorPenaltyFormulation::SIPG)
      {
        value_flux = -0.5 * viscosity * (jump_tensor + transpose(jump_tensor));
      }
      else
      {
        AssertThrow(false,
                    ExcMessage("Specified interior penalty formulation is not implemented."));
      }
    }
    else
    {
      AssertThrow(false, ExcMessage("Specified formulation of viscous term is not implemented."));
    }

    return value_flux;
  }

  /*
   *  This function calculates the gradient in normal direction on element e
   *  depending on the formulation of the viscous term.
   */
  template<typename Integrator>
  inline DEAL_II_ALWAYS_INLINE //
    vector
    calculate_normal_gradient(unsigned int const q, Integrator & integrator) const
  {
    tensor gradient;

    if(operator_data.formulation_viscous_term == FormulationViscousTerm::DivergenceFormulation)
    {
      /*
       * F = 2 * nu * symmetric_gradient
       *   = 2.0 * nu * 1/2 (grad(u) + grad(u)^T)
       */
      gradient = make_vectorized_array<Number>(2.0) * integrator.get_symmetric_gradient(q);
    }
    else if(operator_data.formulation_viscous_term == FormulationViscousTerm::LaplaceFormulation)
    {
      /*
       *  F = nu * grad(u)
       */
      gradient = integrator.get_gradient(q);
    }
    else
    {
      AssertThrow(operator_data.formulation_viscous_term ==
                      FormulationViscousTerm::DivergenceFormulation ||
                    operator_data.formulation_viscous_term ==
                      FormulationViscousTerm::LaplaceFormulation,
                  ExcMessage("Specified formulation of viscous term is not implemented."));
    }

    vector normal_gradient = gradient * integrator.get_normal_vector(q);

    return normal_gradient;
  }

  /*
   *  Calculation of gradient flux. Strictly speaking, this value is not a numerical flux since
   *  the flux is multiplied by the normal vector, i.e., "gradient_flux" = numerical_flux * normal,
   *  where normal denotes the normal vector of element e⁻.
   */
  inline DEAL_II_ALWAYS_INLINE //
    vector
    calculate_gradient_flux(vector const & normal_gradient_m,
                            vector const & normal_gradient_p,
                            vector const & value_m,
                            vector const & value_p,
                            vector const & normal,
                            scalar const & viscosity,
                            scalar const & penalty_parameter) const
  {
    vector gradient_flux;

    vector jump_value              = value_m - value_p;
    vector average_normal_gradient = 0.5 * (normal_gradient_m + normal_gradient_p);

    if(operator_data.formulation_viscous_term == FormulationViscousTerm::DivergenceFormulation)
    {
      if(operator_data.penalty_term_div_formulation ==
         PenaltyTermDivergenceFormulation::Symmetrized)
      {
        gradient_flux =
          viscosity * average_normal_gradient -
          viscosity * penalty_parameter * (jump_value + (jump_value * normal) * normal);
      }
      else if(operator_data.penalty_term_div_formulation ==
              PenaltyTermDivergenceFormulation::NotSymmetrized)
      {
        gradient_flux =
          viscosity * average_normal_gradient - viscosity * penalty_parameter * jump_value;
      }
      else
      {
        AssertThrow(operator_data.penalty_term_div_formulation ==
                        PenaltyTermDivergenceFormulation::Symmetrized ||
                      operator_data.penalty_term_div_formulation ==
                        PenaltyTermDivergenceFormulation::NotSymmetrized,
                    ExcMessage("Specified formulation of viscous term is not implemented."));
      }
    }
    else if(operator_data.formulation_viscous_term == FormulationViscousTerm::LaplaceFormulation)
    {
      gradient_flux =
        viscosity * average_normal_gradient - viscosity * penalty_parameter * jump_value;
    }
    else
    {
      AssertThrow(operator_data.formulation_viscous_term ==
                      FormulationViscousTerm::DivergenceFormulation ||
                    operator_data.formulation_viscous_term ==
                      FormulationViscousTerm::LaplaceFormulation,
                  ExcMessage("Specified formulation of viscous term is not implemented."));
    }

    return gradient_flux;
  }

  // clang-format off
  /*
   *  These two functions calculates the velocity gradient in normal
   *  direction depending on the operator type, the type of the boundary face
   *  and the given boundary conditions.
   *
   *  Divergence formulation: F(u) = nu * ( grad(u) + grad(u)^T )
   *  Laplace formulation: F(u) = nu * grad(u)
   *
   *                            +---------------------------------+---------------------------------------+----------------------------------------------------+
   *                            | Dirichlet boundaries            | Neumann boundaries                    | symmetry boundaries                                |
   *  +-------------------------+---------------------------------+---------------------------------------+----------------------------------------------------+
   *  | full operator           | F(u⁺)*n = F(u⁻)*n               | F(u⁺)*n = -F(u⁻)*n + 2h               | F(u⁺)*n = -F(u⁻)*n + 2*[(F(u⁻)*n)*n]n              |
   *  +-------------------------+---------------------------------+---------------------------------------+----------------------------------------------------+
   *  | homogeneous operator    | F(u⁺)*n = F(u⁻)*n               | F(u⁺)*n = -F(u⁻)*n                    | F(u⁺)*n = -F(u⁻)*n + 2*[(F(u⁻)*n)*n]n              |
   *  +-------------------------+---------------------------------+---------------------------------------+----------------------------------------------------+
   *  | inhomogeneous operator  | F(u⁺)*n = F(u⁻)*n, F(u⁻)*n = 0  | F(u⁺)*n = -F(u⁻)*n + 2h , F(u⁻)*n = 0 | F(u⁺)*n = -F(u⁻)*n + 2*[(F(u⁻)*n)*n]n, F(u⁻)*n = 0 |
   *  +-------------------------+---------------------------------+---------------------------------------+----------------------------------------------------+
   *
   *                            +---------------------------------+---------------------------------------+----------------------------------------------------+
   *                            | Dirichlet boundaries            | Neumann boundaries                    | symmetry boundaries                                |
   *  +-------------------------+---------------------------------+---------------------------------------+----------------------------------------------------+
   *  | full operator           | {{F(u)}}*n = F(u⁻)*n            | {{F(u)}}*n = h                        | {{F(u)}}*n = 2*[(F(u⁻)*n)*n]n                      |
   *  +-------------------------+---------------------------------+---------------------------------------+----------------------------------------------------+
   *  | homogeneous operator    | {{F(u)}}*n = F(u⁻)*n            | {{F(u)}}*n = 0                        | {{F(u)}}*n = 2*[(F(u⁻)*n)*n]n                      |
   *  +-------------------------+---------------------------------+---------------------------------------+----------------------------------------------------+
   *  | inhomogeneous operator  | {{F(u)}}*n = 0                  | {{F(u)}}*n = h                        | {{F(u)}}*n = 0                                     |
   *  +-------------------------+---------------------------------+---------------------------------------+----------------------------------------------------+
   */
  // clang-format on
  template<typename Integrator>
  inline DEAL_II_ALWAYS_INLINE //
    vector
    calculate_interior_normal_gradient(unsigned int const   q,
                                       Integrator const &   integrator,
                                       OperatorType const & operator_type) const
  {
    vector normal_gradient_m;

    if(operator_type == OperatorType::full || operator_type == OperatorType::homogeneous)
    {
      normal_gradient_m = calculate_normal_gradient(q, integrator);
    }
    else if(operator_type == OperatorType::inhomogeneous)
    {
      // do nothing, normal_gradient_m is already initialized with 0
    }
    else
    {
      AssertThrow(false, ExcMessage("Specified OperatorType is not implemented!"));
    }

    return normal_gradient_m;
  }

  void
  cell_loop(MatrixFree<dim, Number> const & matrix_free,
            VectorType &                    dst,
            VectorType const &              src,
            Range const &                   cell_range) const
  {
    CellIntegratorU integrator(matrix_free, operator_data.dof_index, operator_data.quad_index);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      integrator.reinit(cell);
      integrator.gather_evaluate(src, false, true, false);

      do_cell_integral(integrator, cell);

      integrator.integrate_scatter(false, true, dst);
    }
  }

  void
  face_loop(MatrixFree<dim, Number> const & matrix_free,
            VectorType &                    dst,
            VectorType const &              src,
            Range const &                   face_range) const
  {
    FaceIntegratorU integrator_m(matrix_free,
                                 true,
                                 operator_data.dof_index,
                                 operator_data.quad_index);

    FaceIntegratorU integrator_p(matrix_free,
                                 false,
                                 operator_data.dof_index,
                                 operator_data.quad_index);

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      integrator_m.reinit(face);
      integrator_p.reinit(face);

      integrator_m.gather_evaluate(src, true, true);
      integrator_p.gather_evaluate(src, true, true);

      do_face_integral(integrator_m, integrator_p, face);

      integrator_m.integrate_scatter(true, true, dst);
      integrator_p.integrate_scatter(true, true, dst);
    }
  }

  void
  boundary_face_loop_hom_operator(MatrixFree<dim, Number> const & matrix_free,
                                  VectorType &                    dst,
                                  VectorType const &              src,
                                  Range const &                   face_range) const
  {
    FaceIntegratorU integrator(matrix_free,
                               true,
                               operator_data.dof_index,
                               operator_data.quad_index);

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      types::boundary_id boundary_id = matrix_free.get_boundary_id(face);

      integrator.reinit(face);

      integrator.gather_evaluate(src, true, true);

      do_boundary_integral(integrator, OperatorType::homogeneous, boundary_id, face);

      integrator.integrate_scatter(true, true, dst);
    }
  }

  void
  boundary_face_loop_full_operator(MatrixFree<dim, Number> const & matrix_free,
                                   VectorType &                    dst,
                                   VectorType const &              src,
                                   Range const &                   face_range) const
  {
    FaceIntegratorU integrator(matrix_free,
                               true,
                               operator_data.dof_index,
                               operator_data.quad_index);

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      types::boundary_id boundary_id = matrix_free.get_boundary_id(face);

      integrator.reinit(face);

      integrator.gather_evaluate(src, true, true);

      do_boundary_integral(integrator, OperatorType::full, boundary_id, face);

      integrator.integrate_scatter(true, true, dst);
    }
  }

  void
  cell_loop_empty(MatrixFree<dim, Number> const &,
                  VectorType &,
                  VectorType const &,
                  Range const &) const
  {
  }

  void
  face_loop_empty(MatrixFree<dim, Number> const &,
                  VectorType &,
                  VectorType const &,
                  Range const &) const
  {
  }

  void
  boundary_face_loop_inhom_operator(MatrixFree<dim, Number> const & matrix_free,
                                    VectorType &                    dst,
                                    VectorType const &,
                                    Range const & face_range) const
  {
    FaceIntegratorU integrator(matrix_free,
                               true,
                               operator_data.dof_index,
                               operator_data.quad_index);

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      types::boundary_id boundary_id = matrix_free.get_boundary_id(face);

      integrator.reinit(face);

      do_boundary_integral(integrator, OperatorType::inhomogeneous, boundary_id, face);

      integrator.integrate_scatter(true, true, dst);
    }
  }

  /*
   *  Calculation of diagonal.
   */
  void
  cell_loop_diagonal(MatrixFree<dim, Number> const & matrix_free,
                     VectorType &                    dst,
                     VectorType const &,
                     Range const & cell_range) const
  {
    CellIntegratorU integrator(matrix_free, operator_data.dof_index, operator_data.quad_index);

    unsigned int const    dofs_per_cell = integrator.dofs_per_cell;
    AlignedVector<scalar> local_diagonal_vector(dofs_per_cell);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      integrator.reinit(cell);

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          integrator.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
        integrator.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

        integrator.evaluate(false, true);

        do_cell_integral(integrator, cell);

        integrator.integrate(false, true);

        local_diagonal_vector[j] = integrator.begin_dof_values()[j];
      }
      for(unsigned int j = 0; j < dofs_per_cell; ++j)
        integrator.begin_dof_values()[j] = local_diagonal_vector[j];

      integrator.distribute_local_to_global(dst);
    }
  }

  void
  face_loop_diagonal(MatrixFree<dim, Number> const & matrix_free,
                     VectorType &                    dst,
                     VectorType const &,
                     Range const & face_range) const
  {
    FaceIntegratorU integrator_m(matrix_free,
                                 true,
                                 operator_data.dof_index,
                                 operator_data.quad_index);

    FaceIntegratorU integrator_p(matrix_free,
                                 false,
                                 operator_data.dof_index,
                                 operator_data.quad_index);

    unsigned int const    dofs_per_cell = integrator_m.dofs_per_cell;
    AlignedVector<scalar> local_diagonal_vector(dofs_per_cell);

    // perform face integrals for element e⁻
    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      integrator_m.reinit(face);
      integrator_p.reinit(face);

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        // set dof value j of element- to 1 and all other dof values of element- to zero
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          integrator_m.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
        integrator_m.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

        integrator_m.evaluate(true, true);

        do_face_int_integral(integrator_m, integrator_p, face);

        integrator_m.integrate(true, true);

        local_diagonal_vector[j] = integrator_m.begin_dof_values()[j];
      }
      for(unsigned int j = 0; j < dofs_per_cell; ++j)
        integrator_m.begin_dof_values()[j] = local_diagonal_vector[j];

      integrator_m.distribute_local_to_global(dst);
    }

    unsigned int const                     dofs_per_cell_neighbor = integrator_p.dofs_per_cell;
    AlignedVector<VectorizedArray<Number>> local_diagonal_vector_neighbor(dofs_per_cell_neighbor);

    // Perform face integrals for element e⁺.
    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      integrator_m.reinit(face);
      integrator_p.reinit(face);

      for(unsigned int j = 0; j < dofs_per_cell_neighbor; ++j)
      {
        // set dof value j of element+ to 1 and all other dof values of element+ to zero
        for(unsigned int i = 0; i < dofs_per_cell_neighbor; ++i)
          integrator_p.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
        integrator_p.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

        integrator_p.evaluate(true, true);

        do_face_ext_integral(integrator_m, integrator_p, face);

        integrator_p.integrate(true, true);

        local_diagonal_vector_neighbor[j] = integrator_p.begin_dof_values()[j];
      }
      for(unsigned int j = 0; j < dofs_per_cell_neighbor; ++j)
        integrator_p.begin_dof_values()[j] = local_diagonal_vector_neighbor[j];

      integrator_p.distribute_local_to_global(dst);
    }
  }

  void
  boundary_face_loop_diagonal(MatrixFree<dim, Number> const & matrix_free,
                              VectorType &                    dst,
                              VectorType const &,
                              Range const & face_range) const
  {
    FaceIntegratorU integrator(matrix_free,
                               true,
                               operator_data.dof_index,
                               operator_data.quad_index);

    unsigned int const    dofs_per_cell = integrator.dofs_per_cell;
    AlignedVector<scalar> local_diagonal_vector(dofs_per_cell);

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      types::boundary_id boundary_id = matrix_free.get_boundary_id(face);

      integrator.reinit(face);

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        // set dof value j of element- to 1 and all other dof values of element- to zero
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          integrator.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
        integrator.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

        integrator.evaluate(true, true);

        do_boundary_integral(integrator, OperatorType::homogeneous, boundary_id, face);

        integrator.integrate(true, true);

        local_diagonal_vector[j] = integrator.begin_dof_values()[j];
      }
      for(unsigned int j = 0; j < dofs_per_cell; ++j)
        integrator.begin_dof_values()[j] = local_diagonal_vector[j];

      integrator.distribute_local_to_global(dst);
    }
  }

  /*
   *  Block-jacobi operator: re-implement face_loop; cell_loop and boundary_face_loop are
   *  identical to homogeneous operator.
   */
  void
  face_loop_block_jacobi(MatrixFree<dim, Number> const & matrix_free,
                         VectorType &                    dst,
                         VectorType const &              src,
                         Range const &                   face_range) const
  {
    FaceIntegratorU integrator_m(matrix_free,
                                 true,
                                 operator_data.dof_index,
                                 operator_data.quad_index);

    FaceIntegratorU integrator_p(matrix_free,
                                 false,
                                 operator_data.dof_index,
                                 operator_data.quad_index);

    // perform face integral for element e⁻
    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      integrator_m.reinit(face);
      integrator_p.reinit(face);

      integrator_m.gather_evaluate(src, true, true);

      do_face_int_integral(integrator_m, integrator_p, face);

      integrator_m.integrate_scatter(true, true, dst);
    }

    // perform face integral for element e⁺
    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      integrator_m.reinit(face);
      integrator_p.reinit(face);

      integrator_p.gather_evaluate(src, true, true);

      do_face_ext_integral(integrator_m, integrator_p, face);

      integrator_p.integrate_scatter(true, true, dst);
    }
  }

  void
  cell_loop_calculate_block_diagonal(MatrixFree<dim, Number> const &         matrix_free,
                                     std::vector<LAPACKFullMatrix<Number>> & matrices,
                                     VectorType const &,
                                     Range const & cell_range) const
  {
    CellIntegratorU integrator(matrix_free, operator_data.dof_index, operator_data.quad_index);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      integrator.reinit(cell);

      unsigned int dofs_per_cell = integrator.dofs_per_cell;

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          integrator.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
        integrator.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

        integrator.evaluate(false, true);

        do_cell_integral(integrator, cell);

        integrator.integrate(false, true);

        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          for(unsigned int v = 0; v < VectorizedArray<Number>::n_array_elements; ++v)
            matrices[cell * VectorizedArray<Number>::n_array_elements + v](i, j) +=
              integrator.begin_dof_values()[i][v];
      }
    }
  }

  void
  face_loop_calculate_block_diagonal(MatrixFree<dim, Number> const &         matrix_free,
                                     std::vector<LAPACKFullMatrix<Number>> & matrices,
                                     VectorType const &,
                                     Range const & face_range) const
  {
    FaceIntegratorU integrator_m(matrix_free,
                                 true,
                                 operator_data.dof_index,
                                 operator_data.quad_index);

    FaceIntegratorU integrator_p(matrix_free,
                                 false,
                                 operator_data.dof_index,
                                 operator_data.quad_index);

    // Perform face integrals for element e⁻.
    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      integrator_m.reinit(face);
      integrator_p.reinit(face);

      unsigned int dofs_per_cell = integrator_m.dofs_per_cell;

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        // set dof value j of element- to 1 and all other dof values of element- to zero
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          integrator_m.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
        integrator_m.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

        integrator_m.evaluate(true, true);

        do_face_int_integral(integrator_m, integrator_p, face);

        integrator_m.integrate(true, true);

        for(unsigned int v = 0; v < VectorizedArray<Number>::n_array_elements; ++v)
        {
          const unsigned int cell_number = matrix_free.get_face_info(face).cells_interior[v];
          if(cell_number != numbers::invalid_unsigned_int)
            for(unsigned int i = 0; i < dofs_per_cell; ++i)
              matrices[cell_number](i, j) += integrator_m.begin_dof_values()[i][v];
        }
      }
    }

    // Perform face integrals for element e⁺.
    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      integrator_m.reinit(face);
      integrator_p.reinit(face);

      unsigned int dofs_per_cell = integrator_p.dofs_per_cell;

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        // set dof value j of element+ to 1 and all other dof values of element+ to zero
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          integrator_p.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
        integrator_p.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

        integrator_p.evaluate(true, true);

        do_face_ext_integral(integrator_m, integrator_p, face);

        integrator_p.integrate(true, true);

        for(unsigned int v = 0; v < VectorizedArray<Number>::n_array_elements; ++v)
        {
          const unsigned int cell_number = matrix_free.get_face_info(face).cells_exterior[v];
          if(cell_number != numbers::invalid_unsigned_int)
            for(unsigned int i = 0; i < dofs_per_cell; ++i)
              matrices[cell_number](i, j) += integrator_p.begin_dof_values()[i][v];
        }
      }
    }
  }

  void
  boundary_face_loop_calculate_block_diagonal(MatrixFree<dim, Number> const &         matrix_free,
                                              std::vector<LAPACKFullMatrix<Number>> & matrices,
                                              VectorType const &,
                                              Range const & face_range) const
  {
    FaceIntegratorU integrator(matrix_free,
                               true,
                               operator_data.dof_index,
                               operator_data.quad_index);

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      types::boundary_id boundary_id = matrix_free.get_boundary_id(face);

      integrator.reinit(face);

      unsigned int dofs_per_cell = integrator.dofs_per_cell;

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        // set dof value j of element- to 1 and all other dof values of element- to zero
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          integrator.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
        integrator.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

        integrator.evaluate(true, true);

        do_boundary_integral(integrator, OperatorType::homogeneous, boundary_id, face);

        integrator.integrate(true, true);

        for(unsigned int v = 0; v < VectorizedArray<Number>::n_array_elements; ++v)
        {
          const unsigned int cell_number = matrix_free.get_face_info(face).cells_interior[v];
          if(cell_number != numbers::invalid_unsigned_int)
            for(unsigned int i = 0; i < dofs_per_cell; ++i)
              matrices[cell_number](i, j) += integrator.begin_dof_values()[i][v];
        }
      }
    }
  }

  void
  cell_based_loop_calculate_block_diagonal(MatrixFree<dim, Number> const &         matrix_free,
                                           std::vector<LAPACKFullMatrix<Number>> & matrices,
                                           VectorType const &,
                                           Range const & cell_range) const
  {
    // TODO
    AssertThrow(viscosity_is_variable() == false,
                ExcMessage(
                  "For cell-based face loops, the data structures for the variable viscosity field "
                  "have to be changed, i.e., these data structures also have to be cell-based."));

    CellIntegratorU integrator(matrix_free, operator_data.dof_index, operator_data.quad_index);
    FaceIntegratorU integrator_m(matrix_free,
                                 true,
                                 operator_data.dof_index,
                                 operator_data.quad_index);
    FaceIntegratorU integrator_p(matrix_free,
                                 false,
                                 operator_data.dof_index,
                                 operator_data.quad_index);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      // cell integral
      unsigned int const n_filled_lanes = matrix_free.n_active_entries_per_cell_batch(cell);

      integrator.reinit(cell);

      unsigned int dofs_per_cell = integrator.dofs_per_cell;

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          integrator.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
        integrator.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

        integrator.evaluate(false, true);

        do_cell_integral(integrator, cell);

        integrator.integrate(false, true);

        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          for(unsigned int v = 0; v < n_filled_lanes; ++v)
            matrices[cell * VectorizedArray<Number>::n_array_elements + v](i, j) +=
              integrator.begin_dof_values()[i][v];
      }

      // loop over all faces
      unsigned int const n_faces = GeometryInfo<dim>::faces_per_cell;
      for(unsigned int face = 0; face < n_faces; ++face)
      {
        integrator_m.reinit(cell, face);
        integrator_p.reinit(cell, face);
        auto bids = matrix_free.get_faces_by_cells_boundary_id(cell, face);
        auto bid  = bids[0];

        for(unsigned int j = 0; j < dofs_per_cell; ++j)
        {
          for(unsigned int i = 0; i < dofs_per_cell; ++i)
            integrator_m.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
          integrator_m.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

          integrator_m.evaluate(true, true);

          if(bid == numbers::internal_face_boundary_id) // internal face
          {
            // TODO specify the correct cell, face indices to obtain the correct, variable viscosity
            do_face_int_integral(integrator_m, integrator_p, 0 /* cell, face */);
          }
          else // boundary face
          {
            // TODO specify the correct cell, face indices to obtain the correct, variable viscosity
            do_boundary_integral(integrator_m, OperatorType::homogeneous, bid, 0 /* cell, face */);
          }

          integrator_m.integrate(true, true);

          for(unsigned int i = 0; i < dofs_per_cell; ++i)
            for(unsigned int v = 0; v < n_filled_lanes; ++v)
              matrices[cell * VectorizedArray<Number>::n_array_elements + v](i, j) +=
                integrator_m.begin_dof_values()[i][v];
        }
      }
    }
  }

private:
  MatrixFree<dim, Number> const * matrix_free;
  ViscousOperatorData<dim>        operator_data;

  // penalty parameter
  AlignedVector<scalar> array_penalty_parameter;

  // viscosity
  Number const_viscosity;

  // variable viscosity
  Table<2, scalar> viscous_coefficient_cell;
  Table<2, scalar> viscous_coefficient_face;
  Table<2, scalar> viscous_coefficient_face_neighbor;

  // TODO cell-based for loops
  std::shared_ptr<Table<3, scalar>> viscous_coefficient_face_cell_based;

  // evaluation time (time dependent problems)
  mutable Number eval_time;

  unsigned int n_mpi_processes;

  // required for elementwise block Jacobi operation
  std::shared_ptr<CellIntegratorU> integrator;
  std::shared_ptr<FaceIntegratorU> integrator_m;
  std::shared_ptr<FaceIntegratorU> integrator_p;
};

} // namespace IncNS

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_VISCOUS_OPERATOR_H_ \
        */
