/*
 * viscous_operator.h
 *
 *  Created on: Nov 5, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_VISCOUS_OPERATOR_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_VISCOUS_OPERATOR_H_

#include <deal.II/matrix_free/fe_evaluation.h>

#include "../../../functionalities/evaluate_functions.h"
#include "../../../operators/interior_penalty_parameter.h"
#include "../../../operators/operator_type.h"
#include "../../user_interface/boundary_descriptor.h"
#include "../../user_interface/input_parameters.h"

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
  double                           IP_factor;
  unsigned int                     dof_index;
  unsigned int                     quad_index;

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

template<int dim, int degree, typename Number>
class ViscousOperator
{
public:
  typedef ViscousOperator<dim, degree, Number> This;

  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef VectorizedArray<Number>                 scalar;
  typedef Tensor<1, dim, VectorizedArray<Number>> vector;
  typedef Tensor<2, dim, VectorizedArray<Number>> tensor;

  typedef std::pair<unsigned int, unsigned int> Range;

  typedef Number value_type;

  typedef FEEvaluation<dim, degree, degree + 1, dim, Number>     FEEvalCell;
  typedef FEFaceEvaluation<dim, degree, degree + 1, dim, Number> FEEvalFace;

  ViscousOperator()
    : data(nullptr),
      const_viscosity(-1.0),
      eval_time(0.0),
      n_mpi_processes(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
  {
  }

  void
  initialize(Mapping<dim> const &             mapping,
             MatrixFree<dim, Number> const &  mf_data,
             ViscousOperatorData<dim> const & operator_data_in)
  {
    this->data          = &mf_data;
    this->operator_data = operator_data_in;

    IP::calculate_penalty_parameter<dim, degree, Number>(array_penalty_parameter,
                                                         *this->data,
                                                         mapping,
                                                         operator_data.dof_index);

    const_viscosity = operator_data.viscosity;

    // Block Jacobi elementwise
    fe_eval.reset(
      new FEEvalCell(*this->data, this->operator_data.dof_index, this->operator_data.quad_index));
    fe_eval_m.reset(new FEEvalFace(
      *this->data, true, this->operator_data.dof_index, this->operator_data.quad_index));
    fe_eval_p.reset(new FEEvalFace(
      *this->data, false, this->operator_data.dof_index, this->operator_data.quad_index));
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
    this->viscous_coefficient_cell.reinit(this->data->n_cell_batches(),
                                          Utilities::pow(degree + 1, dim));
    this->viscous_coefficient_cell.fill(make_vectorized_array<Number>(const_viscosity));

    this->viscous_coefficient_face.reinit(this->data->n_inner_face_batches() +
                                            this->data->n_boundary_face_batches(),
                                          Utilities::pow(degree + 1, dim - 1));
    this->viscous_coefficient_face.fill(make_vectorized_array<Number>(const_viscosity));

    this->viscous_coefficient_face_neighbor.reinit(this->data->n_inner_face_batches(),
                                                   Utilities::pow(degree + 1, dim - 1));
    this->viscous_coefficient_face_neighbor.fill(make_vectorized_array<Number>(const_viscosity));

    // TODO
    //    this->viscous_coefficient_face_cell_based.reset(new
    //    Table<3,VectorizedArray<Number>>(this->data->n_cell_batches(),
    //                                                                                         2*dim,
    //                                                                                         Utilities::pow(n_actual_q_points_vel_linear,
    //                                                                                         dim -
    //                                                                                         1)));
    //    this->viscous_coefficient_face_cell_based->fill(make_vectorized_array<Number>(const_viscosity));
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
    data->loop(&This::cell_loop,
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
    data->loop(&This::cell_loop,
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
    data->loop(&This::cell_loop,
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
    data->loop(&This::cell_loop,
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

    data->loop(&This::cell_loop_empty,
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

    data->loop(&This::cell_loop,
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

    data->loop(&This::cell_loop,
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

    data->loop(&This::cell_loop_diagonal,
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

    data->loop(&This::cell_loop_diagonal,
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
      data->cell_loop(&This::cell_based_loop_calculate_block_diagonal, this, matrices, src);
    }
    else
    {
      AssertThrow(
        n_mpi_processes == 1,
        ExcMessage(
          "Block diagonal calculation with separate loops over cells and faces only works in serial. "
          "Use cell based loops for parallel computations."));

      data->loop(&This::cell_loop_calculate_block_diagonal,
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
    unsigned int dofs_per_cell = fe_eval->dofs_per_cell;

    fe_eval->reinit(cell);

    for(unsigned int i = 0; i < dofs_per_cell; ++i)
      fe_eval->begin_dof_values()[i] = src[i];

    fe_eval->evaluate(false, true, false);

    do_cell_integral(*fe_eval, cell);

    fe_eval->integrate(false, true);

    for(unsigned int i = 0; i < dofs_per_cell; ++i)
      dst[i] += fe_eval->begin_dof_values()[i];

    // loop over all faces
    unsigned int const n_faces = GeometryInfo<dim>::faces_per_cell;
    for(unsigned int face = 0; face < n_faces; ++face)
    {
      fe_eval_m->reinit(cell, face);
      fe_eval_p->reinit(cell, face);

      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        fe_eval_m->begin_dof_values()[i] = src[i];

      // do not need to read dof values for fe_eval_p (already initialized with 0)

      fe_eval_m->evaluate(true, true);

      auto bids = data->get_faces_by_cells_boundary_id(cell, face);
      auto bid  = bids[0];

      if(bid == numbers::internal_face_boundary_id) // internal face
      {
        // TODO specify the correct cell, face indices to obtain the correct, variable viscosity
        do_face_int_integral(*fe_eval_m, *fe_eval_p, 0 /* cell, face */);
      }
      else // boundary face
      {
        // TODO specify the correct cell, face indices to obtain the correct, variable viscosity
        do_boundary_integral(*fe_eval_m, OperatorType::homogeneous, bid, 0 /* cell, face */);
      }

      fe_eval_m->integrate(true, true);

      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        dst[i] += fe_eval_m->begin_dof_values()[i];
    }
  }

private:
  template<typename FEEvaluation>
  inline void
  do_cell_integral(FEEvaluation & fe_eval, unsigned int const cell) const
  {
    AssertThrow(const_viscosity >= 0.0, ExcMessage("Constant viscosity has not been set!"));

    for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
    {
      scalar viscosity = make_vectorized_array<Number>(const_viscosity);
      if(viscosity_is_variable())
        viscosity = viscous_coefficient_cell[cell][q];

      if(operator_data.formulation_viscous_term == FormulationViscousTerm::DivergenceFormulation)
      {
        fe_eval.submit_gradient(viscosity * make_vectorized_array<Number>(2.) *
                                  fe_eval.get_symmetric_gradient(q),
                                q);
      }
      else if(operator_data.formulation_viscous_term == FormulationViscousTerm::LaplaceFormulation)
      {
        fe_eval.submit_gradient(viscosity * fe_eval.get_gradient(q), q);
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

  template<typename FEEvaluation>
  void
  do_face_integral(FEEvaluation &     fe_eval_m,
                   FEEvaluation &     fe_eval_p,
                   unsigned int const face) const
  {
    scalar penalty_parameter = IP::get_penalty_factor<Number>(degree, operator_data.IP_factor) *
                               std::max(fe_eval_m.read_cell_data(array_penalty_parameter),
                                        fe_eval_p.read_cell_data(array_penalty_parameter));

    for(unsigned int q = 0; q < fe_eval_m.n_q_points; ++q)
    {
      scalar average_viscosity = make_vectorized_array<Number>(const_viscosity);
      if(viscosity_is_variable())
        average_viscosity = calculate_average_viscosity(face, q);

      vector value_m = fe_eval_m.get_value(q);
      vector value_p = fe_eval_p.get_value(q);
      vector normal  = fe_eval_m.get_normal_vector(q);

      tensor value_flux = calculate_value_flux(value_m, value_p, normal, average_viscosity);

      vector normal_gradient_m = calculate_normal_gradient(q, fe_eval_m);
      vector normal_gradient_p = calculate_normal_gradient(q, fe_eval_p);

      vector gradient_flux = calculate_gradient_flux(normal_gradient_m,
                                                     normal_gradient_p,
                                                     value_m,
                                                     value_p,
                                                     normal,
                                                     average_viscosity,
                                                     penalty_parameter);

      fe_eval_m.submit_gradient(value_flux, q);
      fe_eval_p.submit_gradient(value_flux, q);

      fe_eval_m.submit_value(-gradient_flux, q);
      fe_eval_p.submit_value(gradient_flux, q); // + sign since n⁺ = -n⁻
    }
  }

  template<typename FEEvaluation>
  void
  do_face_int_integral(FEEvaluation &     fe_eval_m,
                       FEEvaluation &     fe_eval_p,
                       unsigned int const face) const
  {
    scalar penalty_parameter = IP::get_penalty_factor<Number>(degree, operator_data.IP_factor) *
                               std::max(fe_eval_m.read_cell_data(array_penalty_parameter),
                                        fe_eval_p.read_cell_data(array_penalty_parameter));

    for(unsigned int q = 0; q < fe_eval_m.n_q_points; ++q)
    {
      scalar average_viscosity = make_vectorized_array<Number>(const_viscosity);
      if(viscosity_is_variable())
        average_viscosity = calculate_average_viscosity(face, q);

      // set exterior values to zero
      vector value_m = fe_eval_m.get_value(q);
      vector value_p;

      vector normal_m = fe_eval_m.get_normal_vector(q);

      tensor value_flux = calculate_value_flux(value_m, value_p, normal_m, average_viscosity);

      vector normal_gradient_m = calculate_normal_gradient(q, fe_eval_m);
      vector normal_gradient_p; // set exterior gradient to zero

      vector gradient_flux = calculate_gradient_flux(normal_gradient_m,
                                                     normal_gradient_p,
                                                     value_m,
                                                     value_p,
                                                     normal_m,
                                                     average_viscosity,
                                                     penalty_parameter);

      fe_eval_m.submit_gradient(value_flux, q);
      fe_eval_m.submit_value(-gradient_flux, q);
    }
  }


  template<typename FEEvaluation>
  void
  do_face_ext_integral(FEEvaluation &     fe_eval_m,
                       FEEvaluation &     fe_eval_p,
                       unsigned int const face) const
  {
    scalar penalty_parameter = IP::get_penalty_factor<Number>(degree, operator_data.IP_factor) *
                               std::max(fe_eval_m.read_cell_data(array_penalty_parameter),
                                        fe_eval_p.read_cell_data(array_penalty_parameter));

    for(unsigned int q = 0; q < fe_eval_p.n_q_points; ++q)
    {
      scalar average_viscosity = make_vectorized_array<Number>(const_viscosity);
      if(viscosity_is_variable())
        average_viscosity = calculate_average_viscosity(face, q);

      // set exterior values to zero
      vector value_m;
      vector value_p = fe_eval_p.get_value(q);
      // multiply by -1.0 to get the correct normal vector !!!
      vector normal_p = -fe_eval_p.get_normal_vector(q);

      tensor value_flux = calculate_value_flux(value_p, value_m, normal_p, average_viscosity);

      // set exterior gradient to zero
      vector normal_gradient_m;
      // multiply by -1.0 since normal vector n⁺ = -n⁻ !!!
      vector normal_gradient_p = -calculate_normal_gradient(q, fe_eval_p);

      vector gradient_flux = calculate_gradient_flux(normal_gradient_p,
                                                     normal_gradient_m,
                                                     value_p,
                                                     value_m,
                                                     normal_p,
                                                     average_viscosity,
                                                     penalty_parameter);

      fe_eval_p.submit_gradient(value_flux, q);
      fe_eval_p.submit_value(-gradient_flux, q);
    }
  }

  template<typename FEEvaluation>
  void
  do_boundary_integral(FEEvaluation &             fe_eval,
                       OperatorType const &       operator_type,
                       types::boundary_id const & boundary_id,
                       unsigned int const         face) const
  {
    BoundaryTypeU boundary_type = operator_data.bc->get_boundary_type(boundary_id);

    scalar penalty_parameter = IP::get_penalty_factor<Number>(degree, operator_data.IP_factor) *
                               fe_eval.read_cell_data(array_penalty_parameter);

    for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
    {
      scalar viscosity = make_vectorized_array<Number>(const_viscosity);
      if(viscosity_is_variable())
        viscosity = viscous_coefficient_face[face][q];

      vector value_m = calculate_interior_value(q, fe_eval, operator_type);
      vector value_p =
        calculate_exterior_value(value_m, q, fe_eval, operator_type, boundary_type, boundary_id);

      vector normal = fe_eval.get_normal_vector(q);

      tensor value_flux = calculate_value_flux(value_m, value_p, normal, viscosity);

      vector normal_gradient_m = calculate_interior_normal_gradient(q, fe_eval, operator_type);
      vector normal_gradient_p = calculate_exterior_normal_gradient(
        normal_gradient_m, q, fe_eval, operator_type, boundary_type, boundary_id);

      vector gradient_flux = calculate_gradient_flux(normal_gradient_m,
                                                     normal_gradient_p,
                                                     value_m,
                                                     value_p,
                                                     normal,
                                                     viscosity,
                                                     penalty_parameter);

      fe_eval.submit_gradient(value_flux, q);
      fe_eval.submit_value(-gradient_flux, q);
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

  // clang-format off
  /*
   *  The following two functions calculate the interior/exterior value for boundary faces depending on the
   *  operator type, the type of the boundary face and the given boundary conditions.
   *
   *                            +-------------------------+--------------------+------------------------------+
   *                            | Dirichlet boundaries    | Neumann boundaries | symmetry boundaries          |
   *  +-------------------------+-------------------------+--------------------+------------------------------+
   *  | full operator           | u⁺ = -u⁻ + 2g           | u⁺ = u⁻            | u⁺ = u⁻ - 2 (u⁻*n)n          |
   *  +-------------------------+-------------------------+--------------------+------------------------------+
   *  | homogeneous operator    | u⁺ = -u⁻                | u⁺ = u⁻            | u⁺ = u⁻ - 2 (u⁻*n)n          |
   *  +-------------------------+-------------------------+--------------------+------------------------------+
   *  | inhomogeneous operator  | u⁺ = -u⁻ + 2g , u⁻ = 0  | u⁺ = u⁻ , u⁻ = 0   | u⁺ = u⁻ - 2 (u⁻*n)n , u⁻ = 0 |
   *  +-------------------------+-------------------------+--------------------+------------------------------+
   *
   */
  // clang-format on
  template<typename FEEvaluationVelocity>
  inline DEAL_II_ALWAYS_INLINE //
    vector
    calculate_interior_value(unsigned int const           q,
                             FEEvaluationVelocity const & fe_eval_velocity,
                             OperatorType const &         operator_type) const
  {
    // element e⁻
    vector value_m;

    if(operator_type == OperatorType::full || operator_type == OperatorType::homogeneous)
    {
      value_m = fe_eval_velocity.get_value(q);
    }
    else if(operator_type == OperatorType::inhomogeneous)
    {
      // do nothing, value_m is already initialized with zeros
    }
    else
    {
      AssertThrow(false, ExcMessage("Specified OperatorType is not implemented!"));
    }

    return value_m;
  }

  template<typename FEEvaluationVelocity>
  inline DEAL_II_ALWAYS_INLINE //
    vector
    calculate_exterior_value(vector const &               value_m,
                             unsigned int const           q,
                             FEEvaluationVelocity const & fe_eval_velocity,
                             OperatorType const &         operator_type,
                             BoundaryTypeU const &        boundary_type,
                             types::boundary_id const     boundary_id = types::boundary_id()) const
  {
    // element e⁺
    vector value_p;

    if(boundary_type == BoundaryTypeU::Dirichlet)
    {
      if(operator_type == OperatorType::full || operator_type == OperatorType::inhomogeneous)
      {
        vector g;

        typename std::map<types::boundary_id, std::shared_ptr<Function<dim>>>::iterator it;
        it                          = operator_data.bc->dirichlet_bc.find(boundary_id);
        Point<dim, scalar> q_points = fe_eval_velocity.quadrature_point(q);
        evaluate_vectorial_function(g, it->second, q_points, eval_time);

        value_p = -value_m + make_vectorized_array<Number>(2.0) * g;
      }
      else if(operator_type == OperatorType::homogeneous)
      {
        value_p = -value_m;
      }
      else
      {
        AssertThrow(false, ExcMessage("Specified OperatorType is not implemented!"));
      }
    }
    else if(boundary_type == BoundaryTypeU::Neumann)
    {
      value_p = value_m;
    }
    else if(boundary_type == BoundaryTypeU::Symmetry)
    {
      vector normal_m = fe_eval_velocity.get_normal_vector(q);

      value_p = value_m - 2.0 * (value_m * normal_m) * normal_m;
    }
    else
    {
      AssertThrow(false, ExcMessage("Boundary type of face is invalid or not implemented."));
    }

    return value_p;
  }

  /*
   *  This function calculates the gradient in normal direction on element e
   *  depending on the formulation of the viscous term.
   */
  template<typename FEEvaluation>
  inline DEAL_II_ALWAYS_INLINE //
    vector
    calculate_normal_gradient(unsigned int const q, FEEvaluation & fe_eval) const
  {
    tensor gradient;

    if(operator_data.formulation_viscous_term == FormulationViscousTerm::DivergenceFormulation)
    {
      /*
       * F = 2 * nu * symmetric_gradient
       *   = 2.0 * nu * 1/2 (grad(u) + grad(u)^T)
       */
      gradient = make_vectorized_array<Number>(2.0) * fe_eval.get_symmetric_gradient(q);
    }
    else if(operator_data.formulation_viscous_term == FormulationViscousTerm::LaplaceFormulation)
    {
      /*
       *  F = nu * grad(u)
       */
      gradient = fe_eval.get_gradient(q);
    }
    else
    {
      AssertThrow(operator_data.formulation_viscous_term ==
                      FormulationViscousTerm::DivergenceFormulation ||
                    operator_data.formulation_viscous_term ==
                      FormulationViscousTerm::LaplaceFormulation,
                  ExcMessage("Specified formulation of viscous term is not implemented."));
    }

    vector normal_gradient = gradient * fe_eval.get_normal_vector(q);

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
  template<typename FEEvaluation>
  inline DEAL_II_ALWAYS_INLINE //
    vector
    calculate_interior_normal_gradient(unsigned int const   q,
                                       FEEvaluation const & fe_eval,
                                       OperatorType const & operator_type) const
  {
    vector normal_gradient_m;

    if(operator_type == OperatorType::full || operator_type == OperatorType::homogeneous)
    {
      normal_gradient_m = calculate_normal_gradient(q, fe_eval);
    }
    else if(operator_type == OperatorType::inhomogeneous)
    {
      // do nothing, normal_gradient_m is already intialized with 0
    }
    else
    {
      AssertThrow(false, ExcMessage("Specified OperatorType is not implemented!"));
    }

    return normal_gradient_m;
  }

  template<typename FEEvaluation>
  inline DEAL_II_ALWAYS_INLINE //
    vector
    calculate_exterior_normal_gradient(
      vector const &           normal_gradient_m,
      unsigned int const       q,
      FEEvaluation const &     fe_eval,
      OperatorType const &     operator_type,
      BoundaryTypeU const &    boundary_type,
      types::boundary_id const boundary_id = types::boundary_id()) const
  {
    vector normal_gradient_p;

    if(boundary_type == BoundaryTypeU::Dirichlet)
    {
      normal_gradient_p = normal_gradient_m;
    }
    else if(boundary_type == BoundaryTypeU::Neumann)
    {
      if(operator_type == OperatorType::full || operator_type == OperatorType::inhomogeneous)
      {
        vector h;

        typename std::map<types::boundary_id, std::shared_ptr<Function<dim>>>::iterator it;
        it                          = operator_data.bc->neumann_bc.find(boundary_id);
        Point<dim, scalar> q_points = fe_eval.quadrature_point(q);
        evaluate_vectorial_function(h, it->second, q_points, eval_time);

        normal_gradient_p = -normal_gradient_m + 2.0 * h;
      }
      else if(operator_type == OperatorType::homogeneous)
      {
        normal_gradient_p = -normal_gradient_m;
      }
      else
      {
        AssertThrow(false, ExcMessage("Specified OperatorType is not implemented!"));
      }
    }
    else if(boundary_type == BoundaryTypeU::Symmetry)
    {
      vector normal_m   = fe_eval.get_normal_vector(q);
      normal_gradient_p = -normal_gradient_m + 2.0 * (normal_gradient_m * normal_m) * normal_m;
    }
    else
    {
      AssertThrow(false, ExcMessage("Boundary type of face is invalid or not implemented."));
    }

    return normal_gradient_p;
  }

  void
  cell_loop(MatrixFree<dim, Number> const & data,
            VectorType &                    dst,
            VectorType const &              src,
            Range const &                   cell_range) const
  {
    FEEvalCell fe_eval(data, operator_data.dof_index, operator_data.quad_index);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);
      fe_eval.gather_evaluate(src, false, true, false);

      do_cell_integral(fe_eval, cell);

      fe_eval.integrate_scatter(false, true, dst);
    }
  }

  void
  face_loop(MatrixFree<dim, Number> const & data,
            VectorType &                    dst,
            VectorType const &              src,
            Range const &                   face_range) const
  {
    FEEvalFace fe_eval(data, true, operator_data.dof_index, operator_data.quad_index);

    FEEvalFace fe_eval_neighbor(data, false, operator_data.dof_index, operator_data.quad_index);

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      fe_eval.reinit(face);
      fe_eval_neighbor.reinit(face);

      fe_eval.gather_evaluate(src, true, true);
      fe_eval_neighbor.gather_evaluate(src, true, true);

      do_face_integral(fe_eval, fe_eval_neighbor, face);

      fe_eval.integrate_scatter(true, true, dst);
      fe_eval_neighbor.integrate_scatter(true, true, dst);
    }
  }

  void
  boundary_face_loop_hom_operator(MatrixFree<dim, Number> const & data,
                                  VectorType &                    dst,
                                  VectorType const &              src,
                                  Range const &                   face_range) const
  {
    FEEvalFace fe_eval(data, true, operator_data.dof_index, operator_data.quad_index);

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      types::boundary_id boundary_id = data.get_boundary_id(face);

      fe_eval.reinit(face);

      fe_eval.gather_evaluate(src, true, true);

      do_boundary_integral(fe_eval, OperatorType::homogeneous, boundary_id, face);

      fe_eval.integrate_scatter(true, true, dst);
    }
  }

  void
  boundary_face_loop_full_operator(MatrixFree<dim, Number> const & data,
                                   VectorType &                    dst,
                                   VectorType const &              src,
                                   Range const &                   face_range) const
  {
    FEEvalFace fe_eval(data, true, operator_data.dof_index, operator_data.quad_index);

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      types::boundary_id boundary_id = data.get_boundary_id(face);

      fe_eval.reinit(face);

      fe_eval.gather_evaluate(src, true, true);

      do_boundary_integral(fe_eval, OperatorType::full, boundary_id, face);

      fe_eval.integrate_scatter(true, true, dst);
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
  boundary_face_loop_inhom_operator(MatrixFree<dim, Number> const & data,
                                    VectorType &                    dst,
                                    VectorType const &,
                                    Range const & face_range) const
  {
    FEEvalFace fe_eval(data, true, operator_data.dof_index, operator_data.quad_index);

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      types::boundary_id boundary_id = data.get_boundary_id(face);

      fe_eval.reinit(face);

      do_boundary_integral(fe_eval, OperatorType::inhomogeneous, boundary_id, face);

      fe_eval.integrate_scatter(true, true, dst);
    }
  }

  /*
   *  Calculation of diagonal.
   */
  void
  cell_loop_diagonal(MatrixFree<dim, Number> const & data,
                     VectorType &                    dst,
                     VectorType const &,
                     Range const & cell_range) const
  {
    FEEvalCell fe_eval(data, operator_data.dof_index, operator_data.quad_index);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);

      unsigned int            dofs_per_cell = fe_eval.dofs_per_cell;
      VectorizedArray<Number> local_diagonal_vector[fe_eval.tensor_dofs_per_cell];
      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          fe_eval.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
        fe_eval.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

        fe_eval.evaluate(false, true);

        do_cell_integral(fe_eval, cell);

        fe_eval.integrate(false, true);

        local_diagonal_vector[j] = fe_eval.begin_dof_values()[j];
      }
      for(unsigned int j = 0; j < dofs_per_cell; ++j)
        fe_eval.begin_dof_values()[j] = local_diagonal_vector[j];

      fe_eval.distribute_local_to_global(dst);
    }
  }

  void
  face_loop_diagonal(MatrixFree<dim, Number> const & data,
                     VectorType &                    dst,
                     VectorType const &,
                     Range const & face_range) const
  {
    FEEvalFace fe_eval(data, true, operator_data.dof_index, operator_data.quad_index);

    FEEvalFace fe_eval_neighbor(data, false, operator_data.dof_index, operator_data.quad_index);

    // perform face integrals for element e⁻
    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      fe_eval.reinit(face);
      fe_eval_neighbor.reinit(face);

      unsigned int            dofs_per_cell = fe_eval.dofs_per_cell;
      VectorizedArray<Number> local_diagonal_vector[fe_eval.tensor_dofs_per_cell];
      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        // set dof value j of element- to 1 and all other dof values of element- to zero
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          fe_eval.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
        fe_eval.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

        fe_eval.evaluate(true, true);

        do_face_int_integral(fe_eval, fe_eval_neighbor, face);

        fe_eval.integrate(true, true);

        local_diagonal_vector[j] = fe_eval.begin_dof_values()[j];
      }
      for(unsigned int j = 0; j < dofs_per_cell; ++j)
        fe_eval.begin_dof_values()[j] = local_diagonal_vector[j];

      fe_eval.distribute_local_to_global(dst);
    }

    // Perform face integrals for element e⁺.
    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      fe_eval.reinit(face);
      fe_eval_neighbor.reinit(face);

      unsigned int            dofs_per_cell = fe_eval_neighbor.dofs_per_cell;
      VectorizedArray<Number> local_diagonal_vector_neighbor[fe_eval_neighbor.tensor_dofs_per_cell];
      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        // set dof value j of element+ to 1 and all other dof values of element+ to zero
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          fe_eval_neighbor.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
        fe_eval_neighbor.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

        fe_eval_neighbor.evaluate(true, true);

        do_face_ext_integral(fe_eval, fe_eval_neighbor, face);

        fe_eval_neighbor.integrate(true, true);

        local_diagonal_vector_neighbor[j] = fe_eval_neighbor.begin_dof_values()[j];
      }
      for(unsigned int j = 0; j < dofs_per_cell; ++j)
        fe_eval_neighbor.begin_dof_values()[j] = local_diagonal_vector_neighbor[j];

      fe_eval_neighbor.distribute_local_to_global(dst);
    }
  }

  void
  boundary_face_loop_diagonal(MatrixFree<dim, Number> const & data,
                              VectorType &                    dst,
                              VectorType const &,
                              Range const & face_range) const
  {
    FEEvalFace fe_eval(data, true, operator_data.dof_index, operator_data.quad_index);

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      types::boundary_id boundary_id = data.get_boundary_id(face);

      fe_eval.reinit(face);

      unsigned int            dofs_per_cell = fe_eval.dofs_per_cell;
      VectorizedArray<Number> local_diagonal_vector[fe_eval.tensor_dofs_per_cell];
      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        // set dof value j of element- to 1 and all other dof values of element- to zero
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          fe_eval.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
        fe_eval.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

        fe_eval.evaluate(true, true);

        do_boundary_integral(fe_eval, OperatorType::homogeneous, boundary_id, face);

        fe_eval.integrate(true, true);

        local_diagonal_vector[j] = fe_eval.begin_dof_values()[j];
      }
      for(unsigned int j = 0; j < dofs_per_cell; ++j)
        fe_eval.begin_dof_values()[j] = local_diagonal_vector[j];

      fe_eval.distribute_local_to_global(dst);
    }
  }

  /*
   *  Block-jacobi operator: re-implement face_loop; cell_loop and boundary_face_loop are
   *  identical to homogeneous operator.
   */
  void
  face_loop_block_jacobi(MatrixFree<dim, Number> const & data,
                         VectorType &                    dst,
                         VectorType const &              src,
                         Range const &                   face_range) const
  {
    FEEvalFace fe_eval(data, true, operator_data.dof_index, operator_data.quad_index);

    FEEvalFace fe_eval_neighbor(data, false, operator_data.dof_index, operator_data.quad_index);

    // perform face integral for element e⁻
    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      fe_eval.reinit(face);
      fe_eval_neighbor.reinit(face);

      fe_eval.gather_evaluate(src, true, true);

      do_face_int_integral(fe_eval, fe_eval_neighbor, face);

      fe_eval.integrate_scatter(true, true, dst);
    }

    // perform face integral for element e⁺
    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      fe_eval.reinit(face);
      fe_eval_neighbor.reinit(face);

      fe_eval_neighbor.gather_evaluate(src, true, true);

      do_face_ext_integral(fe_eval, fe_eval_neighbor, face);

      fe_eval_neighbor.integrate_scatter(true, true, dst);
    }
  }

  void
  cell_loop_calculate_block_diagonal(MatrixFree<dim, Number> const &         data,
                                     std::vector<LAPACKFullMatrix<Number>> & matrices,
                                     VectorType const &,
                                     Range const & cell_range) const
  {
    FEEvalCell fe_eval(data, operator_data.dof_index, operator_data.quad_index);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);

      unsigned int dofs_per_cell = fe_eval.dofs_per_cell;

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          fe_eval.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
        fe_eval.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

        fe_eval.evaluate(false, true);

        do_cell_integral(fe_eval, cell);

        fe_eval.integrate(false, true);

        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          for(unsigned int v = 0; v < VectorizedArray<Number>::n_array_elements; ++v)
            matrices[cell * VectorizedArray<Number>::n_array_elements + v](i, j) +=
              fe_eval.begin_dof_values()[i][v];
      }
    }
  }

  void
  face_loop_calculate_block_diagonal(MatrixFree<dim, Number> const &         data,
                                     std::vector<LAPACKFullMatrix<Number>> & matrices,
                                     VectorType const &,
                                     Range const & face_range) const
  {
    FEEvalFace fe_eval(data, true, operator_data.dof_index, operator_data.quad_index);

    FEEvalFace fe_eval_neighbor(data, false, operator_data.dof_index, operator_data.quad_index);

    // Perform face integrals for element e⁻.
    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      fe_eval.reinit(face);
      fe_eval_neighbor.reinit(face);

      unsigned int dofs_per_cell = fe_eval.dofs_per_cell;

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        // set dof value j of element- to 1 and all other dof values of element- to zero
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          fe_eval.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
        fe_eval.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

        fe_eval.evaluate(true, true);

        do_face_int_integral(fe_eval, fe_eval_neighbor, face);

        fe_eval.integrate(true, true);

        for(unsigned int v = 0; v < VectorizedArray<Number>::n_array_elements; ++v)
        {
          const unsigned int cell_number = data.get_face_info(face).cells_interior[v];
          if(cell_number != numbers::invalid_unsigned_int)
            for(unsigned int i = 0; i < dofs_per_cell; ++i)
              matrices[cell_number](i, j) += fe_eval.begin_dof_values()[i][v];
        }
      }
    }

    // Perform face integrals for element e⁺.
    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      fe_eval.reinit(face);
      fe_eval_neighbor.reinit(face);

      unsigned int dofs_per_cell = fe_eval_neighbor.dofs_per_cell;

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        // set dof value j of element+ to 1 and all other dof values of element+ to zero
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          fe_eval_neighbor.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
        fe_eval_neighbor.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

        fe_eval_neighbor.evaluate(true, true);

        do_face_ext_integral(fe_eval, fe_eval_neighbor, face);

        fe_eval_neighbor.integrate(true, true);

        for(unsigned int v = 0; v < VectorizedArray<Number>::n_array_elements; ++v)
        {
          const unsigned int cell_number = data.get_face_info(face).cells_exterior[v];
          if(cell_number != numbers::invalid_unsigned_int)
            for(unsigned int i = 0; i < dofs_per_cell; ++i)
              matrices[cell_number](i, j) += fe_eval_neighbor.begin_dof_values()[i][v];
        }
      }
    }
  }

  void
  boundary_face_loop_calculate_block_diagonal(MatrixFree<dim, Number> const &         data,
                                              std::vector<LAPACKFullMatrix<Number>> & matrices,
                                              VectorType const &,
                                              Range const & face_range) const
  {
    FEEvalFace fe_eval(data, true, operator_data.dof_index, operator_data.quad_index);

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      types::boundary_id boundary_id = data.get_boundary_id(face);

      fe_eval.reinit(face);

      unsigned int dofs_per_cell = fe_eval.dofs_per_cell;

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        // set dof value j of element- to 1 and all other dof values of element- to zero
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          fe_eval.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
        fe_eval.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

        fe_eval.evaluate(true, true);

        do_boundary_integral(fe_eval, OperatorType::homogeneous, boundary_id, face);

        fe_eval.integrate(true, true);

        for(unsigned int v = 0; v < VectorizedArray<Number>::n_array_elements; ++v)
        {
          const unsigned int cell_number = data.get_face_info(face).cells_interior[v];
          if(cell_number != numbers::invalid_unsigned_int)
            for(unsigned int i = 0; i < dofs_per_cell; ++i)
              matrices[cell_number](i, j) += fe_eval.begin_dof_values()[i][v];
        }
      }
    }
  }

  void
  cell_based_loop_calculate_block_diagonal(MatrixFree<dim, Number> const &         data,
                                           std::vector<LAPACKFullMatrix<Number>> & matrices,
                                           VectorType const &,
                                           Range const & cell_range) const
  {
    // TODO
    AssertThrow(viscosity_is_variable() == false,
                ExcMessage(
                  "For cell-based face loops, the data structures for the variable viscosity field "
                  "have to be changed, i.e., these data structures also have to be cell-based."));

    FEEvalCell fe_eval(data, operator_data.dof_index, operator_data.quad_index);
    FEEvalFace fe_eval_m(data, true, operator_data.dof_index, operator_data.quad_index);
    FEEvalFace fe_eval_p(data, false, operator_data.dof_index, operator_data.quad_index);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      // cell integral
      unsigned int const n_filled_lanes = data.n_active_entries_per_cell_batch(cell);

      fe_eval.reinit(cell);

      unsigned int dofs_per_cell = fe_eval.dofs_per_cell;

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          fe_eval.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
        fe_eval.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

        fe_eval.evaluate(false, true);

        do_cell_integral(fe_eval, cell);

        fe_eval.integrate(false, true);

        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          for(unsigned int v = 0; v < n_filled_lanes; ++v)
            matrices[cell * VectorizedArray<Number>::n_array_elements + v](i, j) +=
              fe_eval.begin_dof_values()[i][v];
      }

      // loop over all faces
      unsigned int const n_faces = GeometryInfo<dim>::faces_per_cell;
      for(unsigned int face = 0; face < n_faces; ++face)
      {
        fe_eval_m.reinit(cell, face);
        fe_eval_p.reinit(cell, face);
        auto bids = data.get_faces_by_cells_boundary_id(cell, face);
        auto bid  = bids[0];

        for(unsigned int j = 0; j < dofs_per_cell; ++j)
        {
          for(unsigned int i = 0; i < dofs_per_cell; ++i)
            fe_eval_m.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
          fe_eval_m.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

          fe_eval_m.evaluate(true, true);

          if(bid == numbers::internal_face_boundary_id) // internal face
          {
            // TODO specify the correct cell, face indices to obtain the correct, variable viscosity
            do_face_int_integral(fe_eval_m, fe_eval_p, 0 /* cell, face */);
          }
          else // boundary face
          {
            // TODO specify the correct cell, face indices to obtain the correct, variable viscosity
            do_boundary_integral(fe_eval_m, OperatorType::homogeneous, bid, 0 /* cell, face */);
          }

          fe_eval_m.integrate(true, true);

          for(unsigned int i = 0; i < dofs_per_cell; ++i)
            for(unsigned int v = 0; v < n_filled_lanes; ++v)
              matrices[cell * VectorizedArray<Number>::n_array_elements + v](i, j) +=
                fe_eval_m.begin_dof_values()[i][v];
        }
      }
    }
  }

private:
  MatrixFree<dim, Number> const * data;
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
  std::shared_ptr<FEEvalCell> fe_eval;
  std::shared_ptr<FEEvalFace> fe_eval_m;
  std::shared_ptr<FEEvalFace> fe_eval_p;
};

} // namespace IncNS

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_VISCOUS_OPERATOR_H_ \
        */
