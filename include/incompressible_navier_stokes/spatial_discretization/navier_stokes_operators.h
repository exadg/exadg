/*
 * navier_stokes_operators.h
 *
 *  Created on: Jun 6, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_NAVIER_STOKES_OPERATORS_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_NAVIER_STOKES_OPERATORS_H_

#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/geometry_info.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/point.h>
#include <deal.II/base/table.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/types.h>
#include <deal.II/base/vectorization.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <utility>
#include <vector>

#include "../../operators/operator_type.h"
#include "../user_interface/boundary_descriptor.h"
#include "../user_interface/input_parameters.h"

#include "../../functionalities/evaluate_functions.h"
#include "../../operators/interior_penalty_parameter.h"

namespace IncNS
{
template<int dim>
struct BodyForceOperatorData
{
  BodyForceOperatorData() : dof_index(0), quad_index(0)
  {
  }

  unsigned int dof_index;

  unsigned int quad_index;

  std::shared_ptr<Function<dim>> rhs;
};

template<int dim, int degree, typename Number>
class BodyForceOperator
{
public:
  typedef BodyForceOperator<dim, degree, Number> This;

  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef VectorizedArray<Number>                 scalar;
  typedef Tensor<1, dim, VectorizedArray<Number>> vector;

  typedef std::pair<unsigned int, unsigned int> Range;

  typedef FEEvaluation<dim, degree, degree + 1, dim, Number> FEEval;

  BodyForceOperator() : data(nullptr), eval_time(0.0)
  {
  }

  void
  initialize(MatrixFree<dim, Number> const &    mf_data,
             BodyForceOperatorData<dim> const & operator_data_in)
  {
    this->data          = &mf_data;
    this->operator_data = operator_data_in;
  }

  void
  evaluate(VectorType & dst, Number const evaluation_time) const
  {
    this->eval_time = evaluation_time;

    VectorType src;
    data->cell_loop(&This::cell_loop, this, dst, src, true /*zero_dst_vector = true*/);
  }

  void
  evaluate_add(VectorType & dst, Number const evaluation_time) const
  {
    this->eval_time = evaluation_time;

    VectorType src;
    data->cell_loop(&This::cell_loop, this, dst, src, false /*zero_dst_vector = false*/);
  }

private:
  template<typename FEEvaluation>
  void
  do_cell_integral(FEEvaluation & fe_eval) const
  {
    for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
    {
      Point<dim, scalar> q_points = fe_eval.quadrature_point(q);

      vector rhs;

      evaluate_vectorial_function(rhs, operator_data.rhs, q_points, eval_time);

      fe_eval.submit_value(rhs, q);
    }
  }

  void
  cell_loop(MatrixFree<dim, Number> const & data,
            VectorType &                    dst,
            VectorType const &,
            Range const & cell_range) const
  {
    FEEval fe_eval(data, operator_data.dof_index, operator_data.quad_index);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);

      do_cell_integral(fe_eval);

      fe_eval.integrate_scatter(true, false, dst);
    }
  }

  MatrixFree<dim, Number> const * data;

  BodyForceOperatorData<dim> operator_data;

  mutable Number eval_time;
};

struct MassMatrixOperatorData
{
  MassMatrixOperatorData() : dof_index(0), quad_index(0)
  {
  }

  unsigned int dof_index;
  unsigned int quad_index;
};

template<int dim, int degree, typename Number>
class MassMatrixOperator
{
public:
  typedef MassMatrixOperator<dim, degree, Number> This;

  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef VectorizedArray<Number> scalar;

  typedef std::pair<unsigned int, unsigned int> Range;

  typedef FEEvaluation<dim, degree, degree + 1, dim, Number> FEEval;

  MassMatrixOperator() : data(nullptr), scaling_factor(1.0)
  {
  }

  MassMatrixOperatorData const &
  get_operator_data() const
  {
    return operator_data;
  }

  MatrixFree<dim, Number> const &
  get_data() const
  {
    return *data;
  }

  void
  initialize(MatrixFree<dim, Number> const & mf_data,
             MassMatrixOperatorData const &  operator_data_in)
  {
    this->data          = &mf_data;
    this->operator_data = operator_data_in;

    // Block Jacobi elementwise
    fe_eval.reset(new FEEval(*data, this->operator_data.dof_index, this->operator_data.quad_index));
  }

  // apply matrix vector multiplication
  void
  apply(VectorType & dst, VectorType const & src) const
  {
    AssertThrow(std::abs(scaling_factor - 1.0) < 1.e-12,
                ExcMessage("Invalid parameter scaling_factor."));

    data->cell_loop(&This::cell_loop, this, dst, src, true /*zero_dst_vector = true*/);
  }

  void
  apply_scale(VectorType & dst, Number const & factor, VectorType const & src) const
  {
    scaling_factor = factor;

    data->cell_loop(&This::cell_loop, this, dst, src, true /*zero_dst_vector = true*/);

    scaling_factor = 1.0;
  }

  void
  apply_add(VectorType & dst, VectorType const & src) const
  {
    AssertThrow(std::abs(scaling_factor - 1.0) < 1.e-12,
                ExcMessage("Invalid parameter scaling_factor."));

    data->cell_loop(&This::cell_loop, this, dst, src, false /*zero_dst_vector = false*/);
  }

  void
  apply_scale_add(VectorType & dst, Number const & factor, VectorType const & src) const
  {
    scaling_factor = factor;

    data->cell_loop(&This::cell_loop, this, dst, src, false /*zero_dst_vector = false*/);

    scaling_factor = 1.0;
  }

  void
  calculate_diagonal(VectorType & diagonal) const
  {
    AssertThrow(std::abs(scaling_factor - 1.0) < 1.e-12,
                ExcMessage("Invalid parameter scaling_factor."));

    VectorType src;
    data->cell_loop(
      &This::cell_loop_diagonal, this, diagonal, src, true /*zero_dst_vector = true*/);
  }

  void
  add_diagonal(VectorType & diagonal) const
  {
    AssertThrow(std::abs(scaling_factor - 1.0) < 1.e-12,
                ExcMessage("Invalid parameter scaling_factor."));

    VectorType src;
    data->cell_loop(
      &This::cell_loop_diagonal, this, diagonal, src, false /*zero_dst_vector = false*/);
  }

  void
  add_block_diagonal_matrices(std::vector<LAPACKFullMatrix<Number>> & matrices) const
  {
    AssertThrow(std::abs(scaling_factor - 1.0) < 1.e-12,
                ExcMessage("Invalid parameter scaling_factor."));

    VectorType src;

    data->cell_loop(&This::cell_loop_calculate_block_diagonal, this, matrices, src);
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

    fe_eval->evaluate(true, false, false);

    do_cell_integral(*fe_eval);

    fe_eval->integrate(true, false);

    for(unsigned int i = 0; i < dofs_per_cell; ++i)
      dst[i] += fe_eval->begin_dof_values()[i];
  }

private:
  template<typename FEEvaluation>
  void
  do_cell_integral(FEEvaluation & fe_eval) const
  {
    for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
    {
      fe_eval.submit_value(scaling_factor * fe_eval.get_value(q), q);
    }
  }

  void
  cell_loop(MatrixFree<dim, Number> const & data,
            VectorType &                    dst,
            VectorType const &              src,
            Range const &                   cell_range) const
  {
    FEEval fe_eval(data, operator_data.dof_index, operator_data.quad_index);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);

      fe_eval.gather_evaluate(src, true, false, false);

      do_cell_integral(fe_eval);

      fe_eval.integrate_scatter(true, false, dst);
    }
  }

  void
  cell_loop_diagonal(MatrixFree<dim, Number> const & data,
                     VectorType &                    dst,
                     VectorType const &,
                     Range const & cell_range) const
  {
    FEEval fe_eval(data, operator_data.dof_index, operator_data.quad_index);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);

      unsigned int dofs_per_cell = fe_eval.dofs_per_cell;
      // tensor_dofs_per_cell >= dofs_per_cell
      VectorizedArray<Number> local_diagonal_vector[fe_eval.tensor_dofs_per_cell];

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          fe_eval.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
        fe_eval.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

        fe_eval.evaluate(true, false, false);

        do_cell_integral(fe_eval);

        fe_eval.integrate(true, false);

        local_diagonal_vector[j] = fe_eval.begin_dof_values()[j];
      }
      for(unsigned int j = 0; j < dofs_per_cell; ++j)
        fe_eval.begin_dof_values()[j] = local_diagonal_vector[j];

      fe_eval.distribute_local_to_global(dst);
    }
  }

  void
  cell_loop_calculate_block_diagonal(MatrixFree<dim, Number> const &         data,
                                     std::vector<LAPACKFullMatrix<Number>> & matrices,
                                     VectorType const &,
                                     Range const & cell_range) const
  {
    FEEval fe_eval(data, operator_data.dof_index, operator_data.quad_index);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);

      unsigned int dofs_per_cell = fe_eval.dofs_per_cell;

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          fe_eval.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
        fe_eval.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

        fe_eval.evaluate(true, false, false);

        do_cell_integral(fe_eval);

        fe_eval.integrate(true, false);

        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          for(unsigned int v = 0; v < VectorizedArray<Number>::n_array_elements; ++v)
            matrices[cell * VectorizedArray<Number>::n_array_elements + v](i, j) +=
              fe_eval.begin_dof_values()[i][v];
      }
    }
  }

  MatrixFree<dim, Number> const * data;

  MassMatrixOperatorData operator_data;

  mutable Number scaling_factor;

  // required for elementwise block Jacobi operation
  std::shared_ptr<FEEval> fe_eval;
};

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
    this->viscous_coefficient_cell.reinit(this->data->n_macro_cells(),
                                          Utilities::pow(degree + 1, dim));
    this->viscous_coefficient_cell.fill(make_vectorized_array<Number>(const_viscosity));

    this->viscous_coefficient_face.reinit(this->data->n_macro_inner_faces() +
                                            this->data->n_macro_boundary_faces(),
                                          Utilities::pow(degree + 1, dim - 1));
    this->viscous_coefficient_face.fill(make_vectorized_array<Number>(const_viscosity));

    this->viscous_coefficient_face_neighbor.reinit(this->data->n_macro_inner_faces(),
                                                   Utilities::pow(degree + 1, dim - 1));
    this->viscous_coefficient_face_neighbor.fill(make_vectorized_array<Number>(const_viscosity));

    // TODO
    //    this->viscous_coefficient_face_cell_based.reset(new
    //    Table<3,VectorizedArray<Number>>(this->data->n_macro_cells(),
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


template<int dim>
struct GradientOperatorData
{
  GradientOperatorData()
    : dof_index_velocity(0),
      dof_index_pressure(1),
      quad_index(0),
      integration_by_parts(true),
      use_boundary_data(true)
  {
  }

  unsigned int dof_index_velocity;
  unsigned int dof_index_pressure;

  unsigned int quad_index;

  bool integration_by_parts;
  bool use_boundary_data;

  std::shared_ptr<BoundaryDescriptorP<dim>> bc;
};

template<int dim, int degree_u, int degree_p, typename Number>
class GradientOperator
{
public:
  typedef GradientOperator<dim, degree_u, degree_p, Number> This;

  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef VectorizedArray<Number>                 scalar;
  typedef Tensor<1, dim, VectorizedArray<Number>> vector;

  typedef std::pair<unsigned int, unsigned int> Range;

  typedef FEEvaluation<dim, degree_u, degree_u + 1, dim, Number> FEEvalVelocity;
  typedef FEEvaluation<dim, degree_p, degree_u + 1, 1, Number>   FEEvalPressure;

  typedef FEFaceEvaluation<dim, degree_u, degree_u + 1, dim, Number> FEFaceEvalVelocity;
  typedef FEFaceEvaluation<dim, degree_p, degree_u + 1, 1, Number>   FEFaceEvalPressure;

  GradientOperator() : data(nullptr), eval_time(0.0), inverse_scaling_factor_pressure(1.0)
  {
  }

  void
  initialize(MatrixFree<dim, Number> const &   mf_data,
             GradientOperatorData<dim> const & operator_data_in)
  {
    this->data          = &mf_data;
    this->operator_data = operator_data_in;
  }

  void
  set_scaling_factor_pressure(Number const & scaling_factor)
  {
    inverse_scaling_factor_pressure = 1.0 / scaling_factor;
  }

  void
  apply(VectorType & dst, const VectorType & src) const
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
  apply_add(VectorType & dst, const VectorType & src) const
  {
    data->loop(&This::cell_loop,
               &This::face_loop,
               &This::boundary_face_loop_hom_operator,
               this,
               dst,
               src,
               false /*zero_dst_vector = false*/);
  }

  void
  rhs(VectorType & dst, Number const evaluation_time) const
  {
    dst = 0;
    rhs_add(dst, evaluation_time);
  }

  void
  rhs_add(VectorType & dst, Number const evaluation_time) const
  {
    this->eval_time = evaluation_time;

    VectorType tmp;
    tmp.reinit(dst, false /* init with 0 */);

    data->loop(&This::cell_loop_inhom_operator,
               &This::face_loop_inhom_operator,
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

private:
  template<typename FEEvaluationPressure, typename FEEvaluationVelocity>
  void
  do_cell_integral_weak(FEEvaluationPressure & fe_eval_pressure,
                        FEEvaluationVelocity & fe_eval_velocity) const
  {
    for(unsigned int q = 0; q < fe_eval_velocity.n_q_points; ++q)
    {
      fe_eval_velocity.submit_divergence(-fe_eval_pressure.get_value(q), q);
    }
  }

  template<typename FEEvaluationPressure, typename FEEvaluationVelocity>
  void
  do_cell_integral_strong(FEEvaluationPressure & fe_eval_pressure,
                          FEEvaluationVelocity & fe_eval_velocity) const
  {
    for(unsigned int q = 0; q < fe_eval_velocity.n_q_points; ++q)
    {
      fe_eval_velocity.submit_value(fe_eval_pressure.get_gradient(q), q);
    }
  }

  template<typename FEEvaluationPressure, typename FEEvaluationVelocity>
  void
  do_face_integral(FEEvaluationPressure & fe_eval_pressure_m,
                   FEEvaluationPressure & fe_eval_pressure_p,
                   FEEvaluationVelocity & fe_eval_velocity_m,
                   FEEvaluationVelocity & fe_eval_velocity_p) const
  {
    for(unsigned int q = 0; q < fe_eval_velocity_m.n_q_points; ++q)
    {
      scalar value_m = fe_eval_pressure_m.get_value(q);
      scalar value_p = fe_eval_pressure_p.get_value(q);

      scalar flux = calculate_flux(value_m, value_p);

      vector flux_times_normal = flux * fe_eval_pressure_m.get_normal_vector(q);

      fe_eval_velocity_m.submit_value(flux_times_normal, q);
      // minus sign since n⁺ = - n⁻
      fe_eval_velocity_p.submit_value(-flux_times_normal, q);
    }
  }

  template<typename FEEvaluationPressure, typename FEEvaluationVelocity>
  void
  do_boundary_integral(FEEvaluationPressure &     fe_eval_pressure,
                       FEEvaluationVelocity &     fe_eval_velocity,
                       OperatorType const &       operator_type,
                       types::boundary_id const & boundary_id) const
  {
    BoundaryTypeP boundary_type = operator_data.bc->get_boundary_type(boundary_id);

    for(unsigned int q = 0; q < fe_eval_velocity.n_q_points; ++q)
    {
      scalar flux = make_vectorized_array<Number>(0.0);

      if(operator_data.use_boundary_data == true)
      {
        scalar value_m = calculate_interior_value(q, fe_eval_pressure, operator_type);

        scalar value_p = calculate_exterior_value(
          value_m, q, fe_eval_pressure, operator_type, boundary_type, boundary_id);

        flux = calculate_flux(value_m, value_p);
      }
      else // use_boundary_data == false
      {
        scalar value_m = fe_eval_pressure.get_value(q);

        flux = calculate_flux(value_m, value_m /* value_p = value_m */);
      }

      vector flux_times_normal = flux * fe_eval_pressure.get_normal_vector(q);

      fe_eval_velocity.submit_value(flux_times_normal, q);
    }
  }


  /*
   *  This function implements the central flux as numerical flux function.
   */
  inline DEAL_II_ALWAYS_INLINE //
    scalar
    calculate_flux(scalar const & value_m, scalar const & value_p) const
  {
    return 0.5 * (value_m + value_p);
  }

  /*
   *  These two function calculate the interior/exterior value on boundary faces depending on the
   * operator type, the type of the boundary face and the given boundary conditions.
   *
   *                            +--------------------+----------------------+
   *                            | Neumann boundaries | Dirichlet boundaries |
   *  +-------------------------+--------------------+----------------------+
   *  | full operator           | p⁺ = p⁻            | p⁺ = - p⁻ + 2g       |
   *  +-------------------------+--------------------+----------------------+
   *  | homogeneous operator    | p⁺ = p⁻            | p⁺ = - p⁻            |
   *  +-------------------------+--------------------+----------------------+
   *  | inhomogeneous operator  | p⁺ = 0 , p⁻ = 0    | p⁺ = 2g , p⁻ = 0     |
   *  +-------------------------+--------------------+----------------------+
   *
   */
  template<typename FEEvaluationPressure>
  inline DEAL_II_ALWAYS_INLINE //
    scalar
    calculate_interior_value(unsigned int const           q,
                             FEEvaluationPressure const & fe_eval_pressure,
                             OperatorType const &         operator_type) const
  {
    // element e⁻
    scalar value_m = make_vectorized_array<Number>(0.0);

    if(operator_type == OperatorType::full || operator_type == OperatorType::homogeneous)
    {
      value_m = fe_eval_pressure.get_value(q);
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


  template<typename FEEvaluationPressure>
  inline DEAL_II_ALWAYS_INLINE //
    scalar
    calculate_exterior_value(scalar const &               value_m,
                             unsigned int const           q,
                             FEEvaluationPressure const & fe_eval_pressure,
                             OperatorType const &         operator_type,
                             BoundaryTypeP const &        boundary_type,
                             types::boundary_id const     boundary_id = types::boundary_id()) const
  {
    scalar value_p = make_vectorized_array<Number>(0.0);

    if(boundary_type == BoundaryTypeP::Dirichlet)
    {
      if(operator_type == OperatorType::full || operator_type == OperatorType::inhomogeneous)
      {
        scalar g = make_vectorized_array<Number>(0.0);
        typename std::map<types::boundary_id, std::shared_ptr<Function<dim>>>::iterator it;
        it                          = operator_data.bc->dirichlet_bc.find(boundary_id);
        Point<dim, scalar> q_points = fe_eval_pressure.quadrature_point(q);
        evaluate_scalar_function(g, it->second, q_points, eval_time);

        value_p = -value_m + 2.0 * inverse_scaling_factor_pressure * g;
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
    else if(boundary_type == BoundaryTypeP::Neumann)
    {
      value_p = value_m;
    }
    else
    {
      AssertThrow(false, ExcMessage("Boundary type of face is invalid or not implemented."));
    }

    return value_p;
  }

  void
  cell_loop(MatrixFree<dim, Number> const & data,
            VectorType &                    dst,
            VectorType const &              src,
            Range const &                   cell_range) const
  {
    FEEvalVelocity fe_eval_velocity(data,
                                    operator_data.dof_index_velocity,
                                    operator_data.quad_index);
    FEEvalPressure fe_eval_pressure(data,
                                    operator_data.dof_index_pressure,
                                    operator_data.quad_index);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      fe_eval_velocity.reinit(cell);
      fe_eval_pressure.reinit(cell);

      if(operator_data.integration_by_parts == true)
      {
        fe_eval_pressure.gather_evaluate(src, true, false);

        do_cell_integral_weak(fe_eval_pressure, fe_eval_velocity);

        fe_eval_velocity.integrate_scatter(false, true, dst);
      }
      else // integration_by_parts == false
      {
        fe_eval_pressure.gather_evaluate(src, false, true);

        do_cell_integral_strong(fe_eval_pressure, fe_eval_velocity);

        fe_eval_velocity.integrate_scatter(true, false, dst);
      }
    }
  }

  void
  face_loop(MatrixFree<dim, Number> const & data,
            VectorType &                    dst,
            VectorType const &              src,
            Range const &                   face_range) const
  {
    if(operator_data.integration_by_parts == true)
    {
      FEFaceEvalVelocity fe_eval_velocity(data,
                                          true,
                                          operator_data.dof_index_velocity,
                                          operator_data.quad_index);
      FEFaceEvalVelocity fe_eval_velocity_neighbor(data,
                                                   false,
                                                   operator_data.dof_index_velocity,
                                                   operator_data.quad_index);

      FEFaceEvalPressure fe_eval_pressure(data,
                                          true,
                                          operator_data.dof_index_pressure,
                                          operator_data.quad_index);
      FEFaceEvalPressure fe_eval_pressure_neighbor(data,
                                                   false,
                                                   operator_data.dof_index_pressure,
                                                   operator_data.quad_index);

      for(unsigned int face = face_range.first; face < face_range.second; face++)
      {
        fe_eval_velocity.reinit(face);
        fe_eval_velocity_neighbor.reinit(face);

        fe_eval_pressure.reinit(face);
        fe_eval_pressure_neighbor.reinit(face);

        fe_eval_pressure.gather_evaluate(src, true, false);
        fe_eval_pressure_neighbor.gather_evaluate(src, true, false);

        do_face_integral(fe_eval_pressure,
                         fe_eval_pressure_neighbor,
                         fe_eval_velocity,
                         fe_eval_velocity_neighbor);

        fe_eval_velocity.integrate_scatter(true, false, dst);
        fe_eval_velocity_neighbor.integrate_scatter(true, false, dst);
      }
    }
  }

  void
  boundary_face_loop_hom_operator(MatrixFree<dim, Number> const & data,
                                  VectorType &                    dst,
                                  VectorType const &              src,
                                  Range const &                   face_range) const
  {
    if(operator_data.integration_by_parts == true)
    {
      FEFaceEvalVelocity fe_eval_velocity(data,
                                          true,
                                          operator_data.dof_index_velocity,
                                          operator_data.quad_index);
      FEFaceEvalPressure fe_eval_pressure(data,
                                          true,
                                          operator_data.dof_index_pressure,
                                          operator_data.quad_index);

      for(unsigned int face = face_range.first; face < face_range.second; face++)
      {
        fe_eval_velocity.reinit(face);
        fe_eval_pressure.reinit(face);

        fe_eval_pressure.gather_evaluate(src, true, false);

        do_boundary_integral(fe_eval_pressure,
                             fe_eval_velocity,
                             OperatorType::homogeneous,
                             data.get_boundary_id(face));

        fe_eval_velocity.integrate_scatter(true, false, dst);
      }
    }
  }

  void
  boundary_face_loop_full_operator(MatrixFree<dim, Number> const & data,
                                   VectorType &                    dst,
                                   VectorType const &              src,
                                   Range const &                   face_range) const
  {
    if(operator_data.integration_by_parts == true)
    {
      FEFaceEvalVelocity fe_eval_velocity(data,
                                          true,
                                          operator_data.dof_index_velocity,
                                          operator_data.quad_index);
      FEFaceEvalPressure fe_eval_pressure(data,
                                          true,
                                          operator_data.dof_index_pressure,
                                          operator_data.quad_index);

      for(unsigned int face = face_range.first; face < face_range.second; face++)
      {
        fe_eval_velocity.reinit(face);
        fe_eval_pressure.reinit(face);

        fe_eval_pressure.gather_evaluate(src, true, false);

        do_boundary_integral(fe_eval_pressure,
                             fe_eval_velocity,
                             OperatorType::full,
                             data.get_boundary_id(face));

        fe_eval_velocity.integrate_scatter(true, false, dst);
      }
    }
  }

  void
  cell_loop_inhom_operator(MatrixFree<dim, Number> const &,
                           VectorType &,
                           VectorType const &,
                           Range const &) const
  {
  }

  void
  face_loop_inhom_operator(MatrixFree<dim, Number> const &,
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
    if(operator_data.integration_by_parts == true)
    {
      FEFaceEvalVelocity fe_eval_velocity(data,
                                          true,
                                          operator_data.dof_index_velocity,
                                          operator_data.quad_index);
      FEFaceEvalPressure fe_eval_pressure(data,
                                          true,
                                          operator_data.dof_index_pressure,
                                          operator_data.quad_index);

      for(unsigned int face = face_range.first; face < face_range.second; face++)
      {
        fe_eval_velocity.reinit(face);
        fe_eval_pressure.reinit(face);

        do_boundary_integral(fe_eval_pressure,
                             fe_eval_velocity,
                             OperatorType::inhomogeneous,
                             data.get_boundary_id(face));

        fe_eval_velocity.integrate_scatter(true, false, dst);
      }
    }
  }

  MatrixFree<dim, Number> const * data;

  GradientOperatorData<dim> operator_data;

  mutable Number eval_time;

  // if the continuity equation of the incompressible Navier-Stokes
  // equations is scaled by a constant factor, the system of equations
  // is solved for a modified pressure p^* = 1/scaling_factor * p. Hence,
  // when applying the gradient operator to this modified pressure we have
  // to make sure that we also apply the correct boundary conditions for p^*,
  // i.e., g_p^* = 1/scaling_factor * g_p
  Number inverse_scaling_factor_pressure;
};

template<int dim>
struct DivergenceOperatorData
{
  DivergenceOperatorData()
    : dof_index_velocity(0),
      dof_index_pressure(1),
      quad_index(0),
      integration_by_parts(true),
      use_boundary_data(true)
  {
  }

  unsigned int dof_index_velocity;
  unsigned int dof_index_pressure;

  unsigned int quad_index;

  bool integration_by_parts;
  bool use_boundary_data;

  std::shared_ptr<BoundaryDescriptorU<dim>> bc;
};

template<int dim, int degree_u, int degree_p, typename Number>
class DivergenceOperator
{
public:
  typedef DivergenceOperator<dim, degree_u, degree_p, Number> This;

  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef VectorizedArray<Number>                 scalar;
  typedef Tensor<1, dim, VectorizedArray<Number>> vector;

  typedef std::pair<unsigned int, unsigned int> Range;

  typedef FEEvaluation<dim, degree_u, degree_u + 1, dim, Number> FEEvalVelocity;
  typedef FEEvaluation<dim, degree_p, degree_u + 1, 1, Number>   FEEvalPressure;

  typedef FEFaceEvaluation<dim, degree_u, degree_u + 1, dim, Number> FEFaceEvalVelocity;
  typedef FEFaceEvaluation<dim, degree_p, degree_u + 1, 1, Number>   FEFaceEvalPressure;

  DivergenceOperator() : data(nullptr), eval_time(0.0)
  {
  }

  void
  initialize(MatrixFree<dim, Number> const &     mf_data,
             DivergenceOperatorData<dim> const & operator_data_in)
  {
    this->data          = &mf_data;
    this->operator_data = operator_data_in;
  }

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

    data->loop(&This::cell_loop_inhom_operator,
               &This::face_loop_inhom_operator,
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

private:
  template<typename FEEvaluationPressure, typename FEEvaluationVelocity>
  void
  do_cell_integral_weak(FEEvaluationPressure & fe_eval_pressure,
                        FEEvaluationVelocity & fe_eval_velocity) const
  {
    for(unsigned int q = 0; q < fe_eval_velocity.n_q_points; ++q)
    {
      // minus sign due to integration by parts
      fe_eval_pressure.submit_gradient(-fe_eval_velocity.get_value(q), q);
    }
  }

  template<typename FEEvaluationPressure, typename FEEvaluationVelocity>
  void
  do_cell_integral_strong(FEEvaluationPressure & fe_eval_pressure,
                          FEEvaluationVelocity & fe_eval_velocity) const
  {
    for(unsigned int q = 0; q < fe_eval_velocity.n_q_points; ++q)
    {
      fe_eval_pressure.submit_value(fe_eval_velocity.get_divergence(q), q);
    }
  }

  template<typename FEEvaluationPressure, typename FEEvaluationVelocity>
  void
  do_face_integral(FEEvaluationVelocity & fe_eval_velocity_m,
                   FEEvaluationVelocity & fe_eval_velocity_p,
                   FEEvaluationPressure & fe_eval_pressure_m,
                   FEEvaluationPressure & fe_eval_pressure_p) const
  {
    for(unsigned int q = 0; q < fe_eval_velocity_m.n_q_points; ++q)
    {
      vector value_m = fe_eval_velocity_m.get_value(q);
      vector value_p = fe_eval_velocity_p.get_value(q);

      vector flux = calculate_flux(value_m, value_p);

      scalar flux_times_normal = flux * fe_eval_velocity_m.get_normal_vector(q);

      fe_eval_pressure_m.submit_value(flux_times_normal, q);
      // minus sign since n⁺ = - n⁻
      fe_eval_pressure_p.submit_value(-flux_times_normal, q);
    }
  }

  template<typename FEEvaluationPressure, typename FEEvaluationVelocity>
  void
  do_boundary_integral(FEEvaluationVelocity &     fe_eval_velocity,
                       FEEvaluationPressure &     fe_eval_pressure,
                       OperatorType const &       operator_type,
                       types::boundary_id const & boundary_id) const
  {
    BoundaryTypeU boundary_type = operator_data.bc->get_boundary_type(boundary_id);

    for(unsigned int q = 0; q < fe_eval_pressure.n_q_points; ++q)
    {
      vector flux;

      if(operator_data.use_boundary_data == true)
      {
        vector value_m = calculate_interior_value(q, fe_eval_velocity, operator_type);
        vector value_p = calculate_exterior_value(
          value_m, q, fe_eval_velocity, operator_type, boundary_type, boundary_id);

        flux = calculate_flux(value_m, value_p);
      }
      else // use_boundary_data == false
      {
        vector value_m = fe_eval_velocity.get_value(q);

        flux = calculate_flux(value_m, value_m /* value_p = value_m */);
      }

      scalar flux_times_normal = flux * fe_eval_velocity.get_normal_vector(q);
      fe_eval_pressure.submit_value(flux_times_normal, q);
    }
  }

  /*
   *  This function implements the central flux as numerical flux function.
   */
  inline DEAL_II_ALWAYS_INLINE //
    vector
    calculate_flux(vector const & value_m, vector const & value_p) const
  {
    return 0.5 * (value_m + value_p);
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

  void
  cell_loop(MatrixFree<dim, Number> const & data,
            VectorType &                    dst,
            VectorType const &              src,
            Range const &                   cell_range) const
  {
    FEEvalVelocity fe_eval_velocity(data,
                                    operator_data.dof_index_velocity,
                                    operator_data.quad_index);
    FEEvalPressure fe_eval_pressure(data,
                                    operator_data.dof_index_pressure,
                                    operator_data.quad_index);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      fe_eval_pressure.reinit(cell);

      fe_eval_velocity.reinit(cell);

      if(operator_data.integration_by_parts == true)
      {
        fe_eval_velocity.gather_evaluate(src, true, false, false);

        do_cell_integral_weak(fe_eval_pressure, fe_eval_velocity);

        fe_eval_pressure.integrate_scatter(false, true, dst);
      }
      else // integration_by_parts == false
      {
        fe_eval_velocity.gather_evaluate(src, false, true, false);

        do_cell_integral_strong(fe_eval_pressure, fe_eval_velocity);

        fe_eval_pressure.integrate_scatter(true, false, dst);
      }
    }
  }

  void
  face_loop(MatrixFree<dim, Number> const & data,
            VectorType &                    dst,
            VectorType const &              src,
            Range const &                   face_range) const
  {
    if(operator_data.integration_by_parts == true)
    {
      FEFaceEvalVelocity fe_eval_velocity(data,
                                          true,
                                          operator_data.dof_index_velocity,
                                          operator_data.quad_index);
      FEFaceEvalVelocity fe_eval_velocity_neighbor(data,
                                                   false,
                                                   operator_data.dof_index_velocity,
                                                   operator_data.quad_index);

      FEFaceEvalPressure fe_eval_pressure(data,
                                          true,
                                          operator_data.dof_index_pressure,
                                          operator_data.quad_index);
      FEFaceEvalPressure fe_eval_pressure_neighbor(data,
                                                   false,
                                                   operator_data.dof_index_pressure,
                                                   operator_data.quad_index);

      for(unsigned int face = face_range.first; face < face_range.second; face++)
      {
        fe_eval_pressure.reinit(face);
        fe_eval_pressure_neighbor.reinit(face);

        fe_eval_velocity.reinit(face);
        fe_eval_velocity_neighbor.reinit(face);

        fe_eval_velocity.gather_evaluate(src, true, false);
        fe_eval_velocity_neighbor.gather_evaluate(src, true, false);

        do_face_integral(fe_eval_velocity,
                         fe_eval_velocity_neighbor,
                         fe_eval_pressure,
                         fe_eval_pressure_neighbor);

        fe_eval_pressure.integrate_scatter(true, false, dst);
        fe_eval_pressure_neighbor.integrate_scatter(true, false, dst);
      }
    }
  }

  void
  boundary_face_loop_hom_operator(MatrixFree<dim, Number> const & data,
                                  VectorType &                    dst,
                                  VectorType const &              src,
                                  Range const &                   face_range) const
  {
    if(operator_data.integration_by_parts == true)
    {
      FEFaceEvalVelocity fe_eval_velocity(data,
                                          true,
                                          operator_data.dof_index_velocity,
                                          operator_data.quad_index);

      FEFaceEvalPressure fe_eval_pressure(data,
                                          true,
                                          operator_data.dof_index_pressure,
                                          operator_data.quad_index);

      for(unsigned int face = face_range.first; face < face_range.second; face++)
      {
        fe_eval_pressure.reinit(face);
        fe_eval_velocity.reinit(face);

        fe_eval_velocity.gather_evaluate(src, true, false);

        do_boundary_integral(fe_eval_velocity,
                             fe_eval_pressure,
                             OperatorType::homogeneous,
                             data.get_boundary_id(face));

        fe_eval_pressure.integrate_scatter(true, false, dst);
      }
    }
  }

  void
  boundary_face_loop_full_operator(MatrixFree<dim, Number> const & data,
                                   VectorType &                    dst,
                                   VectorType const &              src,
                                   Range const &                   face_range) const
  {
    if(operator_data.integration_by_parts == true)
    {
      FEFaceEvalVelocity fe_eval_velocity(data,
                                          true,
                                          operator_data.dof_index_velocity,
                                          operator_data.quad_index);

      FEFaceEvalPressure fe_eval_pressure(data,
                                          true,
                                          operator_data.dof_index_pressure,
                                          operator_data.quad_index);

      for(unsigned int face = face_range.first; face < face_range.second; face++)
      {
        fe_eval_pressure.reinit(face);
        fe_eval_velocity.reinit(face);

        fe_eval_velocity.gather_evaluate(src, true, false);

        do_boundary_integral(fe_eval_velocity,
                             fe_eval_pressure,
                             OperatorType::full,
                             data.get_boundary_id(face));

        fe_eval_pressure.integrate_scatter(true, false, dst);
      }
    }
  }

  void
  cell_loop_inhom_operator(MatrixFree<dim, Number> const &,
                           VectorType &,
                           VectorType const &,
                           Range const &) const
  {
  }

  void
  face_loop_inhom_operator(MatrixFree<dim, Number> const &,
                           VectorType &,
                           VectorType const &,
                           Range const &) const
  {
  }

  void
  boundary_face_loop_inhom_operator(MatrixFree<dim, Number> const & data,
                                    VectorType &                    dst,
                                    VectorType const &,
                                    std::pair<unsigned int, unsigned int> const & face_range) const
  {
    if(operator_data.integration_by_parts == true)
    {
      FEFaceEvalVelocity fe_eval_velocity(data,
                                          true,
                                          operator_data.dof_index_velocity,
                                          operator_data.quad_index);

      FEFaceEvalPressure fe_eval_pressure(data,
                                          true,
                                          operator_data.dof_index_pressure,
                                          operator_data.quad_index);

      for(unsigned int face = face_range.first; face < face_range.second; face++)
      {
        fe_eval_pressure.reinit(face);
        fe_eval_velocity.reinit(face);

        do_boundary_integral(fe_eval_velocity,
                             fe_eval_pressure,
                             OperatorType::inhomogeneous,
                             data.get_boundary_id(face));

        fe_eval_pressure.integrate_scatter(true, false, dst);
      }
    }
  }

  MatrixFree<dim, Number> const * data;

  DivergenceOperatorData<dim> operator_data;

  mutable Number eval_time;
};

template<int dim>
struct ConvectiveOperatorData
{
  ConvectiveOperatorData()
    : formulation(FormulationConvectiveTerm::DivergenceFormulation),
      dof_index(0),
      quad_index(0),
      upwind_factor(1.0),
      use_outflow_bc(false),
      type_dirichlet_bc(TypeDirichletBCs::Mirror),
      use_cell_based_loops(false)
  {
  }

  FormulationConvectiveTerm formulation;

  unsigned int dof_index;

  unsigned int quad_index;

  double upwind_factor;

  bool use_outflow_bc;

  TypeDirichletBCs type_dirichlet_bc;

  std::shared_ptr<BoundaryDescriptorU<dim>> bc;

  // use cell based loops
  bool use_cell_based_loops;
};



template<int dim, int degree, typename Number>
class ConvectiveOperator
{
public:
  typedef ConvectiveOperator<dim, degree, Number> This;

  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef VectorizedArray<Number>                 scalar;
  typedef Tensor<1, dim, VectorizedArray<Number>> vector;
  typedef Tensor<2, dim, VectorizedArray<Number>> tensor;

  typedef std::pair<unsigned int, unsigned int> Range;

  static const unsigned int n_q_points_overint = degree + (degree + 2) / 2;

  typedef FEEvaluation<dim, degree, n_q_points_overint, dim, Number>     FEEvalCellOverint;
  typedef FEFaceEvaluation<dim, degree, n_q_points_overint, dim, Number> FEEvalFaceOverint;

  ConvectiveOperator()
    : data(nullptr),
      eval_time(0.0),
      n_mpi_processes(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
  {
  }

  ConvectiveOperatorData<dim> const &
  get_operator_data() const
  {
    return operator_data;
  }

  void
  set_solution_linearization(VectorType const & src) const
  {
    velocity_linearization = src;
  }

  VectorType const &
  get_solution_linearization() const
  {
    return velocity_linearization;
  }

  void
  initialize(MatrixFree<dim, Number> const &     data_in,
             ConvectiveOperatorData<dim> const & operator_data_in)
  {
    data          = &data_in;
    operator_data = operator_data_in;

    data->initialize_dof_vector(velocity_linearization, operator_data.dof_index);

    // Block Jacobi elementwise
    fe_eval.reset(new FEEvalCellOverint(*data, operator_data.dof_index, operator_data.quad_index));
    fe_eval_m.reset(
      new FEEvalFaceOverint(*data, true, operator_data.dof_index, operator_data.quad_index));
    fe_eval_p.reset(
      new FEEvalFaceOverint(*data, false, operator_data.dof_index, operator_data.quad_index));
    fe_eval_linearization.reset(
      new FEEvalCellOverint(*data, operator_data.dof_index, operator_data.quad_index));
    fe_eval_linearization_m.reset(
      new FEEvalFaceOverint(*data, true, operator_data.dof_index, operator_data.quad_index));
    fe_eval_linearization_p.reset(
      new FEEvalFaceOverint(*data, false, operator_data.dof_index, operator_data.quad_index));
  }

  /*
   * Evaluate nonlinear convective operator.
   */
  void
  evaluate(VectorType & dst, VectorType const & src, Number const evaluation_time) const
  {
    this->eval_time = evaluation_time;

    data->loop(&This::cell_loop_nonlinear_operator,
               &This::face_loop_nonlinear_operator,
               &This::boundary_face_loop_nonlinear_operator,
               this,
               dst,
               src,
               true /*zero_dst_vector = true*/,
               MatrixFree<dim, Number>::only_values,
               MatrixFree<dim, Number>::only_values);
  }

  void
  evaluate_add(VectorType & dst, VectorType const & src, Number const evaluation_time) const
  {
    this->eval_time = evaluation_time;

    data->loop(&This::cell_loop_nonlinear_operator,
               &This::face_loop_nonlinear_operator,
               &This::boundary_face_loop_nonlinear_operator,
               this,
               dst,
               src,
               false /*zero_dst_vector = false*/,
               MatrixFree<dim, Number>::only_values,
               MatrixFree<dim, Number>::only_values);
  }

  /*
   * Apply linearized convective operator.
   */
  void
  apply(VectorType & dst, VectorType const & src, Number const evaluation_time) const
  {
    AssertThrow(operator_data.formulation_convective_term ==
                  FormulationConvectiveTerm::DivergenceFormulation,
                ExcMessage("Only divergence formulation is implemented."));

    this->eval_time = evaluation_time;

    data->loop(&This::cell_loop_linearized_operator,
               &This::face_loop_linearized_operator,
               &This::boundary_face_loop_linearized_operator,
               this,
               dst,
               src,
               true /*zero_dst_vector = true*/,
               MatrixFree<dim, Number>::only_values,
               MatrixFree<dim, Number>::only_values);
  }

  void
  apply_add(VectorType & dst, VectorType const & src, Number const evaluation_time) const
  {
    AssertThrow(operator_data.formulation == FormulationConvectiveTerm::DivergenceFormulation,
                ExcMessage("Only divergence formulation is implemented."));

    this->eval_time = evaluation_time;

    data->loop(&This::cell_loop_linearized_operator,
               &This::face_loop_linearized_operator,
               &This::boundary_face_loop_linearized_operator,
               this,
               dst,
               src,
               false /*zero_dst_vector = false*/,
               MatrixFree<dim, Number>::only_values,
               MatrixFree<dim, Number>::only_values);
  }

  /*
   * Calculate diagonal of linearized convective operator.
   */
  void
  calculate_diagonal(VectorType & diagonal, Number const evaluation_time) const
  {
    AssertThrow(operator_data.formulation == FormulationConvectiveTerm::DivergenceFormulation,
                ExcMessage("Only divergence formulation is implemented."));

    this->eval_time = evaluation_time;

    VectorType src;

    data->loop(&This::cell_loop_diagonal,
               &This::face_loop_diagonal,
               &This::boundary_face_loop_diagonal,
               this,
               diagonal,
               src,
               true /*zero_dst_vector = true*/,
               MatrixFree<dim, Number>::only_values,
               MatrixFree<dim, Number>::only_values);
  }

  void
  add_diagonal(VectorType & diagonal, Number const evaluation_time) const
  {
    AssertThrow(operator_data.formulation == FormulationConvectiveTerm::DivergenceFormulation,
                ExcMessage("Only divergence formulation is implemented."));

    this->eval_time = evaluation_time;

    VectorType src;

    data->loop(&This::cell_loop_diagonal,
               &This::face_loop_diagonal,
               &This::boundary_face_loop_diagonal,
               this,
               diagonal,
               src,
               false /*zero_dst_vector = false*/,
               MatrixFree<dim, Number>::only_values,
               MatrixFree<dim, Number>::only_values);
  }

  /*
   * Assemble block diagonal matrices for linearized convective operator.
   */
  void
  add_block_diagonal_matrices(std::vector<LAPACKFullMatrix<Number>> & matrices,
                              Number const                            evaluation_time) const
  {
    AssertThrow(operator_data.formulation == FormulationConvectiveTerm::DivergenceFormulation,
                ExcMessage("Only divergence formulation is implemented."));

    this->eval_time = evaluation_time;

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

  /*
   *  Apply block-diagonal operator as a global operation (only needed in order to test the
   * implementation for assembling the block-diagonal).
   */
  void
  apply_block_diagonal(VectorType & dst, VectorType const & src, Number const evaluation_time) const
  {
    AssertThrow(operator_data.formulation == FormulationConvectiveTerm::DivergenceFormulation,
                ExcMessage("Only divergence formulation is implemented."));

    this->eval_time = evaluation_time;

    data->loop(&This::cell_loop_linearized_operator,
               &This::face_loop_apply_block_diagonal_linearized_operator,
               &This::boundary_face_loop_linearized_operator,
               this,
               dst,
               src,
               true /*zero_dst_vector = true*/,
               MatrixFree<dim, Number>::only_values,
               MatrixFree<dim, Number>::only_values);
  }

  void
  apply_block_diagonal_add(VectorType &       dst,
                           VectorType const & src,
                           Number const       evaluation_time) const
  {
    AssertThrow(operator_data.formulation == FormulationConvectiveTerm::DivergenceFormulation,
                ExcMessage("Only divergence formulation is implemented."));

    this->eval_time = evaluation_time;

    data->loop(&This::cell_loop_linearized_operator,
               &This::face_loop_apply_block_diagonal_linearized_operator,
               &This::boundary_face_loop_linearized_operator,
               this,
               dst,
               src,
               false /*zero_dst_vector = false*/,
               MatrixFree<dim, Number>::only_values,
               MatrixFree<dim, Number>::only_values);
  }

  void
  apply_add_block_diagonal_elementwise(unsigned int const   cell,
                                       scalar * const       dst,
                                       scalar const * const src,
                                       unsigned int const   problem_size = 1) const
  {
    unsigned int dofs_per_cell = fe_eval->dofs_per_cell;

    fe_eval_linearization->reinit(cell);
    fe_eval_linearization->gather_evaluate(velocity_linearization, true, false, false);

    fe_eval->reinit(cell);

    for(unsigned int i = 0; i < dofs_per_cell; ++i)
      fe_eval->begin_dof_values()[i] = src[i];

    fe_eval->evaluate(true, false, false);

    do_cell_integral_linearized_operator(*fe_eval, *fe_eval_linearization);

    fe_eval->integrate(false, true);

    for(unsigned int i = 0; i < dofs_per_cell; ++i)
      dst[i] += fe_eval->begin_dof_values()[i];

    // loop over all faces
    unsigned int const n_faces = GeometryInfo<dim>::faces_per_cell;
    for(unsigned int face = 0; face < n_faces; ++face)
    {
      fe_eval_linearization_m->reinit(cell, face);
      fe_eval_linearization_m->gather_evaluate(velocity_linearization, true, false);

      // TODO
      //        AssertThrow(false, ExcMessage("We have to evaluate the linearized velocity field on
      //        the neighbor. This functionality is not implemented in deal.II/matrix_free."));
      //        fe_eval_linearization_p->reinit(cell, face);
      //        fe_eval_linearization_p->gather_evaluate(velocity_linearization, true, false);

      fe_eval_m->reinit(cell, face);
      fe_eval_p->reinit(cell, face);

      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        fe_eval_m->begin_dof_values()[i] = src[i];

      // do not need to read dof values for fe_eval_p (already initialized with 0)

      fe_eval_m->evaluate(true, false);

      auto bids        = data->get_faces_by_cells_boundary_id(cell, face);
      auto boundary_id = bids[0];

      if(boundary_id == numbers::internal_face_boundary_id) // internal face
      {
        // TODO
        //            do_face_int_integral_linearized_operator(*fe_eval_m, *fe_eval_p,
        //            *fe_eval_linearization_m, *fe_eval_linearization_p);

        // plug in fe_eval_linearization_m twice to avoid the above problem with accessing dofs of
        // the neighboring element
        do_face_int_integral_linearized_operator(*fe_eval_m,
                                                 *fe_eval_p,
                                                 *fe_eval_linearization_m,
                                                 *fe_eval_linearization_m);
      }
      else // boundary face
      {
        do_boundary_integral_linearized_operator(*fe_eval_m, *fe_eval_linearization_m, boundary_id);
      }

      fe_eval_m->integrate(true, false);

      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        dst[i] += fe_eval_m->begin_dof_values()[i];
    }
  }

private:
  template<typename FEEvaluation>
  void
  do_cell_integral_nonlinear_operator(FEEvaluation & fe_eval) const
  {
    if(operator_data.formulation == FormulationConvectiveTerm::DivergenceFormulation)
    {
      fe_eval.evaluate(true, false, false);
      for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
      {
        // nonlinear convective flux F(u) = uu
        vector u = fe_eval.get_value(q);
        tensor F = outer_product(u, u);
        // minus sign due to integration by parts
        fe_eval.submit_gradient(-F, q);
      }
      fe_eval.integrate(false, true);
    }
    else if(operator_data.formulation == FormulationConvectiveTerm::ConvectiveFormulation)
    {
      fe_eval.evaluate(true, true, false);
      for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
      {
        // convective formulation: (u * grad) u = grad(u) * u
        vector u          = fe_eval.get_value(q);
        tensor gradient_u = fe_eval.get_gradient(q);
        vector F          = gradient_u * u;

        // plus sign since the strong formulation is used, i.e.
        // integration by parts is performed twice
        fe_eval.submit_value(F, q);
      }
      fe_eval.integrate(true, false);
    }
    else if(operator_data.formulation == FormulationConvectiveTerm::EnergyPreservingFormulation)
    {
      fe_eval.evaluate(true, true, false);
      for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
      {
        // nonlinear convective flux F(u) = uu
        vector u          = fe_eval.get_value(q);
        tensor F          = outer_product(u, u);
        scalar divergence = fe_eval.get_divergence(q);
        vector div_term   = -0.5 * divergence * u;
        // minus sign due to integration by parts
        fe_eval.submit_gradient(-F, q);
        fe_eval.submit_value(div_term, q);
      }
      fe_eval.integrate(true, true);
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }
  }

  template<typename FEEvaluation>
  void
  do_cell_integral_linearized_operator(FEEvaluation & fe_eval,
                                       FEEvaluation & fe_eval_linearization) const
  {
    for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
    {
      vector delta_u = fe_eval.get_value(q);
      vector u       = fe_eval_linearization.get_value(q);
      tensor F       = outer_product(u, delta_u);
      fe_eval.submit_gradient(-(F + transpose(F)), q); // minus sign due to integration by parts
    }
  }

  template<typename FEEvaluation>
  void
  do_face_integral_nonlinear_operator(FEEvaluation & fe_eval_m, FEEvaluation & fe_eval_p) const
  {
    if(operator_data.formulation == FormulationConvectiveTerm::DivergenceFormulation)
    {
      for(unsigned int q = 0; q < fe_eval_m.n_q_points; ++q)
      {
        vector uM     = fe_eval_m.get_value(q);
        vector uP     = fe_eval_p.get_value(q);
        vector normal = fe_eval_m.get_normal_vector(q);

        vector flux = calculate_lax_friedrichs_flux(uM, uP, normal);

        fe_eval_m.submit_value(flux, q);
        fe_eval_p.submit_value(-flux, q); // minus sign since n⁺ = - n⁻
      }
    }
    else if(operator_data.formulation == FormulationConvectiveTerm::ConvectiveFormulation)
    {
      for(unsigned int q = 0; q < fe_eval_m.n_q_points; ++q)
      {
        vector uM     = fe_eval_m.get_value(q);
        vector uP     = fe_eval_p.get_value(q);
        vector normal = fe_eval_m.get_normal_vector(q);

        vector flux_times_normal       = calculate_upwind_flux(uM, uP, normal);
        scalar average_normal_velocity = 0.5 * (uM + uP) * normal;

        // second term appears since the strong formulation is implemented (integration by parts
        // is performed twice)
        fe_eval_m.submit_value(flux_times_normal - average_normal_velocity * uM, q);
        // opposite signs since n⁺ = - n⁻
        fe_eval_p.submit_value(-flux_times_normal + average_normal_velocity * uP, q);
      }
    }
    else if(operator_data.formulation == FormulationConvectiveTerm::EnergyPreservingFormulation)
    {
      for(unsigned int q = 0; q < fe_eval_m.n_q_points; ++q)
      {
        vector uM     = fe_eval_m.get_value(q);
        vector uP     = fe_eval_p.get_value(q);
        vector jump   = uM - uP;
        vector normal = fe_eval_m.get_normal_vector(q);

        vector flux = calculate_lax_friedrichs_flux(uM, uP, normal);

        // corrections to obtain an energy preserving flux (which is not conservative!)
        vector flux_m = flux + 0.25 * jump * normal * uP;
        vector flux_p = -flux + 0.25 * jump * normal * uM;

        fe_eval_m.submit_value(flux_m, q);
        fe_eval_p.submit_value(flux_p, q);
      }
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }
  }

  template<typename FEEvaluation>
  void
  do_boundary_integral_nonlinear_operator(FEEvaluation &             fe_eval,
                                          types::boundary_id const & boundary_id) const
  {
    BoundaryTypeU boundary_type = operator_data.bc->get_boundary_type(boundary_id);

    if(operator_data.formulation == FormulationConvectiveTerm::DivergenceFormulation)
    {
      for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
      {
        vector uM = fe_eval.get_value(q);
        vector uP = calculate_exterior_value(uM, q, fe_eval, boundary_type, boundary_id);

        vector normalM = fe_eval.get_normal_vector(q);

        // calculate flux
        vector flux;

        if(operator_data.use_outflow_bc == true && boundary_type == BoundaryTypeU::Neumann)
        {
          // outflow BC according to Gravemeier et al. (2012):
          // we need a factor indicating whether we have inflow or outflow
          // on the Neumann part of the boundary.
          // outflow: factor =  1.0 (do nothing, neutral element of multiplication)
          // inflow:  factor = -1.0 (set convective flux to zero)
          scalar outflow_indicator = make_vectorized_array<Number>(1.0);

          scalar uM_n = uM * normalM;

          for(unsigned int v = 0; v < VectorizedArray<Number>::n_array_elements; ++v)
          {
            if(uM_n[v] < 0.0) // inflow
              outflow_indicator[v] = -1.0;
          }

          flux = calculate_lax_friedrichs_flux(uM, uP, normalM, outflow_indicator);
        }
        else // standard
        {
          flux = calculate_lax_friedrichs_flux(uM, uP, normalM);
        }

        fe_eval.submit_value(flux, q);
      }
    }
    else if(operator_data.formulation == FormulationConvectiveTerm::ConvectiveFormulation)
    {
      for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
      {
        vector uM = fe_eval.get_value(q);
        vector uP = calculate_exterior_value(uM, q, fe_eval, boundary_type, boundary_id);

        vector normal = fe_eval.get_normal_vector(q);

        vector flux_times_normal       = calculate_upwind_flux(uM, uP, normal);
        scalar average_normal_velocity = 0.5 * (uM + uP) * normal;

        // second term appears since the strong formulation is implemented (integration by parts
        // is performed twice)
        fe_eval.submit_value(flux_times_normal - average_normal_velocity * uM, q);
      }
    }
    else if(operator_data.formulation == FormulationConvectiveTerm::EnergyPreservingFormulation)
    {
      for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
      {
        vector uM      = fe_eval.get_value(q);
        vector uP      = calculate_exterior_value(uM, q, fe_eval, boundary_type, boundary_id);
        vector normalM = fe_eval.get_normal_vector(q);

        vector flux = calculate_lax_friedrichs_flux(uM, uP, normalM);

        // corrections to obtain an energy preserving flux (which is not conservative!)
        flux = flux + 0.25 * (uM - uP) * normalM * uP;
        fe_eval.submit_value(flux, q);
      }
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }
  }

  template<typename FEEvaluation>
  void
  do_face_integral_linearized_operator(FEEvaluation & fe_eval_m,
                                       FEEvaluation & fe_eval_p,
                                       FEEvaluation & fe_eval_linearization_m,
                                       FEEvaluation & fe_eval_linearization_p) const
  {
    for(unsigned int q = 0; q < fe_eval_m.n_q_points; ++q)
    {
      vector uM = fe_eval_linearization_m.get_value(q);
      vector uP = fe_eval_linearization_p.get_value(q);

      vector delta_uM = fe_eval_m.get_value(q);
      vector delta_uP = fe_eval_p.get_value(q);

      vector normal = fe_eval_m.get_normal_vector(q);

      vector flux = calculate_lax_friedrichs_flux_linearized(uM, uP, delta_uM, delta_uP, normal);

      fe_eval_m.submit_value(flux, q);
      fe_eval_p.submit_value(-flux, q); // minus sign since n⁺ = -n⁻
    }
  }

  template<typename FEEvaluation>
  void
  do_face_int_integral_linearized_operator(FEEvaluation & fe_eval_m,
                                           FEEvaluation & /* fe_eval_p */,
                                           FEEvaluation & fe_eval_linearization_m,
                                           FEEvaluation & fe_eval_linearization_p) const
  {
    for(unsigned int q = 0; q < fe_eval_m.n_q_points; ++q)
    {
      vector uM = fe_eval_linearization_m.get_value(q);
      vector uP = fe_eval_linearization_p.get_value(q);

      vector delta_uM = fe_eval_m.get_value(q);
      vector delta_uP; // set exterior value to zero

      vector normal_m = fe_eval_m.get_normal_vector(q);

      vector flux = calculate_lax_friedrichs_flux_linearized(uM, uP, delta_uM, delta_uP, normal_m);

      fe_eval_m.submit_value(flux, q);
    }
  }

  template<typename FEEvaluation>
  void
  do_face_ext_integral_linearized_operator(FEEvaluation & /* fe_eval_m */,
                                           FEEvaluation & fe_eval_p,
                                           FEEvaluation & fe_eval_linearization_m,
                                           FEEvaluation & fe_eval_linearization_p) const
  {
    for(unsigned int q = 0; q < fe_eval_p.n_q_points; ++q)
    {
      vector uM = fe_eval_linearization_m.get_value(q);
      vector uP = fe_eval_linearization_p.get_value(q);

      vector delta_uM; // set exterior value to zero
      vector delta_uP = fe_eval_p.get_value(q);

      vector normal_p = -fe_eval_p.get_normal_vector(q);

      vector flux = calculate_lax_friedrichs_flux_linearized(uP, uM, delta_uP, delta_uM, normal_p);

      fe_eval_p.submit_value(flux, q);
    }
  }

  template<typename FEEvaluation>
  void
  do_boundary_integral_linearized_operator(FEEvaluation &             fe_eval,
                                           FEEvaluation &             fe_eval_linearization,
                                           types::boundary_id const & boundary_id) const
  {
    BoundaryTypeU boundary_type = operator_data.bc->get_boundary_type(boundary_id);

    for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
    {
      vector uM = fe_eval_linearization.get_value(q);
      vector uP =
        calculate_exterior_value(uM, q, fe_eval_linearization, boundary_type, boundary_id);

      vector delta_uM = fe_eval.get_value(q);
      vector delta_uP = calculate_exterior_value_linearized(delta_uM, q, fe_eval, boundary_type);

      vector normal = fe_eval.get_normal_vector(q);

      vector flux = calculate_lax_friedrichs_flux_linearized(uM, uP, delta_uM, delta_uP, normal);

      fe_eval.submit_value(flux, q);
    }
  }

  /*
   *  This function calculates the exterior velocity on boundary faces
   *  according to:
   *
   *  Dirichlet boundary: u⁺ = -u⁻ + 2g
   *  Neumann boundary:   u⁺ = u⁻
   *  symmetry boundary:  u⁺ = u⁻ -(u⁻*n)n - (u⁻*n)n = u⁻ - 2 (u⁻*n)n
   */
  template<typename FEEvaluation>
  inline DEAL_II_ALWAYS_INLINE //
    vector
    calculate_exterior_value(vector const &           uM,
                             unsigned int const       q,
                             FEEvaluation &           fe_eval,
                             BoundaryTypeU const &    boundary_type,
                             types::boundary_id const boundary_id) const
  {
    vector uP;

    if(boundary_type == BoundaryTypeU::Dirichlet)
    {
      vector g;

      typename std::map<types::boundary_id, std::shared_ptr<Function<dim>>>::iterator it;
      it                          = operator_data.bc->dirichlet_bc.find(boundary_id);
      Point<dim, scalar> q_points = fe_eval.quadrature_point(q);
      evaluate_vectorial_function(g, it->second, q_points, eval_time);

      if(operator_data.type_dirichlet_bc == TypeDirichletBCs::Mirror)
      {
        uP = -uM + make_vectorized_array<Number>(2.0) * g;
      }
      else if(operator_data.type_dirichlet_bc == TypeDirichletBCs::Direct)
      {
        uP = g;
      }
      else
      {
        AssertThrow(
          false,
          ExcMessage(
            "Type of imposition of Dirichlet BC's for convective term is not implemented."));
      }
    }
    else if(boundary_type == BoundaryTypeU::Neumann)
    {
      uP = uM;
    }
    else if(boundary_type == BoundaryTypeU::Symmetry)
    {
      vector normalM = fe_eval.get_normal_vector(q);

      uP = uM - 2. * (uM * normalM) * normalM;
    }
    else
    {
      AssertThrow(false, ExcMessage("Boundary type of face is invalid or not implemented."));
    }

    return uP;
  }

  /*
   *  Calculate Lax-Friedrichs flux for linearized operator on boundary faces.
   *
   *  Homogeneous linearized operator:
   *  Dirichlet boundary: delta_u⁺ = - delta_u⁻
   *  Neumann boundary:   delta_u⁺ = + delta_u⁻
   *  symmetry boundary:  delta_u⁺ = delta_u⁻ - 2 (delta_u⁻*n)n
   */
  template<typename FEEvaluation>
  inline DEAL_II_ALWAYS_INLINE //
    vector
    calculate_exterior_value_linearized(vector &              delta_uM,
                                        unsigned int const    q,
                                        FEEvaluation &        fe_eval,
                                        BoundaryTypeU const & boundary_type) const
  {
    // element e⁺
    vector delta_uP;

    if(boundary_type == BoundaryTypeU::Dirichlet)
    {
      if(operator_data.type_dirichlet_bc == TypeDirichletBCs::Mirror)
      {
        delta_uP = -delta_uM;
      }
      else if(operator_data.type_dirichlet_bc == TypeDirichletBCs::Direct)
      {
        // delta_uP = 0
        // do nothing, delta_uP is already initialized with zero
      }
      else
      {
        AssertThrow(
          false,
          ExcMessage(
            "Type of imposition of Dirichlet BC's for convective term is not implemented."));
      }
    }
    else if(boundary_type == BoundaryTypeU::Neumann)
    {
      delta_uP = delta_uM;
    }
    else if(boundary_type == BoundaryTypeU::Symmetry)
    {
      vector normalM = fe_eval.get_normal_vector(q);
      delta_uP       = delta_uM - 2. * (delta_uM * normalM) * normalM;
    }
    else
    {
      AssertThrow(false, ExcMessage("Boundary type of face is invalid or not implemented."));
    }

    return delta_uP;
  }

  /*
   *  Lax-Friedrichs flux (divergence formulation)
   *  Calculation of lambda according to Shahbazi et al.:
   *  lambda = max ( max |lambda(flux_jacobian_M)| , max |lambda(flux_jacobian_P)| )
   *         = max ( | 2*(uM)^T*normal | , | 2*(uP)^T*normal | )
   */
  inline DEAL_II_ALWAYS_INLINE //
    scalar
    calculate_lambda(scalar const & uM_n, scalar const & uP_n) const
  {
    return this->operator_data.upwind_factor * 2.0 * std::max(std::abs(uM_n), std::abs(uP_n));
  }

  /*
   *  Calculate Lax-Friedrichs flux for nonlinear operator (divergence formulation).
   */
  inline DEAL_II_ALWAYS_INLINE //
    vector
    calculate_lax_friedrichs_flux(
      vector const & uM,
      vector const & uP,
      vector const & normalM,
      scalar const & outflow_indicator = make_vectorized_array<Number>(1.0)) const
  {
    scalar uM_n = uM * normalM;
    scalar uP_n = uP * normalM;

    vector average_normal_flux =
      make_vectorized_array<Number>(0.5) * (uM * uM_n + outflow_indicator * uP * uP_n);

    vector jump_value = uM - uP;

    scalar lambda = calculate_lambda(uM_n, uP_n);

    return (average_normal_flux + 0.5 * lambda * jump_value);
  }

  /*
   *  Calculate Lax-Friedrichs flux for linearized operator (divergence formulation).
   */
  inline DEAL_II_ALWAYS_INLINE //
    vector
    calculate_lax_friedrichs_flux_linearized(vector const & uM,
                                             vector const & uP,
                                             vector const & delta_uM,
                                             vector const & delta_uP,
                                             vector const & normalM) const
  {
    scalar uM_n = uM * normalM;
    scalar uP_n = uP * normalM;

    scalar delta_uM_n = delta_uM * normalM;
    scalar delta_uP_n = delta_uP * normalM;

    vector average_normal_flux =
      make_vectorized_array<Number>(0.5) *
      (uM * delta_uM_n + delta_uM * uM_n + uP * delta_uP_n + delta_uP * uP_n);

    vector jump_value = delta_uM - delta_uP;

    scalar lambda = calculate_lambda(uM_n, uP_n);

    return (average_normal_flux + 0.5 * lambda * jump_value);
  }

  /*
   *  Calculate upwind flux for nonlinear operator (convective formulation).
   */
  inline DEAL_II_ALWAYS_INLINE //
    vector
    calculate_upwind_flux(vector const & uM, vector const & uP, vector const & normalM) const
  {
    vector average_velocity = 0.5 * (uM + uP);

    scalar average_normal_velocity = average_velocity * normalM;

    vector jump_value = uM - uP;

    return (average_normal_velocity * average_velocity + this->operator_data.upwind_factor * 0.5 *
                                                           std::abs(average_normal_velocity) *
                                                           jump_value);
  }

  /*
   *  Calculate upwind flux for linearized operator (convective formulation).
   */
  inline DEAL_II_ALWAYS_INLINE //
    vector
    calculate_upwind_flux_linearized(vector const & uM,
                                     vector const & uP,
                                     vector const & delta_uM,
                                     vector const & delta_uP,
                                     vector const & normalM) const
  {
    vector average_velocity       = 0.5 * (uM + uP);
    vector delta_average_velocity = 0.5 * (delta_uM + delta_uP);

    scalar average_normal_velocity       = average_velocity * normalM;
    scalar delta_average_normal_velocity = delta_average_velocity * normalM;

    vector jump_value = delta_uM - delta_uP;

    return (average_normal_velocity * delta_average_velocity +
            delta_average_normal_velocity * average_velocity +
            this->operator_data.upwind_factor * 0.5 * std::abs(average_normal_velocity) *
              jump_value);
  }

  /*
   *  Evaluation of nonlinear convective operator.
   */
  void
  cell_loop_nonlinear_operator(MatrixFree<dim, Number> const & data,
                               VectorType &                    dst,
                               VectorType const &              src,
                               Range const &                   cell_range) const
  {
    FEEvalCellOverint fe_eval(data, operator_data.dof_index, operator_data.quad_index);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);
      fe_eval.read_dof_values(src);

      do_cell_integral_nonlinear_operator(fe_eval);

      fe_eval.distribute_local_to_global(dst);
    }
  }

  void
  face_loop_nonlinear_operator(MatrixFree<dim, Number> const & data,
                               VectorType &                    dst,
                               VectorType const &              src,
                               Range const &                   face_range) const
  {
    FEEvalFaceOverint fe_eval(data, true, operator_data.dof_index, operator_data.quad_index);
    FEEvalFaceOverint fe_eval_neighbor(data,
                                       false,
                                       operator_data.dof_index,
                                       operator_data.quad_index);

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      fe_eval.reinit(face);
      fe_eval_neighbor.reinit(face);

      fe_eval.gather_evaluate(src, true, false);
      fe_eval_neighbor.gather_evaluate(src, true, false);

      do_face_integral_nonlinear_operator(fe_eval, fe_eval_neighbor);

      fe_eval.integrate_scatter(true, false, dst);
      fe_eval_neighbor.integrate_scatter(true, false, dst);
    }
  }

  void
  boundary_face_loop_nonlinear_operator(MatrixFree<dim, Number> const & data,
                                        VectorType &                    dst,
                                        VectorType const &              src,
                                        Range const &                   face_range) const
  {
    FEEvalFaceOverint fe_eval(data, true, operator_data.dof_index, operator_data.quad_index);

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      fe_eval.reinit(face);

      fe_eval.gather_evaluate(src, true, false);

      do_boundary_integral_nonlinear_operator(fe_eval, data.get_boundary_id(face));

      fe_eval.integrate_scatter(true, false, dst);
    }
  }

  /*
   *  Evaluate linearized convective operator.
   */
  void
  cell_loop_linearized_operator(MatrixFree<dim, Number> const & data,
                                VectorType &                    dst,
                                VectorType const &              src,
                                Range const &                   cell_range) const
  {
    FEEvalCellOverint fe_eval(data, operator_data.dof_index, operator_data.quad_index);

    FEEvalCellOverint fe_eval_linearization(data,
                                            operator_data.dof_index,
                                            operator_data.quad_index);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);
      fe_eval.gather_evaluate(src, true, false, false);

      fe_eval_linearization.reinit(cell);
      fe_eval_linearization.gather_evaluate(velocity_linearization, true, false, false);

      do_cell_integral_linearized_operator(fe_eval, fe_eval_linearization);

      fe_eval.integrate_scatter(false, true, dst);
    }
  }

  void
  face_loop_linearized_operator(MatrixFree<dim, Number> const & data,
                                VectorType &                    dst,
                                VectorType const &              src,
                                Range const &                   face_range) const
  {
    FEEvalFaceOverint fe_eval(data, true, operator_data.dof_index, operator_data.quad_index);

    FEEvalFaceOverint fe_eval_neighbor(data,
                                       false,
                                       operator_data.dof_index,
                                       operator_data.quad_index);

    FEEvalFaceOverint fe_eval_linearization(data,
                                            true,
                                            operator_data.dof_index,
                                            operator_data.quad_index);

    FEEvalFaceOverint fe_eval_linearization_neighbor(data,
                                                     false,
                                                     operator_data.dof_index,
                                                     operator_data.quad_index);

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      fe_eval.reinit(face);
      fe_eval_neighbor.reinit(face);

      fe_eval_linearization.reinit(face);
      fe_eval_linearization_neighbor.reinit(face);

      fe_eval.gather_evaluate(src, true, false);
      fe_eval_neighbor.gather_evaluate(src, true, false);

      fe_eval_linearization.gather_evaluate(velocity_linearization, true, false);
      fe_eval_linearization_neighbor.gather_evaluate(velocity_linearization, true, false);

      do_face_integral_linearized_operator(fe_eval,
                                           fe_eval_neighbor,
                                           fe_eval_linearization,
                                           fe_eval_linearization_neighbor);

      fe_eval.integrate_scatter(true, false, dst);
      fe_eval_neighbor.integrate_scatter(true, false, dst);
    }
  }

  void
  boundary_face_loop_linearized_operator(MatrixFree<dim, Number> const & data,
                                         VectorType &                    dst,
                                         VectorType const &              src,
                                         Range const &                   face_range) const
  {
    FEEvalFaceOverint fe_eval(data, true, operator_data.dof_index, operator_data.quad_index);

    FEEvalFaceOverint fe_eval_linearization(data,
                                            true,
                                            operator_data.dof_index,
                                            operator_data.quad_index);

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      fe_eval.reinit(face);
      fe_eval_linearization.reinit(face);

      fe_eval.gather_evaluate(src, true, false);
      fe_eval_linearization.gather_evaluate(velocity_linearization, true, false);

      do_boundary_integral_linearized_operator(fe_eval,
                                               fe_eval_linearization,
                                               data.get_boundary_id(face));

      fe_eval.integrate_scatter(true, false, dst);
    }
  }


  /*
   *  Calculation of diagonal of linearized convective operator.
   */
  void
  cell_loop_diagonal(MatrixFree<dim, Number> const & data,
                     VectorType &                    dst,
                     VectorType const &,
                     Range const & cell_range) const
  {
    FEEvalCellOverint fe_eval(data, operator_data.dof_index, operator_data.quad_index);

    FEEvalCellOverint fe_eval_linearization(data,
                                            operator_data.dof_index,
                                            operator_data.quad_index);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      fe_eval_linearization.reinit(cell);
      fe_eval_linearization.gather_evaluate(velocity_linearization, true, false, false);

      fe_eval.reinit(cell);

      scalar       local_diagonal_vector[fe_eval.tensor_dofs_per_cell];
      unsigned int dofs_per_cell = fe_eval.dofs_per_cell;

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          fe_eval.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
        fe_eval.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

        fe_eval.evaluate(true, false, false);

        do_cell_integral_linearized_operator(fe_eval, fe_eval_linearization);

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
    FEEvalFaceOverint fe_eval(data, true, operator_data.dof_index, operator_data.quad_index);

    FEEvalFaceOverint fe_eval_neighbor(data,
                                       false,
                                       operator_data.dof_index,
                                       operator_data.quad_index);

    FEEvalFaceOverint fe_eval_linearization(data,
                                            true,
                                            operator_data.dof_index,
                                            operator_data.quad_index);

    FEEvalFaceOverint fe_eval_linearization_neighbor(data,
                                                     false,
                                                     operator_data.dof_index,
                                                     operator_data.quad_index);

    // Perform face integrals for element e⁻
    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      fe_eval_linearization.reinit(face);
      fe_eval_linearization.gather_evaluate(velocity_linearization, true, false);

      fe_eval_linearization_neighbor.reinit(face);
      fe_eval_linearization_neighbor.gather_evaluate(velocity_linearization, true, false);

      fe_eval.reinit(face);

      scalar       local_diagonal_vector[fe_eval.tensor_dofs_per_cell];
      unsigned int dofs_per_cell = fe_eval.dofs_per_cell;

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        // set dof value j of element- to 1 and all other dof values of element- to zero
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          fe_eval.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
        fe_eval.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

        fe_eval.evaluate(true, false);

        do_face_int_integral_linearized_operator(fe_eval,
                                                 fe_eval_neighbor,
                                                 fe_eval_linearization,
                                                 fe_eval_linearization_neighbor);

        fe_eval.integrate(true, false);

        local_diagonal_vector[j] = fe_eval.begin_dof_values()[j];
      }

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
        fe_eval.begin_dof_values()[j] = local_diagonal_vector[j];

      fe_eval.distribute_local_to_global(dst);
    }

    // Perform face integrals for element e⁺
    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      fe_eval_linearization.reinit(face);
      fe_eval_linearization.gather_evaluate(velocity_linearization, true, false);

      fe_eval_linearization_neighbor.reinit(face);
      fe_eval_linearization_neighbor.gather_evaluate(velocity_linearization, true, false);

      fe_eval_neighbor.reinit(face);

      scalar       local_diagonal_vector_neighbor[fe_eval_neighbor.tensor_dofs_per_cell];
      unsigned int dofs_per_cell = fe_eval_neighbor.dofs_per_cell;

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        // set dof value j of element+ to 1 and all other dof values of element+ to zero
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          fe_eval_neighbor.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
        fe_eval_neighbor.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

        fe_eval_neighbor.evaluate(true, false);

        do_face_ext_integral_linearized_operator(fe_eval,
                                                 fe_eval_neighbor,
                                                 fe_eval_linearization,
                                                 fe_eval_linearization_neighbor);

        fe_eval_neighbor.integrate(true, false);

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
    FEEvalFaceOverint fe_eval(data, true, operator_data.dof_index, operator_data.quad_index);

    FEEvalFaceOverint fe_eval_linearization(data,
                                            true,
                                            operator_data.dof_index,
                                            operator_data.quad_index);

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      fe_eval.reinit(face);

      fe_eval_linearization.reinit(face);
      fe_eval_linearization.gather_evaluate(velocity_linearization, true, false);

      scalar       local_diagonal_vector[fe_eval.tensor_dofs_per_cell];
      unsigned int dofs_per_cell = fe_eval.dofs_per_cell;
      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        // set dof value j of element- to 1 and all other dof values of element- to zero
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          fe_eval.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
        fe_eval.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

        fe_eval.evaluate(true, false);

        do_boundary_integral_linearized_operator(fe_eval,
                                                 fe_eval_linearization,
                                                 data.get_boundary_id(face));

        fe_eval.integrate(true, false);

        local_diagonal_vector[j] = fe_eval.begin_dof_values()[j];
      }

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
        fe_eval.begin_dof_values()[j] = local_diagonal_vector[j];

      fe_eval.distribute_local_to_global(dst);
    }
  }

  /*
   *  Apply block-diagonal operator (only needed for testing): re-implement face_loop, cell_loop and
   * boundary_face_loop are identical to linearized homogeneous operator.
   */
  void
  face_loop_apply_block_diagonal_linearized_operator(MatrixFree<dim, Number> const & data,
                                                     VectorType &                    dst,
                                                     VectorType const &              src,
                                                     Range const & face_range) const
  {
    FEEvalFaceOverint fe_eval(data, true, operator_data.dof_index, operator_data.quad_index);

    FEEvalFaceOverint fe_eval_neighbor(data,
                                       false,
                                       operator_data.dof_index,
                                       operator_data.quad_index);

    FEEvalFaceOverint fe_eval_linearization(data,
                                            true,
                                            operator_data.dof_index,
                                            operator_data.quad_index);

    FEEvalFaceOverint fe_eval_linearization_neighbor(data,
                                                     false,
                                                     operator_data.dof_index,
                                                     operator_data.quad_index);

    // Perform face integral for element e⁻
    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      fe_eval_linearization.reinit(face);
      fe_eval_linearization.gather_evaluate(velocity_linearization, true, false);

      fe_eval_linearization_neighbor.reinit(face);
      fe_eval_linearization_neighbor.gather_evaluate(velocity_linearization, true, false);

      fe_eval.reinit(face);
      fe_eval.gather_evaluate(src, true, false);

      do_face_int_integral_linearized_operator(fe_eval,
                                               fe_eval_neighbor,
                                               fe_eval_linearization,
                                               fe_eval_linearization_neighbor);

      fe_eval.integrate_scatter(true, false, dst);
    }

    // Perform face integral for element e⁺
    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      fe_eval_linearization.reinit(face);
      fe_eval_linearization.gather_evaluate(velocity_linearization, true, false);

      fe_eval_linearization_neighbor.reinit(face);
      fe_eval_linearization_neighbor.gather_evaluate(velocity_linearization, true, false);

      fe_eval_neighbor.reinit(face);
      fe_eval_neighbor.gather_evaluate(src, true, false);

      do_face_ext_integral_linearized_operator(fe_eval,
                                               fe_eval_neighbor,
                                               fe_eval_linearization,
                                               fe_eval_linearization_neighbor);

      fe_eval_neighbor.integrate_scatter(true, false, dst);
    }
  }

  /*
   * Calculate/assemble block-diagonal matrices via matrix-free operator evaluation.
   */
  void
  cell_loop_calculate_block_diagonal(MatrixFree<dim, Number> const &         data,
                                     std::vector<LAPACKFullMatrix<Number>> & matrices,
                                     VectorType const &,
                                     Range const & cell_range) const
  {
    FEEvalCellOverint fe_eval(data, operator_data.dof_index, operator_data.quad_index);

    FEEvalCellOverint fe_eval_linearization(data,
                                            operator_data.dof_index,
                                            operator_data.quad_index);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      fe_eval_linearization.reinit(cell);
      fe_eval_linearization.gather_evaluate(velocity_linearization, true, false, false);

      fe_eval.reinit(cell);

      unsigned int dofs_per_cell = fe_eval.dofs_per_cell;

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          fe_eval.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
        fe_eval.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

        fe_eval.evaluate(true, false, false);

        do_cell_integral_linearized_operator(fe_eval, fe_eval_linearization);

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
    FEEvalFaceOverint fe_eval(data, true, operator_data.dof_index, operator_data.quad_index);

    FEEvalFaceOverint fe_eval_neighbor(data,
                                       false,
                                       operator_data.dof_index,
                                       operator_data.quad_index);

    FEEvalFaceOverint fe_eval_linearization(data,
                                            true,
                                            operator_data.dof_index,
                                            operator_data.quad_index);

    FEEvalFaceOverint fe_eval_linearization_neighbor(data,
                                                     false,
                                                     operator_data.dof_index,
                                                     operator_data.quad_index);

    // Perform face integrals for element e⁻.
    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      fe_eval_linearization.reinit(face);
      fe_eval_linearization.gather_evaluate(velocity_linearization, true, false);

      fe_eval_linearization_neighbor.reinit(face);
      fe_eval_linearization_neighbor.gather_evaluate(velocity_linearization, true, false);

      fe_eval.reinit(face);

      unsigned int dofs_per_cell = fe_eval.dofs_per_cell;

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        // set dof value j of element- to 1 and all other dof values of element- to zero
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          fe_eval.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
        fe_eval.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

        fe_eval.evaluate(true, false);

        do_face_int_integral_linearized_operator(fe_eval,
                                                 fe_eval_neighbor,
                                                 fe_eval_linearization,
                                                 fe_eval_linearization_neighbor);

        fe_eval.integrate(true, false);

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
      fe_eval_linearization.reinit(face);
      fe_eval_linearization.gather_evaluate(velocity_linearization, true, false);

      fe_eval_linearization_neighbor.reinit(face);
      fe_eval_linearization_neighbor.gather_evaluate(velocity_linearization, true, false);

      fe_eval_neighbor.reinit(face);

      unsigned int dofs_per_cell = fe_eval_neighbor.dofs_per_cell;

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        // set dof value j of element+ to 1 and all other dof values of element+ to zero
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          fe_eval_neighbor.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
        fe_eval_neighbor.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

        fe_eval_neighbor.evaluate(true, false);

        do_face_ext_integral_linearized_operator(fe_eval,
                                                 fe_eval_neighbor,
                                                 fe_eval_linearization,
                                                 fe_eval_linearization_neighbor);

        fe_eval_neighbor.integrate(true, false);

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
    FEEvalFaceOverint fe_eval(data, true, operator_data.dof_index, operator_data.quad_index);

    FEEvalFaceOverint fe_eval_linearization(data,
                                            true,
                                            operator_data.dof_index,
                                            operator_data.quad_index);

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      fe_eval.reinit(face);

      fe_eval_linearization.reinit(face);
      fe_eval_linearization.gather_evaluate(velocity_linearization, true, false);

      unsigned int dofs_per_cell = fe_eval.dofs_per_cell;

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        // set dof value j of element- to 1 and all other dof values of element- to zero
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          fe_eval.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
        fe_eval.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

        fe_eval.evaluate(true, false);

        do_boundary_integral_linearized_operator(fe_eval,
                                                 fe_eval_linearization,
                                                 data.get_boundary_id(face));

        fe_eval.integrate(true, false);

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
    FEEvalCellOverint fe_eval(data, operator_data.dof_index, operator_data.quad_index);
    FEEvalFaceOverint fe_eval_m(data, true, operator_data.dof_index, operator_data.quad_index);
    FEEvalFaceOverint fe_eval_p(data, false, operator_data.dof_index, operator_data.quad_index);

    FEEvalCellOverint fe_eval_linearization(data,
                                            operator_data.dof_index,
                                            operator_data.quad_index);
    FEEvalFaceOverint fe_eval_linearization_m(data,
                                              true,
                                              operator_data.dof_index,
                                              operator_data.quad_index);
    FEEvalFaceOverint fe_eval_linearization_p(data,
                                              false,
                                              operator_data.dof_index,
                                              operator_data.quad_index);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      // cell integral
      unsigned int const n_filled_lanes = data.n_active_entries_per_cell_batch(cell);

      fe_eval_linearization.reinit(cell);
      fe_eval_linearization.gather_evaluate(velocity_linearization, true, false, false);

      fe_eval.reinit(cell);

      unsigned int dofs_per_cell = fe_eval.dofs_per_cell;

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          fe_eval.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
        fe_eval.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

        fe_eval.evaluate(true, false, false);

        do_cell_integral_linearized_operator(fe_eval, fe_eval_linearization);

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
        fe_eval_linearization_m.reinit(cell, face);
        fe_eval_linearization_m.gather_evaluate(velocity_linearization, true, false);

        // TODO
        //        AssertThrow(false, ExcMessage("We have to evaluate the linearized velocity field
        //        on the neighbor. This functionality is not implemented in deal.II/matrix_free."));
        //        fe_eval_linearization_p.reinit(cell, face);
        //        fe_eval_linearization_p.gather_evaluate(velocity_linearization, true, false);

        fe_eval_m.reinit(cell, face);
        fe_eval_p.reinit(cell, face);

        auto bids        = data.get_faces_by_cells_boundary_id(cell, face);
        auto boundary_id = bids[0];

        for(unsigned int j = 0; j < dofs_per_cell; ++j)
        {
          for(unsigned int i = 0; i < dofs_per_cell; ++i)
            fe_eval_m.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
          fe_eval_m.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

          fe_eval_m.evaluate(true, false);

          if(boundary_id == numbers::internal_face_boundary_id) // internal face
          {
            // TODO
            //            do_face_int_integral_linearized_operator(fe_eval_m,
            //            fe_eval_p,fe_eval_linearization_m, fe_eval_linearization_p);

            // plug in fe_eval_linearization_m twice to avoid the above problem with accessing dofs
            // of the neighboring element
            do_face_int_integral_linearized_operator(fe_eval_m,
                                                     fe_eval_p,
                                                     fe_eval_linearization_m,
                                                     fe_eval_linearization_m);
          }
          else // boundary face
          {
            do_boundary_integral_linearized_operator(fe_eval_m,
                                                     fe_eval_linearization_m,
                                                     boundary_id);
          }

          fe_eval_m.integrate(true, false);

          for(unsigned int i = 0; i < dofs_per_cell; ++i)
            for(unsigned int v = 0; v < n_filled_lanes; ++v)
              matrices[cell * VectorizedArray<Number>::n_array_elements + v](i, j) +=
                fe_eval_m.begin_dof_values()[i][v];
        }
      }
    }
  }

  MatrixFree<dim, Number> const * data;

  ConvectiveOperatorData<dim> operator_data;

  mutable Number eval_time;

  mutable VectorType velocity_linearization;

  unsigned int n_mpi_processes;

  std::shared_ptr<FEEvalCellOverint> fe_eval;
  std::shared_ptr<FEEvalFaceOverint> fe_eval_m;
  std::shared_ptr<FEEvalFaceOverint> fe_eval_p;

  std::shared_ptr<FEEvalCellOverint> fe_eval_linearization;
  std::shared_ptr<FEEvalFaceOverint> fe_eval_linearization_m;
  std::shared_ptr<FEEvalFaceOverint> fe_eval_linearization_p;
};


} // namespace IncNS


#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_NAVIER_STOKES_OPERATORS_H_ \
        */
