/*
 * ScalarConvectionDiffusionOperators.h
 *
 *  Created on: Jul 29, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_SCALARCONVECTIONDIFFUSIONOPERATORS_H_
#define INCLUDE_SCALARCONVECTIONDIFFUSIONOPERATORS_H_

#include <deal.II/lac/parallel_vector.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>

#include "../include/BoundaryDescriptorConvDiff.h"
#include "InputParametersConvDiff.h"

#include "EvaluateFunctions.h"

#include "VerifyCalculationOfDiagonal.h"
#include "InvertDiagonal.h"

namespace ScalarConvDiffOperators
{

using namespace ConvDiff;

struct MassMatrixOperatorData
{
  MassMatrixOperatorData ()
    :
    dof_index(0),
    quad_index(0)
  {}

  unsigned int dof_index;
  unsigned int quad_index;
};

template <int dim, int fe_degree, typename value_type>
class MassMatrixOperator
{
public:
  typedef MassMatrixOperator<dim,fe_degree,value_type> This;

  MassMatrixOperator()
    :
    data(nullptr)
  {}

  void initialize(MatrixFree<dim,value_type> const &mf_data,
                  MassMatrixOperatorData const     &mass_matrix_operator_data_in)
  {
    this->data = &mf_data;
    this->mass_matrix_operator_data = mass_matrix_operator_data_in;
  }

  // apply matrix vector multiplication
  void apply (parallel::distributed::Vector<value_type>       &dst,
              const parallel::distributed::Vector<value_type> &src) const
  {
    dst = 0;
    apply_add(dst,src);
  }

  void apply_add (parallel::distributed::Vector<value_type>       &dst,
                  const parallel::distributed::Vector<value_type> &src) const
  {
    data->cell_loop(&This::cell_loop, this, dst, src);
  }

  void calculate_diagonal(parallel::distributed::Vector<value_type> &diagonal) const
  {
    diagonal = 0;

    add_diagonal(diagonal);
  }

  void add_diagonal(parallel::distributed::Vector<value_type> &diagonal) const
  {
    parallel::distributed::Vector<value_type>  src_dummy(diagonal);

    data->cell_loop(&This::cell_loop_diagonal, this, diagonal, src_dummy);
  }

  void add_block_jacobi_matrices(std::vector<LAPACKFullMatrix<value_type> > &matrices) const
  {
    parallel::distributed::Vector<value_type>  src;

    data->cell_loop(&This::cell_loop_calculate_block_jacobi_matrices, this, matrices, src);
  }

  MassMatrixOperatorData const & get_operator_data() const
  {
    return mass_matrix_operator_data;
  }


private:
  template<typename FEEvaluation>
  inline void do_cell_integral(FEEvaluation &fe_eval) const
  {
    fe_eval.evaluate (true,false,false);

    for (unsigned int q=0; q<fe_eval.n_q_points; ++q)
    {
      fe_eval.submit_value (fe_eval.get_value(q), q);
    }
    fe_eval.integrate (true,false);
  }

  void cell_loop (const MatrixFree<dim,value_type>                &data,
                  parallel::distributed::Vector<value_type>       &dst,
                  const parallel::distributed::Vector<value_type> &src,
                  const std::pair<unsigned int,unsigned int>      &cell_range) const
  {
    FEEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval(data,
                                                                 mass_matrix_operator_data.dof_index,
                                                                 mass_matrix_operator_data.quad_index);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);
      fe_eval.read_dof_values(src);

      do_cell_integral(fe_eval);

      fe_eval.distribute_local_to_global (dst);
    }
  }

  void cell_loop_diagonal (const MatrixFree<dim,value_type>                 &data,
                           parallel::distributed::Vector<value_type>        &dst,
                           const parallel::distributed::Vector<value_type>  &,
                           const std::pair<unsigned int,unsigned int>       &cell_range) const
  {
    FEEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval(data,
                                                                 mass_matrix_operator_data.dof_index,
                                                                 mass_matrix_operator_data.quad_index);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);

      VectorizedArray<value_type> local_diagonal_vector[fe_eval.tensor_dofs_per_cell];
      for (unsigned int j=0; j<fe_eval.dofs_per_cell; ++j)
      {
        for (unsigned int i=0; i<fe_eval.dofs_per_cell; ++i)
          fe_eval.begin_dof_values()[i] = make_vectorized_array<value_type>(0.);
        fe_eval.begin_dof_values()[j] = make_vectorized_array<value_type>(1.);

        do_cell_integral(fe_eval);

        local_diagonal_vector[j] = fe_eval.begin_dof_values()[j];
      }
      for (unsigned int j=0; j<fe_eval.dofs_per_cell; ++j)
        fe_eval.begin_dof_values()[j] = local_diagonal_vector[j];

      fe_eval.distribute_local_to_global (dst);
    }
  }

  void cell_loop_calculate_block_jacobi_matrices (const MatrixFree<dim,value_type>                 &data,
                                                  std::vector<LAPACKFullMatrix<value_type> >       &matrices,
                                                  const parallel::distributed::Vector<value_type>  &,
                                                  const std::pair<unsigned int,unsigned int>       &cell_range) const
  {
    FEEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval(data,
                                                                 mass_matrix_operator_data.dof_index,
                                                                 mass_matrix_operator_data.quad_index);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);

      for (unsigned int j=0; j<fe_eval.dofs_per_cell; ++j)
      {
        for (unsigned int i=0; i<fe_eval.dofs_per_cell; ++i)
          fe_eval.begin_dof_values()[i] = make_vectorized_array<value_type>(0.);
        fe_eval.begin_dof_values()[j] = make_vectorized_array<value_type>(1.);

        do_cell_integral(fe_eval);

        for(unsigned int i=0; i<fe_eval.dofs_per_cell; ++i)
          for (unsigned int v=0; v<VectorizedArray<value_type>::n_array_elements; ++v)
            matrices[cell*VectorizedArray<value_type>::n_array_elements+v](i,j) += fe_eval.begin_dof_values()[i][v];
      }
    }
  }

  MatrixFree<dim,value_type> const * data;
  MassMatrixOperatorData mass_matrix_operator_data;
};

template<int dim>
struct RHSOperatorData
{
  RHSOperatorData ()
    :
    dof_index(0),
    quad_index(0)
  {}

  unsigned int dof_index;
  unsigned int quad_index;
  std_cxx11::shared_ptr<Function<dim> > rhs;
};

template <int dim, int fe_degree, typename value_type>
class RHSOperator
{
public:
  typedef RHSOperator<dim,fe_degree, value_type> This;

  RHSOperator()
    :
    data(nullptr),
    eval_time(0.0)
  {}

  void initialize(MatrixFree<dim,value_type> const &mf_data,
                  RHSOperatorData<dim> const       &operator_data_in)
  {
    this->data = &mf_data;
    this->operator_data = operator_data_in;
  }

  // apply matrix vector multiplication
  void evaluate (parallel::distributed::Vector<value_type> &dst,
                 const value_type                          evaluation_time) const
  {
    dst = 0;
    evaluate_add(dst,evaluation_time);
  }

  void evaluate_add (parallel::distributed::Vector<value_type> &dst,
                     const value_type                          evaluation_time) const
  {
    this->eval_time = evaluation_time;

    parallel::distributed::Vector<value_type> src;
    data->cell_loop(&This::local_evaluate, this, dst, src);
  }

private:

  void local_evaluate (const MatrixFree<dim,value_type>                &data,
                       parallel::distributed::Vector<value_type>       &dst,
                       const parallel::distributed::Vector<value_type> &,
                       const std::pair<unsigned int,unsigned int>      &cell_range) const
  {
    FEEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval(data,
                                                                 operator_data.dof_index,
                                                                 operator_data.quad_index);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);

      for (unsigned int q=0; q<fe_eval.n_q_points; ++q)
      {
        Point<dim,VectorizedArray<value_type> > q_points = fe_eval.quadrature_point(q);
        VectorizedArray<value_type> rhs = make_vectorized_array<value_type>(0.0);
        evaluate_scalar_function(rhs,operator_data.rhs,q_points,eval_time);

        fe_eval.submit_value (rhs, q);
      }
      fe_eval.integrate (true,false);
      fe_eval.distribute_local_to_global (dst);
    }
  }

  MatrixFree<dim,value_type> const * data;
  RHSOperatorData<dim> operator_data;
  value_type mutable eval_time;
};


template<int dim>
struct DiffusiveOperatorData
{
  DiffusiveOperatorData ()
    :
    dof_index(0),
    quad_index(0),
    IP_factor(1.0),
    diffusivity(1.0)
  {}

  unsigned int dof_index;
  unsigned int quad_index;

  double IP_factor;

  std_cxx11::shared_ptr<BoundaryDescriptorConvDiff<dim> > bc;

  double diffusivity;
};

template <int dim, int fe_degree, typename value_type>
class DiffusiveOperator
{
public:
  typedef DiffusiveOperator<dim,fe_degree, value_type> This;

  enum class OperatorType {
    full,
    homogeneous,
    inhomogeneous
  };

  enum class BoundaryType {
    undefined,
    dirichlet,
    neumann
  };

  DiffusiveOperator()
    :
    data(nullptr),
    diffusivity(-1.0)
  {}

  void initialize(Mapping<dim> const               &mapping,
                  MatrixFree<dim,value_type> const &mf_data,
                  DiffusiveOperatorData<dim> const &operator_data_in)
  {
    this->data = &mf_data;
    this->operator_data = operator_data_in;

    compute_array_penalty_parameter(mapping);

    diffusivity = operator_data.diffusivity;
  }

  // apply matrix vector multiplication
  void apply (parallel::distributed::Vector<value_type>       &dst,
              const parallel::distributed::Vector<value_type> &src) const
  {
    dst = 0;
    apply_add(dst,src);
  }

  void apply_add (parallel::distributed::Vector<value_type>       &dst,
                  const parallel::distributed::Vector<value_type> &src) const
  {
    AssertThrow(diffusivity > 0.0,ExcMessage("Diffusivity has not been set!"));

    data->loop(&This::cell_loop,&This::face_loop,
               &This::boundary_face_loop_hom_operator,this, dst, src);
  }

  // apply "block Jacobi" matrix vector multiplication
  void apply_block_Jacobi (parallel::distributed::Vector<value_type>       &dst,
                           const parallel::distributed::Vector<value_type> &src) const
  {
    dst = 0;
    apply_block_jacobi_add(dst,src);
  }

  void apply_block_jacobi_add (parallel::distributed::Vector<value_type>       &dst,
                               const parallel::distributed::Vector<value_type> &src) const
  {
    AssertThrow(diffusivity > 0.0,ExcMessage("Diffusivity has not been set!"));

    data->loop(&This::cell_loop,&This::face_loop_block_jacobi,
               &This::boundary_face_loop_hom_operator,this, dst, src);
  }

  void add_block_jacobi_matrices(std::vector<LAPACKFullMatrix<value_type> > &matrices) const
  {
    parallel::distributed::Vector<value_type>  src;

    data->loop(&This::cell_loop_calculate_block_jacobi_matrices,&This::face_loop_calculate_block_jacobi_matrices,
                    &This::boundary_face_loop_calculate_block_jacobi_matrices, this, matrices, src);
  }

  void rhs (parallel::distributed::Vector<value_type> &dst,
            value_type const                          evaluation_time) const
  {
    dst = 0;
    rhs_add(dst,evaluation_time);
  }

  void rhs_add (parallel::distributed::Vector<value_type> &dst,
                value_type const                          evaluation_time) const
  {
    this->eval_time = evaluation_time;

    parallel::distributed::Vector<value_type> src;

    data->loop(&This::cell_loop_inhom_operator,&This::face_loop_inhom_operator,
               &This::boundary_face_loop_inhom_operator,this, dst, src);
  }

  void evaluate (parallel::distributed::Vector<value_type>       &dst,
                 const parallel::distributed::Vector<value_type> &src,
                 value_type const                                evaluation_time) const
  {
    dst = 0;
    evaluate_add(dst,src,evaluation_time);
  }

  void evaluate_add (parallel::distributed::Vector<value_type>       &dst,
                     const parallel::distributed::Vector<value_type> &src,
                     value_type const                                evaluation_time) const
  {
    this->eval_time = evaluation_time;

    data->loop(&This::cell_loop,&This::face_loop,
               &This::boundary_face_loop_full_operator, this, dst, src);
  }

  void calculate_diagonal(parallel::distributed::Vector<value_type> &diagonal) const
  {
    diagonal = 0;

    add_diagonal(diagonal);
  }

  void add_diagonal(parallel::distributed::Vector<value_type> &diagonal) const
  {
    parallel::distributed::Vector<value_type>  src_dummy(diagonal);

    data->loop(&This::cell_loop_diagonal,&This::face_loop_diagonal,
               &This::boundary_face_loop_diagonal,
               this, diagonal, src_dummy);
  }

  DiffusiveOperatorData<dim> const & get_operator_data() const
  {
    return operator_data;
  }

private:
  void compute_array_penalty_parameter(const Mapping<dim> &mapping)
  {
    // Compute penalty parameter for each cell
    array_penalty_parameter.resize(data->n_macro_cells()+data->n_macro_ghost_cells());
    QGauss<dim> quadrature(fe_degree+1);
    FEValues<dim> fe_values(mapping,data->get_dof_handler(operator_data.dof_index).get_fe(),quadrature, update_JxW_values);
    QGauss<dim-1> face_quadrature(fe_degree+1);
    FEFaceValues<dim> fe_face_values(mapping, data->get_dof_handler(operator_data.dof_index).get_fe(), face_quadrature, update_JxW_values);

    for (unsigned int i=0; i<data->n_macro_cells()+data->n_macro_ghost_cells(); ++i)
    {
      for (unsigned int v=0; v<data->n_components_filled(i); ++v)
      {
        typename DoFHandler<dim>::cell_iterator cell = data->get_cell_iterator(i,v,operator_data.dof_index);
        fe_values.reinit(cell);
        double volume = 0;
        for (unsigned int q=0; q<quadrature.size(); ++q)
          volume += fe_values.JxW(q);
        double surface_area = 0;
        for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
        {
          fe_face_values.reinit(cell, f);
          const double factor = (cell->at_boundary(f) && !cell->has_periodic_neighbor(f)) ? 1. : 0.5;
          for (unsigned int q=0; q<face_quadrature.size(); ++q)
            surface_area += fe_face_values.JxW(q) * factor;
        }
        array_penalty_parameter[i][v] = surface_area / volume;
      }
    }
  }

  double get_penalty_factor() const
  {
    return operator_data.IP_factor * (fe_degree + 1.0) * (fe_degree + 1.0);
  }


  template<typename FEEvaluation>
  inline void do_cell_integral(FEEvaluation &fe_eval) const
  {
    fe_eval.evaluate (false,true,false);

    for (unsigned int q=0; q<fe_eval.n_q_points; ++q)
    {
      fe_eval.submit_gradient (make_vectorized_array<value_type>(diffusivity)*fe_eval.get_gradient(q), q);
    }
    fe_eval.integrate (false,true);
  }

  /*
   *  Calculation of "value_flux".
   */
  inline void calculate_value_flux (VectorizedArray<value_type>       &value_flux,
                                    VectorizedArray<value_type> const &jump_value) const
  {
    value_flux = -0.5 * diffusivity * jump_value;
  }

  /*
   *  This function calculates the jump value = interior_value - exterior_value
   *  depending on the operator type, the type of the boundary face
   *  and the given boundary conditions. The jump_value has to be calculated in order
   *  to evaluate both the value_flux and the gradient_flux.
   *
   *                            +----------------------+--------------------+
   *                            | Dirichlet boundaries | Neumann boundaries |
   *  +-------------------------+----------------------+--------------------+
   *  | full operator           | phi⁺ = -phi⁻ + 2g    | phi⁺ = phi⁻        |
   *  +-------------------------+----------------------+--------------------+
   *  | homogeneous operator    | phi⁺ = -phi⁻         | phi⁺ = phi⁻        |
   *  +-------------------------+----------------------+--------------------+
   *  | inhomogeneous operator  | phi⁻ = 0, phi⁺ = 2g  | phi⁻ = 0, phi⁺ = 0 |
   *  +-------------------------+----------------------+--------------------+
   */
  template<typename FEEvaluation>
  inline void calculate_jump_value_boundary_face(VectorizedArray<value_type>       &jump_value,
                                                 unsigned int const                q,
                                                 FEEvaluation const                &fe_eval,
                                                 OperatorType const                &operator_type,
                                                 BoundaryType const                &boundary_type,
                                                 types::boundary_id const          boundary_id = types::boundary_id()) const
  {
    // element e⁻
    VectorizedArray<value_type> value_m = make_vectorized_array<value_type>(0.0);

    if(operator_type == OperatorType::full ||
       operator_type == OperatorType::homogeneous)
    {
      value_m = fe_eval.get_value(q);
    }
    else if(operator_type == OperatorType::inhomogeneous)
    {
      // do nothing, value_m, normal_gradient_m are already initialized with zeros
    }
    else
    {
      AssertThrow(false, ExcMessage("Specified OperatorType is not implemented!"));
    }


    // element e⁺
    VectorizedArray<value_type> value_p = make_vectorized_array<value_type>(0.0);

    if(operator_type == OperatorType::full)
    {
      if(boundary_type == BoundaryType::dirichlet)
      {
        VectorizedArray<value_type> g = make_vectorized_array<value_type>(0.0);
        typename std::map<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >::iterator it;
        it = operator_data.bc->dirichlet_bc.find(boundary_id);
        Point<dim,VectorizedArray<value_type> > q_points = fe_eval.quadrature_point(q);
        evaluate_scalar_function(g, it->second, q_points, eval_time);

        value_p = - value_m + 2.0*g;
      }
      else if(boundary_type == BoundaryType::neumann)
      {
        value_p = value_m;
      }
      else
      {
        AssertThrow(false,ExcMessage("Boundary type of face is invalid or not implemented."));
      }
    }
    else if(operator_type == OperatorType::homogeneous)
    {
      if(boundary_type == BoundaryType::dirichlet)
      {
        value_p = - value_m;
      }
      else if(boundary_type == BoundaryType::neumann)
      {
        value_p = value_m;
      }
      else
      {
        AssertThrow(false,ExcMessage("Boundary type of face is invalid or not implemented."));
      }
    }
    else if(operator_type == OperatorType::inhomogeneous)
    {
      if(boundary_type == BoundaryType::dirichlet)
      {
        VectorizedArray<value_type> g = make_vectorized_array<value_type>(0.0);
        typename std::map<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >::iterator it;
        it = operator_data.bc->dirichlet_bc.find(boundary_id);
        Point<dim,VectorizedArray<value_type> > q_points = fe_eval.quadrature_point(q);
        evaluate_scalar_function(g,it->second,q_points,eval_time);

        value_p = 2.0*g;
      }
      else if(boundary_type == BoundaryType::neumann)
      {
        // do nothing, value_p is already initialized with zeros
      }
      else
      {
        AssertThrow(false,ExcMessage("Boundary type of face is invalid or not implemented."));
      }
    }
    else
    {
      AssertThrow(false, ExcMessage("Specified OperatorType is not implemented!"));
    }

    jump_value = value_m - value_p;
  }


  /*
   *  Calculation of gradient flux. Strictly speaking, this value is not a numerical flux since
   *  the flux is multiplied by the normal vector, i.e., "gradient_flux" = numerical_flux * normal,
   *  where normal denotes the normal vector of element e⁻.
   */
  inline void calculate_gradient_flux (VectorizedArray<value_type>       &gradient_flux,
                                       VectorizedArray<value_type> const &average_normal_gradient,
                                       VectorizedArray<value_type> const &jump_value,
                                       VectorizedArray<value_type> const &penalty_parameter) const
  {
    gradient_flux = diffusivity * average_normal_gradient - diffusivity * penalty_parameter * jump_value;
  }

  /*
   *  This function calculates the average velocity gradient in normal
   *  direction depending on the operator type, the type of the boundary face
   *  and the given boundary conditions. The average normal gradient has to
   *  be calculated in order to evaluate the gradient flux.
   *
   *                            +-------------------------------------+---------------------------------------+
   *                            | Dirichlet boundaries                | Neumann boundaries                    |
   *  +-------------------------+-------------------------------------+---------------------------------------+
   *  | full operator           | grad(phi⁺)*n = grad(phi⁻)*n         | grad(phi⁺)*n = -grad(phi⁻)*n + 2h     |
   *  +-------------------------+-------------------------------------+---------------------------------------+
   *  | homogeneous operator    | grad(phi⁺)*n = grad(phi⁻)*n         | grad(phi⁺)*n = -grad(phi⁻)*n          |
   *  +-------------------------+-------------------------------------+---------------------------------------+
   *  | inhomogeneous operator  | grad(phi⁺)*n  = 0, grad(phi⁻)*n = 0 | grad(phi⁺)*n  = 0, grad(phi⁻)*n  = 2h |
   *  +-------------------------+-------------------------------------+---------------------------------------+
   *
   *                            +-------------------------------------+---------------------------------------+
   *                            | Dirichlet boundaries                | Neumann boundaries                    |
   *  +-------------------------+-------------------------------------+---------------------------------------+
   *  | full operator           | {{grad(phi)}}*n = grad(phi⁻)*n      | {{grad(phi)}}*n = h                   |
   *  +-------------------------+-------------------------------------+---------------------------------------+
   *  | homogeneous operator    | {{grad(phi)}}*n = grad(phi⁻)*n      | {{grad(phi)}}*n = 0                   |
   *  +-------------------------+-------------------------------------+---------------------------------------+
   *  | inhomogeneous operator  | {{grad(phi)}}*n = 0                 | {{grad(phi)}}*n = h                   |
   *  +-------------------------+-------------------------------------+---------------------------------------+
   */
  template<typename FEEvaluation>
  inline void calculate_average_normal_gradient_boundary_face(
      VectorizedArray<value_type> &average_normal_gradient,
      unsigned int const          q,
      FEEvaluation const          &fe_eval,
      OperatorType const          &operator_type,
      BoundaryType const          &boundary_type,
      types::boundary_id const    boundary_id = types::boundary_id()) const
  {
    if(operator_type == OperatorType::full)
    {
      if(boundary_type == BoundaryType::dirichlet)
      {
        average_normal_gradient = fe_eval.get_normal_gradient(q);
      }
      else if(boundary_type == BoundaryType::neumann)
      {
        VectorizedArray<value_type> h = make_vectorized_array<value_type>(0.0);
        typename std::map<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >::iterator it;
        it = operator_data.bc->neumann_bc.find(boundary_id);
        Point<dim,VectorizedArray<value_type> > q_points = fe_eval.quadrature_point(q);
        evaluate_scalar_function(h, it->second, q_points, eval_time);

        average_normal_gradient = h;
      }
      else
      {
        AssertThrow(false,ExcMessage("Boundary type of face is invalid or not implemented."));
      }
    }
    else if(operator_type == OperatorType::homogeneous)
    {
      if(boundary_type == BoundaryType::dirichlet)
      {
        average_normal_gradient = fe_eval.get_normal_gradient(q);
      }
      else if(boundary_type == BoundaryType::neumann)
      {
        average_normal_gradient = make_vectorized_array<value_type>(0.0);
      }
      else
      {
        AssertThrow(false,ExcMessage("Boundary type of face is invalid or not implemented."));
      }
    }
    else if(operator_type == OperatorType::inhomogeneous)
    {
      if(boundary_type == BoundaryType::dirichlet)
      {
        average_normal_gradient = make_vectorized_array<value_type>(0.0);
      }
      else if(boundary_type == BoundaryType::neumann)
      {
        VectorizedArray<value_type> h = make_vectorized_array<value_type>(0.0);
        typename std::map<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >::iterator it;
        it = operator_data.bc->neumann_bc.find(boundary_id);
        Point<dim,VectorizedArray<value_type> > q_points = fe_eval.quadrature_point(q);
        evaluate_scalar_function(h, it->second, q_points, eval_time);

        average_normal_gradient = h;
      }
      else
      {
        AssertThrow(false,ExcMessage("Boundary type of face is invalid or not implemented."));
      }
    }
    else
    {
      AssertThrow(false, ExcMessage("Specified OperatorType is not implemented!"));
    }
  }


  /*
   *  Evaluate homogeneous operator.
   */
  void cell_loop (const MatrixFree<dim,value_type>                 &data,
                  parallel::distributed::Vector<value_type>        &dst,
                  const parallel::distributed::Vector<value_type>  &src,
                  const std::pair<unsigned int,unsigned int>       &cell_range) const
  {
    FEEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval(data,
                                                                 operator_data.dof_index,
                                                                 operator_data.quad_index);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);
      fe_eval.read_dof_values(src);

      do_cell_integral(fe_eval);

      fe_eval.distribute_local_to_global (dst);
    }
  }

  void face_loop (const MatrixFree<dim,value_type>                &data,
                  parallel::distributed::Vector<value_type>       &dst,
                  const parallel::distributed::Vector<value_type> &src,
                  const std::pair<unsigned int,unsigned int>      &face_range) const
  {
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval(data,true,operator_data.dof_index,operator_data.quad_index);
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval_neighbor(data,false,operator_data.dof_index,operator_data.quad_index);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval.reinit (face);
      fe_eval_neighbor.reinit (face);

      fe_eval.read_dof_values(src);
      fe_eval_neighbor.read_dof_values(src);

      fe_eval.evaluate(true,true);
      fe_eval_neighbor.evaluate(true,true);

      VectorizedArray<value_type> tau_IP = std::max(fe_eval.read_cell_data(array_penalty_parameter),
                                                    fe_eval_neighbor.read_cell_data(array_penalty_parameter))
                                              * get_penalty_factor();

      for(unsigned int q=0;q<fe_eval.n_q_points;++q)
      {
        VectorizedArray<value_type> jump_value = fe_eval.get_value(q) - fe_eval_neighbor.get_value(q);
        VectorizedArray<value_type> value_flux;
        calculate_value_flux(value_flux, jump_value);

        VectorizedArray<value_type> average_normal_gradient =
            0.5 * (fe_eval.get_normal_gradient(q) + fe_eval_neighbor.get_normal_gradient(q));

        VectorizedArray<value_type> gradient_flux;
        calculate_gradient_flux(gradient_flux, average_normal_gradient, jump_value, tau_IP);

        fe_eval.submit_normal_gradient(value_flux,q);
        fe_eval_neighbor.submit_normal_gradient(value_flux,q);

        fe_eval.submit_value(-gradient_flux,q);
        fe_eval_neighbor.submit_value(gradient_flux,q); // + sign since n⁺ = -n⁻
      }
      fe_eval.integrate(true,true);
      fe_eval_neighbor.integrate(true,true);

      fe_eval.distribute_local_to_global(dst);
      fe_eval_neighbor.distribute_local_to_global(dst);
    }
  }

  void boundary_face_loop_hom_operator (const MatrixFree<dim,value_type>                &data,
                                        parallel::distributed::Vector<value_type>       &dst,
                                        const parallel::distributed::Vector<value_type> &src,
                                        const std::pair<unsigned int,unsigned int>      &face_range) const
  {
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval(data,true,operator_data.dof_index,operator_data.quad_index);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      types::boundary_id boundary_id = data.get_boundary_indicator(face);
      BoundaryType boundary_type = BoundaryType::undefined;

      if(operator_data.bc->dirichlet_bc.find(boundary_id) != operator_data.bc->dirichlet_bc.end())
        boundary_type = BoundaryType::dirichlet;
      else if(operator_data.bc->neumann_bc.find(boundary_id) != operator_data.bc->neumann_bc.end())
        boundary_type = BoundaryType::neumann;

      AssertThrow(boundary_type != BoundaryType::undefined,
          ExcMessage("Boundary type of face is invalid or not implemented."));

      fe_eval.reinit (face);
      fe_eval.read_dof_values(src);
      fe_eval.evaluate(true,true);

      VectorizedArray<value_type> tau_IP = fe_eval.read_cell_data(array_penalty_parameter)
                                              * get_penalty_factor();

      for(unsigned int q=0;q<fe_eval.n_q_points;++q)
      {
        VectorizedArray<value_type> jump_value = make_vectorized_array<value_type>(0.0);
        calculate_jump_value_boundary_face(jump_value,q,fe_eval,OperatorType::homogeneous,boundary_type);
        VectorizedArray<value_type> value_flux = make_vectorized_array<value_type>(0.0);
        calculate_value_flux(value_flux, jump_value);

        VectorizedArray<value_type> average_normal_gradient = make_vectorized_array<value_type>(0.0);
        calculate_average_normal_gradient_boundary_face(average_normal_gradient,q,fe_eval,OperatorType::homogeneous,boundary_type);

        VectorizedArray<value_type> gradient_flux = make_vectorized_array<value_type>(0.0);
        calculate_gradient_flux(gradient_flux, average_normal_gradient, jump_value, tau_IP);

        fe_eval.submit_normal_gradient(value_flux,q);
        fe_eval.submit_value(-gradient_flux,q);
      }
      fe_eval.integrate(true,true);
      fe_eval.distribute_local_to_global(dst);
    }
  }

  /*
   *  face integrals for block-jacobi, use homogeneous operator for cell and boundary face integrals
   */
  void face_loop_block_jacobi (const MatrixFree<dim,value_type>                &data,
                               parallel::distributed::Vector<value_type>       &dst,
                               const parallel::distributed::Vector<value_type> &src,
                               const std::pair<unsigned int,unsigned int>      &face_range) const
  {
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval(data,true,operator_data.dof_index,operator_data.quad_index);
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval_neighbor(data,false,operator_data.dof_index,operator_data.quad_index);

    // perform face integral for element e⁻
    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval.reinit (face);
      fe_eval.read_dof_values(src);
      fe_eval.evaluate(true,true);

      fe_eval_neighbor.reinit (face);

      VectorizedArray<value_type> tau_IP = std::max(fe_eval.read_cell_data(array_penalty_parameter),
                                                    fe_eval_neighbor.read_cell_data(array_penalty_parameter))
                                              * get_penalty_factor();

      for(unsigned int q=0;q<fe_eval_neighbor.n_q_points;++q)
      {
        // set exterior value to zero
        VectorizedArray<value_type> jump_value = fe_eval.get_value(q);
        VectorizedArray<value_type> value_flux = make_vectorized_array<value_type>(0.0);
        calculate_value_flux(value_flux, jump_value);

        // set exterior value to zero
        VectorizedArray<value_type> average_normal_gradient = 0.5 * fe_eval.get_normal_gradient(q);

        VectorizedArray<value_type> gradient_flux = make_vectorized_array<value_type>(0.0);
        calculate_gradient_flux(gradient_flux, average_normal_gradient, jump_value, tau_IP);

        fe_eval.submit_normal_gradient(value_flux,q);
        fe_eval.submit_value(-gradient_flux,q);
      }
      fe_eval.integrate(true,true);
      fe_eval.distribute_local_to_global(dst);
    }

    // TODO: This has to be removed as soon as the new infrastructure is used that
    // allows to perform face integrals over all faces of the current element.
    // perform face integral for element e⁺
    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval.reinit (face);

      fe_eval_neighbor.reinit (face);
      fe_eval_neighbor.read_dof_values(src);
      fe_eval_neighbor.evaluate(true,true);

      VectorizedArray<value_type> tau_IP = std::max(fe_eval.read_cell_data(array_penalty_parameter),
                                                    fe_eval_neighbor.read_cell_data(array_penalty_parameter))
                                              * get_penalty_factor();

      for(unsigned int q=0;q<fe_eval_neighbor.n_q_points;++q)
      {
        // set value_m to zero
        VectorizedArray<value_type> jump_value = fe_eval_neighbor.get_value(q);
        VectorizedArray<value_type> value_flux = make_vectorized_array<value_type>(0.0);
        calculate_value_flux(value_flux, jump_value);
        // set gradient_m to zero, minus sign to get the correct normal vector n⁺ = -n⁻
        VectorizedArray<value_type> average_normal_gradient = - 0.5 * fe_eval_neighbor.get_normal_gradient(q);

        VectorizedArray<value_type> gradient_flux = make_vectorized_array<value_type>(0.0);
        calculate_gradient_flux(gradient_flux, average_normal_gradient, jump_value, tau_IP);
        // minus sign since n⁺ = -n⁻
        fe_eval_neighbor.submit_normal_gradient(-value_flux,q);
        fe_eval_neighbor.submit_value(-gradient_flux,q);
      }
      fe_eval_neighbor.integrate(true,true);
      fe_eval_neighbor.distribute_local_to_global(dst);
    }
  }



  /*
   *  calculation of diagonal
   */
  void cell_loop_diagonal (const MatrixFree<dim,value_type>                 &data,
                           parallel::distributed::Vector<value_type>        &dst,
                           const parallel::distributed::Vector<value_type>  &,
                           const std::pair<unsigned int,unsigned int>       &cell_range) const
  {
    FEEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval(data,
                                                                 operator_data.dof_index,
                                                                 operator_data.quad_index);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);

      VectorizedArray<value_type> local_diagonal_vector[fe_eval.tensor_dofs_per_cell];
      for (unsigned int j=0; j<fe_eval.dofs_per_cell; ++j)
      {
        for (unsigned int i=0; i<fe_eval.dofs_per_cell; ++i)
          fe_eval.begin_dof_values()[i] = make_vectorized_array<value_type>(0.);
        fe_eval.begin_dof_values()[j] = make_vectorized_array<value_type>(1.);

        do_cell_integral(fe_eval);

        local_diagonal_vector[j] = fe_eval.begin_dof_values()[j];
      }
      for (unsigned int j=0; j<fe_eval.dofs_per_cell; ++j)
        fe_eval.begin_dof_values()[j] = local_diagonal_vector[j];

      fe_eval.distribute_local_to_global (dst);
    }
  }

  void face_loop_diagonal (const MatrixFree<dim,value_type>                &data,
                           parallel::distributed::Vector<value_type>       &dst,
                           const parallel::distributed::Vector<value_type> &,
                           const std::pair<unsigned int,unsigned int>      &face_range) const
  {
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval(data,true,operator_data.dof_index,operator_data.quad_index);
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval_neighbor(data,false,operator_data.dof_index,operator_data.quad_index);


    // Perform face intergrals for element e⁻.
    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval.reinit (face);
      fe_eval_neighbor.reinit (face);

      VectorizedArray<value_type> tau_IP = std::max(fe_eval.read_cell_data(array_penalty_parameter),
                                                    fe_eval_neighbor.read_cell_data(array_penalty_parameter))
                                              * get_penalty_factor();

      VectorizedArray<value_type> local_diagonal_vector[fe_eval.tensor_dofs_per_cell];
      for (unsigned int j=0; j<fe_eval.dofs_per_cell; ++j)
      {
        // set dof value j of element- to 1 and all other dof values of element- to zero
        for (unsigned int i=0; i<fe_eval.dofs_per_cell; ++i)
          fe_eval.begin_dof_values()[i] = make_vectorized_array<value_type>(0.);
        fe_eval.begin_dof_values()[j] = make_vectorized_array<value_type>(1.);

        fe_eval.evaluate(true,true);

        for(unsigned int q=0;q<fe_eval.n_q_points;++q)
        {
          // set exterior value to zero
          VectorizedArray<value_type> jump_value = fe_eval.get_value(q);
          VectorizedArray<value_type> value_flux = make_vectorized_array<value_type>(0.0);
          calculate_value_flux(value_flux, jump_value);

          // set exterior value to zero
          VectorizedArray<value_type> average_normal_gradient = 0.5 * fe_eval.get_normal_gradient(q);

          VectorizedArray<value_type> gradient_flux = make_vectorized_array<value_type>(0.0);
          calculate_gradient_flux(gradient_flux, average_normal_gradient, jump_value, tau_IP);

          fe_eval.submit_normal_gradient(value_flux,q);
          fe_eval.submit_value(-gradient_flux,q);
        }
        fe_eval.integrate(true,true);

        local_diagonal_vector[j] = fe_eval.begin_dof_values()[j];
      }
      for (unsigned int j=0; j<fe_eval.dofs_per_cell; ++j)
        fe_eval.begin_dof_values()[j] = local_diagonal_vector[j];

      fe_eval.distribute_local_to_global(dst);
    }



    // TODO: This has to be removed as soon as the new infrastructure is used that
    // allows to perform face integrals over all faces of the current element.
    // Perform face intergrals for element e⁺.
    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval.reinit (face);
      fe_eval_neighbor.reinit (face);

      VectorizedArray<value_type> tau_IP = std::max(fe_eval.read_cell_data(array_penalty_parameter),
                                                    fe_eval_neighbor.read_cell_data(array_penalty_parameter))
                                              * get_penalty_factor();

      VectorizedArray<value_type> local_diagonal_vector_neighbor[fe_eval_neighbor.tensor_dofs_per_cell];
      for (unsigned int j=0; j<fe_eval_neighbor.dofs_per_cell; ++j)
      {
        // set dof value j of element+ to 1 and all other dof values of element+ to zero
        for (unsigned int i=0; i<fe_eval_neighbor.dofs_per_cell; ++i)
          fe_eval_neighbor.begin_dof_values()[i] = make_vectorized_array<value_type>(0.);
        fe_eval_neighbor.begin_dof_values()[j] = make_vectorized_array<value_type>(1.);

        fe_eval_neighbor.evaluate(true,true);

        for(unsigned int q=0;q<fe_eval.n_q_points;++q)
        {
          // set value_m to zero
          VectorizedArray<value_type> jump_value = fe_eval_neighbor.get_value(q);
          VectorizedArray<value_type> value_flux = make_vectorized_array<value_type>(0.0);
          calculate_value_flux(value_flux, jump_value);
          // set gradient_m to zero, minus sign to get the correct normal vector n⁺ = -n⁻
          VectorizedArray<value_type> average_normal_gradient = - 0.5 * fe_eval_neighbor.get_normal_gradient(q);

          VectorizedArray<value_type> gradient_flux = make_vectorized_array<value_type>(0.0);
          calculate_gradient_flux(gradient_flux, average_normal_gradient, jump_value, tau_IP);
          // minus sign since n⁺ = -n⁻
          fe_eval_neighbor.submit_normal_gradient(-value_flux,q);
          fe_eval_neighbor.submit_value(-gradient_flux,q);
        }
        fe_eval_neighbor.integrate(true,true);

        local_diagonal_vector_neighbor[j] = fe_eval_neighbor.begin_dof_values()[j];
      }
      for (unsigned int j=0; j<fe_eval_neighbor.dofs_per_cell; ++j)
        fe_eval_neighbor.begin_dof_values()[j] = local_diagonal_vector_neighbor[j];

      fe_eval_neighbor.distribute_local_to_global(dst);
    }
  }

  // TODO: This function has to be removed as soon as the new infrastructure is used that
  // allows to perform face integrals over all faces of the current element.
  void boundary_face_loop_diagonal (const MatrixFree<dim,value_type>                &data,
                                    parallel::distributed::Vector<value_type>       &dst,
                                    const parallel::distributed::Vector<value_type> &/*src*/,
                                    const std::pair<unsigned int,unsigned int>      &face_range) const
  {
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval(data,true,operator_data.dof_index,operator_data.quad_index);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      types::boundary_id boundary_id = data.get_boundary_indicator(face);
      BoundaryType boundary_type = BoundaryType::undefined;

      if(operator_data.bc->dirichlet_bc.find(boundary_id) != operator_data.bc->dirichlet_bc.end())
        boundary_type = BoundaryType::dirichlet;
      else if(operator_data.bc->neumann_bc.find(boundary_id) != operator_data.bc->neumann_bc.end())
        boundary_type = BoundaryType::neumann;

      AssertThrow(boundary_type != BoundaryType::undefined,
          ExcMessage("Boundary type of face is invalid or not implemented."));

      fe_eval.reinit (face);

      VectorizedArray<value_type> tau_IP = fe_eval.read_cell_data(array_penalty_parameter)
                                              * get_penalty_factor();

      VectorizedArray<value_type> local_diagonal_vector[fe_eval.tensor_dofs_per_cell];
      for (unsigned int j=0; j<fe_eval.dofs_per_cell; ++j)
      {
        // set dof value j of element- to 1 and all other dof values of element- to zero
        for (unsigned int i=0; i<fe_eval.dofs_per_cell; ++i)
          fe_eval.begin_dof_values()[i] = make_vectorized_array<value_type>(0.);
        fe_eval.begin_dof_values()[j] = make_vectorized_array<value_type>(1.);

        fe_eval.evaluate(true,true);

        for(unsigned int q=0;q<fe_eval.n_q_points;++q)
        {
          VectorizedArray<value_type> jump_value = make_vectorized_array<value_type>(0.0);
          calculate_jump_value_boundary_face(jump_value,q,fe_eval,OperatorType::homogeneous,boundary_type);
          VectorizedArray<value_type> value_flux = make_vectorized_array<value_type>(0.0);
          calculate_value_flux(value_flux, jump_value);

          VectorizedArray<value_type> average_normal_gradient = make_vectorized_array<value_type>(0.0);
          calculate_average_normal_gradient_boundary_face(average_normal_gradient,q,fe_eval,OperatorType::homogeneous,boundary_type);

          VectorizedArray<value_type> gradient_flux = make_vectorized_array<value_type>(0.0);
          calculate_gradient_flux(gradient_flux, average_normal_gradient, jump_value, tau_IP);

          fe_eval.submit_normal_gradient(value_flux,q);
          fe_eval.submit_value(-gradient_flux,q);
        }
        fe_eval.integrate(true,true);

        local_diagonal_vector[j] = fe_eval.begin_dof_values()[j];
      }
      for (unsigned int j=0; j<fe_eval.dofs_per_cell; ++j)
        fe_eval.begin_dof_values()[j] = local_diagonal_vector[j];

      fe_eval.distribute_local_to_global(dst);
    }
  }


  void cell_loop_calculate_block_jacobi_matrices (const MatrixFree<dim,value_type>                 &data,
                                                  std::vector<LAPACKFullMatrix<value_type> >       &matrices,
                                                  const parallel::distributed::Vector<value_type>  &,
                                                  const std::pair<unsigned int,unsigned int>       &cell_range) const
  {
    FEEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval(data,
                                                                 operator_data.dof_index,
                                                                 operator_data.quad_index);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);

      for (unsigned int j=0; j<fe_eval.dofs_per_cell; ++j)
      {
        for (unsigned int i=0; i<fe_eval.dofs_per_cell; ++i)
          fe_eval.begin_dof_values()[i] = make_vectorized_array<value_type>(0.);
        fe_eval.begin_dof_values()[j] = make_vectorized_array<value_type>(1.);

        do_cell_integral(fe_eval);

        for(unsigned int i=0; i<fe_eval.dofs_per_cell; ++i)
          for (unsigned int v=0; v<VectorizedArray<value_type>::n_array_elements; ++v)
            matrices[cell*VectorizedArray<value_type>::n_array_elements+v](i,j) += fe_eval.begin_dof_values()[i][v];
      }
    }
  }

  void face_loop_calculate_block_jacobi_matrices (const MatrixFree<dim,value_type>                &data,
                                                  std::vector<LAPACKFullMatrix<value_type> >      &matrices,
                                                  const parallel::distributed::Vector<value_type> &,
                                                  const std::pair<unsigned int,unsigned int>      &face_range) const
  {
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval(data,true,operator_data.dof_index,operator_data.quad_index);
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval_neighbor(data,false,operator_data.dof_index,operator_data.quad_index);

    // Perform face intergrals for element e⁻.
    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval.reinit (face);
      fe_eval_neighbor.reinit (face);

      VectorizedArray<value_type> tau_IP = std::max(fe_eval.read_cell_data(array_penalty_parameter),
                                                    fe_eval_neighbor.read_cell_data(array_penalty_parameter))
                                              * get_penalty_factor();

      for (unsigned int j=0; j<fe_eval.dofs_per_cell; ++j)
      {
        // set dof value j of element- to 1 and all other dof values of element- to zero
        for (unsigned int i=0; i<fe_eval.dofs_per_cell; ++i)
          fe_eval.begin_dof_values()[i] = make_vectorized_array<value_type>(0.);
        fe_eval.begin_dof_values()[j] = make_vectorized_array<value_type>(1.);

        fe_eval.evaluate(true,true);

        for(unsigned int q=0;q<fe_eval.n_q_points;++q)
        {
          // set exterior value to zero
          VectorizedArray<value_type> jump_value = fe_eval.get_value(q);
          VectorizedArray<value_type> value_flux = make_vectorized_array<value_type>(0.0);
          calculate_value_flux(value_flux, jump_value);

          // set exterior value to zero
          VectorizedArray<value_type> average_normal_gradient = 0.5 * fe_eval.get_normal_gradient(q);

          VectorizedArray<value_type> gradient_flux = make_vectorized_array<value_type>(0.0);
          calculate_gradient_flux(gradient_flux, average_normal_gradient, jump_value, tau_IP);

          fe_eval.submit_normal_gradient(value_flux,q);
          fe_eval.submit_value(-gradient_flux,q);
        }
        fe_eval.integrate(true,true);

        for (unsigned int v=0; v<VectorizedArray<value_type>::n_array_elements; ++v)
        {
          const unsigned int cell_number = data.faces[face].left_cell[v];
          if (cell_number != numbers::invalid_unsigned_int)
            for(unsigned int i=0; i<fe_eval.dofs_per_cell; ++i)
              matrices[cell_number](i,j) += fe_eval.begin_dof_values()[i][v];
        }
      }
    }



    // TODO: This has to be removed as soon as the new infrastructure is used that
    // allows to perform face integrals over all faces of the current element.
    // Perform face intergrals for element e⁺.
    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval.reinit (face);
      fe_eval_neighbor.reinit (face);

      VectorizedArray<value_type> tau_IP = std::max(fe_eval.read_cell_data(array_penalty_parameter),
                                                    fe_eval_neighbor.read_cell_data(array_penalty_parameter))
                                              * get_penalty_factor();

      for (unsigned int j=0; j<fe_eval_neighbor.dofs_per_cell; ++j)
      {
        // set dof value j of element+ to 1 and all other dof values of element+ to zero
        for (unsigned int i=0; i<fe_eval_neighbor.dofs_per_cell; ++i)
          fe_eval_neighbor.begin_dof_values()[i] = make_vectorized_array<value_type>(0.);
        fe_eval_neighbor.begin_dof_values()[j] = make_vectorized_array<value_type>(1.);

        fe_eval_neighbor.evaluate(true,true);

        for(unsigned int q=0;q<fe_eval.n_q_points;++q)
        {
          // set value_m to zero
          VectorizedArray<value_type> jump_value = fe_eval_neighbor.get_value(q);
          VectorizedArray<value_type> value_flux = make_vectorized_array<value_type>(0.0);
          calculate_value_flux(value_flux, jump_value);
          // set gradient_m to zero, minus sign to get the correct normal vector n⁺ = -n⁻
          VectorizedArray<value_type> average_normal_gradient = - 0.5 * fe_eval_neighbor.get_normal_gradient(q);

          VectorizedArray<value_type> gradient_flux = make_vectorized_array<value_type>(0.0);
          calculate_gradient_flux(gradient_flux, average_normal_gradient, jump_value, tau_IP);
          // minus sign since n⁺ = -n⁻
          fe_eval_neighbor.submit_normal_gradient(-value_flux,q);
          fe_eval_neighbor.submit_value(-gradient_flux,q);
        }
        fe_eval_neighbor.integrate(true,true);

        for (unsigned int v=0; v<VectorizedArray<value_type>::n_array_elements; ++v)
        {
          const unsigned int cell_number = data.faces[face].right_cell[v];
          if (cell_number != numbers::invalid_unsigned_int)
            for(unsigned int i=0; i<fe_eval_neighbor.dofs_per_cell; ++i)
              matrices[cell_number](i,j) += fe_eval_neighbor.begin_dof_values()[i][v];
        }
      }
    }
  }

  // TODO: This function has to be removed as soon as the new infrastructure is used that
  // allows to perform face integrals over all faces of the current element.
  void boundary_face_loop_calculate_block_jacobi_matrices (const MatrixFree<dim,value_type>                &data,
                                                           std::vector<LAPACKFullMatrix<value_type> >      &matrices,
                                                           const parallel::distributed::Vector<value_type> &,
                                                           const std::pair<unsigned int,unsigned int>      &face_range) const
  {
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval(data,true,operator_data.dof_index,operator_data.quad_index);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      types::boundary_id boundary_id = data.get_boundary_indicator(face);
      BoundaryType boundary_type = BoundaryType::undefined;

      if(operator_data.bc->dirichlet_bc.find(boundary_id) != operator_data.bc->dirichlet_bc.end())
        boundary_type = BoundaryType::dirichlet;
      else if(operator_data.bc->neumann_bc.find(boundary_id) != operator_data.bc->neumann_bc.end())
        boundary_type = BoundaryType::neumann;

      AssertThrow(boundary_type != BoundaryType::undefined,
          ExcMessage("Boundary type of face is invalid or not implemented."));

      fe_eval.reinit (face);

      VectorizedArray<value_type> tau_IP = fe_eval.read_cell_data(array_penalty_parameter)
                                              * get_penalty_factor();

      for (unsigned int j=0; j<fe_eval.dofs_per_cell; ++j)
      {
        // set dof value j of element- to 1 and all other dof values of element- to zero
        for (unsigned int i=0; i<fe_eval.dofs_per_cell; ++i)
          fe_eval.begin_dof_values()[i] = make_vectorized_array<value_type>(0.);
        fe_eval.begin_dof_values()[j] = make_vectorized_array<value_type>(1.);

        fe_eval.evaluate(true,true);

        for(unsigned int q=0;q<fe_eval.n_q_points;++q)
        {
          VectorizedArray<value_type> jump_value = make_vectorized_array<value_type>(0.0);
          calculate_jump_value_boundary_face(jump_value,q,fe_eval,OperatorType::homogeneous,boundary_type);
          VectorizedArray<value_type> value_flux = make_vectorized_array<value_type>(0.0);
          calculate_value_flux(value_flux, jump_value);

          VectorizedArray<value_type> average_normal_gradient = make_vectorized_array<value_type>(0.0);
          calculate_average_normal_gradient_boundary_face(average_normal_gradient,q,fe_eval,OperatorType::homogeneous,boundary_type);

          VectorizedArray<value_type> gradient_flux = make_vectorized_array<value_type>(0.0);
          calculate_gradient_flux(gradient_flux, average_normal_gradient, jump_value, tau_IP);

          fe_eval.submit_normal_gradient(value_flux,q);
          fe_eval.submit_value(-gradient_flux,q);
        }
        fe_eval.integrate(true,true);

        for (unsigned int v=0; v<VectorizedArray<value_type>::n_array_elements; ++v)
        {
          const unsigned int cell_number = data.faces[face].left_cell[v];
          if (cell_number != numbers::invalid_unsigned_int)
            for(unsigned int i=0; i<fe_eval.dofs_per_cell; ++i)
              matrices[cell_number](i,j) += fe_eval.begin_dof_values()[i][v];
        }
      }
    }
  }

  /*
   *  evaluate boundary face integrals for full operator (homogeneous + inhomogeneous parts)
   */
  void boundary_face_loop_full_operator (const MatrixFree<dim,value_type>                &data,
                                         parallel::distributed::Vector<value_type>       &dst,
                                         const parallel::distributed::Vector<value_type> &src,
                                         const std::pair<unsigned int,unsigned int>      &face_range) const
  {
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval(data,true,operator_data.dof_index,operator_data.quad_index);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      types::boundary_id boundary_id = data.get_boundary_indicator(face);
      BoundaryType boundary_type = BoundaryType::undefined;

      if(operator_data.bc->dirichlet_bc.find(boundary_id) != operator_data.bc->dirichlet_bc.end())
        boundary_type = BoundaryType::dirichlet;
      else if(operator_data.bc->neumann_bc.find(boundary_id) != operator_data.bc->neumann_bc.end())
        boundary_type = BoundaryType::neumann;

      AssertThrow(boundary_type != BoundaryType::undefined,
          ExcMessage("Boundary type of face is invalid or not implemented."));

      fe_eval.reinit (face);
      fe_eval.read_dof_values(src);
      fe_eval.evaluate(true,true);

      VectorizedArray<value_type> tau_IP = fe_eval.read_cell_data(array_penalty_parameter)
                                              * get_penalty_factor();

      for(unsigned int q=0;q<fe_eval.n_q_points;++q)
      {
        VectorizedArray<value_type> jump_value = make_vectorized_array<value_type>(0.0);
        calculate_jump_value_boundary_face(jump_value,q,fe_eval,OperatorType::full,boundary_type,boundary_id);
        VectorizedArray<value_type> value_flux = make_vectorized_array<value_type>(0.0);
        calculate_value_flux(value_flux, jump_value);

        VectorizedArray<value_type> average_normal_gradient = make_vectorized_array<value_type>(0.0);
        calculate_average_normal_gradient_boundary_face(average_normal_gradient,q,fe_eval,OperatorType::full,boundary_type,boundary_id);

        VectorizedArray<value_type> gradient_flux = make_vectorized_array<value_type>(0.0);
        calculate_gradient_flux(gradient_flux, average_normal_gradient, jump_value, tau_IP);

        fe_eval.submit_normal_gradient(value_flux,q);
        fe_eval.submit_value(-gradient_flux,q);
      }
      fe_eval.integrate(true,true);
      fe_eval.distribute_local_to_global(dst);
    }
  }


  /*
   *  Evaluate inhomogeneous operator. Note that these integrals are multiplied by
   *  a factor of -1.0 since these integrals apppear on the right-hand side of the equations.
   */
  void cell_loop_inhom_operator (const MatrixFree<dim,value_type>                 &,
                                 parallel::distributed::Vector<value_type>        &,
                                 const parallel::distributed::Vector<value_type>  &,
                                 const std::pair<unsigned int,unsigned int>       &) const
  {}

  void face_loop_inhom_operator (const MatrixFree<dim,value_type>                &,
                                 parallel::distributed::Vector<value_type>       &,
                                 const parallel::distributed::Vector<value_type> &,
                                 const std::pair<unsigned int,unsigned int>      &) const
  {}

  void boundary_face_loop_inhom_operator (const MatrixFree<dim,value_type>                &data,
                                          parallel::distributed::Vector<value_type>       &dst,
                                          const parallel::distributed::Vector<value_type> &/*src*/,
                                          const std::pair<unsigned int,unsigned int>      &face_range) const
  {
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval(data,true,operator_data.dof_index,operator_data.quad_index);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      types::boundary_id boundary_id = data.get_boundary_indicator(face);
      BoundaryType boundary_type = BoundaryType::undefined;

      if(operator_data.bc->dirichlet_bc.find(boundary_id) != operator_data.bc->dirichlet_bc.end())
        boundary_type = BoundaryType::dirichlet;
      else if(operator_data.bc->neumann_bc.find(boundary_id) != operator_data.bc->neumann_bc.end())
        boundary_type = BoundaryType::neumann;

      AssertThrow(boundary_type != BoundaryType::undefined,
          ExcMessage("Boundary type of face is invalid or not implemented."));

      fe_eval.reinit (face);

      VectorizedArray<value_type> tau_IP = fe_eval.read_cell_data(array_penalty_parameter)
                                              * get_penalty_factor();

      for(unsigned int q=0;q<fe_eval.n_q_points;++q)
      {
        VectorizedArray<value_type> jump_value = make_vectorized_array<value_type>(0.0);
        calculate_jump_value_boundary_face(jump_value,q,fe_eval,OperatorType::inhomogeneous,boundary_type,boundary_id);
        VectorizedArray<value_type> value_flux = make_vectorized_array<value_type>(0.0);
        calculate_value_flux(value_flux, jump_value);

        VectorizedArray<value_type> average_normal_gradient = make_vectorized_array<value_type>(0.0);
        calculate_average_normal_gradient_boundary_face(average_normal_gradient,q,fe_eval,OperatorType::inhomogeneous,boundary_type,boundary_id);

        VectorizedArray<value_type> gradient_flux = make_vectorized_array<value_type>(0.0);
        calculate_gradient_flux(gradient_flux, average_normal_gradient, jump_value, tau_IP);

        // -value_flux since this term appears on the rhs of the equation !!
        fe_eval.submit_normal_gradient(-value_flux,q);
        // +gradient_flux since this term appears on the rhs of the equation !!
        fe_eval.submit_value(gradient_flux,q);
      }
      fe_eval.integrate(true,true);
      fe_eval.distribute_local_to_global(dst);
    }
  }

  MatrixFree<dim,value_type> const * data;
  DiffusiveOperatorData<dim> operator_data;
  AlignedVector<VectorizedArray<value_type> > array_penalty_parameter;
  double diffusivity;
  mutable value_type eval_time;
};


template<int dim>
struct ConvectiveOperatorData
{
  ConvectiveOperatorData ()
    :
    dof_index(0),
    quad_index(0),
    numerical_flux_formulation(NumericalFluxConvectiveOperator::Undefined)
  {}

  unsigned int dof_index;
  unsigned int quad_index;
  NumericalFluxConvectiveOperator numerical_flux_formulation;

  std_cxx11::shared_ptr<BoundaryDescriptorConvDiff<dim> > bc;
  std_cxx11::shared_ptr<Function<dim> > velocity;
};

template <int dim, int fe_degree, typename value_type>
class ConvectiveOperator
{
public:
  typedef ConvectiveOperator<dim,fe_degree, value_type> This;

  enum class OperatorType {
    full,
    homogeneous,
    inhomogeneous
  };

  enum class BoundaryType {
    undefined,
    dirichlet,
    neumann
  };

  ConvectiveOperator()
    :
    data(nullptr)
  {}

  void initialize(MatrixFree<dim,value_type> const  &mf_data,
                  ConvectiveOperatorData<dim> const &operator_data_in)
  {
    this->data = &mf_data;
    this->operator_data = operator_data_in;
  }

  // apply matrix vector multiplication
  void apply (parallel::distributed::Vector<value_type>       &dst,
              const parallel::distributed::Vector<value_type> &src,
              value_type const                                evaluation_time) const
  {
    dst = 0;
    apply_add(dst,src,evaluation_time);
  }

  void apply_add (parallel::distributed::Vector<value_type>       &dst,
                  const parallel::distributed::Vector<value_type> &src,
                  value_type const                                evaluation_time) const
  {
    this->eval_time = evaluation_time;

    data->loop(&This::cell_loop,&This::face_loop,
               &This::boundary_face_loop_hom_operator,this, dst, src);
  }

  // apply "block Jacobi" matrix vector multiplication
  void apply_block_jacobi (parallel::distributed::Vector<value_type>       &dst,
                           const parallel::distributed::Vector<value_type> &src,
                           value_type const                                evaluation_time) const
  {
    dst = 0;
    apply_block_jacobi_add(dst,src,evaluation_time);
  }

  void apply_block_jacobi_add (parallel::distributed::Vector<value_type>       &dst,
                               const parallel::distributed::Vector<value_type> &src,
                               value_type const                                evaluation_time) const
  {
    this->eval_time = evaluation_time;

    data->loop(&This::cell_loop,&This::face_loop_block_jacobi,
               &This::boundary_face_loop_hom_operator,this, dst, src);
  }

  void add_block_jacobi_matrices(std::vector<LAPACKFullMatrix<value_type> > &matrices,
                                 value_type const                           evaluation_time) const
  {
    this->eval_time = evaluation_time;

    parallel::distributed::Vector<value_type>  src;

    data->loop(&This::cell_loop_calculate_block_jacobi_matrices,&This::face_loop_calculate_block_jacobi_matrices,
               &This::boundary_face_loop_calculate_block_jacobi_matrices, this, matrices, src);
  }

  void evaluate (parallel::distributed::Vector<value_type>       &dst,
                 const parallel::distributed::Vector<value_type> &src,
                 value_type const                                evaluation_time) const
  {
    dst = 0;
    evaluate_add(dst,src,evaluation_time);
  }

  void evaluate_add (parallel::distributed::Vector<value_type>       &dst,
                     const parallel::distributed::Vector<value_type> &src,
                     value_type const                                evaluation_time) const
  {
    this->eval_time = evaluation_time;

    data->loop(&This::cell_loop,&This::face_loop,
               &This::boundary_face_loop_full_operator,this, dst, src);
  }

  void calculate_diagonal (parallel::distributed::Vector<value_type>       &diagonal,
                           value_type const                                evaluation_time) const
  {
    diagonal = 0.0;

    add_diagonal(diagonal,evaluation_time);
  }

  void add_diagonal(parallel::distributed::Vector<value_type>       &diagonal,
                    value_type const                                evaluation_time) const
  {
    this->eval_time = evaluation_time;

    parallel::distributed::Vector<value_type>  src_dummy(diagonal);

    data->loop(&This::cell_loop_diagonal,&This::face_loop_diagonal,
               &This::boundary_face_loop_diagonal,this,diagonal,src_dummy);
  }

  void rhs (parallel::distributed::Vector<value_type> &dst,
            value_type const                          evaluation_time) const
  {
    dst = 0;
    rhs_add(dst,evaluation_time);
  }

  void rhs_add (parallel::distributed::Vector<value_type> &dst,
                value_type const                          evaluation_time) const
  {
    this->eval_time = evaluation_time;

    parallel::distributed::Vector<value_type> src;

    data->loop(&This::cell_loop_inhom_operator,&This::face_loop_inhom_operator,
               &This::boundary_face_loop_inhom_operator,this, dst, src);
  }

  ConvectiveOperatorData<dim> const & get_operator_data() const
  {
    return this->operator_data;
  }

private:
  template<typename FEEvaluation>
  inline void do_cell_integral(FEEvaluation &fe_eval) const
  {
    fe_eval.evaluate (true,false,false);

    for (unsigned int q=0; q<fe_eval.n_q_points; ++q)
    {
      Point<dim,VectorizedArray<value_type> > q_points = fe_eval.quadrature_point(q);
      Tensor<1,dim,VectorizedArray<value_type> > velocity;

      evaluate_vectorial_function(velocity,operator_data.velocity,q_points,eval_time);

      fe_eval.submit_gradient(-fe_eval.get_value(q)*velocity,q);
    }

    fe_eval.integrate (false,true);
  }

  /*
   *  This function calculates the numerical flux for interior faces
   *  using the central flux.
   */
  inline void calculate_central_flux(VectorizedArray<value_type> &flux,
                                     VectorizedArray<value_type> &value_m,
                                     VectorizedArray<value_type> &value_p,
                                     VectorizedArray<value_type> &normal_velocity) const
  {
    VectorizedArray<value_type> average_value = 0.5*(value_m + value_p);
    flux = normal_velocity*average_value;
  }

  /*
   *  This function calculates the numerical flux for interior faces
   *  using the Lax-Friedrichs flux.
   */
  inline void calculate_lax_friedrichs_flux(VectorizedArray<value_type> &flux,
                                            VectorizedArray<value_type> &value_m,
                                            VectorizedArray<value_type> &value_p,
                                            VectorizedArray<value_type> &normal_velocity) const
  {
    VectorizedArray<value_type> average_value = 0.5*(value_m + value_p);
    VectorizedArray<value_type> jump_value = value_m - value_p;
    VectorizedArray<value_type> lambda = std::abs(normal_velocity);
    flux = normal_velocity*average_value + 0.5*lambda*jump_value;
  }

  /*
   *  This function calculates the numerical flux for interior faces where
   *  the type of the numerical flux depends on the specified input parameter.
   */
  template<typename FEEvaluation>
  inline void calculate_flux(VectorizedArray<value_type> &flux,
                             unsigned int const          q,
                             FEEvaluation                &fe_eval,
                             VectorizedArray<value_type> &value_m,
                             VectorizedArray<value_type> &value_p) const
  {
    Point<dim,VectorizedArray<value_type> > q_points = fe_eval.quadrature_point(q);
    Tensor<1,dim,VectorizedArray<value_type> > velocity;

    evaluate_vectorial_function(velocity,operator_data.velocity,q_points,eval_time);

    Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval.get_normal_vector(q);
    VectorizedArray<value_type> normal_velocity = velocity*normal;

    if(this->operator_data.numerical_flux_formulation
        == NumericalFluxConvectiveOperator::CentralFlux)
    {
      calculate_central_flux(flux,value_m,value_p,normal_velocity);
    }
    else if(this->operator_data.numerical_flux_formulation
        == NumericalFluxConvectiveOperator::LaxFriedrichsFlux)
    {
      calculate_lax_friedrichs_flux(flux,value_m,value_p,normal_velocity);
    }
  }

  /*
   *  This function calculates the numerical flux depending on the operator type,
   *  the type of the boundary face and the given boundary conditions.
   *
   *  Definition of exterior values on boundary faces:
   *
   *                            +----------------------+--------------------+
   *                            | Dirichlet boundaries | Neumann boundaries |
   *  +-------------------------+----------------------+--------------------+
   *  | full operator           | phi⁺ = -phi⁻ + 2g    | phi⁺ = phi⁻        |
   *  +-------------------------+----------------------+--------------------+
   *  | homogeneous operator    | phi⁺ = -phi⁻         | phi⁺ = phi⁻        |
   *  +-------------------------+----------------------+--------------------+
   *  | inhomogeneous operator  | phi⁻ = 0, phi⁺ = 2g  | phi⁻ = 0, phi⁺ = 0 |
   *  +-------------------------+----------------------+--------------------+
   */
  template<typename FEEvaluation>
  inline void calculate_flux_boundary_face(VectorizedArray<value_type> &flux,
                                           unsigned int const          q,
                                           FEEvaluation                &fe_eval,
                                           OperatorType const          &operator_type,
                                           BoundaryType const          &boundary_type,
                                           types::boundary_id const    boundary_id = types::boundary_id()) const
  {
    // element e⁻
    VectorizedArray<value_type> value_m = make_vectorized_array<value_type>(0.0);

    if(operator_type == OperatorType::full || operator_type == OperatorType::homogeneous)
    {
      value_m = fe_eval.get_value(q);
    }
    else if(operator_type == OperatorType::inhomogeneous)
    {
      // do nothing, value_m is already initialized with zeros
    }
    else
    {
      AssertThrow(false, ExcMessage("Specified ExteriorValuesType is not implemented!"));
    }

    // element e⁺
    VectorizedArray<value_type> value_p = make_vectorized_array<value_type>(0.0);

    if(operator_type == OperatorType::full)
    {
      if(boundary_type == BoundaryType::dirichlet)
      {
        VectorizedArray<value_type> g = make_vectorized_array<value_type>(0.0);
        typename std::map<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >::iterator it;
        it = operator_data.bc->dirichlet_bc.find(boundary_id);
        Point<dim,VectorizedArray<value_type> > q_points = fe_eval.quadrature_point(q);
        evaluate_scalar_function(g,it->second,q_points,eval_time);

        value_p = - value_m + 2.0*g;
      }
      else if(boundary_type == BoundaryType::neumann)
      {
        value_p = value_m;
      }
      else
      {
        AssertThrow(false,ExcMessage("Boundary type of face is invalid or not implemented."));
      }
    }
    else if(operator_type == OperatorType::homogeneous)
    {
      if(boundary_type == BoundaryType::dirichlet)
      {
        value_p = - value_m;
      }
      else if(boundary_type == BoundaryType::neumann)
      {
        value_p = value_m;
      }
      else
      {
        AssertThrow(false,ExcMessage("Boundary type of face is invalid or not implemented."));
      }
    }
    else if(operator_type == OperatorType::inhomogeneous)
    {
      if(boundary_type == BoundaryType::dirichlet)
      {
        VectorizedArray<value_type> g = make_vectorized_array<value_type>(0.0);
        typename std::map<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >::iterator it;
        it = operator_data.bc->dirichlet_bc.find(boundary_id);
        Point<dim,VectorizedArray<value_type> > q_points = fe_eval.quadrature_point(q);
        evaluate_scalar_function(g,it->second,q_points,eval_time);

        value_p = 2.0*g;
      }
      else if(boundary_type == BoundaryType::neumann)
      {
        // do nothing, value_p is already initialized with zeros
      }
      else
      {
        AssertThrow(false,ExcMessage("Boundary type of face is invalid or not implemented."));
      }
    }
    else
    {
      AssertThrow(false, ExcMessage("Specified ExteriorValuesType is not implemented!"));
    }

    calculate_flux(flux,q,fe_eval,value_m,value_p);
  }


  /*
   *  Evaluate homogeneous operator
   */
  void cell_loop (const MatrixFree<dim,value_type>                 &data,
                  parallel::distributed::Vector<value_type>        &dst,
                  const parallel::distributed::Vector<value_type>  &src,
                  const std::pair<unsigned int,unsigned int>       &cell_range) const
  {
    FEEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval(data,
                                                                 operator_data.dof_index,
                                                                 operator_data.quad_index);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);
      fe_eval.read_dof_values(src);

      do_cell_integral(fe_eval);

      fe_eval.distribute_local_to_global (dst);
    }
  }

  void face_loop (const MatrixFree<dim,value_type>                &data,
                  parallel::distributed::Vector<value_type>       &dst,
                  const parallel::distributed::Vector<value_type> &src,
                  const std::pair<unsigned int,unsigned int>      &face_range) const
  {
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval(data,true,operator_data.dof_index,operator_data.quad_index);
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval_neighbor(data,false,operator_data.dof_index,operator_data.quad_index);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval.reinit(face);
      fe_eval.read_dof_values(src);

      fe_eval_neighbor.reinit(face);
      fe_eval_neighbor.read_dof_values(src);

      fe_eval.evaluate(true,false);
      fe_eval_neighbor.evaluate(true,false);

      for(unsigned int q=0;q<fe_eval.n_q_points;++q)
      {
        VectorizedArray<value_type> numerical_flux = make_vectorized_array<value_type>(0.0);
        VectorizedArray<value_type> value_m = fe_eval.get_value(q);
        VectorizedArray<value_type> value_p = fe_eval_neighbor.get_value(q);
        calculate_flux(numerical_flux, q, fe_eval, value_m, value_p);

        fe_eval.submit_value(numerical_flux,q);
        fe_eval_neighbor.submit_value(-numerical_flux,q);
      }
      fe_eval.integrate(true,false);
      fe_eval_neighbor.integrate(true,false);

      fe_eval.distribute_local_to_global(dst);
      fe_eval_neighbor.distribute_local_to_global(dst);
    }
  }

  void boundary_face_loop_hom_operator (const MatrixFree<dim,value_type>                &data,
                                        parallel::distributed::Vector<value_type>       &dst,
                                        const parallel::distributed::Vector<value_type> &src,
                                        const std::pair<unsigned int,unsigned int>      &face_range) const
  {
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval(data,true,operator_data.dof_index,operator_data.quad_index);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      types::boundary_id boundary_id = data.get_boundary_indicator(face);
      BoundaryType boundary_type = BoundaryType::undefined;

      if(operator_data.bc->dirichlet_bc.find(boundary_id) != operator_data.bc->dirichlet_bc.end())
        boundary_type = BoundaryType::dirichlet;
      else if(operator_data.bc->neumann_bc.find(boundary_id) != operator_data.bc->neumann_bc.end())
        boundary_type = BoundaryType::neumann;

      AssertThrow(boundary_type != BoundaryType::undefined,
          ExcMessage("Boundary type of face is invalid or not implemented."));

      fe_eval.reinit (face);
      fe_eval.read_dof_values(src);
      fe_eval.evaluate(true,false);

      for(unsigned int q=0;q<fe_eval.n_q_points;++q)
      {
        VectorizedArray<value_type> numerical_flux = make_vectorized_array<value_type>(0.0);

        calculate_flux_boundary_face(numerical_flux, q, fe_eval, OperatorType::homogeneous, boundary_type);

        fe_eval.submit_value(numerical_flux,q);
      }

      fe_eval.integrate(true,false);
      fe_eval.distribute_local_to_global(dst);
    }
  }


  /*
   *  face integrals for block-jacobi, use homogeneous operator for cell and boundary face integrals
   */
  void face_loop_block_jacobi (const MatrixFree<dim,value_type>                &data,
                               parallel::distributed::Vector<value_type>       &dst,
                               const parallel::distributed::Vector<value_type> &src,
                               const std::pair<unsigned int,unsigned int>      &face_range) const
  {
    // perform face integral for element e⁻
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval(data,true,operator_data.dof_index,operator_data.quad_index);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval.reinit (face);
      fe_eval.read_dof_values(src);
      fe_eval.evaluate(true,false);

      for(unsigned int q=0;q<fe_eval.n_q_points;++q)
      {
        VectorizedArray<value_type> numerical_flux = make_vectorized_array<value_type>(0.0);
        VectorizedArray<value_type> value_m = fe_eval.get_value(q);
        // set value_p to zero
        VectorizedArray<value_type> value_p = make_vectorized_array<value_type>(0.0);
        calculate_flux(numerical_flux, q, fe_eval, value_m, value_p);

        fe_eval.submit_value(numerical_flux,q);
      }
      fe_eval.integrate(true,false);
      fe_eval.distribute_local_to_global(dst);
    }

    // TODO: this has to be removed as soon as the new infrastructure is used that
    // allows to perform face integrals over all faces of the current element
    // perform face integral for element e⁺
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval_neighbor(data,false,operator_data.dof_index,operator_data.quad_index);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval_neighbor.reinit (face);
      fe_eval_neighbor.read_dof_values(src);
      fe_eval_neighbor.evaluate(true,false);

      for(unsigned int q=0;q<fe_eval.n_q_points;++q)
      {
        VectorizedArray<value_type> numerical_flux = make_vectorized_array<value_type>(0.0);
        // set value_m to zero
        VectorizedArray<value_type> value_m = make_vectorized_array<value_type>(0.0);
        VectorizedArray<value_type> value_p = fe_eval_neighbor.get_value(q);
        calculate_flux(numerical_flux, q, fe_eval_neighbor, value_m, value_p);

        // hack (minus sign) since n⁺ = -n⁻
        fe_eval_neighbor.submit_value(-numerical_flux,q);
      }
      fe_eval_neighbor.integrate(true,false);
      fe_eval_neighbor.distribute_local_to_global(dst);
    }
  }


  /*
   *  calculation of diagonal
   */
  void cell_loop_diagonal (const MatrixFree<dim,value_type>                 &data,
                           parallel::distributed::Vector<value_type>        &dst,
                           const parallel::distributed::Vector<value_type>  &,
                           const std::pair<unsigned int,unsigned int>       &cell_range) const
  {
    FEEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval(data,
                                                                 operator_data.dof_index,
                                                                 operator_data.quad_index);

    // set the correct time for the evaluation of the velocity field
    operator_data.velocity->set_time(eval_time);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);

      VectorizedArray<value_type> local_diagonal_vector[fe_eval.tensor_dofs_per_cell];
      for (unsigned int j=0; j<fe_eval.dofs_per_cell; ++j)
      {
        for (unsigned int i=0; i<fe_eval.dofs_per_cell; ++i)
          fe_eval.begin_dof_values()[i] = make_vectorized_array<value_type>(0.);
        fe_eval.begin_dof_values()[j] = make_vectorized_array<value_type>(1.);

        do_cell_integral(fe_eval);

        local_diagonal_vector[j] = fe_eval.begin_dof_values()[j];
      }
      for (unsigned int j=0; j<fe_eval.dofs_per_cell; ++j)
        fe_eval.begin_dof_values()[j] = local_diagonal_vector[j];

      fe_eval.distribute_local_to_global (dst);
    }
  }

  void face_loop_diagonal (const MatrixFree<dim,value_type>                &data,
                           parallel::distributed::Vector<value_type>       &dst,
                           const parallel::distributed::Vector<value_type> &,
                           const std::pair<unsigned int,unsigned int>      &face_range) const
  {
    // perform face intergrals for element e⁻
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval(data,true,operator_data.dof_index,operator_data.quad_index);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval.reinit (face);

      VectorizedArray<value_type> local_diagonal_vector[fe_eval.tensor_dofs_per_cell];
      for (unsigned int j=0; j<fe_eval.dofs_per_cell; ++j)
      {
        // set dof value j of element- to 1 and all other dof values of element- to zero
        for (unsigned int i=0; i<fe_eval.dofs_per_cell; ++i)
          fe_eval.begin_dof_values()[i] = make_vectorized_array<value_type>(0.);
        fe_eval.begin_dof_values()[j] = make_vectorized_array<value_type>(1.);

        fe_eval.evaluate(true,false);

        for(unsigned int q=0;q<fe_eval.n_q_points;++q)
        {
          VectorizedArray<value_type> numerical_flux = make_vectorized_array<value_type>(0.0);
          VectorizedArray<value_type> value_m = fe_eval.get_value(q);
          // set value_p to zero
          VectorizedArray<value_type> value_p = make_vectorized_array<value_type>(0.0);
          calculate_flux(numerical_flux, q, fe_eval, value_m, value_p);

          fe_eval.submit_value(numerical_flux,q);
        }
        fe_eval.integrate(true,false);

        local_diagonal_vector[j] = fe_eval.begin_dof_values()[j];
      }
      for (unsigned int j=0; j<fe_eval.dofs_per_cell; ++j)
        fe_eval.begin_dof_values()[j] = local_diagonal_vector[j];

      fe_eval.distribute_local_to_global(dst);
    }

    // TODO: this has to be removed as soon as the new infrastructure is used that
    // allows to perform face integrals over all faces of the current element
    // perform face intergrals for element e⁺
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval_neighbor(data,false,operator_data.dof_index,operator_data.quad_index);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval_neighbor.reinit (face);

      VectorizedArray<value_type> local_diagonal_vector_neighbor[fe_eval_neighbor.tensor_dofs_per_cell];
      for (unsigned int j=0; j<fe_eval_neighbor.dofs_per_cell; ++j)
      {
        // set dof value j of element+ to 1 and all other dof values of element+ to zero
        for (unsigned int i=0; i<fe_eval_neighbor.dofs_per_cell; ++i)
          fe_eval_neighbor.begin_dof_values()[i] = make_vectorized_array<value_type>(0.);
        fe_eval_neighbor.begin_dof_values()[j] = make_vectorized_array<value_type>(1.);

        fe_eval_neighbor.evaluate(true,false);

        for(unsigned int q=0;q<fe_eval.n_q_points;++q)
        {
          VectorizedArray<value_type> numerical_flux = make_vectorized_array<value_type>(0.0);
          // set value_m to zero
          VectorizedArray<value_type> value_m = make_vectorized_array<value_type>(0.0);
          VectorizedArray<value_type> value_p = fe_eval_neighbor.get_value(q);
          calculate_flux(numerical_flux, q, fe_eval_neighbor, value_m, value_p);

          // hack (minus sign) since n⁺ = -n⁻
          fe_eval_neighbor.submit_value(-numerical_flux,q);
        }
        fe_eval_neighbor.integrate(true,false);

        local_diagonal_vector_neighbor[j] = fe_eval_neighbor.begin_dof_values()[j];
      }
      for (unsigned int j=0; j<fe_eval_neighbor.dofs_per_cell; ++j)
        fe_eval_neighbor.begin_dof_values()[j] = local_diagonal_vector_neighbor[j];

      fe_eval_neighbor.distribute_local_to_global(dst);
    }
  }

  // TODO: this function has to be removed as soon as the new infrastructure is used that
  // allows to perform face integrals over all faces of the current element
  void boundary_face_loop_diagonal (const MatrixFree<dim,value_type>                &data,
                                    parallel::distributed::Vector<value_type>       &dst,
                                    const parallel::distributed::Vector<value_type> &,
                                    const std::pair<unsigned int,unsigned int>      &face_range) const
  {
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval(data,true,operator_data.dof_index,operator_data.quad_index);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      types::boundary_id boundary_id = data.get_boundary_indicator(face);
      BoundaryType boundary_type = BoundaryType::undefined;

      if(operator_data.bc->dirichlet_bc.find(boundary_id) != operator_data.bc->dirichlet_bc.end())
        boundary_type = BoundaryType::dirichlet;
      else if(operator_data.bc->neumann_bc.find(boundary_id) != operator_data.bc->neumann_bc.end())
        boundary_type = BoundaryType::neumann;

      AssertThrow(boundary_type != BoundaryType::undefined,
          ExcMessage("Boundary type of face is invalid or not implemented."));

      fe_eval.reinit (face);

      VectorizedArray<value_type> local_diagonal_vector[fe_eval.tensor_dofs_per_cell];
      for (unsigned int j=0; j<fe_eval.dofs_per_cell; ++j)
      {
        // set dof value j of element- to 1 and all other dof values of element- to zero
        for (unsigned int i=0; i<fe_eval.dofs_per_cell; ++i)
          fe_eval.begin_dof_values()[i] = make_vectorized_array<value_type>(0.);
        fe_eval.begin_dof_values()[j] = make_vectorized_array<value_type>(1.);

        fe_eval.evaluate(true,false);
        for(unsigned int q=0;q<fe_eval.n_q_points;++q)
        {
          VectorizedArray<value_type> numerical_flux = make_vectorized_array<value_type>(0.0);

          calculate_flux_boundary_face(numerical_flux, q, fe_eval, OperatorType::homogeneous, boundary_type);

          fe_eval.submit_value(numerical_flux,q);
        }
        fe_eval.integrate(true,false);

        local_diagonal_vector[j] = fe_eval.begin_dof_values()[j];
      }
      for (unsigned int j=0; j<fe_eval.dofs_per_cell; ++j)
        fe_eval.begin_dof_values()[j] = local_diagonal_vector[j];

      fe_eval.distribute_local_to_global(dst);
    }
  }

  void cell_loop_calculate_block_jacobi_matrices (const MatrixFree<dim,value_type>                 &data,
                                                  std::vector<LAPACKFullMatrix<value_type> >       &matrices,
                                                  const parallel::distributed::Vector<value_type>  &,
                                                  const std::pair<unsigned int,unsigned int>       &cell_range) const
  {
    FEEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval(data,
                                                                 operator_data.dof_index,
                                                                 operator_data.quad_index);

    // set the correct time for the evaluation of the velocity field
    operator_data.velocity->set_time(eval_time);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);

      for (unsigned int j=0; j<fe_eval.dofs_per_cell; ++j)
      {
        for (unsigned int i=0; i<fe_eval.dofs_per_cell; ++i)
          fe_eval.begin_dof_values()[i] = make_vectorized_array<value_type>(0.);
        fe_eval.begin_dof_values()[j] = make_vectorized_array<value_type>(1.);

        do_cell_integral(fe_eval);

        for(unsigned int i=0; i<fe_eval.dofs_per_cell; ++i)
          for (unsigned int v=0; v<VectorizedArray<value_type>::n_array_elements; ++v)
            matrices[cell*VectorizedArray<value_type>::n_array_elements+v](i,j) += fe_eval.begin_dof_values()[i][v];
      }
    }
  }

  void face_loop_calculate_block_jacobi_matrices (const MatrixFree<dim,value_type>                &data,
                                                  std::vector<LAPACKFullMatrix<value_type> >      &matrices,
                                                  const parallel::distributed::Vector<value_type> &,
                                                  const std::pair<unsigned int,unsigned int>      &face_range) const
  {
    // perform face intergrals for element e⁻
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval(data,true,operator_data.dof_index,operator_data.quad_index);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval.reinit (face);

      for (unsigned int j=0; j<fe_eval.dofs_per_cell; ++j)
      {
        // set dof value j of element- to 1 and all other dof values of element- to zero
        for (unsigned int i=0; i<fe_eval.dofs_per_cell; ++i)
          fe_eval.begin_dof_values()[i] = make_vectorized_array<value_type>(0.);
        fe_eval.begin_dof_values()[j] = make_vectorized_array<value_type>(1.);

        fe_eval.evaluate(true,false);

        for(unsigned int q=0;q<fe_eval.n_q_points;++q)
        {
          VectorizedArray<value_type> numerical_flux = make_vectorized_array<value_type>(0.0);
          VectorizedArray<value_type> value_m = fe_eval.get_value(q);
          // set value_p to zero
          VectorizedArray<value_type> value_p = make_vectorized_array<value_type>(0.0);
          calculate_flux(numerical_flux, q, fe_eval, value_m, value_p);

          fe_eval.submit_value(numerical_flux,q);
        }
        fe_eval.integrate(true,false);

        for (unsigned int v=0; v<VectorizedArray<value_type>::n_array_elements; ++v)
        {
          const unsigned int cell_number = data.faces[face].left_cell[v];
          if (cell_number != numbers::invalid_unsigned_int)
            for(unsigned int i=0; i<fe_eval.dofs_per_cell; ++i)
              matrices[cell_number](i,j) += fe_eval.begin_dof_values()[i][v];
        }
      }
    }

    // TODO: this has to be removed as soon as the new infrastructure is used that
    // allows to perform face integrals over all faces of the current element
    // perform face intergrals for element e⁺
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval_neighbor(data,false,operator_data.dof_index,operator_data.quad_index);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval_neighbor.reinit (face);

      for (unsigned int j=0; j<fe_eval_neighbor.dofs_per_cell; ++j)
      {
        // set dof value j of element+ to 1 and all other dof values of element+ to zero
        for (unsigned int i=0; i<fe_eval_neighbor.dofs_per_cell; ++i)
          fe_eval_neighbor.begin_dof_values()[i] = make_vectorized_array<value_type>(0.);
        fe_eval_neighbor.begin_dof_values()[j] = make_vectorized_array<value_type>(1.);

        fe_eval_neighbor.evaluate(true,false);

        for(unsigned int q=0;q<fe_eval.n_q_points;++q)
        {
          VectorizedArray<value_type> numerical_flux = make_vectorized_array<value_type>(0.0);
          // set value_m to zero
          VectorizedArray<value_type> value_m = make_vectorized_array<value_type>(0.0);
          VectorizedArray<value_type> value_p = fe_eval_neighbor.get_value(q);
          calculate_flux(numerical_flux, q, fe_eval_neighbor, value_m, value_p);

          // hack (minus sign) since n⁺ = -n⁻
          fe_eval_neighbor.submit_value(-numerical_flux,q);
        }
        fe_eval_neighbor.integrate(true,false);

        for (unsigned int v=0; v<VectorizedArray<value_type>::n_array_elements; ++v)
        {
          const unsigned int cell_number = data.faces[face].right_cell[v];
          if (cell_number != numbers::invalid_unsigned_int)
            for(unsigned int i=0; i<fe_eval_neighbor.dofs_per_cell; ++i)
              matrices[cell_number](i,j) += fe_eval_neighbor.begin_dof_values()[i][v];
        }
      }
    }
  }

  // TODO: this function has to be removed as soon as the new infrastructure is used that
  // allows to perform face integrals over all faces of the current element
  void boundary_face_loop_calculate_block_jacobi_matrices (const MatrixFree<dim,value_type>                &data,
                                                           std::vector<LAPACKFullMatrix<value_type> >      &matrices,
                                                           const parallel::distributed::Vector<value_type> &,
                                                           const std::pair<unsigned int,unsigned int>      &face_range) const
  {
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval(data,true,operator_data.dof_index,operator_data.quad_index);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      types::boundary_id boundary_id = data.get_boundary_indicator(face);
      BoundaryType boundary_type = BoundaryType::undefined;

      if(operator_data.bc->dirichlet_bc.find(boundary_id) != operator_data.bc->dirichlet_bc.end())
        boundary_type = BoundaryType::dirichlet;
      else if(operator_data.bc->neumann_bc.find(boundary_id) != operator_data.bc->neumann_bc.end())
        boundary_type = BoundaryType::neumann;

      AssertThrow(boundary_type != BoundaryType::undefined,
          ExcMessage("Boundary type of face is invalid or not implemented."));

      fe_eval.reinit (face);

      for (unsigned int j=0; j<fe_eval.dofs_per_cell; ++j)
      {
        // set dof value j of element- to 1 and all other dof values of element- to zero
        for (unsigned int i=0; i<fe_eval.dofs_per_cell; ++i)
          fe_eval.begin_dof_values()[i] = make_vectorized_array<value_type>(0.);
        fe_eval.begin_dof_values()[j] = make_vectorized_array<value_type>(1.);

        fe_eval.evaluate(true,false);
        for(unsigned int q=0;q<fe_eval.n_q_points;++q)
        {
          VectorizedArray<value_type> numerical_flux = make_vectorized_array<value_type>(0.0);

          calculate_flux_boundary_face(numerical_flux, q, fe_eval, OperatorType::homogeneous, boundary_type);

          fe_eval.submit_value(numerical_flux,q);
        }
        fe_eval.integrate(true,false);

        for (unsigned int v=0; v<VectorizedArray<value_type>::n_array_elements; ++v)
        {
          const unsigned int cell_number = data.faces[face].left_cell[v];
          if (cell_number != numbers::invalid_unsigned_int)
            for(unsigned int i=0; i<fe_eval.dofs_per_cell; ++i)
              matrices[cell_number](i,j) += fe_eval.begin_dof_values()[i][v];
        }
      }
    }
  }

  /*
   *  evaluate boundary face integrals for full operator (homogeneous + inhomogeneous parts)
   */
  void boundary_face_loop_full_operator (const MatrixFree<dim,value_type>                &data,
                                         parallel::distributed::Vector<value_type>       &dst,
                                         const parallel::distributed::Vector<value_type> &src,
                                         const std::pair<unsigned int,unsigned int>      &face_range) const
  {
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval(data,true,operator_data.dof_index,operator_data.quad_index);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      types::boundary_id boundary_id = data.get_boundary_indicator(face);
      BoundaryType boundary_type = BoundaryType::undefined;

      if(operator_data.bc->dirichlet_bc.find(boundary_id) != operator_data.bc->dirichlet_bc.end())
        boundary_type = BoundaryType::dirichlet;
      else if(operator_data.bc->neumann_bc.find(boundary_id) != operator_data.bc->neumann_bc.end())
        boundary_type = BoundaryType::neumann;

      AssertThrow(boundary_type != BoundaryType::undefined,
          ExcMessage("Boundary type of face is invalid or not implemented."));

      fe_eval.reinit (face);
      fe_eval.read_dof_values(src);
      fe_eval.evaluate(true,false);

      for(unsigned int q=0;q<fe_eval.n_q_points;++q)
      {
        VectorizedArray<value_type> numerical_flux = make_vectorized_array<value_type>(0.0);

        calculate_flux_boundary_face(numerical_flux, q, fe_eval, OperatorType::full, boundary_type, boundary_id);

        fe_eval.submit_value(numerical_flux,q);
      }

      fe_eval.integrate(true,false);
      fe_eval.distribute_local_to_global(dst);
    }
  }

  /*
   *  Evaluate inhomogeneous operator. Note that these integrals are multiplied by
   *  a factor of -1.0 since these integrals appear on the right-hand side of the equations.
   */
  void cell_loop_inhom_operator (const MatrixFree<dim,value_type>                 &,
                                 parallel::distributed::Vector<value_type>        &,
                                 const parallel::distributed::Vector<value_type>  &,
                                 const std::pair<unsigned int,unsigned int>       &) const
  {}

  void face_loop_inhom_operator (const MatrixFree<dim,value_type>                &,
                                 parallel::distributed::Vector<value_type>       &,
                                 const parallel::distributed::Vector<value_type> &,
                                 const std::pair<unsigned int,unsigned int>      &) const
  {}

  void boundary_face_loop_inhom_operator (const MatrixFree<dim,value_type>                &data,
                                          parallel::distributed::Vector<value_type>       &dst,
                                          const parallel::distributed::Vector<value_type> &,
                                          const std::pair<unsigned int,unsigned int>      &face_range) const
  {
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval(data,true,operator_data.dof_index,operator_data.quad_index);

    // set the correct time for the evaluation of the velocity field
    operator_data.velocity->set_time(eval_time);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      types::boundary_id boundary_id = data.get_boundary_indicator(face);
      BoundaryType boundary_type = BoundaryType::undefined;

      if(operator_data.bc->dirichlet_bc.find(boundary_id) != operator_data.bc->dirichlet_bc.end())
        boundary_type = BoundaryType::dirichlet;
      else if(operator_data.bc->neumann_bc.find(boundary_id) != operator_data.bc->neumann_bc.end())
        boundary_type = BoundaryType::neumann;

      AssertThrow(boundary_type != BoundaryType::undefined,
          ExcMessage("Boundary type of face is invalid or not implemented."));

      fe_eval.reinit (face);

      for(unsigned int q=0;q<fe_eval.n_q_points;++q)
      {
        VectorizedArray<value_type> numerical_flux = make_vectorized_array<value_type>(0.0);

        calculate_flux_boundary_face(numerical_flux, q, fe_eval, OperatorType::inhomogeneous, boundary_type, boundary_id);

        // -numerical_flux since this term appears on the rhs of the equation !!
        fe_eval.submit_value(-numerical_flux,q);
      }

      fe_eval.integrate(true,false);
      fe_eval.distribute_local_to_global(dst);
    }
  }

  MatrixFree<dim,value_type> const * data;
  ConvectiveOperatorData<dim> operator_data;
  mutable value_type eval_time;
};

template<int dim>
struct ConvectiveOperatorDataDiscontinuousVelocity
{
  ConvectiveOperatorDataDiscontinuousVelocity ()
    :
    dof_index(0),
    dof_index_velocity(0),
    quad_index(0)
  {}

  unsigned int dof_index;
  unsigned int dof_index_velocity;
  unsigned int quad_index;

  std_cxx11::shared_ptr<BoundaryDescriptorConvDiff<dim> > bc;
};

template <int dim, int fe_degree, int fe_degree_velocity, typename value_type>
class ConvectiveOperatorDiscontinuousVelocity
{
public:
  typedef ConvectiveOperatorDiscontinuousVelocity<dim,fe_degree, fe_degree_velocity, value_type> This;

  ConvectiveOperatorDiscontinuousVelocity()
    :
    data(nullptr),
    velocity(nullptr)
  {}

  void initialize(MatrixFree<dim,value_type> const                       &mf_data,
                  ConvectiveOperatorDataDiscontinuousVelocity<dim> const &operator_data_in)
  {
    this->data = &mf_data;
    this->operator_data = operator_data_in;
  }

  // apply matrix vector multiplication
  void apply (parallel::distributed::Vector<value_type>       &dst,
              parallel::distributed::Vector<value_type> const &src,
              parallel::distributed::Vector<value_type> const *vector) const
  {
    dst = 0;
    apply_add(dst,src,vector);
  }

  void apply_add (parallel::distributed::Vector<value_type>       &dst,
                  const parallel::distributed::Vector<value_type> &src,
                  parallel::distributed::Vector<value_type> const *vector) const
  {
    velocity = vector;

    data->loop(&This::cell_loop,&This::face_loop,&This::boundary_face_loop_hom_operator,this, dst, src);

    velocity = nullptr;
  }

private:
  void cell_loop (const MatrixFree<dim,value_type>                 &data,
                  parallel::distributed::Vector<value_type>        &dst,
                  const parallel::distributed::Vector<value_type>  &src,
                  const std::pair<unsigned int,unsigned int>       &cell_range) const
  {
    FEEvaluation<dim,fe_degree,fe_degree+1,1,value_type>
      fe_eval(data, operator_data.dof_index, operator_data.quad_index);

    FEEvaluation<dim,fe_degree_velocity,fe_degree+1,dim,value_type>
      fe_eval_velocity(data, operator_data.dof_index_velocity, operator_data.quad_index);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);
      fe_eval.read_dof_values(src);
      fe_eval.evaluate(true,false,false);

      fe_eval_velocity.reinit(cell);
      fe_eval_velocity.read_dof_values(*velocity);
      fe_eval_velocity.evaluate(true,false,false);

      for (unsigned int q=0; q<fe_eval.n_q_points; ++q)
      {
        fe_eval.submit_gradient(-fe_eval.get_value(q)*fe_eval_velocity.get_value(q),q);
      }
      fe_eval.integrate (false,true);
      fe_eval.distribute_local_to_global (dst);
    }
  }

  void face_loop (const MatrixFree<dim,value_type>                &data,
                  parallel::distributed::Vector<value_type>       &dst,
                  const parallel::distributed::Vector<value_type> &src,
                  const std::pair<unsigned int,unsigned int>      &face_range) const
  {
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,1,value_type>
      fe_eval(data,true,operator_data.dof_index,operator_data.quad_index);
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,1,value_type>
      fe_eval_neighbor(data,false,operator_data.dof_index,operator_data.quad_index);

    FEFaceEvaluation<dim,fe_degree_velocity,fe_degree+1,dim,value_type>
      fe_eval_velocity(data,true,operator_data.dof_index_velocity,operator_data.quad_index);
    FEFaceEvaluation<dim,fe_degree_velocity,fe_degree+1,dim,value_type>
      fe_eval_velocity_neighbor(data,false,operator_data.dof_index_velocity,operator_data.quad_index);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval.reinit (face);
      fe_eval.read_dof_values(src);
      fe_eval.evaluate(true,false);

      fe_eval_neighbor.reinit (face);
      fe_eval_neighbor.read_dof_values(src);
      fe_eval_neighbor.evaluate(true,false);

      fe_eval_velocity.reinit(face);
      fe_eval_velocity.read_dof_values(*velocity);
      fe_eval_velocity.evaluate(true,false);

      fe_eval_velocity_neighbor.reinit(face);
      fe_eval_velocity_neighbor.read_dof_values(*velocity);
      fe_eval_velocity_neighbor.evaluate(true,false);

      for(unsigned int q=0;q<fe_eval.n_q_points;++q)
      {
        Tensor<1,dim,VectorizedArray<value_type> > velocity_m = fe_eval_velocity.get_value(q);
        Tensor<1,dim,VectorizedArray<value_type> > velocity_p = fe_eval_velocity_neighbor.get_value(q);

        Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval.get_normal_vector(q);
        VectorizedArray<value_type> velocity_m_normal = velocity_m*normal;
        VectorizedArray<value_type> velocity_p_normal = velocity_p*normal;

        VectorizedArray<value_type> value_m = fe_eval.get_value(q);
        VectorizedArray<value_type> value_p = fe_eval_neighbor.get_value(q);
        VectorizedArray<value_type> average_value = 0.5*(velocity_m_normal*value_m + velocity_p_normal*value_p);
        VectorizedArray<value_type> jump_value = value_m - value_p;
        VectorizedArray<value_type> lambda = std::max(std::abs(velocity_m_normal),std::abs(velocity_p_normal));
        VectorizedArray<value_type> lf_flux = average_value + 0.5*lambda*jump_value;

        fe_eval.submit_value(lf_flux,q);
        fe_eval_neighbor.submit_value(-lf_flux,q);
      }
      fe_eval.integrate(true,false);
      fe_eval.distribute_local_to_global(dst);

      fe_eval_neighbor.integrate(true,false);
      fe_eval_neighbor.distribute_local_to_global(dst);
    }
  }

  void boundary_face_loop_hom_operator (const MatrixFree<dim,value_type>                &data,
                                        parallel::distributed::Vector<value_type>       &dst,
                                        const parallel::distributed::Vector<value_type> &src,
                                        const std::pair<unsigned int,unsigned int>      &face_range) const
  {
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,1,value_type>
      fe_eval(data,true,operator_data.dof_index,operator_data.quad_index);

    FEFaceEvaluation<dim,fe_degree_velocity,fe_degree+1,dim,value_type>
      fe_eval_velocity(data,true,operator_data.dof_index_velocity,operator_data.quad_index);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval.reinit (face);
      fe_eval.read_dof_values(src);
      fe_eval.evaluate(true,false);

      fe_eval_velocity.reinit(face);
      fe_eval_velocity.read_dof_values(*velocity);
      fe_eval_velocity.evaluate(true,false);

      typename std::map<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >::iterator it;
      types::boundary_id boundary_id = data.get_boundary_indicator(face);

      for(unsigned int q=0;q<fe_eval.n_q_points;++q)
      {
        Tensor<1,dim,VectorizedArray<value_type> > velocity_m = fe_eval_velocity.get_value(q);

        Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval.get_normal_vector(q);
        VectorizedArray<value_type> velocity_m_normal = velocity_m*normal;

        VectorizedArray<value_type> value_m = fe_eval.get_value(q);
        VectorizedArray<value_type> value_p = make_vectorized_array<value_type>(0.0);

        it = operator_data.bc->dirichlet_bc.find(boundary_id);
        if(it != operator_data.bc->dirichlet_bc.end())
        {
          // on GammaD: phi⁺ = -phi⁻ + 2g -> {{phi}} = g, [phi] = 2 phi⁻ - 2 g
          // homogeneous part: phi⁺ = -phi⁻ -> {{phi}} = 0, [phi] = 2 phi⁻
          // inhomogenous part: phi⁺ = 2g -> {{phi}} = g, [phi] = -2 g
          value_p = - value_m;
        }
        it = operator_data.bc->neumann_bc.find(boundary_id);
        if(it != operator_data.bc->neumann_bc.end())
        {
          // on GammaN: phi⁺ = phi⁻-> {{phi}} = phi⁻, [phi] = 0
          // homogeneous part: phi⁺ = phi⁻ -> {{phi}} = phi⁻, [phi] = 0
          // inhomogenous part: phi⁺ = 0 -> {{phi}} = 0, [phi] = 0
          value_p = value_m;
        }

        VectorizedArray<value_type> average_value = 0.5*(value_m + value_p)*velocity_m_normal;
        VectorizedArray<value_type> jump_value = value_m - value_p;
        VectorizedArray<value_type> lambda = std::abs(velocity_m_normal);
        VectorizedArray<value_type> lf_flux = average_value + 0.5*lambda*jump_value;

        fe_eval.submit_value(lf_flux,q);
      }
      fe_eval.integrate(true,false);
      fe_eval.distribute_local_to_global(dst);
    }
  }

  MatrixFree<dim,value_type> const * data;
  ConvectiveOperatorDataDiscontinuousVelocity<dim> operator_data;
  mutable parallel::distributed::Vector<value_type> const * velocity;
};

template<int dim>
struct HelmholtzOperatorData
{
  HelmholtzOperatorData ()
    :
    unsteady_problem(true),
    dof_index(0)
  {}

  bool unsteady_problem;

  unsigned int dof_index;
};

#include "MatrixOperatorBase.h"

template <int dim, int fe_degree, typename Number = double>
class HelmholtzOperator : public MatrixOperatorBase
{
public:
  typedef Number value_type;

  HelmholtzOperator()
    :
    data(nullptr),
    mass_matrix_operator(nullptr),
    diffusive_operator(nullptr),
    scaling_factor_time_derivative_term(-1.0)
  {}

  void initialize(MatrixFree<dim,Number> const                      &mf_data_in,
                  HelmholtzOperatorData<dim> const                  &helmholtz_operator_data_in,
                  MassMatrixOperator<dim, fe_degree, Number>  const &mass_matrix_operator_in,
                  DiffusiveOperator<dim, fe_degree, Number> const   &diffusive_operator_in)
  {
    // copy parameters into element variables
    this->data = &mf_data_in;
    this->helmholtz_operator_data = helmholtz_operator_data_in;
    this->mass_matrix_operator = &mass_matrix_operator_in;
    this->diffusive_operator = &diffusive_operator_in;
  }


  /*
   *  This function is called by the multigrid algorithm to initialize the
   *  matrices on all levels. To construct the matrices, and object of
   *  type UnderlyingOperator is used that provides all the information for
   *  the setup, i.e., the information that is needed to call the
   *  member function initialize(...).
   */
  template<typename UnderlyingOperator>
  void initialize_mg_matrix (unsigned int const                              level,
                             DoFHandler<dim> const                           &dof_handler,
                             Mapping<dim> const                              &mapping,
                             UnderlyingOperator const                        &underlying_operator,
                             const std::vector<GridTools::PeriodicFacePair<
                               typename Triangulation<dim>::cell_iterator> > &periodic_face_pairs_level0)
  {
    // setup own matrix free object
    const QGauss<1> quad(dof_handler.get_fe().degree+1);
    typename MatrixFree<dim,Number>::AdditionalData addit_data;
    addit_data.tasks_parallel_scheme = MatrixFree<dim,Number>::AdditionalData::none;
    if (dof_handler.get_fe().dofs_per_vertex == 0)
      addit_data.build_face_info = true;

    // TODO
//    addit_data.mapping_update_flags = (update_gradients | update_JxW_values |
//                                       update_quadrature_points | update_normal_vectors |
//                                       update_values);

    addit_data.level_mg_handler = level;
    addit_data.mpi_communicator =
      dynamic_cast<const parallel::Triangulation<dim> *>(&dof_handler.get_triangulation()) ?
      (dynamic_cast<const parallel::Triangulation<dim> *>(&dof_handler.get_triangulation()))->get_communicator() : MPI_COMM_SELF;
    addit_data.periodic_face_pairs_level_0 = periodic_face_pairs_level0;

    ConstraintMatrix constraints;
    // reinit
    own_matrix_free_storage.reinit(mapping, dof_handler, constraints, quad, addit_data);

    // setup own mass matrix operator
    MassMatrixOperatorData mass_matrix_operator_data = underlying_operator.get_mass_matrix_operator_data();
    mass_matrix_operator_data.dof_index = 0;
    mass_matrix_operator_data.quad_index = 0;
    own_mass_matrix_operator_storage.initialize(own_matrix_free_storage,mass_matrix_operator_data);

    // setup own viscous operator
    DiffusiveOperatorData<dim> diffusive_operator_data = underlying_operator.get_diffusive_operator_data();
    diffusive_operator_data.dof_index = 0;
    diffusive_operator_data.quad_index = 0;
    own_diffusive_operator_storage.initialize(mapping,own_matrix_free_storage,diffusive_operator_data);

    // setup Helmholtz operator
    HelmholtzOperatorData<dim> operator_data = underlying_operator.get_helmholtz_operator_data();
    initialize(own_matrix_free_storage, operator_data, own_mass_matrix_operator_storage, own_diffusive_operator_storage);

    // Initialize other variables:

    // mass matrix term: set scaling factor time derivative term
    set_scaling_factor_time_derivative_term(underlying_operator.get_scaling_factor_time_derivative_term());

    // initialize temp vector: this is done in this function because
    // the vector temp is only used in the function vmult_add(), i.e.,
    // when using the multigrid preconditioner
    initialize_dof_vector(temp);
  }

  /*
   *  Scaling factor of time derivative term (mass matrix term)
   */
  void set_scaling_factor_time_derivative_term(double const &factor)
  {
    scaling_factor_time_derivative_term = factor;
  }

  double get_scaling_factor_time_derivative_term() const
  {
    return scaling_factor_time_derivative_term;
  }

  /*
   *  Operator data
   */
  HelmholtzOperatorData<dim> const & get_helmholtz_operator_data() const
  {
    return this->helmholtz_operator_data;
  }

  /*
   *  Operator data of basic operators: mass matrix, diffusive operator
   */
  MassMatrixOperatorData const & get_mass_matrix_operator_data() const
  {
    return mass_matrix_operator->get_operator_data();
  }

  DiffusiveOperatorData<dim> const & get_diffusive_operator_data() const
  {
    return diffusive_operator->get_operator_data();
  }

  /*
   *  MatrixFree data
   */
  MatrixFree<dim,value_type> const & get_data() const
  {
    return *data;
  }

  /*
   *  This function does nothing in case of the HelmholtzOperator.
   *  IT is only necessary due to the interface of the multigrid preconditioner
   *  and especially the coarse grid solver that calls this function.
   */
  void apply_nullspace_projection(parallel::distributed::Vector<Number> &/*vec*/) const {}

  // apply matrix vector multiplication
  void vmult (parallel::distributed::Vector<Number>       &dst,
              const parallel::distributed::Vector<Number> &src) const
  {
    // helmholtz operator = mass_matrix_operator + viscous_operator
    if(helmholtz_operator_data.unsteady_problem == true)
    {
      AssertThrow(scaling_factor_time_derivative_term > 0.0,
        ExcMessage("Scaling factor of time derivative term has not been initialized for Helmholtz operator!"));

      mass_matrix_operator->apply(dst,src);
      dst *= scaling_factor_time_derivative_term;
    }
    else
    {
      dst = 0.0;
    }

    diffusive_operator->apply_add(dst,src);
  }

  void vmult_add(parallel::distributed::Vector<Number>       &dst,
                 const parallel::distributed::Vector<Number> &src) const
  {
    // helmholtz operator = mass_matrix_operator + viscous_operator
    if(helmholtz_operator_data.unsteady_problem == true)
    {
      AssertThrow(scaling_factor_time_derivative_term > 0.0,
        ExcMessage("Scaling factor of time derivative term has not been initialized for Helmholtz operator!"));

      mass_matrix_operator->apply(temp,src);
      temp *= scaling_factor_time_derivative_term;
      dst += temp;
    }

    diffusive_operator->apply_add(dst,src);
  }

  void vmult_interface_down(parallel::distributed::Vector<Number>       &dst,
                            const parallel::distributed::Vector<Number> &src) const
  {
    vmult(dst,src);
  }

  void vmult_add_interface_up(parallel::distributed::Vector<Number>       &dst,
                              const parallel::distributed::Vector<Number> &src) const
  {
    vmult_add(dst,src);
  }

  types::global_dof_index m() const
  {
    return data->get_vector_partitioner(helmholtz_operator_data.dof_index)->size();
  }

  Number el (const unsigned int,  const unsigned int) const
  {
    AssertThrow(false, ExcMessage("Matrix-free does not allow for entry access"));
    return Number();
  }

  /*
   *  This function initializes a global dof-vector.
   */
  void initialize_dof_vector(parallel::distributed::Vector<Number> &vector) const
  {
    data->initialize_dof_vector(vector,helmholtz_operator_data.dof_index);
  }

  /*
   *  Calculation of inverse diagonal (needed for smoothers and preconditioners)
   */
  void calculate_inverse_diagonal(parallel::distributed::Vector<Number> &diagonal) const
  {
    calculate_diagonal(diagonal);

    // verify_calculation_of_diagonal(*this,diagonal);

    invert_diagonal(diagonal);
  }

  /*
   *  Apply block Jacobi preconditioner
   */
  void apply_block_jacobi (parallel::distributed::Vector<Number>       &dst,
                           parallel::distributed::Vector<Number> const &src) const
  {
    AssertThrow(false,ExcMessage("Block Jacobi preconditioner not implemented for scalar reaction-diffusion operator"));
  }

  /*
   *  Update block Jacobi preconditioner
   */
  void update_block_jacobi () const
  {
    AssertThrow(false,ExcMessage("Function update_block_jacobi() has not been implemented."));
  }

private:
  /*
   *  This function calculates the diagonal of the discrete operator representing the
   *  scalar reaction-diffusion operator.
   */
  void calculate_diagonal(parallel::distributed::Vector<Number> &diagonal) const
  {
    if(helmholtz_operator_data.unsteady_problem == true)
    {
      AssertThrow(scaling_factor_time_derivative_term > 0.0,
        ExcMessage("Scaling factor of time derivative term has not been initialized for Helmholtz operator!"));

      mass_matrix_operator->calculate_diagonal(diagonal);
      diagonal *= scaling_factor_time_derivative_term;
    }
    else
    {
      diagonal = 0.0;
    }

    diffusive_operator->add_diagonal(diagonal);
  }

  MatrixFree<dim,Number> const * data;
  MassMatrixOperator<dim, fe_degree, Number>  const *mass_matrix_operator;
  DiffusiveOperator<dim, fe_degree, Number>  const *diffusive_operator;
  HelmholtzOperatorData<dim> helmholtz_operator_data;
  parallel::distributed::Vector<Number> mutable temp;
  double scaling_factor_time_derivative_term;

  /*
   * The following variables are necessary when applying the multigrid
   * preconditioner to the Helmholtz operator. In that case, the
   * Helmholtz has to be generated for each level of the multigrid algorithm.
   * Accordingly, in a first step one has to setup own objects of
   * MatrixFree, MassMatrixOperator, DiffusiveOperator,
   *   e.g., own_matrix_free_storage.reinit(...);
   * and later initialize the HelmholtzOperator with these
   * ojects by setting the above pointers to the own_objects_storage,
   *   e.g., data = &own_matrix_free_storage;
   */
  MatrixFree<dim,Number> own_matrix_free_storage;
  MassMatrixOperator<dim, fe_degree, Number> own_mass_matrix_operator_storage;
  DiffusiveOperator<dim, fe_degree, Number> own_diffusive_operator_storage;
};




template<typename UnderlyingOperator, typename Number>
class ConvectionDiffusionBlockJacobiOperator
{
public:
  ConvectionDiffusionBlockJacobiOperator(UnderlyingOperator const &underlying_operator_in)
    : underlying_operator(underlying_operator_in)
  {}

  void vmult (parallel::distributed::Vector<Number>       &dst,
              const parallel::distributed::Vector<Number> &src) const
  {
    underlying_operator.vmult_block_jacobi(dst,src);
  }

private:
  UnderlyingOperator const &underlying_operator;
};


template<int dim>
struct ConvectionDiffusionOperatorData
{
  ConvectionDiffusionOperatorData()
    :
    unsteady_problem(true),
    convective_problem(true),
    diffusive_problem(true),
    dof_index(0)
  {}

  bool unsteady_problem;
  bool convective_problem;
  bool diffusive_problem;
  unsigned int dof_index;
};

template <int dim, int fe_degree, typename Number = double>
class ConvectionDiffusionOperator : public MatrixOperatorBase
{
public:
  typedef Number value_type;

  ConvectionDiffusionOperator()
    :
    block_jacobi_matrices_have_been_initialized(false),
    data(nullptr),
    mass_matrix_operator(nullptr),
    convective_operator(nullptr),
    diffusive_operator(nullptr),
    scaling_factor_time_derivative_term(-1.0),
    evaluation_time(0.0)
  {}

  void initialize(MatrixFree<dim,Number> const                      &mf_data_in,
                  ConvectionDiffusionOperatorData<dim> const        &operator_data_in,
                  MassMatrixOperator<dim, fe_degree, Number>  const &mass_matrix_operator_in,
                  ConvectiveOperator<dim, fe_degree, Number> const  &convective_operator_in,
                  DiffusiveOperator<dim, fe_degree, Number> const   &diffusive_operator_in)
  {
    // copy parameters into element variables
    this->data = &mf_data_in;
    this->operator_data = operator_data_in;
    this->mass_matrix_operator = &mass_matrix_operator_in;
    this->convective_operator = &convective_operator_in;
    this->diffusive_operator = &diffusive_operator_in;
  }


  /*
   *  This function is called by the multigrid algorithm to initialize the
   *  matrices on all levels. To construct the matrices, and object of
   *  type UnderlyingOperator is used that provides all the information for
   *  the setup, i.e., the information that is needed to call the
   *  member function initialize(...).
   */
  template<typename UnderlyingOperator>
  void initialize_mg_matrix (unsigned int const                              level,
                             DoFHandler<dim> const                           &dof_handler,
                             Mapping<dim> const                              &mapping,
                             UnderlyingOperator const                        &underlying_operator,
                             const std::vector<GridTools::PeriodicFacePair<
                               typename Triangulation<dim>::cell_iterator> > &periodic_face_pairs_level0)
  {
    // setup own matrix free object
    const QGauss<1> quad(dof_handler.get_fe().degree+1);
    typename MatrixFree<dim,Number>::AdditionalData addit_data;
    addit_data.tasks_parallel_scheme = MatrixFree<dim,Number>::AdditionalData::none;
    if (dof_handler.get_fe().dofs_per_vertex == 0)
      addit_data.build_face_info = true;
    // TODO
    addit_data.mapping_update_flags = (update_gradients | update_JxW_values |
                                       update_quadrature_points | update_normal_vectors |
                                       update_values);

    addit_data.level_mg_handler = level;
    addit_data.mpi_communicator =
      dynamic_cast<const parallel::Triangulation<dim> *>(&dof_handler.get_triangulation()) ?
      (dynamic_cast<const parallel::Triangulation<dim> *>(&dof_handler.get_triangulation()))->get_communicator() : MPI_COMM_SELF;
    addit_data.periodic_face_pairs_level_0 = periodic_face_pairs_level0;

    ConstraintMatrix constraints;
    // reinit
    own_matrix_free_storage.reinit(mapping, dof_handler, constraints, quad, addit_data);

    // setup own mass matrix operator
    MassMatrixOperatorData mass_matrix_operator_data = underlying_operator.get_mass_matrix_operator_data();
    mass_matrix_operator_data.dof_index = 0;
    mass_matrix_operator_data.quad_index = 0;
    own_mass_matrix_operator_storage.initialize(own_matrix_free_storage,mass_matrix_operator_data);

    // setup own convective operator
    ConvectiveOperatorData<dim> convective_operator_data = underlying_operator.get_convective_operator_data();
    convective_operator_data.dof_index = 0;
    convective_operator_data.quad_index = 0;
    own_convective_operator_storage.initialize(own_matrix_free_storage,convective_operator_data);

    // setup own viscous operator
    DiffusiveOperatorData<dim> diffusive_operator_data = underlying_operator.get_diffusive_operator_data();
    diffusive_operator_data.dof_index = 0;
    diffusive_operator_data.quad_index = 0;
    own_diffusive_operator_storage.initialize(mapping,own_matrix_free_storage,diffusive_operator_data);

    // setup convection-diffusion operator
    ConvectionDiffusionOperatorData<dim> my_operator_data = underlying_operator.get_convection_diffusion_operator_data();
    initialize(own_matrix_free_storage,
               my_operator_data,
               own_mass_matrix_operator_storage,
               own_convective_operator_storage,
               own_diffusive_operator_storage);

    // Initialize other variables:

    // mass matrix term: set scaling factor time derivative term
    set_scaling_factor_time_derivative_term(underlying_operator.get_scaling_factor_time_derivative_term());

    // convective term: evaluation_time
    // This variables is not set here. If the convective term
    // is considered, this variables has to be updated anyway,
    // which is done somewhere else.

    // viscous term:

    // initialize temp vector: this is done in this function because
    // the vector temp is only used in the function vmult_add(), i.e.,
    // when using the multigrid preconditioner
    initialize_dof_vector(temp);
  }

  /*
   *  Scaling factor of time derivative term (mass matrix term)
   */
  void set_scaling_factor_time_derivative_term(double const &factor)
  {
    scaling_factor_time_derivative_term = factor;
  }

  double get_scaling_factor_time_derivative_term() const
  {
    return scaling_factor_time_derivative_term;
  }

  /*
   *  Evaluation time that is needed for evaluation of convective operator.
   */
  void set_evaluation_time(double const &evaluation_time_in)
  {
    evaluation_time = evaluation_time_in;
  }

  double get_evaluation_time() const
  {
    return evaluation_time;
  }

  /*
   *  Operator data
   */
  ConvectionDiffusionOperatorData<dim> const & get_convection_diffusion_operator_data() const
  {
    return this->operator_data;
  }

  /*
   *  Helmholtz operator data
   */
  HelmholtzOperatorData<dim> const get_helmholtz_operator_data() const
  {
    ScalarConvDiffOperators::HelmholtzOperatorData<dim> helmholtz_operator_data;
    helmholtz_operator_data.unsteady_problem = this->operator_data.unsteady_problem;
    helmholtz_operator_data.dof_index = 0;

    return helmholtz_operator_data;
  }

  /*
   *  Operator data of basic operators: mass matrix, convective operator, diffusive operator
   */
  MassMatrixOperatorData const & get_mass_matrix_operator_data() const
  {
    return mass_matrix_operator->get_operator_data();
  }

  ConvectiveOperatorData<dim> const & get_convective_operator_data() const
  {
    return convective_operator->get_operator_data();
  }

  DiffusiveOperatorData<dim> const & get_diffusive_operator_data() const
  {
    return diffusive_operator->get_operator_data();
  }

  /*
   *  MatrixFree data
   */
  MatrixFree<dim,value_type> const & get_data() const
  {
    return *data;
  }

  /*
   *  This function does nothing in case of the ConvectionDiffusionOperator.
   *  IT is only necessary due to the interface of the multigrid preconditioner
   *  and especially the coarse grid solver that calls this function.
   */
  void apply_nullspace_projection(parallel::distributed::Vector<Number> &/*vec*/) const {}

  // apply matrix vector multiplication
  void vmult (parallel::distributed::Vector<Number>       &dst,
              const parallel::distributed::Vector<Number> &src) const
  {
    if(operator_data.unsteady_problem == true)
    {
      AssertThrow(scaling_factor_time_derivative_term > 0.0,
        ExcMessage("Scaling factor of time derivative term has not been initialized for convection-diffusion operator!"));

      mass_matrix_operator->apply(dst,src);
      dst *= scaling_factor_time_derivative_term;
    }
    else
    {
      dst = 0.0;
    }

    if(operator_data.diffusive_problem == true)
    {
      diffusive_operator->apply_add(dst,src);
    }

    if(operator_data.convective_problem == true)
    {
      convective_operator->apply_add(dst,src,evaluation_time);
    }
  }

  void vmult_add(parallel::distributed::Vector<Number>       &dst,
                 const parallel::distributed::Vector<Number> &src) const
  {
    if(operator_data.unsteady_problem == true)
    {
      AssertThrow(scaling_factor_time_derivative_term > 0.0,
        ExcMessage("Scaling factor of time derivative term has not been initialized for convection-diffusion operator!"));

      mass_matrix_operator->apply(temp,src);
      temp *= scaling_factor_time_derivative_term;
      dst += temp;
    }

    if(operator_data.diffusive_problem == true)
    {
      diffusive_operator->apply_add(dst,src);
    }

    if(operator_data.convective_problem == true)
    {
      convective_operator->apply_add(dst,src,evaluation_time);
    }
  }

  // Apply matrix vector multiplication for global block Jacobi system.
  // Do that sequentially for the different operators.
  // This function is only needed when solving the global block Jacobi problem
  // iteratively in which case the function vmult_block_jacobi() represents
  // the "vmult()" operation of the linear system of equations.
  void vmult_block_jacobi (parallel::distributed::Vector<Number>       &dst,
                           const parallel::distributed::Vector<Number> &src) const
  {
    if(operator_data.unsteady_problem == true)
    {
      AssertThrow(scaling_factor_time_derivative_term > 0.0,
        ExcMessage("Scaling factor of time derivative term has not been initialized for convection-diffusion operator!"));

      // mass matrix operator has already "block Jacobi form" in DG
      mass_matrix_operator->apply(dst,src);
      dst *= scaling_factor_time_derivative_term;
    }
    else
    {
      dst = 0.0;
    }

    if(operator_data.diffusive_problem == true)
    {
      diffusive_operator->apply_block_jacobi_add(dst,src);
    }

    if(operator_data.convective_problem == true)
    {
      convective_operator->apply_block_jacobi_add(dst,src,evaluation_time);
    }
  }


  // TODO only needed for testing
  void vmult_block_jacobi_2 (parallel::distributed::Vector<Number>       &dst,
                           const parallel::distributed::Vector<Number> &src) const
  {
    data->cell_loop(&ConvectionDiffusionOperator<dim,fe_degree,Number>::vmult_block_jacobi_matrices, this, dst, src);
  }

  // TODO only needed for testing
  void vmult_block_jacobi_matrices (const MatrixFree<dim,value_type>                 &data,
                                    parallel::distributed::Vector<value_type>        &dst,
                                    const parallel::distributed::Vector<value_type>  &src,
                                    const std::pair<unsigned int,unsigned int>       &cell_range) const
  {
    FEEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval(data,
                                                                 mass_matrix_operator->get_operator_data().dof_index,
                                                                 mass_matrix_operator->get_operator_data().quad_index);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);
      fe_eval.read_dof_values(src);

      for (unsigned int v=0; v<VectorizedArray<value_type>::n_array_elements; ++v)
      {
        // fill source vector
        Vector<value_type> src_vector(fe_eval.dofs_per_cell);
        Vector<value_type> dst_vector(fe_eval.dofs_per_cell);
        for (unsigned int j=0; j<fe_eval.dofs_per_cell; ++j)
          src_vector(j) = fe_eval.begin_dof_values()[j][v];

        // apply matrix-vector product
        matrices[cell*VectorizedArray<value_type>::n_array_elements+v].vmult(dst_vector,src_vector,false);

        // write solution to dst-vector
        for (unsigned int j=0; j<fe_eval.dofs_per_cell; ++j)
          fe_eval.begin_dof_values()[j][v] = dst_vector(j);
      }

      fe_eval.set_dof_values (dst);
    }
  }

  void vmult_interface_down(parallel::distributed::Vector<Number>       &dst,
                            const parallel::distributed::Vector<Number> &src) const
  {
    vmult(dst,src);
  }

  void vmult_add_interface_up(parallel::distributed::Vector<Number>       &dst,
                              const parallel::distributed::Vector<Number> &src) const
  {
    vmult_add(dst,src);
  }

  types::global_dof_index m() const
  {
    return data->get_vector_partitioner(operator_data.dof_index)->size();
  }

  Number el (const unsigned int,  const unsigned int) const
  {
    AssertThrow(false, ExcMessage("Matrix-free does not allow for entry access"));
    return Number();
  }

  unsigned int get_dof_index() const
  {
    return operator_data.dof_index;
  }

  unsigned int get_quad_index() const
  {
    // Operator data does not contain quad_index. Hence,
    // ask one of the basic operators (here we choose the mass matrix operator)
    // for the quadrature index.
    return get_mass_matrix_operator_data().quad_index;
  }

  /*
   *  This function initializes a global dof-vector.
   */
  void initialize_dof_vector(parallel::distributed::Vector<Number> &vector) const
  {
    data->initialize_dof_vector(vector,operator_data.dof_index);
  }

  /*
   *  Calculation of inverse diagonal (needed for smoothers and preconditioners)
   */
  void calculate_inverse_diagonal(parallel::distributed::Vector<Number> &diagonal) const
  {
    calculate_diagonal(diagonal);

    // verify_calculation_of_diagonal(*this,diagonal);

    invert_diagonal(diagonal);
  }

  /*
   *  Apply block Jacobi preconditioner
   */
  void apply_block_jacobi (parallel::distributed::Vector<Number>       &dst,
                           parallel::distributed::Vector<Number> const &src) const
  {
    /*
    // VARIANT 1: solve global system of equations using an iterative solver
    IterationNumberControl control (30,1.e-20,1.e-3);
    typename SolverGMRES<parallel::distributed::Vector<Number> >::AdditionalData additional_data;
    additional_data.right_preconditioning = true;
    additional_data.max_n_tmp_vectors = 100;
    SolverGMRES<parallel::distributed::Vector<Number> > solver (control,additional_data);

    typedef ConvectionDiffusionOperator<dim,fe_degree,Number> MY_TYPE;
    ConvectionDiffusionBlockJacobiOperator<MY_TYPE, Number> block_jacobi_operator(*this);

    dst = 0.0;
    solver.solve(block_jacobi_operator,dst,src,PreconditionIdentity());
    //std::cout<<"Number of iterations block Jacobi solve = "<<control.last_step()<<std::endl;
    */

    // VARIANT 2: calculate block jacobi matrices and solve block Jacobi problem
    // elementwise using a direct solver

    // check_block_jacobi_matrices(src);

    // apply_inverse_matrices
    data->cell_loop(&ConvectionDiffusionOperator<dim,fe_degree,Number>::cell_loop_apply_inverse_block_jacobi_matrices, this, dst, src);
  }

  /*
   *  This function updates the block Jacobi preconditioner.
   *  Since this function also initializes the block Jacobi preconditioner,
   *  make sure that the block Jacobi matrices are allocated before calculating
   *  the matrices and the LU factorization.
   */
  void update_block_jacobi () const
  {
    if(block_jacobi_matrices_have_been_initialized == false)
    {
      matrices.resize(data->n_macro_cells()*VectorizedArray<Number>::n_array_elements,
        LAPACKFullMatrix<Number>(data->get_shape_info().dofs_per_cell, data->get_shape_info().dofs_per_cell));

      block_jacobi_matrices_have_been_initialized = true;
    }

    calculate_block_jacobi_matrices();
    calculate_lu_factorization_block_jacobi();
  }

  /*
   *  Initialize block Jacobi matrices
   */
  void initialize_block_jacobi_matrices_with_zero() const
  {
    // initialize matrices
    for(typename std::vector<LAPACKFullMatrix<Number> >::iterator
        it = matrices.begin(); it != matrices.end(); ++it)
    {
      *it = 0;
    }
  }

  void calculate_block_jacobi_matrices() const
  {
    // initialize block Jacobi matrices with zeros
    initialize_block_jacobi_matrices_with_zero();

    // calculate block Jacobi matrices
    if(operator_data.unsteady_problem == true)
    {
      AssertThrow(scaling_factor_time_derivative_term > 0.0,
        ExcMessage("Scaling factor of time derivative term has not been initialized for convection-diffusion operator!"));

      mass_matrix_operator->add_block_jacobi_matrices(matrices);

      for(typename std::vector<LAPACKFullMatrix<Number> >::iterator
          it = matrices.begin(); it != matrices.end(); ++it)
      {
        (*it) *= scaling_factor_time_derivative_term;
      }
    }

    if(operator_data.diffusive_problem == true)
    {
      diffusive_operator->add_block_jacobi_matrices(matrices);
    }

    if(operator_data.convective_problem == true)
    {
      convective_operator->add_block_jacobi_matrices(matrices,evaluation_time);
    }
  }

  void calculate_lu_factorization_block_jacobi() const
  {
    for(typename std::vector<LAPACKFullMatrix<Number> >::iterator
        it = matrices.begin(); it != matrices.end(); ++it)
    {
      LAPACKFullMatrix<Number> copy(*it);
      try // the matrix might be singular
      {
        (*it).compute_lu_factorization();
      }
      catch (std::exception &exc)
      {
        // add a small, positive value to the diagonal of the LU
        // factorized matrix
        for(unsigned int i=0;i<(*it).m();++i)
        {
          for(unsigned int j=0;j<(*it).n();++j)
          {
            if(i==j)
              (*it)(i,j) += 1.e-4;
          }
        }
      }
    }
  }

  void cell_loop_apply_inverse_block_jacobi_matrices (const MatrixFree<dim,value_type>                 &data,
                                                      parallel::distributed::Vector<value_type>        &dst,
                                                      const parallel::distributed::Vector<value_type>  &src,
                                                      const std::pair<unsigned int,unsigned int>       &cell_range) const
  {
    // apply inverse block matrices
    FEEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval(data,
                                                                 mass_matrix_operator->get_operator_data().dof_index,
                                                                 mass_matrix_operator->get_operator_data().quad_index);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);
      fe_eval.read_dof_values(src);

      for (unsigned int v=0; v<VectorizedArray<value_type>::n_array_elements; ++v)
      {
        // fill source vector
        Vector<value_type> src_vector(fe_eval.dofs_per_cell);
        for (unsigned int j=0; j<fe_eval.dofs_per_cell; ++j)
          src_vector(j) = fe_eval.begin_dof_values()[j][v];

        // apply inverse matrix
        matrices[cell*VectorizedArray<value_type>::n_array_elements+v].apply_lu_factorization(src_vector,false);

        // write solution to dst-vector
        for (unsigned int j=0; j<fe_eval.dofs_per_cell; ++j)
          fe_eval.begin_dof_values()[j][v] = src_vector(j);
      }

      fe_eval.set_dof_values (dst);
    }
  }

  void check_block_jacobi_matrices(parallel::distributed::Vector<Number> const &src) const
  {
    calculate_block_jacobi_matrices();

    // test matrix-vector product for block Jacobi problem by comparing
    // matrix-free matrix-vector product and matrix-based matrix-vector product
    // (where the matrices are generated using the matrix-free implementation)
    parallel::distributed::Vector<Number> tmp1(src), tmp2(src), diff(src);
    tmp1 = 0.0;
    tmp2 = 0.0;
    vmult_block_jacobi(tmp1,src);
    vmult_block_jacobi_2(tmp2,src);

    diff = tmp2;
    diff.add(-1.0,tmp1);

    std::cout << "L2 norm variant 1 = " << tmp1.l2_norm() << std::endl
              << "L2 norm variant 2 = " << tmp2.l2_norm() << std::endl
              << "L2 norm v2 - v1 = " << diff.l2_norm() << std::endl << std::endl;
  }

private:
  /*
   *  This function calculates the diagonal of the scalar reaction-convection-diffusion operator.
   */
  void calculate_diagonal(parallel::distributed::Vector<Number> &diagonal) const
  {
    if(operator_data.unsteady_problem == true)
    {
      AssertThrow(scaling_factor_time_derivative_term > 0.0,
        ExcMessage("Scaling factor of time derivative term has not been initialized for convection-diffusion operator!"));

      mass_matrix_operator->calculate_diagonal(diagonal);
      diagonal *= scaling_factor_time_derivative_term;
    }
    else
    {
      diagonal = 0.0;
    }

    if(operator_data.diffusive_problem == true)
    {
      diffusive_operator->add_diagonal(diagonal);
    }

    if(operator_data.convective_problem == true)
    {
      convective_operator->add_diagonal(diagonal,evaluation_time);
    }
  }

  // TODO
  mutable std::vector<LAPACKFullMatrix<Number> > matrices;
  mutable bool block_jacobi_matrices_have_been_initialized;

  MatrixFree<dim,Number> const * data;
  MassMatrixOperator<dim, fe_degree, Number>  const *mass_matrix_operator;
  ConvectiveOperator<dim, fe_degree, Number> const *convective_operator;
  DiffusiveOperator<dim, fe_degree, Number>  const *diffusive_operator;
  ConvectionDiffusionOperatorData<dim> operator_data;
  parallel::distributed::Vector<Number> mutable temp;
  double scaling_factor_time_derivative_term;
  double evaluation_time;

  /*
   * The following variables are necessary when applying the multigrid
   * preconditioner to the convection-diffusion operator. In that case, the
   * Helmholtz has to be generated for each level of the multigrid algorithm.
   * Accordingly, in a first step one has to setup own objects of
   * MatrixFree, MassMatrixOperator, DiffusiveOperator,
   *   e.g., own_matrix_free_storage.reinit(...);
   * and later initialize the convection-diffusion operator with these
   * ojects by setting the above pointers to the own_objects_storage,
   *   e.g., data = &own_matrix_free_storage;
   */
  MatrixFree<dim,Number> own_matrix_free_storage;
  MassMatrixOperator<dim, fe_degree, Number> own_mass_matrix_operator_storage;
  ConvectiveOperator<dim, fe_degree, Number> own_convective_operator_storage;
  DiffusiveOperator<dim, fe_degree, Number> own_diffusive_operator_storage;
};




// Convection-diffusion operator for runtime optimization:
// Evaluate volume and surface integrals of convective term, diffusive term and
// rhs term in one function (local_apply, local_apply_face, local_evaluate_boundary_face)
// instead of implementing each operator seperately and subsequently looping over all operators.
//
// Note: to obtain meaningful results, ensure that ...
//   ... the rhs-function, velocity-field and that the diffusivity is zero
//   if the rhs operator, convective operator or diffusive operator is "inactive".
//   The reason behind is that the volume and surface integrals of these operators
//   will always be evaluated for this "runtime optimization" implementation of the
//   convection-diffusion operator.
//
// Note: This operator is only implemented for the special case of explicit time integration,
//   i.e., when "evaluating" the operators for a given input-vector, at a given time and given
//   boundary conditions. Accordingly, the convective and diffusive operators a multiplied by
//   a factor of -1.0 since these terms are shifted to the right hand side of the equation.
//   The implicit solution of linear systems of equations (in case of implicit time integration)
//   is currently not available for this implementation.

template<int dim>
struct ConvectionDiffusionOperatorDataEfficiency
{
  ConvectionDiffusionOperatorDataEfficiency (){}

  ConvectiveOperatorData<dim> conv_data;
  DiffusiveOperatorData<dim> diff_data;
  RHSOperatorData<dim> rhs_data;
};

template <int dim, int fe_degree, typename value_type>
class ConvectionDiffusionOperatorEfficiency
{
public:
  ConvectionDiffusionOperatorEfficiency()
    :
    data(nullptr),
    diffusivity(-1.0)
  {}

  void initialize(Mapping<dim> const                                   &mapping,
                  MatrixFree<dim,value_type> const                     &mf_data,
                  ConvectionDiffusionOperatorDataEfficiency<dim> const &operator_data_in)
  {
    this->data = &mf_data;
    this->operator_data = operator_data_in;

    compute_array_penalty_parameter(mapping);

    diffusivity = operator_data.diff_data.diffusivity;
  }

  // Note: for this operator only the evaluate functions are implemented (no apply functions, no rhs functions)

  void evaluate (parallel::distributed::Vector<value_type>       &dst,
                 const parallel::distributed::Vector<value_type> &src,
                 value_type const                                evaluation_time) const
  {
    dst = 0;
    evaluate_add(dst,src,evaluation_time);
  }

  void evaluate_add (parallel::distributed::Vector<value_type>       &dst,
                     const parallel::distributed::Vector<value_type> &src,
                     value_type const                                evaluation_time) const
  {
    this->eval_time = evaluation_time;

    data->loop(&ConvectionDiffusionOperatorEfficiency<dim,fe_degree, value_type>::local_apply_cell,
               &ConvectionDiffusionOperatorEfficiency<dim,fe_degree, value_type>::local_apply_face,
               &ConvectionDiffusionOperatorEfficiency<dim,fe_degree, value_type>::local_evaluate_boundary_face, this, dst, src);
  }

private:
  void compute_array_penalty_parameter(const Mapping<dim> &mapping)
  {
    // Compute penalty parameter for each cell
    array_penalty_parameter.resize(data->n_macro_cells()+data->n_macro_ghost_cells());
    QGauss<dim> quadrature(fe_degree+1);
    FEValues<dim> fe_values(mapping,data->get_dof_handler(operator_data.diff_data.dof_index).get_fe(),quadrature, update_JxW_values);
    QGauss<dim-1> face_quadrature(fe_degree+1);
    FEFaceValues<dim> fe_face_values(mapping, data->get_dof_handler(operator_data.diff_data.dof_index).get_fe(), face_quadrature, update_JxW_values);

    for (unsigned int i=0; i<data->n_macro_cells()+data->n_macro_ghost_cells(); ++i)
    {
      for (unsigned int v=0; v<data->n_components_filled(i); ++v)
      {
        typename DoFHandler<dim>::cell_iterator cell = data->get_cell_iterator(i,v,operator_data.diff_data.dof_index);
        fe_values.reinit(cell);
        double volume = 0;
        for (unsigned int q=0; q<quadrature.size(); ++q)
          volume += fe_values.JxW(q);
        double surface_area = 0;
        for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
        {
          fe_face_values.reinit(cell, f);
          const double factor = (cell->at_boundary(f) && !cell->has_periodic_neighbor(f)) ? 1. : 0.5;
          for (unsigned int q=0; q<face_quadrature.size(); ++q)
            surface_area += fe_face_values.JxW(q) * factor;
        }
        array_penalty_parameter[i][v] = surface_area / volume;
      }
    }
  }

  value_type get_penalty_factor() const
  {
    return operator_data.diff_data.IP_factor * (fe_degree + 1.0) * (fe_degree + 1.0);
  }

  void local_apply_cell (const MatrixFree<dim,value_type>                 &data,
                         parallel::distributed::Vector<value_type>        &dst,
                         const parallel::distributed::Vector<value_type>  &src,
                         const std::pair<unsigned int,unsigned int>       &cell_range) const
  {
    FEEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval(data,
                                                                 operator_data.diff_data.dof_index,
                                                                 operator_data.diff_data.quad_index);

    // set the correct time for the evaluation of the velocity field
    operator_data.conv_data.velocity->set_time(eval_time);
    operator_data.rhs_data.rhs->set_time(eval_time);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);
      fe_eval.read_dof_values(src);
      fe_eval.evaluate (true,true,false);

      for (unsigned int q=0; q<fe_eval.n_q_points; ++q)
      {
        Point<dim,VectorizedArray<value_type> > q_points = fe_eval.quadrature_point(q);
        // velocity
        Tensor<1,dim,VectorizedArray<value_type> > velocity;
        for(unsigned int d=0;d<dim;++d)
        {
          value_type array [VectorizedArray<value_type>::n_array_elements];
          for (unsigned int n=0; n<VectorizedArray<value_type>::n_array_elements; ++n)
          {
            Point<dim> q_point;
            for (unsigned int d=0; d<dim; ++d)
              q_point[d] = q_points[d][n];
            array[n] = operator_data.conv_data.velocity->value(q_point,d);
          }
          velocity[d].load(&array[0]);
        }
        // rhs
        VectorizedArray<value_type> rhs = make_vectorized_array<value_type>(0.0);
        value_type array [VectorizedArray<value_type>::n_array_elements];
        for (unsigned int n=0; n<VectorizedArray<value_type>::n_array_elements; ++n)
        {
          Point<dim> q_point;
          for (unsigned int d=0; d<dim; ++d)
          q_point[d] = q_points[d][n];
          array[n] = operator_data.rhs_data.rhs->value(q_point);
        }
        rhs.load(&array[0]);
        //                           |<-    convective term      ->|  |<-                  diffusive term                                  ->|
        fe_eval.submit_gradient(-1.0*(-fe_eval.get_value(q)*velocity + make_vectorized_array<value_type>(diffusivity)*fe_eval.get_gradient(q)),q);
        // rhs term
        fe_eval.submit_value (rhs, q);
      }
      fe_eval.integrate (true,true);
      fe_eval.distribute_local_to_global (dst);
    }
  }

  void local_apply_face (const MatrixFree<dim,value_type>                &data,
                         parallel::distributed::Vector<value_type>       &dst,
                         const parallel::distributed::Vector<value_type> &src,
                         const std::pair<unsigned int,unsigned int>      &face_range) const
  {
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval(data,true,operator_data.diff_data.dof_index,operator_data.diff_data.quad_index);
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval_neighbor(data,false,operator_data.diff_data.dof_index,operator_data.diff_data.quad_index);

    // set the correct time for the evaluation of the velocity field
    operator_data.conv_data.velocity->set_time(eval_time);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval.reinit (face);
      fe_eval_neighbor.reinit (face);

      fe_eval.read_dof_values(src);
      fe_eval.evaluate(true,true);
      fe_eval_neighbor.read_dof_values(src);
      fe_eval_neighbor.evaluate(true,true);

      VectorizedArray<value_type> tau_IP = std::max(fe_eval.read_cell_data(array_penalty_parameter),fe_eval_neighbor.read_cell_data(array_penalty_parameter))
                                              * get_penalty_factor();

      for(unsigned int q=0;q<fe_eval.n_q_points;++q)
      {
        Point<dim,VectorizedArray<value_type> > q_points = fe_eval.quadrature_point(q);
        Tensor<1,dim,VectorizedArray<value_type> > velocity;
        for(unsigned int d=0;d<dim;++d)
        {
          value_type array [VectorizedArray<value_type>::n_array_elements];
          for (unsigned int n=0; n<VectorizedArray<value_type>::n_array_elements; ++n)
          {
            Point<dim> q_point;
            for (unsigned int d=0; d<dim; ++d)
              q_point[d] = q_points[d][n];
            array[n] = operator_data.conv_data.velocity->value(q_point,d);
          }
          velocity[d].load(&array[0]);
        }
        Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval.get_normal_vector(q);
        VectorizedArray<value_type> u_n = velocity*normal;
        VectorizedArray<value_type> value_m = fe_eval.get_value(q);
        VectorizedArray<value_type> value_p = fe_eval_neighbor.get_value(q);
        VectorizedArray<value_type> average_value = 0.5*(value_m + value_p);
        VectorizedArray<value_type> jump_value = value_m - value_p;
        VectorizedArray<value_type> lambda = std::abs(u_n);
        VectorizedArray<value_type> lf_flux = make_vectorized_array<value_type>(0.0);

        if(this->operator_data.conv_data.numerical_flux_formulation == NumericalFluxConvectiveOperator::CentralFlux)
          lf_flux = u_n*average_value;
        else if(this->operator_data.conv_data.numerical_flux_formulation == NumericalFluxConvectiveOperator::LaxFriedrichsFlux)
          lf_flux = u_n*average_value + 0.5*lambda*jump_value;
        else
          AssertThrow(this->operator_data.conv_data.numerical_flux_formulation == NumericalFluxConvectiveOperator::CentralFlux ||
                      this->operator_data.conv_data.numerical_flux_formulation == NumericalFluxConvectiveOperator::LaxFriedrichsFlux,
                      ExcMessage("Specified numerical flux function for convective operator is not implemented!"));

        VectorizedArray<value_type> gradient_flux = ( fe_eval.get_normal_gradient(q) +
                                                      fe_eval_neighbor.get_normal_gradient(q) ) * 0.5;
        gradient_flux = gradient_flux - tau_IP * jump_value;

        fe_eval.submit_normal_gradient(-1.0*(-0.5*diffusivity*jump_value),q);
        fe_eval_neighbor.submit_normal_gradient(-1.0*(-0.5*diffusivity*jump_value),q);

        fe_eval.submit_value(-1.0*(lf_flux - diffusivity*gradient_flux),q);
        fe_eval_neighbor.submit_value(-1.0*(-lf_flux + diffusivity*gradient_flux),q);
      }
      fe_eval.integrate(true,true);
      fe_eval.distribute_local_to_global(dst);
      fe_eval_neighbor.integrate(true,true);
      fe_eval_neighbor.distribute_local_to_global(dst);
    }
  }

  void local_evaluate_boundary_face (const MatrixFree<dim,value_type>                &data,
                                     parallel::distributed::Vector<value_type>       &dst,
                                     const parallel::distributed::Vector<value_type> &src,
                                     const std::pair<unsigned int,unsigned int>      &face_range) const
  {
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval(data,true,operator_data.diff_data.dof_index,operator_data.diff_data.quad_index);

    // set the correct time for the evaluation of the velocity field
    operator_data.conv_data.velocity->set_time(eval_time);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval.reinit (face);

      fe_eval.read_dof_values(src);
      fe_eval.evaluate(true,true);

      VectorizedArray<value_type> tau_IP = fe_eval.read_cell_data(array_penalty_parameter)
                                              * get_penalty_factor();

      typename std::map<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >::iterator it;
      types::boundary_id boundary_id = data.get_boundary_indicator(face);

      for(unsigned int q=0;q<fe_eval.n_q_points;++q)
      {
        it = operator_data.diff_data.bc->dirichlet_bc.find(boundary_id);
        if(it != operator_data.diff_data.bc->dirichlet_bc.end())
        {
          // on GammaD: u⁺ = -u⁻ + 2g -> {{u}} = g, [u] = 2u⁻ - 2g
          // homogeneous part: u⁺ = -u⁻ -> {{u}} = 0, [u] = 2u⁻
          // inhomogenous part: u⁺ = 2g -> {{u}} = g, [u] = -2g

          // on GammaD: grad(u⁺)*n = grad(u⁻)*n -> {{grad(u)}}*n = grad(u⁻)*n
          // homogeneous part: {{grad(u)}}*n = grad(u⁻)*n
          // inhomogeneous part: {{grad(u)}}*n = 0

          Point<dim,VectorizedArray<value_type> > q_points = fe_eval.quadrature_point(q);
          Tensor<1,dim,VectorizedArray<value_type> > velocity;
          for(unsigned int d=0;d<dim;++d)
          {
            value_type array [VectorizedArray<value_type>::n_array_elements];
            for (unsigned int n=0; n<VectorizedArray<value_type>::n_array_elements; ++n)
            {
              Point<dim> q_point;
              for (unsigned int d=0; d<dim; ++d)
                q_point[d] = q_points[d][n];
              array[n] = operator_data.conv_data.velocity->value(q_point,d);
            }
            velocity[d].load(&array[0]);
          }
          Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval.get_normal_vector(q);
          VectorizedArray<value_type> u_n = velocity*normal;
          VectorizedArray<value_type> value_m = fe_eval.get_value(q);

          // set the correct time for the evaluation of the boundary conditions
          it->second->set_time(eval_time);
          VectorizedArray<value_type> g = make_vectorized_array<value_type>(0.0);
          value_type array [VectorizedArray<value_type>::n_array_elements];
          for (unsigned int n=0; n<VectorizedArray<value_type>::n_array_elements; ++n)
          {
            Point<dim> q_point;
            for (unsigned int d=0; d<dim; ++d)
              q_point[d] = q_points[d][n];
            array[n] = it->second->value(q_point);
          }
          g.load(&array[0]);

          VectorizedArray<value_type> value_p = - value_m + 2.0*g;
          VectorizedArray<value_type> average_value = 0.5*(value_m + value_p);
          VectorizedArray<value_type> jump_value = value_m - value_p;
          VectorizedArray<value_type> lambda = std::abs(u_n);
          VectorizedArray<value_type> lf_flux = make_vectorized_array<value_type>(0.0);

          if(this->operator_data.conv_data.numerical_flux_formulation == NumericalFluxConvectiveOperator::CentralFlux)
            lf_flux = u_n*average_value;
          else if(this->operator_data.conv_data.numerical_flux_formulation == NumericalFluxConvectiveOperator::LaxFriedrichsFlux)
            lf_flux = u_n*average_value + 0.5*lambda*jump_value;
          else
            AssertThrow(this->operator_data.conv_data.numerical_flux_formulation == NumericalFluxConvectiveOperator::CentralFlux ||
                        this->operator_data.conv_data.numerical_flux_formulation == NumericalFluxConvectiveOperator::LaxFriedrichsFlux,
                        ExcMessage("Specified numerical flux function for convective operator is not implemented!"));

          VectorizedArray<value_type> gradient_flux = fe_eval.get_normal_gradient(q);
          gradient_flux = gradient_flux - tau_IP * jump_value;

          fe_eval.submit_normal_gradient(-1.0*(-0.5*diffusivity*jump_value),q);

          fe_eval.submit_value(-1.0*(lf_flux -diffusivity*gradient_flux),q);
        }
        it = operator_data.diff_data.bc->neumann_bc.find(boundary_id);
        if(it != operator_data.diff_data.bc->neumann_bc.end())
        {
          // on GammaN: u⁺ = u⁻-> {{u}} = u⁻, [u] = 0
          // homogeneous part: u⁺ = u⁻ -> {{u}} = u⁻, [u] = 0
          // inhomogenous part: u⁺ = 0 -> {{u}} = 0, [u] = 0

          // on GammaN: grad(u⁺)*n = -grad(u⁻)*n + 2h -> {{grad(u)}}*n = h
          // homogeneous part: {{grad(u)}}*n = 0
          // inhomogeneous part: {{grad(u)}}*n = h

          Point<dim,VectorizedArray<value_type> > q_points = fe_eval.quadrature_point(q);
          Tensor<1,dim,VectorizedArray<value_type> > velocity;
          for(unsigned int d=0;d<dim;++d)
          {
            value_type array [VectorizedArray<value_type>::n_array_elements];
            for (unsigned int n=0; n<VectorizedArray<value_type>::n_array_elements; ++n)
            {
              Point<dim> q_point;
              for (unsigned int d=0; d<dim; ++d)
                q_point[d] = q_points[d][n];
              array[n] = operator_data.conv_data.velocity->value(q_point,d);
            }
            velocity[d].load(&array[0]);
          }
          Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval.get_normal_vector(q);
          VectorizedArray<value_type> u_n = velocity*normal;
          VectorizedArray<value_type> value_m = fe_eval.get_value(q);
          VectorizedArray<value_type> value_p = value_m;
          VectorizedArray<value_type> average_value = 0.5*(value_m + value_p);
          VectorizedArray<value_type> jump_value = value_m - value_p;
          VectorizedArray<value_type> lambda = std::abs(u_n);
          VectorizedArray<value_type> lf_flux = make_vectorized_array<value_type>(0.0);

          if(this->operator_data.conv_data.numerical_flux_formulation == NumericalFluxConvectiveOperator::CentralFlux)
            lf_flux = u_n*average_value;
          else if(this->operator_data.conv_data.numerical_flux_formulation == NumericalFluxConvectiveOperator::LaxFriedrichsFlux)
            lf_flux = u_n*average_value + 0.5*lambda*jump_value;
          else
            AssertThrow(this->operator_data.conv_data.numerical_flux_formulation == NumericalFluxConvectiveOperator::CentralFlux ||
                        this->operator_data.conv_data.numerical_flux_formulation == NumericalFluxConvectiveOperator::LaxFriedrichsFlux,
                        ExcMessage("Specified numerical flux function for convective operator is not implemented!"));

          VectorizedArray<value_type> gradient_flux = make_vectorized_array<value_type>(0.0);

          // set time for the correct evaluation of boundary conditions
          it->second->set_time(eval_time);
          value_type array [VectorizedArray<value_type>::n_array_elements];
          for (unsigned int n=0; n<VectorizedArray<value_type>::n_array_elements; ++n)
          {
            Point<dim> q_point;
            for (unsigned int d=0; d<dim; ++d)
              q_point[d] = q_points[d][n];
            array[n] = it->second->value(q_point);
          }
          gradient_flux.load(&array[0]);

          fe_eval.submit_normal_gradient(-1.0*(-0.5*diffusivity*jump_value),q);

          fe_eval.submit_value(-1.0*(lf_flux - diffusivity*gradient_flux),q);
        }
      }
      fe_eval.integrate(true,true);
      fe_eval.distribute_local_to_global(dst);
    }
  }

  MatrixFree<dim,value_type> const * data;
  ConvectionDiffusionOperatorDataEfficiency<dim> operator_data;
  AlignedVector<VectorizedArray<value_type> > array_penalty_parameter;
  double diffusivity;
  mutable value_type eval_time;
};

}

#endif /* INCLUDE_SCALARCONVECTIONDIFFUSIONOPERATORS_H_ */
