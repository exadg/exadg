/*
 * ScalarConvectionDiffusionOperators.h
 *
 *  Created on: Jul 29, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_SCALARCONVECTIONDIFFUSIONOPERATORS_H_
#define INCLUDE_SCALARCONVECTIONDIFFUSIONOPERATORS_H_

#include "../include/BoundaryDescriptorConvDiff.h"
#include "InputParametersConvDiff.h"

// forward declaration
template<int dim> class FEParameters;

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
    apply_mass_matrix(dst,src);
  }

  void calculate_diagonal(parallel::distributed::Vector<value_type> &diagonal) const
  {
    diagonal = 0;

    add_diagonal(diagonal);
  }

  void add_diagonal(parallel::distributed::Vector<value_type> &diagonal) const
  {
    parallel::distributed::Vector<value_type>  src_dummy(diagonal);

    data->cell_loop(&MassMatrixOperator<dim,fe_degree,value_type>::local_diagonal, this, diagonal, src_dummy);
  }

private:
  void apply_mass_matrix (parallel::distributed::Vector<value_type>        &dst,
                          const parallel::distributed::Vector<value_type>  &src) const
  {
    data->cell_loop(&MassMatrixOperator<dim,fe_degree, value_type>::local_apply, this, dst, src);
  }

  void local_apply (const MatrixFree<dim,value_type>                 &data,
                    parallel::distributed::Vector<value_type>        &dst,
                    const parallel::distributed::Vector<value_type>  &src,
                    const std::pair<unsigned int,unsigned int>       &cell_range) const
  {
    FEEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval(data,
                                                                 mass_matrix_operator_data.dof_index,
                                                                 mass_matrix_operator_data.quad_index);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);
      fe_eval.read_dof_values(src);
      fe_eval.evaluate (true,false,false);

      for (unsigned int q=0; q<fe_eval.n_q_points; ++q)
      {
        fe_eval.submit_value (fe_eval.get_value(q), q);
      }
      fe_eval.integrate (true,false);
      fe_eval.distribute_local_to_global (dst);
    }
  }

  void local_diagonal (const MatrixFree<dim,value_type>                 &data,
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

        // copied from local_apply_cell TODO
        fe_eval.evaluate (true,false,false);
        for (unsigned int q=0; q<fe_eval.n_q_points; ++q)
        {
          fe_eval.submit_value (fe_eval.get_value(q), q);
        }
        fe_eval.integrate (true,false);
        // copied from local_apply_cell TODO

        local_diagonal_vector[j] = fe_eval.begin_dof_values()[j];
      }
      for (unsigned int j=0; j<fe_eval.dofs_per_cell; ++j)
        fe_eval.begin_dof_values()[j] = local_diagonal_vector[j];

      fe_eval.distribute_local_to_global (dst);
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
    data->cell_loop(&RHSOperator<dim,fe_degree, value_type>::local_evaluate, this, dst, src);
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

    // set correct evaluation time for the evaluation of the rhs-function
    operator_data.rhs->set_time(eval_time);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);

      for (unsigned int q=0; q<fe_eval.n_q_points; ++q)
      {
        Point<dim,VectorizedArray<value_type> > q_points = fe_eval.quadrature_point(q);
        VectorizedArray<value_type> rhs;

        value_type array [VectorizedArray<value_type>::n_array_elements];
        for (unsigned int n=0; n<VectorizedArray<value_type>::n_array_elements; ++n)
        {
          Point<dim> q_point;
          for (unsigned int d=0; d<dim; ++d)
          q_point[d] = q_points[d][n];
          array[n] = operator_data.rhs->value(q_point);
        }
        rhs.load(&array[0]);

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
    apply_diffusive_operator(dst,src);
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

    data->loop(&DiffusiveOperator<dim,fe_degree, value_type>::local_rhs_cell,
               &DiffusiveOperator<dim,fe_degree, value_type>::local_rhs_face,
               &DiffusiveOperator<dim,fe_degree, value_type>::local_rhs_boundary_face,this, dst, src);
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

    data->loop(&DiffusiveOperator<dim,fe_degree, value_type>::local_apply_cell,
               &DiffusiveOperator<dim,fe_degree, value_type>::local_apply_face,
               &DiffusiveOperator<dim,fe_degree, value_type>::local_evaluate_boundary_face, this, dst, src);
  }

  void calculate_diagonal(parallel::distributed::Vector<value_type> &diagonal) const
  {
    diagonal = 0;

    add_diagonal(diagonal);
  }

  void add_diagonal(parallel::distributed::Vector<value_type> &diagonal) const
  {
    parallel::distributed::Vector<value_type>  src_dummy(diagonal);

    data->loop(&DiffusiveOperator<dim,fe_degree,value_type>::local_diagonal_cell,
               &DiffusiveOperator<dim,fe_degree,value_type>::local_diagonal_face,
               &DiffusiveOperator<dim,fe_degree,value_type>::local_diagonal_boundary_face,
               this, diagonal, src_dummy);
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

  value_type get_penalty_factor() const
  {
    return operator_data.IP_factor * (fe_degree + 1.0) * (fe_degree + 1.0);
  }

  void apply_diffusive_operator (parallel::distributed::Vector<value_type>        &dst,
                                 const parallel::distributed::Vector<value_type>  &src) const
  {
    AssertThrow(diffusivity > 0.0,ExcMessage("Diffusivity has not been set!"));

    data->loop(&DiffusiveOperator<dim,fe_degree, value_type>::local_apply_cell,
               &DiffusiveOperator<dim,fe_degree, value_type>::local_apply_face,
               &DiffusiveOperator<dim,fe_degree, value_type>::local_apply_boundary_face,this, dst, src);
  }

  void local_apply_cell (const MatrixFree<dim,value_type>                 &data,
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
      fe_eval.evaluate (false,true,false);

      for (unsigned int q=0; q<fe_eval.n_q_points; ++q)
      {
        fe_eval.submit_gradient (make_vectorized_array<value_type>(diffusivity)*fe_eval.get_gradient(q), q);
      }
      fe_eval.integrate (false,true);
      fe_eval.distribute_local_to_global (dst);
    }
  }

  void local_apply_face (const MatrixFree<dim,value_type>                &data,
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
      fe_eval.evaluate(true,true);
      fe_eval_neighbor.read_dof_values(src);
      fe_eval_neighbor.evaluate(true,true);

      VectorizedArray<value_type> tau_IP = std::max(fe_eval.read_cell_data(array_penalty_parameter),fe_eval_neighbor.read_cell_data(array_penalty_parameter))
                                              * get_penalty_factor();

      for(unsigned int q=0;q<fe_eval.n_q_points;++q)
      {
        VectorizedArray<value_type> jump_value = (fe_eval.get_value(q) - fe_eval_neighbor.get_value(q));
        VectorizedArray<value_type> gradient_flux = ( fe_eval.get_normal_gradient(q) +
                                        fe_eval_neighbor.get_normal_gradient(q) ) * 0.5;
        gradient_flux = gradient_flux - tau_IP * jump_value;

        fe_eval.submit_normal_gradient(-0.5*diffusivity*jump_value,q);
        fe_eval_neighbor.submit_normal_gradient(-0.5*diffusivity*jump_value,q);

        fe_eval.submit_value(-diffusivity*gradient_flux,q);
        fe_eval_neighbor.submit_value(diffusivity*gradient_flux,q);
      }
      fe_eval.integrate(true,true);
      fe_eval.distribute_local_to_global(dst);
      fe_eval_neighbor.integrate(true,true);
      fe_eval_neighbor.distribute_local_to_global(dst);
    }
  }

  void local_apply_boundary_face (const MatrixFree<dim,value_type>                &data,
                                  parallel::distributed::Vector<value_type>       &dst,
                                  const parallel::distributed::Vector<value_type> &src,
                                  const std::pair<unsigned int,unsigned int>      &face_range) const
  {
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval(data,true,operator_data.dof_index,operator_data.quad_index);

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
        it = operator_data.bc->dirichlet_bc.find(boundary_id);
        if(it != operator_data.bc->dirichlet_bc.end())
        {
          // on GammaD: u⁺ = -u⁻ + 2g -> {{u}} = g, [u] = 2u⁻ - 2g
          // homogeneous part: u⁺ = -u⁻ -> {{u}} = 0, [u] = 2u⁻
          // inhomongenous part: u⁺ = 2g -> {{u}} = g, [u] = -2g

          // on GammaD: grad(u⁺)*n = grad(u⁻)*n -> {{grad(u)}}*n = grad(u⁻)*n
          // homogeneous part: {{grad(u)}}*n = grad(u⁻)*n
          // inhomogeneous part: {{grad(u)}}*n = 0
          VectorizedArray<value_type> jump_value = 2.0*fe_eval.get_value(q);
          VectorizedArray<value_type> gradient_flux = fe_eval.get_normal_gradient(q);
          gradient_flux = gradient_flux - tau_IP * jump_value;

          fe_eval.submit_normal_gradient(-0.5*diffusivity*jump_value,q);
          fe_eval.submit_value(-diffusivity*gradient_flux,q);
        }

        it = operator_data.bc->neumann_bc.find(boundary_id);
        if (it != operator_data.bc->neumann_bc.end())
        {
          // on GammaD: u⁺ = u⁻-> {{u}} = u⁻, [u] = 0
          // homogeneous part: u⁺ = u⁻ -> {{u}} = u⁻, [u] = 0
          // inhomongenous part: u⁺ = 0 -> {{u}} = 0, [u] = 0

          // on GammaD: grad(u⁺)*n = -grad(u⁻)*n + 2h -> {{grad(u)}}*n = h
          // homogeneous part: {{grad(u)}}*n = 0
          // inhomogeneous part: {{grad(u)}}*n = h
          VectorizedArray<value_type> jump_value = make_vectorized_array<value_type>(0.0);
          VectorizedArray<value_type> gradient_flux = make_vectorized_array<value_type>(0.0);

          fe_eval.submit_normal_gradient(-0.5*diffusivity*jump_value,q);
          fe_eval.submit_value(-diffusivity*gradient_flux,q);
        }
      }
      fe_eval.integrate(true,true);
      fe_eval.distribute_local_to_global(dst);
    }
  }

  void local_diagonal_cell (const MatrixFree<dim,value_type>                 &data,
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

        // copied from local_apply_cell TODO
        fe_eval.evaluate (false,true,false);
        for (unsigned int q=0; q<fe_eval.n_q_points; ++q)
        {
          fe_eval.submit_gradient (make_vectorized_array<value_type>(diffusivity)*fe_eval.get_gradient(q), q);
        }
        fe_eval.integrate (false,true);
        // copied from local_apply_cell TODO

        local_diagonal_vector[j] = fe_eval.begin_dof_values()[j];
      }
      for (unsigned int j=0; j<fe_eval.dofs_per_cell; ++j)
        fe_eval.begin_dof_values()[j] = local_diagonal_vector[j];

      fe_eval.distribute_local_to_global (dst);
    }
  }

  void local_diagonal_face (const MatrixFree<dim,value_type>                &data,
                            parallel::distributed::Vector<value_type>       &dst,
                            const parallel::distributed::Vector<value_type> &,
                            const std::pair<unsigned int,unsigned int>      &face_range) const
  {
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval(data,true,operator_data.dof_index,operator_data.quad_index);
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval_neighbor(data,false,operator_data.dof_index,operator_data.quad_index);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval.reinit (face);
      fe_eval_neighbor.reinit (face);

      VectorizedArray<value_type> tau_IP = std::max(fe_eval.read_cell_data(array_penalty_parameter),fe_eval_neighbor.read_cell_data(array_penalty_parameter))
                                              * get_penalty_factor();



      // element-
      VectorizedArray<value_type> local_diagonal_vector[fe_eval.tensor_dofs_per_cell];
      for (unsigned int j=0; j<fe_eval.dofs_per_cell; ++j)
      {
        // set dof value j of element- to 1 and all other dof values of element- to zero
        for (unsigned int i=0; i<fe_eval.dofs_per_cell; ++i)
          fe_eval.begin_dof_values()[i] = make_vectorized_array<value_type>(0.);
        fe_eval.begin_dof_values()[j] = make_vectorized_array<value_type>(1.);

        // set all dof values of element+ to zero
        for (unsigned int i=0; i<fe_eval_neighbor.dofs_per_cell; ++i)
          fe_eval_neighbor.begin_dof_values()[i] = make_vectorized_array<value_type>(0.);

        // copied from local_apply_face (note that fe_eval_neighbor.submit...
        // and fe_eval_neighbor.integrate() has to be removed. TODO
        fe_eval.evaluate(true,true);
        fe_eval_neighbor.evaluate(true,true);

        for(unsigned int q=0;q<fe_eval.n_q_points;++q)
        {
          VectorizedArray<value_type> jump_value = (fe_eval.get_value(q) - fe_eval_neighbor.get_value(q));
          VectorizedArray<value_type> gradient_flux = ( fe_eval.get_normal_gradient(q) +
                                          fe_eval_neighbor.get_normal_gradient(q) ) * 0.5;
          gradient_flux = gradient_flux - tau_IP * jump_value;

          fe_eval.submit_normal_gradient(-0.5*diffusivity*jump_value,q);
          fe_eval.submit_value(-diffusivity*gradient_flux,q);
        }
        fe_eval.integrate(true,true);
        // copied from local_apply_face (note that fe_eval_neighbor.submit...
        // and fe_eval_neighbor.integrate() has to be removed. //TODO

        local_diagonal_vector[j] = fe_eval.begin_dof_values()[j];
      }
      for (unsigned int j=0; j<fe_eval.dofs_per_cell; ++j)
        fe_eval.begin_dof_values()[j] = local_diagonal_vector[j];

      fe_eval.distribute_local_to_global(dst);



      // element+
      VectorizedArray<value_type> local_diagonal_vector_neighbor[fe_eval_neighbor.tensor_dofs_per_cell];
      for (unsigned int j=0; j<fe_eval_neighbor.dofs_per_cell; ++j)
      {
        // set all dof values of element- to zero
        for (unsigned int i=0; i<fe_eval.dofs_per_cell; ++i)
          fe_eval.begin_dof_values()[i] = make_vectorized_array<value_type>(0.);

        // set dof value j of element+ to 1 and all other dof values of element+ to zero
        for (unsigned int i=0; i<fe_eval_neighbor.dofs_per_cell; ++i)
          fe_eval_neighbor.begin_dof_values()[i] = make_vectorized_array<value_type>(0.);
        fe_eval_neighbor.begin_dof_values()[j] = make_vectorized_array<value_type>(1.);

        // copied from local_apply_face (note that fe_eval.submit...
        // and fe_evalintegrate() has to be removed. TODO
        fe_eval.evaluate(true,true);
        fe_eval_neighbor.evaluate(true,true);

        for(unsigned int q=0;q<fe_eval.n_q_points;++q)
        {
         VectorizedArray<value_type> jump_value = (fe_eval.get_value(q) - fe_eval_neighbor.get_value(q));
         VectorizedArray<value_type> gradient_flux = ( fe_eval.get_normal_gradient(q) +
                                         fe_eval_neighbor.get_normal_gradient(q) ) * 0.5;
         gradient_flux = gradient_flux - tau_IP * jump_value;

         fe_eval_neighbor.submit_normal_gradient(-0.5*diffusivity*jump_value,q);
         fe_eval_neighbor.submit_value(diffusivity*gradient_flux,q);
        }
        fe_eval_neighbor.integrate(true,true);
        // copied from local_apply_face (note that fe_eval.submit...
        // and fe_evalintegrate() has to be removed. TODO

        local_diagonal_vector_neighbor[j] = fe_eval_neighbor.begin_dof_values()[j];
      }
      for (unsigned int j=0; j<fe_eval_neighbor.dofs_per_cell; ++j)
        fe_eval_neighbor.begin_dof_values()[j] = local_diagonal_vector_neighbor[j];

      fe_eval_neighbor.distribute_local_to_global(dst);

    }
  }

  void local_diagonal_boundary_face (const MatrixFree<dim,value_type>                &data,
                                     parallel::distributed::Vector<value_type>       &dst,
                                     const parallel::distributed::Vector<value_type> &/*src*/,
                                     const std::pair<unsigned int,unsigned int>      &face_range) const
  {
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval(data,true,operator_data.dof_index,operator_data.quad_index);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval.reinit (face);

      VectorizedArray<value_type> tau_IP = fe_eval.read_cell_data(array_penalty_parameter)
                                              * get_penalty_factor();

      typename std::map<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >::iterator it;
      types::boundary_id boundary_id = data.get_boundary_indicator(face);

      VectorizedArray<value_type> local_diagonal_vector[fe_eval.tensor_dofs_per_cell];
      for (unsigned int j=0; j<fe_eval.dofs_per_cell; ++j)
      {
        // set dof value j of element- to 1 and all other dof values of element- to zero
        for (unsigned int i=0; i<fe_eval.dofs_per_cell; ++i)
          fe_eval.begin_dof_values()[i] = make_vectorized_array<value_type>(0.);
        fe_eval.begin_dof_values()[j] = make_vectorized_array<value_type>(1.);

        // copied from local_apply_boundary_face // TODO
        fe_eval.evaluate(true,true);

        for(unsigned int q=0;q<fe_eval.n_q_points;++q)
        {
          it = operator_data.bc->dirichlet_bc.find(boundary_id);
          if(it != operator_data.bc->dirichlet_bc.end())
          {
            // on GammaD: u⁺ = -u⁻ + 2g -> {{u}} = g, [u] = 2u⁻ - 2g
            // homogeneous part: u⁺ = -u⁻ -> {{u}} = 0, [u] = 2u⁻
            // inhomongenous part: u⁺ = 2g -> {{u}} = g, [u] = -2g

            // on GammaD: grad(u⁺)*n = grad(u⁻)*n -> {{grad(u)}}*n = grad(u⁻)*n
            // homogeneous part: {{grad(u)}}*n = grad(u⁻)*n
            // inhomogeneous part: {{grad(u)}}*n = 0
            VectorizedArray<value_type> jump_value = 2.0*fe_eval.get_value(q);
            VectorizedArray<value_type> gradient_flux = fe_eval.get_normal_gradient(q);
            gradient_flux = gradient_flux - tau_IP * jump_value;

            fe_eval.submit_normal_gradient(-0.5*diffusivity*jump_value,q);
            fe_eval.submit_value(-diffusivity*gradient_flux,q);
          }

          it = operator_data.bc->neumann_bc.find(boundary_id);
          if (it != operator_data.bc->neumann_bc.end())
          {
            // on GammaD: u⁺ = u⁻-> {{u}} = u⁻, [u] = 0
            // homogeneous part: u⁺ = u⁻ -> {{u}} = u⁻, [u] = 0
            // inhomongenous part: u⁺ = 0 -> {{u}} = 0, [u] = 0

            // on GammaD: grad(u⁺)*n = -grad(u⁻)*n + 2h -> {{grad(u)}}*n = h
            // homogeneous part: {{grad(u)}}*n = 0
            // inhomogeneous part: {{grad(u)}}*n = h
            VectorizedArray<value_type> jump_value = make_vectorized_array<value_type>(0.0);
            VectorizedArray<value_type> gradient_flux = make_vectorized_array<value_type>(0.0);

            fe_eval.submit_normal_gradient(-0.5*diffusivity*jump_value,q);
            fe_eval.submit_value(-diffusivity*gradient_flux,q);
          }
        }
        fe_eval.integrate(true,true);
        // copied from local_apply_boundary_face // TODO

        local_diagonal_vector[j] = fe_eval.begin_dof_values()[j];
      }
      for (unsigned int j=0; j<fe_eval.dofs_per_cell; ++j)
        fe_eval.begin_dof_values()[j] = local_diagonal_vector[j];

      fe_eval.distribute_local_to_global(dst);
    }
  }

  void local_rhs_cell (const MatrixFree<dim,value_type>                 &,
                       parallel::distributed::Vector<value_type>        &,
                       const parallel::distributed::Vector<value_type>  &,
                       const std::pair<unsigned int,unsigned int>       &) const
  {}

  void local_rhs_face (const MatrixFree<dim,value_type>                &,
                       parallel::distributed::Vector<value_type>       &,
                       const parallel::distributed::Vector<value_type> &,
                       const std::pair<unsigned int,unsigned int>      &) const
  {}

  void local_rhs_boundary_face (const MatrixFree<dim,value_type>                &data,
                                parallel::distributed::Vector<value_type>       &dst,
                                const parallel::distributed::Vector<value_type> &/*src*/,
                                const std::pair<unsigned int,unsigned int>      &face_range) const
  {
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval(data,true,operator_data.dof_index,operator_data.quad_index);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval.reinit (face);

      VectorizedArray<value_type> tau_IP = fe_eval.read_cell_data(array_penalty_parameter)
                                              * get_penalty_factor();

      typename std::map<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >::iterator it;
      types::boundary_id boundary_id = data.get_boundary_indicator(face);

      for(unsigned int q=0;q<fe_eval.n_q_points;++q)
      {
        it = operator_data.bc->dirichlet_bc.find(boundary_id);
        if(it != operator_data.bc->dirichlet_bc.end())
        {
          // on GammaD: u⁺ = -u⁻ + 2g -> {{u}} = g, [u] = 2u⁻ - 2g
          // homogeneous part: u⁺ = -u⁻ -> {{u}} = 0, [u] = 2u⁻
          // inhomongenous part: u⁺ = 2g -> {{u}} = g, [u] = -2g

          // on GammaD: grad(u⁺)*n = grad(u⁻)*n -> {{grad(u)}}*n = grad(u⁻)*n
          // homogeneous part: {{grad(u)}}*n = grad(u⁻)*n
          // inhomogeneous part: {{grad(u)}}*n = 0

          // set time for the correct evaluation of boundary conditions
          it->second->set_time(eval_time);
          Point<dim,VectorizedArray<value_type> > q_points = fe_eval.quadrature_point(q);
          VectorizedArray<value_type> g;
          value_type array [VectorizedArray<value_type>::n_array_elements];
          for (unsigned int n=0; n<VectorizedArray<value_type>::n_array_elements; ++n)
          {
            Point<dim> q_point;
            for (unsigned int d=0; d<dim; ++d)
              q_point[d] = q_points[d][n];
            array[n] = it->second->value(q_point);
          }
          g.load(&array[0]);

          VectorizedArray<value_type> jump_value = -2.0*g;
          VectorizedArray<value_type> gradient_flux = make_vectorized_array<value_type>(0.0);
          gradient_flux = gradient_flux - tau_IP * jump_value;

          fe_eval.submit_normal_gradient(-0.5*diffusivity*(-jump_value),q); // -jump_value since this term appears on the rhs of the equation
          fe_eval.submit_value(-diffusivity*(-gradient_flux),q); // -gradient_flux since this term appears on the rhs of the equation
        }

        it = operator_data.bc->neumann_bc.find(boundary_id);
        if (it != operator_data.bc->neumann_bc.end())
        {
          // on GammaN: u⁺ = u⁻-> {{u}} = u⁻, [u] = 0
          // homogeneous part: u⁺ = u⁻ -> {{u}} = u⁻, [u] = 0
          // inhomongenous part: u⁺ = 0 -> {{u}} = 0, [u] = 0

          // on GammaN: grad(u⁺)*n = -grad(u⁻)*n + 2h -> {{grad(u)}}*n = h
          // homogeneous part: {{grad(u)}}*n = 0
          // inhomogeneous part: {{grad(u)}}*n = h
          VectorizedArray<value_type> jump_value = make_vectorized_array<value_type>(0.0);
          VectorizedArray<value_type> gradient_flux = make_vectorized_array<value_type>(0.0);

          // set time for the correct evaluation of boundary conditions
          it->second->set_time(eval_time);
          Point<dim,VectorizedArray<value_type> > q_points = fe_eval.quadrature_point(q);
          value_type array [VectorizedArray<value_type>::n_array_elements];
          for (unsigned int n=0; n<VectorizedArray<value_type>::n_array_elements; ++n)
          {
            Point<dim> q_point;
            for (unsigned int d=0; d<dim; ++d)
              q_point[d] = q_points[d][n];
            array[n] = it->second->value(q_point);
          }
          gradient_flux.load(&array[0]);

          fe_eval.submit_normal_gradient(-0.5*diffusivity*(-jump_value),q); // -jump_value since this term appears on the rhs of the equation
          fe_eval.submit_value(-diffusivity*(-gradient_flux),q); // -gradient_flux since this term appears on the rhs of the equation
        }
      }
      fe_eval.integrate(true,true);
      fe_eval.distribute_local_to_global(dst);
    }
  }

  void local_evaluate_boundary_face (const MatrixFree<dim,value_type>                &data,
                                     parallel::distributed::Vector<value_type>       &dst,
                                     const parallel::distributed::Vector<value_type> &src,
                                     const std::pair<unsigned int,unsigned int>      &face_range) const
  {
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval(data,true,operator_data.dof_index,operator_data.quad_index);

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
        it = operator_data.bc->dirichlet_bc.find(boundary_id);
        if(it != operator_data.bc->dirichlet_bc.end())
        {
          // on GammaD: u⁺ = -u⁻ + 2g -> {{u}} = g, [u] = 2u⁻ - 2g
          // homogeneous part: u⁺ = -u⁻ -> {{u}} = 0, [u] = 2u⁻
          // inhomongenous part: u⁺ = 2g -> {{u}} = g, [u] = -2g

          // on GammaD: grad(u⁺)*n = grad(u⁻)*n -> {{grad(u)}}*n = grad(u⁻)*n
          // homogeneous part: {{grad(u)}}*n = grad(u⁻)*n
          // inhomogeneous part: {{grad(u)}}*n = 0

          // set time for the correct evaluation of boundary conditions
          it->second->set_time(eval_time);
          Point<dim,VectorizedArray<value_type> > q_points = fe_eval.quadrature_point(q);
          VectorizedArray<value_type> g;
          value_type array [VectorizedArray<value_type>::n_array_elements];
          for (unsigned int n=0; n<VectorizedArray<value_type>::n_array_elements; ++n)
          {
            Point<dim> q_point;
            for (unsigned int d=0; d<dim; ++d)
              q_point[d] = q_points[d][n];
            array[n] = it->second->value(q_point);
          }
          g.load(&array[0]);

          VectorizedArray<value_type> jump_value = 2.0*(fe_eval.get_value(q)-g);
          VectorizedArray<value_type> gradient_flux = fe_eval.get_normal_gradient(q);
          gradient_flux = gradient_flux - tau_IP * jump_value;

          fe_eval.submit_normal_gradient(-0.5*diffusivity*jump_value,q);
          fe_eval.submit_value(-diffusivity*gradient_flux,q);
        }
        it = operator_data.bc->neumann_bc.find(boundary_id);
        if(it != operator_data.bc->neumann_bc.end())
        {
          // on GammaN: u⁺ = u⁻-> {{u}} = u⁻, [u] = 0
          // homogeneous part: u⁺ = u⁻ -> {{u}} = u⁻, [u] = 0
          // inhomongenous part: u⁺ = 0 -> {{u}} = 0, [u] = 0

          // on GammaN: grad(u⁺)*n = -grad(u⁻)*n + 2h -> {{grad(u)}}*n = h
          // homogeneous part: {{grad(u)}}*n = 0
          // inhomogeneous part: {{grad(u)}}*n = h
          VectorizedArray<value_type> jump_value;
          VectorizedArray<value_type> gradient_flux;

          // set time for the correct evaluation of boundary conditions
          it->second->set_time(eval_time);
          Point<dim,VectorizedArray<value_type> > q_points = fe_eval.quadrature_point(q);
          value_type array [VectorizedArray<value_type>::n_array_elements];
          for (unsigned int n=0; n<VectorizedArray<value_type>::n_array_elements; ++n)
          {
            Point<dim> q_point;
            for (unsigned int d=0; d<dim; ++d)
              q_point[d] = q_points[d][n];
            array[n] = it->second->value(q_point);
          }
          gradient_flux.load(&array[0]);

          fe_eval.submit_normal_gradient(-0.5*diffusivity*jump_value,q);
          fe_eval.submit_value(-diffusivity*gradient_flux,q);
        }
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
    apply_convective_operator(dst,src);
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

    data->loop(&ConvectiveOperator<dim,fe_degree, value_type>::local_apply_cell,
               &ConvectiveOperator<dim,fe_degree, value_type>::local_apply_face,
               &ConvectiveOperator<dim,fe_degree, value_type>::local_evaluate_boundary_face,this, dst, src);
  }

  void rhs (parallel::distributed::Vector<value_type>       &dst,
            value_type const                                evaluation_time) const
  {
    dst = 0;
    rhs_add(dst,evaluation_time);
  }

  void rhs_add (parallel::distributed::Vector<value_type>       &dst,
                value_type const                                evaluation_time) const
  {
    this->eval_time = evaluation_time;

    parallel::distributed::Vector<value_type> src;

    data->loop(&ConvectiveOperator<dim,fe_degree, value_type>::local_rhs_cell,
               &ConvectiveOperator<dim,fe_degree, value_type>::local_rhs_face,
               &ConvectiveOperator<dim,fe_degree, value_type>::local_rhs_boundary_face,this, dst, src);
  }

private:
  void apply_convective_operator (parallel::distributed::Vector<value_type>       &dst,
                                  const parallel::distributed::Vector<value_type> &src) const
  {
    data->loop(&ConvectiveOperator<dim,fe_degree, value_type>::local_apply_cell,
               &ConvectiveOperator<dim,fe_degree, value_type>::local_apply_face,
               &ConvectiveOperator<dim,fe_degree, value_type>::local_apply_boundary_face,this, dst, src);
  }

  void local_apply_cell (const MatrixFree<dim,value_type>                 &data,
                         parallel::distributed::Vector<value_type>        &dst,
                         const parallel::distributed::Vector<value_type>  &src,
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
      fe_eval.read_dof_values(src);
      fe_eval.evaluate (true,false,false);

      for (unsigned int q=0; q<fe_eval.n_q_points; ++q)
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
            array[n] = operator_data.velocity->value(q_point,d);
          }
          velocity[d].load(&array[0]);
        }
        fe_eval.submit_gradient(-fe_eval.get_value(q)*velocity,q);
      }
      fe_eval.integrate (false,true);
      fe_eval.distribute_local_to_global (dst);
    }
  }

  void local_apply_face (const MatrixFree<dim,value_type>                &data,
                         parallel::distributed::Vector<value_type>       &dst,
                         const parallel::distributed::Vector<value_type> &src,
                         const std::pair<unsigned int,unsigned int>      &face_range) const
  {
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval(data,true,operator_data.dof_index,operator_data.quad_index);
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval_neighbor(data,false,operator_data.dof_index,operator_data.quad_index);

    // set the correct time for the evaluation of the velocity field
    operator_data.velocity->set_time(eval_time);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval.reinit (face);
      fe_eval.read_dof_values(src);
      fe_eval.evaluate(true,false);

      fe_eval_neighbor.reinit (face);
      fe_eval_neighbor.read_dof_values(src);
      fe_eval_neighbor.evaluate(true,false);

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
            array[n] = operator_data.velocity->value(q_point,d);
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

        if(this->operator_data.numerical_flux_formulation == NumericalFluxConvectiveOperator::CentralFlux)
          lf_flux = u_n*average_value;
        else if(this->operator_data.numerical_flux_formulation == NumericalFluxConvectiveOperator::LaxFriedrichsFlux)
          lf_flux = u_n*average_value + 0.5*lambda*jump_value;
        else
          AssertThrow(this->operator_data.numerical_flux_formulation == NumericalFluxConvectiveOperator::CentralFlux ||
                      this->operator_data.numerical_flux_formulation == NumericalFluxConvectiveOperator::LaxFriedrichsFlux,
                      ExcMessage("Specified numerical flux function for convective operator is not implemented!"));

        fe_eval.submit_value(lf_flux,q);
        fe_eval_neighbor.submit_value(-lf_flux,q);
      }
      fe_eval.integrate(true,false);
      fe_eval.distribute_local_to_global(dst);

      fe_eval_neighbor.integrate(true,false);
      fe_eval_neighbor.distribute_local_to_global(dst);
    }
  }

  void local_apply_boundary_face (const MatrixFree<dim,value_type>                &data,
                                  parallel::distributed::Vector<value_type>       &dst,
                                  const parallel::distributed::Vector<value_type> &src,
                                  const std::pair<unsigned int,unsigned int>      &face_range) const
  {
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval(data,true,operator_data.dof_index,operator_data.quad_index);

    // set the correct time for the evaluation of the velocity field
    operator_data.velocity->set_time(eval_time);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval.reinit (face);
      fe_eval.read_dof_values(src);
      fe_eval.evaluate(true,false);

      typename std::map<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >::iterator it;
      types::boundary_id boundary_id = data.get_boundary_indicator(face);

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
            array[n] = operator_data.velocity->value(q_point,d);
          }
          velocity[d].load(&array[0]);
        }
        Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval.get_normal_vector(q);
        VectorizedArray<value_type> u_n = velocity*normal;
        VectorizedArray<value_type> value_m = fe_eval.get_value(q);
        VectorizedArray<value_type> value_p = make_vectorized_array<value_type>(0.0);

        it = operator_data.bc->dirichlet_bc.find(boundary_id);
        if(it != operator_data.bc->dirichlet_bc.end())
        {
          // on GammaD: phi⁺ = -phi⁻ + 2g -> {{phi}} = g, [phi] = 2 phi⁻ - 2 g
          // homogeneous part: phi⁺ = -phi⁻ -> {{phi}} = 0, [phi] = 2 phi⁻
          // inhomongenous part: phi⁺ = 2g -> {{phi}} = g, [phi] = -2 g
          value_p = - value_m;
        }
        it = operator_data.bc->neumann_bc.find(boundary_id);
        if(it != operator_data.bc->neumann_bc.end())
        {
          // on GammaN: phi⁺ = phi⁻-> {{phi}} = phi⁻, [phi] = 0
          // homogeneous part: phi⁺ = phi⁻ -> {{phi}} = phi⁻, [phi] = 0
          // inhomongenous part: phi⁺ = 0 -> {{phi}} = 0, [phi] = 0
          value_p = value_m;
        }
        VectorizedArray<value_type> average_value = 0.5*(value_m + value_p);
        VectorizedArray<value_type> jump_value = value_m - value_p;
        VectorizedArray<value_type> lambda = std::abs(u_n);
        VectorizedArray<value_type> lf_flux = make_vectorized_array<value_type>(0.0);

        if(this->operator_data.numerical_flux_formulation == NumericalFluxConvectiveOperator::CentralFlux)
          lf_flux = u_n*average_value;
        else if(this->operator_data.numerical_flux_formulation == NumericalFluxConvectiveOperator::LaxFriedrichsFlux)
          lf_flux = u_n*average_value + 0.5*lambda*jump_value;
        else
          AssertThrow(this->operator_data.numerical_flux_formulation == NumericalFluxConvectiveOperator::CentralFlux ||
                      this->operator_data.numerical_flux_formulation == NumericalFluxConvectiveOperator::LaxFriedrichsFlux,
                      ExcMessage("Specified numerical flux function for convective operator is not implemented!"));

        fe_eval.submit_value(lf_flux,q);
      }
      fe_eval.integrate(true,false);
      fe_eval.distribute_local_to_global(dst);
    }
  }

  void local_evaluate_boundary_face (const MatrixFree<dim,value_type>                &data,
                                     parallel::distributed::Vector<value_type>       &dst,
                                     const parallel::distributed::Vector<value_type> &src,
                                     const std::pair<unsigned int,unsigned int>      &face_range) const
  {
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval(data,true,operator_data.dof_index,operator_data.quad_index);

    // set the correct time for the evaluation of the velocity field
    operator_data.velocity->set_time(eval_time);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval.reinit (face);
      fe_eval.read_dof_values(src);
      fe_eval.evaluate(true,false);

      typename std::map<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >::iterator it;
      types::boundary_id boundary_id = data.get_boundary_indicator(face);

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
            array[n] = operator_data.velocity->value(q_point,d);
          }
          velocity[d].load(&array[0]);
        }
        Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval.get_normal_vector(q);
        VectorizedArray<value_type> u_n = velocity*normal;
        VectorizedArray<value_type> value_m = fe_eval.get_value(q);
        VectorizedArray<value_type> value_p = make_vectorized_array<value_type>(0.0);

        it = operator_data.bc->dirichlet_bc.find(boundary_id);
        if(it != operator_data.bc->dirichlet_bc.end())
        {
          // on GammaD: phi⁺ = -phi⁻ + 2g -> {{phi}} = g, [phi] = 2 phi⁻ - 2 g
          // homogeneous part: phi⁺ = -phi⁻ -> {{phi}} = 0, [phi] = 2 phi⁻
          // inhomongenous part: phi⁺ = 2g -> {{phi}} = g, [phi] = -2 g

          // set the correct time for the evaluation of the boundary conditions
          it->second->set_time(eval_time);
          VectorizedArray<value_type> g;
          value_type array [VectorizedArray<value_type>::n_array_elements];
          for (unsigned int n=0; n<VectorizedArray<value_type>::n_array_elements; ++n)
          {
            Point<dim> q_point;
            for (unsigned int d=0; d<dim; ++d)
              q_point[d] = q_points[d][n];
            array[n] = it->second->value(q_point);
          }
          g.load(&array[0]);

          value_p = - value_m + 2.0*g;
        }
        it = operator_data.bc->neumann_bc.find(boundary_id);
        if(it != operator_data.bc->neumann_bc.end())
        {
          // on GammaN: phi⁺ = phi⁻-> {{phi}} = phi⁻, [phi] = 0
          // homogeneous part: phi⁺ = phi⁻ -> {{phi}} = phi⁻, [phi] = 0
          // inhomongenous part: phi⁺ = 0 -> {{phi}} = 0, [phi] = 0
          value_p = value_m;
        }
        VectorizedArray<value_type> average_value = 0.5*(value_m + value_p);
        VectorizedArray<value_type> jump_value = value_m - value_p;
        VectorizedArray<value_type> lambda = std::abs(u_n);
        VectorizedArray<value_type> lf_flux = make_vectorized_array<value_type>(0.0);

        if(this->operator_data.numerical_flux_formulation == NumericalFluxConvectiveOperator::CentralFlux)
          lf_flux = u_n*average_value;
        else if(this->operator_data.numerical_flux_formulation == NumericalFluxConvectiveOperator::LaxFriedrichsFlux)
          lf_flux = u_n*average_value + 0.5*lambda*jump_value;
        else
          AssertThrow(this->operator_data.numerical_flux_formulation == NumericalFluxConvectiveOperator::CentralFlux ||
                      this->operator_data.numerical_flux_formulation == NumericalFluxConvectiveOperator::LaxFriedrichsFlux,
                      ExcMessage("Specified numerical flux function for convective operator is not implemented!"));

        fe_eval.submit_value(lf_flux,q);
      }
      fe_eval.integrate(true,false);
      fe_eval.distribute_local_to_global(dst);
    }
  }

  void local_rhs_cell (const MatrixFree<dim,value_type>                 &,
                       parallel::distributed::Vector<value_type>        &,
                       const parallel::distributed::Vector<value_type>  &,
                       const std::pair<unsigned int,unsigned int>       &) const
   {}

  void local_rhs_face (const MatrixFree<dim,value_type>                &,
                       parallel::distributed::Vector<value_type>       &,
                       const parallel::distributed::Vector<value_type> &,
                       const std::pair<unsigned int,unsigned int>      &) const
  {}

  void local_rhs_boundary_face (const MatrixFree<dim,value_type>                &data,
                                parallel::distributed::Vector<value_type>       &dst,
                                const parallel::distributed::Vector<value_type> &,
                                const std::pair<unsigned int,unsigned int>      &face_range) const
  {
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval(data,true,operator_data.dof_index,operator_data.quad_index);

    // set the correct time for the evaluation of the velocity field
    operator_data.velocity->set_time(eval_time);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval.reinit (face);

      typename std::map<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >::iterator it;
      types::boundary_id boundary_id = data.get_boundary_indicator(face);

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
            array[n] = operator_data.velocity->value(q_point,d);
          }
          velocity[d].load(&array[0]);
        }
        Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval.get_normal_vector(q);
        VectorizedArray<value_type> u_n = velocity*normal;

        VectorizedArray<value_type> average_value = make_vectorized_array<value_type>(0.0);
        VectorizedArray<value_type> jump_value = make_vectorized_array<value_type>(0.0);

        it = operator_data.bc->dirichlet_bc.find(boundary_id);
        if(it != operator_data.bc->dirichlet_bc.end())
        {
          // on GammaD: phi⁺ = -phi⁻ + 2g -> {{phi}} = g, [phi] = 2 phi⁻ - 2 g
          // homogeneous part: phi⁺ = -phi⁻ -> {{phi}} = 0, [phi] = 2 phi⁻
          // inhomongenous part: phi⁺ = 2g -> {{phi}} = g, [phi] = -2 g

          // set the correct time for the evaluation of the boundary conditions
          it->second->set_time(eval_time);
          VectorizedArray<value_type> g;
          value_type array [VectorizedArray<value_type>::n_array_elements];
          for (unsigned int n=0; n<VectorizedArray<value_type>::n_array_elements; ++n)
          {
            Point<dim> q_point;
            for (unsigned int d=0; d<dim; ++d)
              q_point[d] = q_points[d][n];
            array[n] = it->second->value(q_point);
          }
          g.load(&array[0]);

          average_value = g;
          jump_value = -2.0*g;
        }
        it = operator_data.bc->neumann_bc.find(boundary_id);
        if(it != operator_data.bc->neumann_bc.end())
        {
          // on GammaN: phi⁺ = phi⁻-> {{phi}} = phi⁻, [phi] = 0
          // homogeneous part: phi⁺ = phi⁻ -> {{phi}} = phi⁻, [phi] = 0
          // inhomongenous part: phi⁺ = 0 -> {{phi}} = 0, [phi] = 0

          // do nothing since average_value = 0 and jump_value = 0
        }

        VectorizedArray<value_type> lambda = std::abs(u_n);
        VectorizedArray<value_type> lf_flux = make_vectorized_array<value_type>(0.0);

        if(this->operator_data.numerical_flux_formulation == NumericalFluxConvectiveOperator::CentralFlux)
          lf_flux = u_n*average_value;
        else if(this->operator_data.numerical_flux_formulation == NumericalFluxConvectiveOperator::LaxFriedrichsFlux)
          lf_flux = u_n*average_value + 0.5*lambda*jump_value;
        else
          AssertThrow(this->operator_data.numerical_flux_formulation == NumericalFluxConvectiveOperator::CentralFlux ||
                      this->operator_data.numerical_flux_formulation == NumericalFluxConvectiveOperator::LaxFriedrichsFlux,
                      ExcMessage("Specified numerical flux function for convective operator is not implemented!"));

        fe_eval.submit_value(-lf_flux,q); // -lf_flux since this term appears on the rhs of the equation
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
    dof_index(-1),
    dof_index_velocity(-1),
    quad_index(-1)
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

    apply_convective_operator(dst,src);

    velocity = nullptr;
  }

private:
  void apply_convective_operator (parallel::distributed::Vector<value_type>       &dst,
                                  const parallel::distributed::Vector<value_type> &src) const
  {
    data->loop(&ConvectiveOperatorDiscontinuousVelocity<dim,fe_degree, fe_degree_velocity, value_type>::local_apply_cell,
               &ConvectiveOperatorDiscontinuousVelocity<dim,fe_degree, fe_degree_velocity, value_type>::local_apply_face,
               &ConvectiveOperatorDiscontinuousVelocity<dim,fe_degree, fe_degree_velocity, value_type>::local_apply_boundary_face,this, dst, src);
  }

  void local_apply_cell (const MatrixFree<dim,value_type>                 &data,
                         parallel::distributed::Vector<value_type>        &dst,
                         const parallel::distributed::Vector<value_type>  &src,
                         const std::pair<unsigned int,unsigned int>       &cell_range) const
  {
    FEEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval(
      data, operator_data.dof_index, operator_data.quad_index);

    FEEvaluation<dim,fe_degree_velocity,fe_degree+1,dim,value_type> fe_eval_velocity(
      data, operator_data.dof_index_velocity, operator_data.quad_index);

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

  void local_apply_face (const MatrixFree<dim,value_type>                &data,
                         parallel::distributed::Vector<value_type>       &dst,
                         const parallel::distributed::Vector<value_type> &src,
                         const std::pair<unsigned int,unsigned int>      &face_range) const
  {
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval(
        data,true,operator_data.dof_index,operator_data.quad_index);
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval_neighbor(
        data,false,operator_data.dof_index,operator_data.quad_index);

    FEFaceEvaluation<dim,fe_degree_velocity,fe_degree+1,dim,value_type> fe_eval_velocity(
        data,true,operator_data.dof_index_velocity,operator_data.quad_index);
    FEFaceEvaluation<dim,fe_degree_velocity,fe_degree+1,dim,value_type> fe_eval_velocity_neighbor(
        data,false,operator_data.dof_index_velocity,operator_data.quad_index);

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

  void local_apply_boundary_face (const MatrixFree<dim,value_type>                &data,
                                  parallel::distributed::Vector<value_type>       &dst,
                                  const parallel::distributed::Vector<value_type> &src,
                                  const std::pair<unsigned int,unsigned int>      &face_range) const
  {
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval(
        data,true,operator_data.dof_index,operator_data.quad_index);

    FEFaceEvaluation<dim,fe_degree_velocity,fe_degree+1,dim,value_type> fe_eval_velocity(
        data,true,operator_data.dof_index_velocity,operator_data.quad_index);

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
          // inhomongenous part: phi⁺ = 2g -> {{phi}} = g, [phi] = -2 g
          value_p = - value_m;
        }
        it = operator_data.bc->neumann_bc.find(boundary_id);
        if(it != operator_data.bc->neumann_bc.end())
        {
          // on GammaN: phi⁺ = phi⁻-> {{phi}} = phi⁻, [phi] = 0
          // homogeneous part: phi⁺ = phi⁻ -> {{phi}} = phi⁻, [phi] = 0
          // inhomongenous part: phi⁺ = 0 -> {{phi}} = 0, [phi] = 0
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
    dof_index(0),
    mass_matrix_coefficient(-1.0)
  {}

  unsigned int dof_index;

  MassMatrixOperatorData mass_matrix_operator_data;
  DiffusiveOperatorData<dim> diffusive_operator_data;

  /*
   * This variable 'mass_matrix_coefficient' is only used when initializing the HelmholtzOperator.
   * In order to change/update this coefficient during the simulation (e.g., varying time step sizes)
   * use the element variable 'mass_matrix_coefficient' of HelmholtzOperator.
   */
  double mass_matrix_coefficient;

  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> > periodic_face_pairs_level0;

  void set_dof_index(unsigned int dof_index_in)
  {
    this->dof_index = dof_index_in;

    // don't forget to set the dof_indices of mass_matrix_operator_data
    // and diffusive_operator_data
    mass_matrix_operator_data.dof_index = dof_index_in;
    diffusive_operator_data.dof_index = dof_index_in;
  }
};

template <int dim, int fe_degree, typename Number = double>
class HelmholtzOperator : public Subscriptor
{
public:
  typedef Number value_type;

  HelmholtzOperator()
    :
    data(nullptr),
    mass_matrix_operator(nullptr),
    diffusive_operator(nullptr),
    mass_matrix_coefficient(-1.0)
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

    // set mass matrix coefficient!
    AssertThrow(helmholtz_operator_data.mass_matrix_coefficient > 0.0,
                ExcMessage("Mass matrix coefficient of HelmholtzOperatorData has not been initialized!"));

    this->mass_matrix_coefficient = helmholtz_operator_data.mass_matrix_coefficient;
  }

  void reinit (const DoFHandler<dim>            &dof_handler,
               const Mapping<dim>               &mapping,
               const HelmholtzOperatorData<dim> &operator_data,
               const MGConstrainedDoFs          &/*mg_constrained_dofs*/,
               const unsigned int               level = numbers::invalid_unsigned_int,
               FEParameters<dim> const          &fe_param = FEParameters<dim>())
  {
    (void)fe_param; // avoid compiler warning

    // setup own matrix free object
    const QGauss<1> quad(dof_handler.get_fe().degree+1);
    typename MatrixFree<dim,Number>::AdditionalData addit_data;
    addit_data.tasks_parallel_scheme = MatrixFree<dim,Number>::AdditionalData::none;
    if (dof_handler.get_fe().dofs_per_vertex == 0)
      addit_data.build_face_info = true;
    addit_data.level_mg_handler = level;
    addit_data.mpi_communicator =
      dynamic_cast<const parallel::Triangulation<dim> *>(&dof_handler.get_triangulation()) ?
      (dynamic_cast<const parallel::Triangulation<dim> *>(&dof_handler.get_triangulation()))->get_communicator() : MPI_COMM_SELF;
    addit_data.periodic_face_pairs_level_0 = operator_data.periodic_face_pairs_level0;

    ConstraintMatrix constraints;
    own_matrix_free_storage.reinit(mapping, dof_handler, constraints, quad, addit_data);

    // set the dof index to zero (for the HelmholtzOperator and also
    // for the basic Operators (MassMatrixOperator and ViscousOperator))
    // because the own_matrix_free_storage object has only one dof_handler with dof_index = 0
    HelmholtzOperatorData<dim> my_operator_data = operator_data;
    my_operator_data.set_dof_index(0);

    // setup own mass matrix operator
    MassMatrixOperatorData mass_matrix_operator_data = my_operator_data.mass_matrix_operator_data;
    own_mass_matrix_operator_storage.initialize(own_matrix_free_storage,mass_matrix_operator_data);

    // setup own viscous operator
    DiffusiveOperatorData<dim> diffusive_operator_data = my_operator_data.diffusive_operator_data;
    own_diffusive_operator_storage.initialize(mapping,own_matrix_free_storage,diffusive_operator_data);

    // setup Helmholtz operator
    initialize(own_matrix_free_storage, my_operator_data, own_mass_matrix_operator_storage, own_diffusive_operator_storage);

    // initialize temp vector: this is done in this function because
    // the vector temp is only used in the function vmult_add(), i.e.,
    // when using the multigrid preconditioner
    initialize_dof_vector(temp);
  }

  void apply_nullspace_projection(parallel::distributed::Vector<Number> &/*vec*/) const
  {
    // does nothing in case of the Helmholtz operator
    // this function is only necessary due to the interface of the multigrid preconditioner
    // and especially the coarse grid solver that calls this function
  }

  // apply matrix vector multiplication
  void vmult (parallel::distributed::Vector<Number>       &dst,
              const parallel::distributed::Vector<Number> &src) const
  {
    // helmholtz operator = mass_matrix_operator + viscous_operator
    mass_matrix_operator->apply(dst,src);
    dst *= mass_matrix_coefficient;

    diffusive_operator->apply_add(dst,src);
  }

//  void Tvmult(parallel::distributed::Vector<Number>       &dst,
//              const parallel::distributed::Vector<Number> &src) const
//  {
//    vmult(dst,src);
//  }
//
//  void Tvmult_add(parallel::distributed::Vector<Number>       &dst,
//                  const parallel::distributed::Vector<Number> &src) const
//  {
//    vmult_add(dst,src);
//  }

  void vmult_add(parallel::distributed::Vector<Number>       &dst,
                 const parallel::distributed::Vector<Number> &src) const
  {
    // helmholtz operator = mass_matrix_operator + viscous_operator
    mass_matrix_operator->apply(temp,src);
    temp *= mass_matrix_coefficient;
    dst += temp;

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

//  types::global_dof_index n() const
//  {
//    return data->get_vector_partitioner(helmholtz_operator_data.dof_index)->size();
//  }

  Number el (const unsigned int,  const unsigned int) const
  {
    AssertThrow(false, ExcMessage("Matrix-free does not allow for entry access"));
    return Number();
  }

  unsigned int get_dof_index() const
  {
    return helmholtz_operator_data.dof_index;
  }

  void calculate_diagonal(parallel::distributed::Vector<Number> &diagonal) const
  {
    mass_matrix_operator->calculate_diagonal(diagonal);
    diagonal *= mass_matrix_coefficient;

    diffusive_operator->add_diagonal(diagonal);

    // verify_calculation_of_diagonal(diagonal);
  }

  void verify_calculation_of_diagonal(parallel::distributed::Vector<Number> const &diagonal) const
  {
    parallel::distributed::Vector<Number>  diagonal2(diagonal);
    diagonal2 = 0.0;
    parallel::distributed::Vector<Number>  src(diagonal2);
    parallel::distributed::Vector<Number>  dst(diagonal2);
    for (unsigned int i=0;i<diagonal.local_size();++i)
    {
      src.local_element(i) = 1.0;
      vmult(dst,src);
      diagonal2.local_element(i) = dst.local_element(i);
      src.local_element(i) = 0.0;
    }

    std::cout<<"L2 norm diagonal - Variant 1: "<<diagonal.l2_norm()<<std::endl;
    std::cout<<"L2 norm diagonal - Variant 2: "<<diagonal2.l2_norm()<<std::endl;
    diagonal2.add(-1.0,diagonal);
    std::cout<<"L2 error diagonal: "<<diagonal2.l2_norm()<<std::endl;
  }

  void invert_diagonal(parallel::distributed::Vector<Number> &diagonal) const
  {
    for (unsigned int i=0;i<diagonal.local_size();++i)
    {
      if( std::abs(diagonal.local_element(i)) > 1.0e-10 )
        diagonal.local_element(i) = 1.0/diagonal.local_element(i);
      else
        diagonal.local_element(i) = 1.0;
    }
  }

  void calculate_inverse_diagonal(parallel::distributed::Vector<Number> &diagonal) const
  {
    calculate_diagonal(diagonal);

    invert_diagonal(diagonal);
  }

  void initialize_dof_vector(parallel::distributed::Vector<Number> &vector) const
  {
    data->initialize_dof_vector(vector,get_dof_index());
  }

private:
  MatrixFree<dim,Number> const * data;
  MassMatrixOperator<dim, fe_degree, Number>  const *mass_matrix_operator;
  DiffusiveOperator<dim, fe_degree, Number>  const *diffusive_operator;
  HelmholtzOperatorData<dim> helmholtz_operator_data;
  parallel::distributed::Vector<Number> mutable temp;
  double mass_matrix_coefficient;

  /*
   * The following variables are necessary when applying the multigrid preconditioner to the Helmholtz operator:
   * In that case, the HelmholtzOperator has to be generated for each level of the multigrid algorithm.
   * Accordingly, in a first step one has to setup own objects of MatrixFree, MassMatrixOperator, DiffusiveOperator,
   *  e.g., own_matrix_free_storage.reinit(...);
   * and later initialize the HelmholtzOperator with these ojects by setting the above pointers to the own_objects_storage,
   *  e.g., data = &own_matrix_free_storage;
   */
  MatrixFree<dim,Number> own_matrix_free_storage;
  MassMatrixOperator<dim, fe_degree, Number> own_mass_matrix_operator_storage;
  DiffusiveOperator<dim, fe_degree, Number> own_diffusive_operator_storage;

};


// Convection-diffusion operator for runtime optimization:
// Evaluate volume and surface integrals of convective term, diffusive term and
// rhs term in one function (local_apply, local_apply_face, local_evaluate_boundary_face)
// instead of implementing each operator seperately and subsequently looping over all operators.
//
// Note: to obtain meaningful results, ensure that ...
//   ... the rhs-function, velocity-field and that the diffusivity is zero
//   if the rhs operator, convective operator or diffusive operator is "inactive"
//   The reason behind is that the volume and surface integrals of these operators
//   will always be evaluated for this "runtime optimization" implementation of the
//   convection-diffusion operator
//
// Note: This operator is only implemented for the special case of explicit time integration,
//   i.e., when "evaluating" the operators for a given input-vector, at a given time and given
//   boundary conditions. Accordingly, the convective and diffusive operators a multiplied by
//   a factor of -1.0 since these terms are shifted to the right hand side of the equation.
//   The implicit solution of linear systems of equations (in case of implicit time integration)
//   is currently not available for this implementation.

template<int dim>
struct ConvectionDiffusionOperatorData
{
  ConvectionDiffusionOperatorData (){}

  ConvectiveOperatorData<dim> conv_data;
  DiffusiveOperatorData<dim> diff_data;
  RHSOperatorData<dim> rhs_data;
};

template <int dim, int fe_degree, typename value_type>
class ConvectionDiffusionOperator
{
public:
  ConvectionDiffusionOperator()
    :
    data(nullptr),
    diffusivity(-1.0)
  {}

  void initialize(Mapping<dim> const               &mapping,
                  MatrixFree<dim,value_type> const           &mf_data,
                  ConvectionDiffusionOperatorData<dim> const &operator_data_in)
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

    data->loop(&ConvectionDiffusionOperator<dim,fe_degree, value_type>::local_apply_cell,
               &ConvectionDiffusionOperator<dim,fe_degree, value_type>::local_apply_face,
               &ConvectionDiffusionOperator<dim,fe_degree, value_type>::local_evaluate_boundary_face, this, dst, src);
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
        VectorizedArray<value_type> rhs;
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
          // inhomongenous part: u⁺ = 2g -> {{u}} = g, [u] = -2g

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
          VectorizedArray<value_type> g;
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
          // inhomongenous part: u⁺ = 0 -> {{u}} = 0, [u] = 0

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

          VectorizedArray<value_type> gradient_flux;

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
  ConvectionDiffusionOperatorData<dim> operator_data;
  AlignedVector<VectorizedArray<value_type> > array_penalty_parameter;
  double diffusivity;
  mutable value_type eval_time;
};

}

#endif /* INCLUDE_SCALARCONVECTIONDIFFUSIONOPERATORS_H_ */
