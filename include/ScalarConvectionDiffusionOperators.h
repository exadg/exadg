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

namespace ScalarConvDiffOperators
{

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
          VectorizedArray<value_type> jump_value;
          VectorizedArray<value_type> gradient_flux;

          fe_eval.submit_normal_gradient(-0.5*diffusivity*jump_value,q);
          fe_eval.submit_value(-diffusivity*gradient_flux,q);
        }
      }
      fe_eval.integrate(true,true);
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
                                const parallel::distributed::Vector<value_type> &src,
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
          VectorizedArray<value_type> gradient_flux;
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
            const parallel::distributed::Vector<value_type> &src,
            value_type const                                evaluation_time) const
  {
    dst = 0;
    rhs_add(dst,src,evaluation_time);
  }

  void rhs_add (parallel::distributed::Vector<value_type>       &dst,
                const parallel::distributed::Vector<value_type> &src,
                value_type const                                evaluation_time) const
  {
    this->eval_time = evaluation_time;

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
        VectorizedArray<value_type> lf_flux;

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
        VectorizedArray<value_type> value_p;

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
        VectorizedArray<value_type> lf_flux;

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
        VectorizedArray<value_type> value_p;

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
        VectorizedArray<value_type> lf_flux;

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
                                const parallel::distributed::Vector<value_type> &src,
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

        VectorizedArray<value_type> average_value;
        VectorizedArray<value_type> jump_value;

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
        VectorizedArray<value_type> lf_flux;

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
        VectorizedArray<value_type> value_p;

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
        VectorizedArray<value_type> lf_flux;

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
          VectorizedArray<value_type> lf_flux = u_n*average_value + 0.5*lambda*jump_value;

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
          VectorizedArray<value_type> lf_flux;

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
