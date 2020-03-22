/*
 * rhs_operator.cpp
 *
 *  Created on: 21.03.2020
 *      Author: fehn
 */

#include "rhs_operator.h"

#include "../../../functionalities/evaluate_functions.h"

namespace Structure
{
template<int dim, typename Number>
RHSOperator<dim, Number>::RHSOperator()
  : data(nullptr),
    dof_handler(nullptr),
    constraint_matrix(nullptr),
    mapping(nullptr),
    eval_time(0.0)
{
}

template<int dim, typename Number>
void
RHSOperator<dim, Number>::reinit(MatrixFree<dim, Number> const &   mf_data,
                                 DoFHandler<dim> const &           dof_handler,
                                 AffineConstraints<double> const & constraint_matrix,
                                 Mapping<dim> const &              mapping,
                                 RHSOperatorData<dim> const &      operator_data)
{
  this->data              = &mf_data;
  this->dof_handler       = &dof_handler;
  this->constraint_matrix = &constraint_matrix;
  this->mapping           = &mapping;
  this->operator_data     = operator_data;
}

template<int dim, typename Number>
void
RHSOperator<dim, Number>::evaluate_add(VectorType & dst, double const evaluation_time) const
{
  this->eval_time = evaluation_time;

  if(operator_data.do_rhs)
  {
    VectorType src;
    data->cell_loop(&This::cell_loop, this, dst, src, false /*zero_dst_vector*/);
  }
}

template<int dim, typename Number>
void
RHSOperator<dim, Number>::evaluate_add_nbc(VectorType & dst, double const evaluation_time) const
{
  this->eval_time = evaluation_time;

  // TODO: use MatrixFree
  // data->loop(
  //  &This::cell_loop_empty, &This::face_loop, &This::boundary_loop, this, dst, src, false);
  do_boundary(dst);
}

template<int dim, typename Number>
void
RHSOperator<dim, Number>::cell_loop(MatrixFree<dim, Number> const & data,
                                    VectorType &                    dst,
                                    VectorType const &              src,
                                    Range const &                   cell_range) const
{
  (void)src;

  IntegratorCell fe_eval(data, operator_data.dof_index, operator_data.quad_index);

  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    fe_eval.reinit(cell);

    for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
    {
      auto q_points = fe_eval.quadrature_point(q);
      auto rhs = FunctionEvaluator<1, dim, Number>::value(operator_data.rhs, q_points, eval_time);
      fe_eval.submit_value(rhs, q);
    }
    fe_eval.integrate(true, false);

    fe_eval.distribute_local_to_global(dst);
  }
}

template<int dim, typename Number>
void
RHSOperator<dim, Number>::cell_loop_empty(MatrixFree<dim, Number> const & data,
                                          VectorType &                    dst,
                                          VectorType const &              src,
                                          Range const &                   cell_range) const
{
  (void)data;
  (void)dst;
  (void)src;
  (void)cell_range;
  // nothing to do since to source term has been provided
}

template<int dim, typename Number>
void
RHSOperator<dim, Number>::face_loop(MatrixFree<dim, Number> const & data,
                                    VectorType &                    dst,
                                    VectorType const &              src,
                                    Range const &                   cell_range) const
{
  (void)data;
  (void)dst;
  (void)src;
  (void)cell_range;
  // nothing to do on inner faces
}

template<int dim, typename Number>
void
RHSOperator<dim, Number>::boundary_loop(MatrixFree<dim, Number> const & data,
                                        VectorType &                    dst,
                                        VectorType const &              src,
                                        Range const &                   face_range) const
{
  (void)src;
  IntegratorFace fe_eval(data, true, operator_data.dof_index, operator_data.quad_index);

  for(unsigned int face = face_range.first; face < face_range.second; face++)
  {
    auto                           boundary_id = data.get_boundary_id(face);
    std::shared_ptr<Function<dim>> fu;
    if(this->operator_data.bc->get_boundary_type(boundary_id) == BoundaryType::Neumann)
      fu = this->operator_data.bc->neumann_bc[boundary_id];
    else
      continue;

    fe_eval.reinit(face);

    for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
    {
      auto q_points = fe_eval.quadrature_point(q);
      auto rhs      = FunctionEvaluator<1, dim, Number>::value(fu, q_points, eval_time);
      fe_eval.submit_value(rhs, q);
    }

    fe_eval.integrate_scatter(true, false, dst);
  }
}

template<int dim, typename Number>
void
RHSOperator<dim, Number>::do_boundary(VectorType & dst) const
{
  if(this->operator_data.bc->neumann_bc.size() == 0)
    // none available -> nothing to do
    return;

  auto & fe = this->dof_handler->get_fe(operator_data.dof_index);

  QGauss<dim - 1> face_quadrature(operator_data.degree + 1);

  FEFaceValues<dim> fe_face_values(*mapping,
                                   fe,
                                   face_quadrature,
                                   update_values | update_JxW_values | update_gradients |
                                     update_normal_vectors | update_quadrature_points);


  Vector<double> local_rhs(fe.dofs_per_cell);


  std::vector<types::global_dof_index> dof_indices(fe.dofs_per_cell);


  auto cell = this->dof_handler->begin_active();
  auto endc = this->dof_handler->end();


  for(; cell != endc; ++cell)
  {
    if(!cell->is_locally_owned())
      continue;

    bool did_reinit = false;


    for(unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
    {
      if(!cell->at_boundary(face))
        continue;

      auto boundary_id = cell->face(face)->boundary_id();
      auto boundary    = this->operator_data.bc->get_boundary(boundary_id);

      if(boundary.first != BoundaryType::Neumann)
        continue;

      // set evaluation time for boundary function
      boundary.second->set_time(this->eval_time);

      fe_face_values.reinit(cell, face);

      if(!did_reinit)
        local_rhs = 0.0;
      did_reinit = true;


      for(unsigned int q = 0; q < face_quadrature.size(); ++q)
      {
        const Point<dim> quadrature_point = fe_face_values.quadrature_point(q);

        Tensor<1, dim, double> neumann_value;

        for(unsigned int d = 0; d < dim; d++)
          neumann_value[d] = boundary.second->value(quadrature_point, d) * fe_face_values.JxW(q);

        for(unsigned int i = 0; i < fe.dofs_per_cell; ++i)
        {
          const unsigned int component_i = fe.system_to_component_index(i).first;

          local_rhs(i) += fe_face_values.shape_value(i, q) * neumann_value[component_i];
        }
      }
    }

    if(!did_reinit)
      continue;

    cell->get_dof_indices(dof_indices);

    constraint_matrix->distribute_local_to_global(local_rhs, dof_indices, dst);
  }


  dst.compress(VectorOperation::add);
}

template class RHSOperator<2, float>;
template class RHSOperator<2, double>;

template class RHSOperator<3, float>;
template class RHSOperator<3, double>;

} // namespace Structure
