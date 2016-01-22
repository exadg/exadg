

#include "poisson_solver.h"

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/distributed/tria_base.h>
#include <deal.II/matrix_free/fe_evaluation.h>

template <int dim, typename Number>
LaplaceOperator<dim,Number>::LaplaceOperator ()
  :
  data (0),
  fe_degree (numbers::invalid_unsigned_int)
{}



template <int dim, typename Number>
void LaplaceOperator<dim,Number>::clear()
{
  solver_data = PoissonSolverData<dim>();
  data = 0;
  fe_degree = numbers::invalid_unsigned_int;
  own_matrix_free_storage.clear();
  tmp_projection_vector.reinit(0);
}



template <int dim, typename Number>
void LaplaceOperator<dim,Number>::reinit(const MatrixFree<dim,Number>       &mf_data,
                                         const Mapping<dim>                 &mapping,
                                         const PoissonSolverData<dim>       &solver_data)
{
  this->data = &mf_data;
  this->solver_data = solver_data;
  this->fe_degree = mf_data.get_dof_handler(solver_data.pressure_dof_index).get_fe().degree;
  AssertThrow (Utilities::fixed_power<dim>(fe_degree+1) ==
               mf_data.get_n_q_points(solver_data.pressure_quad_index),
               ExcMessage("Expected pressure_degree+1 quadrature points"));

  check_boundary_conditions(mapping);
}



template <int dim, typename Number>
void LaplaceOperator<dim,Number>::reinit (const DoFHandler<dim> &dof_handler,
                                          const Mapping<dim> &mapping,
                                          const PoissonSolverData<dim> &solver_data,
                                          const unsigned int level)
{
  clear();

  const QGauss<1> quad(dof_handler.get_fe().degree+1);
  typename MatrixFree<dim,Number>::AdditionalData addit_data;
  addit_data.tasks_parallel_scheme = MatrixFree<dim,Number>::AdditionalData::none;
  addit_data.build_face_info = true;
  addit_data.level_mg_handler = level;
  addit_data.mpi_communicator =
    dynamic_cast<const parallel::Triangulation<dim> *>(&dof_handler.get_triangulation()) ?
    (dynamic_cast<const parallel::Triangulation<dim> *>(&dof_handler.get_triangulation()))->get_communicator() : MPI_COMM_SELF;
  addit_data.periodic_face_pairs_level_0 = solver_data.periodic_face_pairs;

  ConstraintMatrix constraints;
  const bool is_feq = dof_handler.get_fe().dofs_per_vertex > 0;
  if (is_feq)
    {
      IndexSet relevant_dofs;
      DoFTools::extract_locally_relevant_dofs(dof_handler, relevant_dofs);
      constraints.reinit(relevant_dofs);
      DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    }
  constraints.close();

  PoissonSolverData<dim> my_solver_data = solver_data;
  my_solver_data.pressure_dof_index = 0;
  my_solver_data.pressure_quad_index = 0;

  own_matrix_free_storage.reinit(mapping, dof_handler, constraints, quad,
                                 addit_data);

  reinit(own_matrix_free_storage, mapping, my_solver_data);
}



template <int dim, typename Number>
void LaplaceOperator<dim,Number>::check_boundary_conditions(const Mapping<dim> &mapping)
{
  // Check that the Dirichlet and Neumann boundary conditions do not overlap
  std::set<types::boundary_id> periodic_boundary_ids;
  for (unsigned int i=0; i<solver_data.periodic_face_pairs.size(); ++i)
    {
      AssertThrow(solver_data.periodic_face_pairs[i].cell[0]->level() == 0,
                  ExcMessage("Received periodic cell pairs on non-zero level"));
      periodic_boundary_ids.insert(solver_data.periodic_face_pairs[i].cell[0]->face(solver_data.periodic_face_pairs[i].face_idx[0])->boundary_id());
      periodic_boundary_ids.insert(solver_data.periodic_face_pairs[i].cell[1]->face(solver_data.periodic_face_pairs[i].face_idx[1])->boundary_id());
    }

  pure_neumann_problem = true;
  const Triangulation<dim> &tria =
    data->get_dof_handler(solver_data.pressure_dof_index).get_triangulation();
  for (typename Triangulation<dim>::cell_iterator cell = tria.begin();
       cell != tria.end(); ++cell)
    for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
      if (cell->at_boundary(f))
        {
          types::boundary_id bid = cell->face(f)->boundary_id();
          if (solver_data.dirichlet_boundaries.find(bid) !=
              solver_data.dirichlet_boundaries.end())
            {
              AssertThrow(solver_data.neumann_boundaries.find(bid) ==
                          solver_data.neumann_boundaries.end(),
                          ExcMessage("Boundary id " + Utilities::to_string((int)bid) +
                                     " wants to set both Dirichlet and Neumann " +
                                     "boundary conditions, which is impossible!"));
              AssertThrow(periodic_boundary_ids.find(bid) ==
                          periodic_boundary_ids.end(),
                          ExcMessage("Boundary id " + Utilities::to_string((int)bid) +
                                     " wants to set both Dirichlet and periodic " +
                                     "boundary conditions, which is impossible!"));
              pure_neumann_problem = false;
              continue;
            }
          if (solver_data.neumann_boundaries.find(bid) !=
              solver_data.neumann_boundaries.end())
            {
              AssertThrow(periodic_boundary_ids.find(bid) ==
                          periodic_boundary_ids.end(),
                          ExcMessage("Boundary id " + Utilities::to_string((int)bid) +
                                     " wants to set both Neumann and periodic " +
                                     "boundary conditions, which is impossible!"));
              continue;
            }
          AssertThrow(periodic_boundary_ids.find(bid) != periodic_boundary_ids.end(),
                      ExcMessage("Boundary id " + Utilities::to_string((int)bid) +
                                 " does neither set Dirichlet, Neumann, no periodic " +
                                 "boundary conditions! Bailing out."));
        }

  // Check for consistency of 'pure_neumann_problem' over all participating
  // processors
  int my_neumann = pure_neumann_problem;
  MPI_Comm mpi_communicator =
    dynamic_cast<const parallel::Triangulation<dim> *>(&tria) ?
    (dynamic_cast<const parallel::Triangulation<dim> *>(&tria))->get_communicator() :
    MPI_COMM_SELF;
  const int max_pure_neumann = Utilities::MPI::max(my_neumann,
                                                   mpi_communicator);
  const int min_pure_neumann = Utilities::MPI::min(my_neumann,
                                                   mpi_communicator);
  AssertThrow(max_pure_neumann == min_pure_neumann,
              ExcMessage("Neumann/Dirichlet assignment over processors does not match."));
  pure_neumann_problem = min_pure_neumann;

  // Compute penalty parameter for each cell
  array_penalty_parameter.resize(data->n_macro_cells()+data->n_macro_ghost_cells());
  QGauss<dim> quadrature(fe_degree+1);
  FEValues<dim> fe_values(mapping,
                          data->get_dof_handler(solver_data.pressure_dof_index).get_fe(),
                          quadrature, update_JxW_values);
  QGauss<dim-1> face_quadrature(fe_degree+1);
  FEFaceValues<dim> fe_face_values(mapping, data->get_dof_handler(solver_data.pressure_dof_index).get_fe(), face_quadrature, update_JxW_values);

  for (unsigned int i=0; i<data->n_macro_cells()+data->n_macro_ghost_cells(); ++i)
    for (unsigned int v=0; v<data->n_components_filled(i); ++v)
      {
        typename DoFHandler<dim>::cell_iterator cell = data->get_cell_iterator(i,v,solver_data.pressure_dof_index);
        fe_values.reinit(cell);
        double volume = 0;
        for (unsigned int q=0; q<quadrature.size(); ++q)
          volume += fe_values.JxW(q);
        double surface_area = 0;
        for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
          {
            fe_face_values.reinit(cell, f);
            const double factor = (cell->at_boundary(f) &&
                                   periodic_boundary_ids.find(cell->face(f)->boundary_id()) ==
                                   periodic_boundary_ids.end()) ? 1. : 0.5;
            for (unsigned int q=0; q<face_quadrature.size(); ++q)
              surface_area += fe_face_values.JxW(q) * factor;
          }
        array_penalty_parameter[i][v] = surface_area / volume;
      }
}



template <int dim, typename Number>
void LaplaceOperator<dim,Number>::vmult(parallel::distributed::Vector<Number> &dst,
                                        const parallel::distributed::Vector<Number> &src) const
{
  dst = 0;
  vmult_add(dst, src);
}



template <int dim, typename Number>
void LaplaceOperator<dim,Number>::Tvmult(parallel::distributed::Vector<Number> &dst,
                                         const parallel::distributed::Vector<Number> &src) const
{
  dst = 0;
  vmult_add(dst, src);
}



template <int dim, typename Number>
void LaplaceOperator<dim,Number>::Tvmult_add(parallel::distributed::Vector<Number> &dst,
                                             const parallel::distributed::Vector<Number> &src) const
{
  vmult_add(dst, src);
}



template <int dim, typename Number>
void LaplaceOperator<dim,Number>::vmult_add(parallel::distributed::Vector<Number> &dst,
                                            const parallel::distributed::Vector<Number> &src) const
{
  Assert(src.partitioners_are_globally_compatible(*data->get_dof_info(solver_data.pressure_dof_index).vector_partitioner), ExcInternalError());
  Assert(dst.partitioners_are_globally_compatible(*data->get_dof_info(solver_data.pressure_dof_index).vector_partitioner), ExcInternalError());

  const parallel::distributed::Vector<Number> *actual_src = &src;
  if(pure_neumann_problem)
    {
      tmp_projection_vector = src;
      apply_nullspace_projection(tmp_projection_vector);
      actual_src = &tmp_projection_vector;
    }

  switch (fe_degree)
    {
    case 0:
      data->loop (&LaplaceOperator::template local_apply<0>,
                  &LaplaceOperator::template local_apply_face<0>,
                  &LaplaceOperator::template local_apply_boundary<0>,
                  this, dst, *actual_src);
      break;
    case 1:
      data->loop (&LaplaceOperator::template local_apply<1>,
                  &LaplaceOperator::template local_apply_face<1>,
                  &LaplaceOperator::template local_apply_boundary<1>,
                  this, dst, *actual_src);
      break;
    case 2:
      data->loop (&LaplaceOperator::template local_apply<2>,
                  &LaplaceOperator::template local_apply_face<2>,
                  &LaplaceOperator::template local_apply_boundary<2>,
                  this, dst, *actual_src);
      break;
    case 3:
      data->loop (&LaplaceOperator::template local_apply<3>,
                  &LaplaceOperator::template local_apply_face<3>,
                  &LaplaceOperator::template local_apply_boundary<3>,
                  this, dst, *actual_src);
      break;
    case 4:
      data->loop (&LaplaceOperator::template local_apply<4>,
                  &LaplaceOperator::template local_apply_face<4>,
                  &LaplaceOperator::template local_apply_boundary<4>,
                  this, dst, *actual_src);
      break;
    case 5:
      data->loop (&LaplaceOperator::template local_apply<5>,
                  &LaplaceOperator::template local_apply_face<5>,
                  &LaplaceOperator::template local_apply_boundary<5>,
                  this, dst, *actual_src);
      break;
    case 6:
      data->loop (&LaplaceOperator::template local_apply<6>,
                  &LaplaceOperator::template local_apply_face<6>,
                  &LaplaceOperator::template local_apply_boundary<6>,
                  this, dst, *actual_src);
      break;
    case 7:
      data->loop (&LaplaceOperator::template local_apply<7>,
                  &LaplaceOperator::template local_apply_face<7>,
                  &LaplaceOperator::template local_apply_boundary<7>,
                  this, dst, *actual_src);
      break;
    case 8:
      data->loop (&LaplaceOperator::template local_apply<8>,
                  &LaplaceOperator::template local_apply_face<8>,
                  &LaplaceOperator::template local_apply_boundary<8>,
                  this, dst, *actual_src);
      break;
    case 9:
      data->loop (&LaplaceOperator::template local_apply<9>,
                  &LaplaceOperator::template local_apply_face<9>,
                  &LaplaceOperator::template local_apply_boundary<9>,
                  this, dst, *actual_src);
      break;
    case 10:
      data->loop (&LaplaceOperator::template local_apply<10>,
                  &LaplaceOperator::template local_apply_face<10>,
                  &LaplaceOperator::template local_apply_boundary<10>,
                  this, dst, *actual_src);
      break;
    default:
      AssertThrow(false, ExcMessage("Only polynomial degrees 0 up to 10 instantiated"));
    }

  if (pure_neumann_problem)
    apply_nullspace_projection(dst);
}



template <int dim, typename Number>
void LaplaceOperator<dim,Number>::apply_nullspace_projection(parallel::distributed::Vector<Number> &vec) const
{
  const Number mean_val = vec.mean_value();
  vec.add(-mean_val);
}



template <int dim, typename Number>
types::global_dof_index LaplaceOperator<dim,Number>::m() const
{
  return data->get_vector_partitioner(solver_data.pressure_dof_index)->size();
}



template <int dim, typename Number>
types::global_dof_index LaplaceOperator<dim,Number>::n() const
{
  return data->get_vector_partitioner(solver_data.pressure_dof_index)->size();
}



template <int dim, typename Number>
Number LaplaceOperator<dim,Number>::el (const unsigned int,  const unsigned int) const
{
  AssertThrow(false, ExcMessage("Matrix-free does not allow for entry access"));
  return Number();
}



template <int dim, typename Number>
void LaplaceOperator<dim,Number>
::initialize_dof_vector(parallel::distributed::Vector<Number> &vector) const
{
  if (!vector.partitioners_are_compatible(*data->get_dof_info(solver_data.pressure_dof_index).vector_partitioner))
    data->initialize_dof_vector(vector, solver_data.pressure_dof_index);
}



template <int dim, typename Number>
void LaplaceOperator<dim,Number>
::compute_inverse_diagonal (parallel::distributed::Vector<Number> &inverse_diagonal_entries)
{
  data->initialize_dof_vector(inverse_diagonal_entries, solver_data.pressure_dof_index);
  unsigned int dummy;
  switch (fe_degree)
    {
    case 0:
      data->loop (&LaplaceOperator::template local_diagonal_cell<0>,
                  &LaplaceOperator::template local_diagonal_face<0>,
                  &LaplaceOperator::template local_diagonal_boundary<0>,
                  this, inverse_diagonal_entries, dummy);
      break;
    case 1:
      data->loop (&LaplaceOperator::template local_diagonal_cell<1>,
                  &LaplaceOperator::template local_diagonal_face<1>,
                  &LaplaceOperator::template local_diagonal_boundary<1>,
                  this, inverse_diagonal_entries, dummy);
      break;
    case 2:
      data->loop (&LaplaceOperator::template local_diagonal_cell<2>,
                  &LaplaceOperator::template local_diagonal_face<2>,
                  &LaplaceOperator::template local_diagonal_boundary<2>,
                  this, inverse_diagonal_entries, dummy);
      break;
    case 3:
      data->loop (&LaplaceOperator::template local_diagonal_cell<3>,
                  &LaplaceOperator::template local_diagonal_face<3>,
                  &LaplaceOperator::template local_diagonal_boundary<3>,
                  this, inverse_diagonal_entries, dummy);
      break;
    case 4:
      data->loop (&LaplaceOperator::template local_diagonal_cell<4>,
                  &LaplaceOperator::template local_diagonal_face<4>,
                  &LaplaceOperator::template local_diagonal_boundary<4>,
                  this, inverse_diagonal_entries, dummy);
      break;
    case 5:
      data->loop (&LaplaceOperator::template local_diagonal_cell<5>,
                  &LaplaceOperator::template local_diagonal_face<5>,
                  &LaplaceOperator::template local_diagonal_boundary<5>,
                  this, inverse_diagonal_entries, dummy);
      break;
    case 6:
      data->loop (&LaplaceOperator::template local_diagonal_cell<6>,
                  &LaplaceOperator::template local_diagonal_face<6>,
                  &LaplaceOperator::template local_diagonal_boundary<6>,
                  this, inverse_diagonal_entries, dummy);
      break;
    case 7:
      data->loop (&LaplaceOperator::template local_diagonal_cell<7>,
                  &LaplaceOperator::template local_diagonal_face<7>,
                  &LaplaceOperator::template local_diagonal_boundary<7>,
                  this, inverse_diagonal_entries, dummy);
      break;
    case 8:
      data->loop (&LaplaceOperator::template local_diagonal_cell<8>,
                  &LaplaceOperator::template local_diagonal_face<8>,
                  &LaplaceOperator::template local_diagonal_boundary<8>,
                  this, inverse_diagonal_entries, dummy);
      break;
    case 9:
      data->loop (&LaplaceOperator::template local_diagonal_cell<9>,
                  &LaplaceOperator::template local_diagonal_face<9>,
                  &LaplaceOperator::template local_diagonal_boundary<9>,
                  this, inverse_diagonal_entries, dummy);
      break;
    case 10:
      data->loop (&LaplaceOperator::template local_diagonal_cell<10>,
                  &LaplaceOperator::template local_diagonal_face<10>,
                  &LaplaceOperator::template local_diagonal_boundary<10>,
                  this, inverse_diagonal_entries, dummy);
      break;
    default:
      AssertThrow(false, ExcMessage("Only polynomial degrees 0 up to 10 instantiated"));
    }

  if(pure_neumann_problem)
    {
      parallel::distributed::Vector<Number> vec1;
      vec1.reinit(inverse_diagonal_entries, true);
      for(unsigned int i=0;i<vec1.local_size();++i)
        vec1.local_element(i) = 1.;
      parallel::distributed::Vector<Number> d;
      d.reinit(inverse_diagonal_entries, true);
      vmult(d,vec1);
      double length = vec1*vec1;
      double factor = vec1*d;
      inverse_diagonal_entries.add(-2./length,d,factor/pow(length,2.),vec1);
    }

  for (unsigned int i=0; i<inverse_diagonal_entries.local_size(); ++i)
    if (std::abs(inverse_diagonal_entries.local_element(i)) > 1e-10)
      inverse_diagonal_entries.local_element(i) = 1./inverse_diagonal_entries.local_element(i);
    else
      inverse_diagonal_entries.local_element(i) = 1.;
}



template <int dim, typename Number>
template <int degree>
void LaplaceOperator<dim,Number>::
local_apply (const MatrixFree<dim,Number>                &data,
             parallel::distributed::Vector<Number>       &dst,
             const parallel::distributed::Vector<Number> &src,
             const std::pair<unsigned int,unsigned int>  &cell_range) const
{
  FEEvaluation<dim,degree,degree+1,1,Number> phi (data,
                                                  solver_data.pressure_dof_index,
                                                  solver_data.pressure_quad_index);

  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      phi.reinit (cell);
      phi.read_dof_values(src);
      phi.evaluate (false,true,false);
      for (unsigned int q=0; q<phi.n_q_points; ++q)
        phi.submit_gradient (phi.get_gradient(q), q);
      phi.integrate (false,true);
      phi.distribute_local_to_global (dst);
    }
}



template <int dim, typename Number>
template <int degree>
void LaplaceOperator<dim,Number>::
local_apply_face (const MatrixFree<dim,Number>                &data,
                  parallel::distributed::Vector<Number>       &dst,
                  const parallel::distributed::Vector<Number> &src,
                  const std::pair<unsigned int,unsigned int>  &face_range) const
{
  FEFaceEvaluation<dim,degree,degree+1,1,Number> fe_eval(data,true,
                                                         solver_data.pressure_dof_index,
                                                         solver_data.pressure_quad_index);
  FEFaceEvaluation<dim,degree,degree+1,1,Number> fe_eval_neighbor(data,false,
                                                                  solver_data.pressure_dof_index,
                                                                  solver_data.pressure_quad_index);

  for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval.reinit (face);
      fe_eval_neighbor.reinit (face);

      fe_eval.read_dof_values(src);
      fe_eval.evaluate(true,true);
      fe_eval_neighbor.read_dof_values(src);
      fe_eval_neighbor.evaluate(true,true);

      VectorizedArray<Number> sigmaF =
        std::max(fe_eval.read_cell_data(array_penalty_parameter),
                 fe_eval_neighbor.read_cell_data(array_penalty_parameter)) *
        get_penalty_factor();

      for(unsigned int q=0;q<fe_eval.n_q_points;++q)
        {
          VectorizedArray<Number> valueM = fe_eval.get_value(q);
          VectorizedArray<Number> valueP = fe_eval_neighbor.get_value(q);

          VectorizedArray<Number> jump_value = valueM - valueP;
          VectorizedArray<Number> average_gradient =
            ( fe_eval.get_normal_gradient(q) + fe_eval_neighbor.get_normal_gradient(q) ) * 0.5;
          average_gradient = average_gradient - jump_value * sigmaF;

          fe_eval.submit_normal_gradient(-0.5*jump_value,q);
          fe_eval_neighbor.submit_normal_gradient(-0.5*jump_value,q);
          fe_eval.submit_value(-average_gradient,q);
          fe_eval_neighbor.submit_value(average_gradient,q);
        }
      fe_eval.integrate(true,true);
      fe_eval.distribute_local_to_global(dst);
      fe_eval_neighbor.integrate(true,true);
      fe_eval_neighbor.distribute_local_to_global(dst);
    }
}



template <int dim, typename Number>
template <int degree>
void LaplaceOperator<dim,Number>::
local_apply_boundary (const MatrixFree<dim,Number>                &data,
                      parallel::distributed::Vector<Number>       &dst,
                      const parallel::distributed::Vector<Number> &src,
                      const std::pair<unsigned int,unsigned int>  &face_range) const
{
  FEFaceEvaluation<dim,degree,degree+1,1,Number> fe_eval(data, true,
                                                         solver_data.pressure_dof_index,
                                                         solver_data.pressure_quad_index);
  for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval.reinit (face);
      fe_eval.read_dof_values(src);
      fe_eval.evaluate(true,true);
      VectorizedArray<Number> sigmaF =
        fe_eval.read_cell_data(array_penalty_parameter) *
        get_penalty_factor();
      const bool is_dirichlet =
        solver_data.dirichlet_boundaries.find(data.get_boundary_indicator(face)) !=
        solver_data.dirichlet_boundaries.end();

      for(unsigned int q=0;q<fe_eval.n_q_points;++q)
        {
          if (!is_dirichlet) // Neumann boundaries
            {
              //set gradient in normal direction to zero, i.e. u+ = ue-, grad+ = -grad-
              VectorizedArray<Number> jump_value = make_vectorized_array<Number>(0.0);
              VectorizedArray<Number> average_gradient = make_vectorized_array<Number>(0.0);
              average_gradient = average_gradient - jump_value * sigmaF;

              fe_eval.submit_normal_gradient(-0.5*jump_value,q);
              fe_eval.submit_value(-average_gradient,q);
            }
          else // Dirichlet boundaries
            {
              //set value to zero, i.e. u+ = - u- , grad+ = grad-
              VectorizedArray<Number> valueM = fe_eval.get_value(q);

              VectorizedArray<Number> jump_value = 2.0*valueM;
              VectorizedArray<Number> average_gradient = fe_eval.get_normal_gradient(q);
              average_gradient = average_gradient - jump_value * sigmaF;

              fe_eval.submit_normal_gradient(-0.5*jump_value,q);
              fe_eval.submit_value(-average_gradient,q);
            }
        }

      fe_eval.integrate(true,true);
      fe_eval.distribute_local_to_global(dst);
    }
}



template <int dim, typename Number>
template <int degree>
void LaplaceOperator<dim,Number>::
local_diagonal_cell (const MatrixFree<dim,Number>                &data,
                     parallel::distributed::Vector<Number>       &dst,
                     const unsigned int  &,
                     const std::pair<unsigned int,unsigned int>  &cell_range) const
{
  FEEvaluation<dim,degree,degree+1,1,Number> phi (data,
                                                  solver_data.pressure_dof_index,
                                                  solver_data.pressure_quad_index);

  VectorizedArray<Number> local_diagonal_vector[phi.tensor_dofs_per_cell];
  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      phi.reinit (cell);

      for (unsigned int i=0; i<phi.dofs_per_cell; ++i)
        {
          for (unsigned int j=0; j<phi.dofs_per_cell; ++j)
            phi.begin_dof_values()[j] = VectorizedArray<Number>();
          phi.begin_dof_values()[i] = 1.;
          phi.evaluate (false,true,false);
          for (unsigned int q=0; q<phi.n_q_points; ++q)
            phi.submit_gradient (phi.get_gradient(q), q);
          phi.integrate (false,true);
          local_diagonal_vector[i] = phi.begin_dof_values()[i];
        }
      for (unsigned int i=0; i<phi.tensor_dofs_per_cell; ++i)
        phi.begin_dof_values()[i] = local_diagonal_vector[i];
      phi.distribute_local_to_global (dst);
    }
}



template <int dim, typename Number>
template <int degree>
void LaplaceOperator<dim,Number>::
local_diagonal_face (const MatrixFree<dim,Number>                &data,
                     parallel::distributed::Vector<Number>       &dst,
                     const unsigned int  &,
                     const std::pair<unsigned int,unsigned int>  &face_range) const
{
  FEFaceEvaluation<dim,degree,degree+1,1,Number> phi(data,true,
                                                     solver_data.pressure_dof_index,
                                                     solver_data.pressure_quad_index);
  FEFaceEvaluation<dim,degree,degree+1,1,Number> phi_outer(data,false,
                                                           solver_data.pressure_dof_index,
                                                           solver_data.pressure_quad_index);

  VectorizedArray<Number> local_diagonal_vector[phi.tensor_dofs_per_cell];
  for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      phi.reinit (face);
      phi_outer.reinit (face);

      VectorizedArray<Number> sigmaF =
        std::max(phi.read_cell_data(array_penalty_parameter),
                 phi_outer.read_cell_data(array_penalty_parameter)) *
        get_penalty_factor();

      // element-
      for (unsigned int j=0; j<phi.dofs_per_cell; ++j)
        phi_outer.begin_dof_values()[j] = VectorizedArray<Number>();
      phi_outer.evaluate(true, true);
      for (unsigned int i=0; i<phi.dofs_per_cell; ++i)
        {
          for (unsigned int j=0; j<phi.dofs_per_cell; ++j)
            phi.begin_dof_values()[j] = VectorizedArray<Number>();
          phi.begin_dof_values()[i] = 1.;
          phi.evaluate(true,true);

          for(unsigned int q=0;q<phi.n_q_points;++q)
            {
              VectorizedArray<Number> valueM = phi.get_value(q);
              VectorizedArray<Number> valueP = phi_outer.get_value(q);

              VectorizedArray<Number> jump_value = valueM - valueP;
              VectorizedArray<Number> average_gradient =
                ( phi.get_normal_gradient(q) + phi_outer.get_normal_gradient(q) ) * 0.5;
              average_gradient = average_gradient - jump_value * sigmaF;

              phi.submit_normal_gradient(-0.5*jump_value,q);
              phi.submit_value(-average_gradient,q);
            }
          phi.integrate(true,true);
          local_diagonal_vector[i] = phi.begin_dof_values()[i];
        }
      for (unsigned int i=0; i<phi.dofs_per_cell; ++i)
        phi.begin_dof_values()[i] = local_diagonal_vector[i];
      phi.distribute_local_to_global(dst);

      // neighbor (element+)
      for (unsigned int j=0; j<phi.dofs_per_cell; ++j)
        phi.begin_dof_values()[j] = VectorizedArray<Number>();
      phi.evaluate(true, true);
      for (unsigned int i=0; i<phi.dofs_per_cell; ++i)
        {
          for (unsigned int j=0; j<phi.dofs_per_cell; ++j)
            phi_outer.begin_dof_values()[j] = VectorizedArray<Number>();
          phi_outer.begin_dof_values()[i] = 1.;
          phi_outer.evaluate(true,true);

          for(unsigned int q=0;q<phi.n_q_points;++q)
            {
              VectorizedArray<Number> valueM = phi.get_value(q);
              VectorizedArray<Number> valueP = phi_outer.get_value(q);

              VectorizedArray<Number> jump_value = valueM - valueP;
              VectorizedArray<Number> average_gradient =
                ( phi.get_normal_gradient(q) + phi_outer.get_normal_gradient(q) ) * 0.5;
              average_gradient = average_gradient - jump_value * sigmaF;

              phi_outer.submit_normal_gradient(-0.5*jump_value,q);
              phi_outer.submit_value(average_gradient,q);
            }
          phi_outer.integrate(true,true);
          local_diagonal_vector[i] = phi_outer.begin_dof_values()[i];
        }
      for (unsigned int i=0; i<phi.dofs_per_cell; ++i)
        phi_outer.begin_dof_values()[i] = local_diagonal_vector[i];
      phi_outer.distribute_local_to_global(dst);
    }
}



template <int dim, typename Number>
template <int degree>
void LaplaceOperator<dim,Number>::
local_diagonal_boundary (const MatrixFree<dim,Number>                &data,
                         parallel::distributed::Vector<Number>       &dst,
                         const unsigned int  &,
                         const std::pair<unsigned int,unsigned int>  &face_range) const
{
  FEFaceEvaluation<dim,degree,degree+1,1,Number> phi (data, true,
                                                      solver_data.pressure_dof_index,
                                                      solver_data.pressure_quad_index);

  VectorizedArray<Number> local_diagonal_vector[phi.tensor_dofs_per_cell];
  for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      phi.reinit (face);

      VectorizedArray<Number> sigmaF =
        phi.read_cell_data(array_penalty_parameter) *
        get_penalty_factor();
      const bool is_dirichlet =
        solver_data.dirichlet_boundaries.find(data.get_boundary_indicator(face)) !=
        solver_data.dirichlet_boundaries.end();

      for (unsigned int i=0; i<phi.dofs_per_cell; ++i)
        {
          for (unsigned int j=0; j<phi.dofs_per_cell; ++j)
            phi.begin_dof_values()[j] = VectorizedArray<Number>();
          phi.begin_dof_values()[i] = 1.;
          phi.evaluate(true,true);

          for(unsigned int q=0;q<phi.n_q_points;++q)
            if (!is_dirichlet) // Neumann boundaries
              {
                //set solution gradient in normal direction to zero, i.e. u+ = u-, grad+ = -grad-
                VectorizedArray<Number> jump_value = make_vectorized_array<Number>(0.0);
                VectorizedArray<Number> average_gradient = make_vectorized_array<Number>(0.0);
                average_gradient = average_gradient - jump_value * sigmaF;

                phi.submit_normal_gradient(-0.5*jump_value,q);
                phi.submit_value(-average_gradient,q);
              }
            else // Dirichlet
              {
                //set value to zero, i.e. u+ = - u- , grad+ = grad-
                VectorizedArray<Number> valueM = phi.get_value(q);

                VectorizedArray<Number> jump_value = 2.0*valueM;
                VectorizedArray<Number> average_gradient = phi.get_normal_gradient(q);
                average_gradient = average_gradient - jump_value * sigmaF;

                phi.submit_normal_gradient(-0.5*jump_value,q);
                phi.submit_value(-average_gradient,q);
              }

          phi.integrate(true,true);
          local_diagonal_vector[i] = phi.begin_dof_values()[i];
        }
      for (unsigned int i=0; i<phi.dofs_per_cell; ++i)
        phi.begin_dof_values()[i] = local_diagonal_vector[i];
      phi.distribute_local_to_global(dst);
    }
}



template <typename Number>
class JacobiPreconditioner
{
public:
  JacobiPreconditioner (const parallel::distributed::Vector<Number> &inv_diagonal)
    :
    inverse_diagonal(inv_diagonal)
  {}

  void vmult(parallel::distributed::Vector<Number> &dst,
             const parallel::distributed::Vector<Number> &src) const
  {
    if (!PointerComparison::equal(&dst, &src))
      dst = src;
    dst.scale(inverse_diagonal);
  }

private:
  const parallel::distributed::Vector<Number> &inverse_diagonal;
};



template<typename Operator>
class MGCoarseIterative : public MGCoarseGridBase<parallel::distributed::Vector<typename Operator::value_type> >
{
public:
  MGCoarseIterative(const Operator &matrix,
                    const parallel::distributed::Vector<typename Operator::value_type> *inv_diagonal)
    :
    coarse_matrix (matrix),
    use_jacobi (inv_diagonal != 0)
  {
    if (use_jacobi)
      {
        inverse_diagonal = *inv_diagonal;
        AssertDimension(inverse_diagonal.size(), coarse_matrix.m());
      }
  }

  virtual void operator() (const unsigned int,
                           parallel::distributed::Vector<typename Operator::value_type> &dst,
                           const parallel::distributed::Vector<typename Operator::value_type> &src) const
  {
    ReductionControl solver_control (1e4, 1e-50, 1e-4);
    SolverCG<parallel::distributed::Vector<typename Operator::value_type> >
      solver_coarse (solver_control, solver_memory);
    if (use_jacobi)
      {
        JacobiPreconditioner<typename Operator::value_type> preconditioner(inverse_diagonal);
        solver_coarse.solve (coarse_matrix, dst, src, preconditioner);
      }
    else
      solver_coarse.solve (coarse_matrix, dst, src, PreconditionIdentity());
  }

private:
  const Operator &coarse_matrix;
  parallel::distributed::Vector<typename Operator::value_type> inverse_diagonal;
  mutable GrowingVectorMemory<parallel::distributed::Vector<typename Operator::value_type> > solver_memory;
  const bool use_jacobi;
};



template<typename VECTOR>
class MGCoarseFromSmoother : public MGCoarseGridBase<VECTOR>
{
public:
  MGCoarseFromSmoother(const MGSmootherBase<VECTOR> &mg_smoother)
    : smoother(mg_smoother)
  {}

  virtual void operator() (const unsigned int   level,
                           VECTOR &dst,
                           const VECTOR &src) const
  {
    Assert(level == 0, ExcNotImplemented());
    smoother.smooth(level, dst, src);
  }

  const MGSmootherBase<VECTOR> &smoother;
};



template <int dim>
void PoissonSolver<dim>::initialize (const Mapping<dim> &mapping,
                                     const MatrixFree<dim,double> &matrix_free,
                                     const PoissonSolverData<dim> &solver_data)
{
  global_matrix.reinit(matrix_free, mapping, solver_data);

  const DoFHandler<dim> &dof_handler = matrix_free.get_dof_handler(solver_data.pressure_dof_index);
  const parallel::Triangulation<dim> *tria =
    dynamic_cast<const parallel::Triangulation<dim> *>(&dof_handler.get_triangulation());
  AssertThrow(tria != 0, ExcMessage("Only works for distributed triangulations"));
  mpi_communicator = tria->get_communicator();

  mg_matrices.resize(0, tria->n_global_levels()-1);
  MGLevelObject<typename SMOOTHER::AdditionalData> smoother_data;
  smoother_data.resize(0, dof_handler.get_triangulation().n_global_levels()-1);
  for (unsigned int level = 0; level<tria->n_global_levels(); ++level)
    {
      mg_matrices[level].reinit(dof_handler, mapping, solver_data, level);

      if (level > 0)
        {
          smoother_data[level].smoothing_range = solver_data.smoother_smoothing_range;
          smoother_data[level].degree = solver_data.smoother_poly_degree;
          smoother_data[level].eig_cg_n_iterations = 20;
        }
      else
        {
          // TODO: here we would like to have an adaptive choice...
          smoother_data[level].smoothing_range = 0.;
          if (dof_handler.n_dofs(0) > 2000)
            {
              smoother_data[level].degree = 100;
              smoother_data[level].eig_cg_n_iterations = 200;
            }
          else
            {
              smoother_data[level].degree = 40;
              smoother_data[level].eig_cg_n_iterations = 100;
            }
        }
      mg_matrices[level].compute_inverse_diagonal(smoother_data[level].matrix_diagonal_inverse);
    }

  mg_smoother.initialize(mg_matrices, smoother_data);

  switch (solver_data.coarse_solver)
    {
    case PoissonSolverData<dim>::coarse_chebyshev_smoother:
      {
        mg_coarse.reset(new MGCoarseFromSmoother<parallel::distributed::Vector<Number> >(mg_smoother));
        break;
      }
    case PoissonSolverData<dim>::coarse_iterative_noprec:
      {
        mg_coarse.reset(new MGCoarseIterative<LevelMatrixType>(mg_matrices[0], 0));
        break;
      }
    case PoissonSolverData<dim>::coarse_iterative_jacobi:
      {
        mg_coarse.reset(new MGCoarseIterative<LevelMatrixType>(mg_matrices[0], &smoother_data[0].matrix_diagonal_inverse));
        break;
      }
    default:
      AssertThrow(false, ExcMessage("Unknown coarse-grid solver given"));
    }

  mg_transfer.set_laplace_operator(mg_matrices);
  mg_transfer.build(dof_handler);

  mg_matrix.reset(new mg::Matrix<parallel::distributed::Vector<Number> > (mg_matrices));

  mg.reset(new Multigrid<parallel::distributed::Vector<Number> > (dof_handler,
                                                                  *mg_matrix,
                                                                  *mg_coarse,
                                                                  mg_transfer,
                                                                  mg_smoother,
                                                                  mg_smoother));

  preconditioner.reset(new PreconditionMG<dim, parallel::distributed::Vector<Number>,
                       MGTransferMF<dim,LevelMatrixType> >
                       (dof_handler, *mg, mg_transfer));

  {
    Utilities::System::MemoryStats stats;
    Utilities::System::get_memory_stats(stats);
    Utilities::MPI::MinMaxAvg memory =
      Utilities::MPI::min_max_avg (stats.VmRSS/1024., tria->get_communicator());
    if (Utilities::MPI::this_mpi_process(tria->get_communicator()) == 0)
      std::cout << "Memory stats [MB]: " << memory.min
                << " [p" << memory.min_index << "] "
                << memory.avg << " " << memory.max
                << " [p" << memory.max_index << "]"
                << std::endl;
  }
}

template <int dim>
unsigned int
PoissonSolver<dim>::solve (parallel::distributed::Vector<double> &dst,
                           const parallel::distributed::Vector<double> &src) const
  {
    Assert(preconditioner.get() != 0,
           ExcNotInitialized());
    ReductionControl solver_control (1e5, 1.e-12, global_matrix.get_solver_data().solver_tolerance); //1.e-5
    GrowingVectorMemory<parallel::distributed::Vector<double> > solver_memory;
    SolverCG<parallel::distributed::Vector<double> > solver (solver_control, solver_memory);
    try
      {
        solver.solve(global_matrix, dst, src, *preconditioner);
      }
    catch (SolverControl::NoConvergence)
      {
        if(Utilities::MPI::this_mpi_process(mpi_communicator)==0)
          std::cout<<"Multigrid failed trying to solve the pressure poisson equation." << std::endl;
      }
    return solver_control.last_step();
  }


// explicit instantiations
template class LaplaceOperator<2,double>;
template class LaplaceOperator<3,double>;
template class LaplaceOperator<2,float>;
template class LaplaceOperator<3,float>;
template class PoissonSolver<2>;
template class PoissonSolver<3>;
