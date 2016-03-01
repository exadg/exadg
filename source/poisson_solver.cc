

#include "poisson_solver.h"

#include <deal.II/base/function_lib.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/distributed/tria_base.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/numerics/vector_tools.h>

template <int dim, typename Number>
LaplaceOperator<dim,Number>::LaplaceOperator ()
  :
  data (0),
  fe_degree (numbers::invalid_unsigned_int),
  needs_mean_value_constraint (false),
  apply_mean_value_constraint_in_matvec (false)
{}



template <int dim, typename Number>
void LaplaceOperator<dim,Number>::clear()
{
  solver_data = PoissonSolverData<dim>();
  data = 0;
  fe_degree = numbers::invalid_unsigned_int;
  needs_mean_value_constraint = false;
  apply_mean_value_constraint_in_matvec = false;
  own_matrix_free_storage.clear();
  tmp_projection_vector.reinit(0);
  have_interface_matrices = false;
}



template <int dim, typename Number>
void LaplaceOperator<dim,Number>::reinit(const MatrixFree<dim,Number>       &mf_data,
                                         const Mapping<dim>                 &mapping,
                                         const PoissonSolverData<dim>       &solver_data)
{
  this->data = &mf_data;
  this->solver_data = solver_data;
  this->fe_degree = mf_data.get_dof_handler(solver_data.poisson_dof_index).get_fe().degree;
  AssertThrow (Utilities::fixed_power<dim>(fe_degree+1) ==
               mf_data.get_n_q_points(solver_data.poisson_quad_index),
               ExcMessage("Expected fe_degree+1 quadrature points"));

  compute_array_penalty_parameter(mapping);

  // Check whether the Poisson matrix is singular when applied to a vector
  // consisting of only ones (except for constrained entries)
  parallel::distributed::Vector<Number> in_vec, out_vec;
  initialize_dof_vector(in_vec);
  initialize_dof_vector(out_vec);
  in_vec = 1;
  const std::vector<unsigned int> &constrained_entries =
    mf_data.get_constrained_dofs(solver_data.poisson_dof_index);
  for (unsigned int i=0; i<constrained_entries.size(); ++i)
    in_vec.local_element(constrained_entries[i]) = 0;
  vmult_add(out_vec, in_vec);
  const double linfty_norm = out_vec.linfty_norm();

  // since we cannot know the magnitude of the entries at this point (the
  // diagonal entries would be a guideline but they are not available here),
  // we instead multiply by a random vector
  for (unsigned int i=0; i<in_vec.local_size(); ++i)
    in_vec.local_element(i) = (double)rand()/RAND_MAX;
  vmult(out_vec, in_vec);
  const double linfty_norm_compare = out_vec.linfty_norm();

  // use mean value constraint if the infty norm with the one vector is very
  // small
  needs_mean_value_constraint =
    linfty_norm / linfty_norm_compare < std::pow(std::numeric_limits<Number>::epsilon(), 2./3.);
  apply_mean_value_constraint_in_matvec = needs_mean_value_constraint;
}



namespace
{
  template <int dim>
  void add_periodicity_constraints(const unsigned int level,
                                   const unsigned int target_level,
                                   const typename DoFHandler<dim>::face_iterator face1,
                                   const typename DoFHandler<dim>::face_iterator face2,
                                   ConstraintMatrix &constraints)
  {
    if (level == 0)
      {
        const unsigned int dofs_per_face = face1->get_fe(0).dofs_per_face;
        std::vector<types::global_dof_index> dofs_1(dofs_per_face);
        std::vector<types::global_dof_index> dofs_2(dofs_per_face);

        face1->get_mg_dof_indices(target_level, dofs_1, 0);
        face2->get_mg_dof_indices(target_level, dofs_2, 0);

        for (unsigned int i=0; i<dofs_per_face; ++i)
          if (constraints.can_store_line(dofs_2[i]) &&
              constraints.can_store_line(dofs_1[i]) &&
              !constraints.is_constrained(dofs_2[i]))
            {
              constraints.add_line(dofs_2[i]);
              constraints.add_entry(dofs_2[i], dofs_1[i], 1.);
            }
      }
    else if (face1->has_children() && face2->has_children())
      {
        for (unsigned int c=0; c<face1->n_children(); ++c)
          add_periodicity_constraints<dim>(level-1, target_level, face1->child(c),
                                           face2->child(c), constraints);
      }
  }
}



template <int dim, typename Number>
void LaplaceOperator<dim,Number>::reinit (const DoFHandler<dim> &dof_handler,
                                          const Mapping<dim> &mapping,
                                          const PoissonSolverData<dim> &solver_data,
                                          const MGConstrainedDoFs &mg_constrained_dofs,
                                          const unsigned int level)
{
  clear();
  this->solver_data = solver_data;

  const QGauss<1> quad(dof_handler.get_fe().degree+1);
  typename MatrixFree<dim,Number>::AdditionalData addit_data;
  addit_data.tasks_parallel_scheme = MatrixFree<dim,Number>::AdditionalData::none;
  addit_data.build_face_info = true;
  addit_data.level_mg_handler = level;
  addit_data.mpi_communicator =
    dynamic_cast<const parallel::Triangulation<dim> *>(&dof_handler.get_triangulation()) ?
    (dynamic_cast<const parallel::Triangulation<dim> *>(&dof_handler.get_triangulation()))->get_communicator() : MPI_COMM_SELF;
  addit_data.periodic_face_pairs_level_0 = solver_data.periodic_face_pairs_level0;

  ConstraintMatrix constraints;
  const bool is_feq = dof_handler.get_fe().dofs_per_vertex > 0;

  // For continuous elements, add the constraints due to hanging nodes and
  // boundary conditions
  if (is_feq && level == numbers::invalid_unsigned_int)
    {
      ZeroFunction<dim> zero_function(dof_handler.get_fe().n_components());
      typename FunctionMap<dim>::type dirichlet_boundary;
      for (std::set<types::boundary_id>::const_iterator it =
             solver_data.dirichlet_boundaries.begin();
           it != solver_data.dirichlet_boundaries.end(); ++it)
        dirichlet_boundary[*it] = &zero_function;

      IndexSet relevant_dofs;
      DoFTools::extract_locally_relevant_dofs(dof_handler, relevant_dofs);
      constraints.reinit(relevant_dofs);
      DoFTools::make_hanging_node_constraints(dof_handler, constraints);

      // add periodicity constraints
      std::vector<GridTools::PeriodicFacePair<typename DoFHandler<dim>::cell_iterator> > periodic_faces;
      for (typename std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> >::const_iterator
             it = solver_data.periodic_face_pairs_level0.begin();
           it != solver_data.periodic_face_pairs_level0.end(); ++it)
        {
          GridTools::PeriodicFacePair<typename DoFHandler<dim>::cell_iterator> periodic;
          for (unsigned int i=0; i<2; ++i)
            {
              periodic.cell[i] = typename DoFHandler<dim>::cell_iterator
                (&dof_handler.get_triangulation(),
                 it->cell[i]->level(), it->cell[i]->index(), &dof_handler);
              periodic.face_idx[i] = it->face_idx[i];
            }
          periodic.orientation = it->orientation;
          periodic.matrix = it->matrix;
          periodic_faces.push_back(periodic);
        }
      DoFTools::make_periodicity_constraints<DoFHandler<dim> > (periodic_faces,
                                                                constraints);

      VectorTools::interpolate_boundary_values(dof_handler, dirichlet_boundary,
                                               constraints);
    }
  else if (is_feq)
    {
      IndexSet relevant_dofs;
      DoFTools::extract_locally_relevant_level_dofs(dof_handler, level,
                                                    relevant_dofs);
      constraints.reinit(relevant_dofs);

      // add periodicity constraints
      for (typename std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> >::const_iterator
             it = solver_data.periodic_face_pairs_level0.begin();
           it != solver_data.periodic_face_pairs_level0.end(); ++it)
        {
          typename DoFHandler<dim>::cell_iterator
            cell1(&dof_handler.get_triangulation(), 0, it->cell[1]->index(), &dof_handler),
            cell0(&dof_handler.get_triangulation(), 0, it->cell[0]->index(), &dof_handler);
          add_periodicity_constraints<dim>(level, level,
                                           cell1->face(it->face_idx[1]),
                                           cell0->face(it->face_idx[0]),
                                           constraints);
        }

      constraints.add_lines(mg_constrained_dofs.get_boundary_indices(level));

      std::vector<types::global_dof_index> interface_indices;
      mg_constrained_dofs.get_refinement_edge_indices(level).fill_index_vector(interface_indices);
      edge_constrained_indices.clear();
      edge_constrained_indices.reserve(interface_indices.size());
      edge_constrained_values.resize(interface_indices.size());
      const IndexSet &locally_owned = dof_handler.locally_owned_mg_dofs(level);
      for (unsigned int i=0; i<interface_indices.size(); ++i)
        if (locally_owned.is_element(interface_indices[i]))
          edge_constrained_indices.push_back(locally_owned.index_within_set(interface_indices[i]));
      have_interface_matrices = Utilities::MPI::max((unsigned int)edge_constrained_indices.size(), MPI_COMM_WORLD) > 0;
    }

  // constraint zeroth DoF in continuous case (the mean value constraint will
  // be applied in the DG case). In case we have interface matrices, there are
  // Dirichlet constraints on parts of the boundary and no such transformation
  // is required.
  if (verify_boundary_conditions(dof_handler, solver_data)
      && is_feq && !have_interface_matrices && constraints.can_store_line(0))
    {
      // if dof 0 is constrained, it must be a periodic dof, so we take the
      // value on the other side
      types::global_dof_index line_index = 0;
      while (true)
        {
          const std::vector<std::pair<types::global_dof_index,double> >* lines =
            constraints.get_constraint_entries(line_index);
          if (lines == 0)
            {
              constraints.add_line(line_index);
              // add the constraint back to the MGConstrainedDoFs field. This
              // is potentially dangerous but we know what we are doing... ;-)
              if (level != numbers::invalid_unsigned_int)
                const_cast<IndexSet &>(mg_constrained_dofs.get_boundary_indices(level))
                  .add_index(line_index);
              break;
            }
          else
            {
              Assert(lines->size() == 1 && std::abs((*lines)[0].second-1.)<1e-15,
                     ExcMessage("Periodic index expected, bailing out"));
              line_index = (*lines)[0].first;
            }
        }
    }

  constraints.close();

  PoissonSolverData<dim> my_solver_data = solver_data;
  my_solver_data.poisson_dof_index = 0;
  my_solver_data.poisson_quad_index = 0;

  own_matrix_free_storage.reinit(mapping, dof_handler, constraints, quad,
                                 addit_data);

  reinit(own_matrix_free_storage, mapping, my_solver_data);
}



template <int dim, typename Number>
bool LaplaceOperator<dim,Number>
::verify_boundary_conditions(const DoFHandler<dim>        &dof_handler,
                             const PoissonSolverData<dim> &solver_data)
{
  // Check that the Dirichlet and Neumann boundary conditions do not overlap
  std::set<types::boundary_id> periodic_boundary_ids;
  for (unsigned int i=0; i<solver_data.periodic_face_pairs_level0.size(); ++i)
    {
      AssertThrow(solver_data.periodic_face_pairs_level0[i].cell[0]->level() == 0,
                  ExcMessage("Received periodic cell pairs on non-zero level"));
      periodic_boundary_ids.insert(solver_data.periodic_face_pairs_level0[i].cell[0]->face(solver_data.periodic_face_pairs_level0[i].face_idx[0])->boundary_id());
      periodic_boundary_ids.insert(solver_data.periodic_face_pairs_level0[i].cell[1]->face(solver_data.periodic_face_pairs_level0[i].face_idx[1])->boundary_id());
    }

  bool pure_neumann_problem = true;
  const Triangulation<dim> &tria = dof_handler.get_triangulation();
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
                                 " does neither set Dirichlet, Neumann, nor periodic " +
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
  int min_pure_neumann = Utilities::MPI::min(my_neumann, mpi_communicator);
  AssertThrow(max_pure_neumann == min_pure_neumann,
              ExcMessage("Neumann/Dirichlet assignment over processors does not match."));

  return pure_neumann_problem;
}



template <int dim, typename Number>
void LaplaceOperator<dim,Number>::compute_array_penalty_parameter(const Mapping<dim> &mapping)
{
  std::set<types::boundary_id> periodic_boundary_ids;
  for (unsigned int i=0; i<solver_data.periodic_face_pairs_level0.size(); ++i)
    {
      AssertThrow(solver_data.periodic_face_pairs_level0[i].cell[0]->level() == 0,
                  ExcMessage("Received periodic cell pairs on non-zero level"));
      periodic_boundary_ids.insert(solver_data.periodic_face_pairs_level0[i].cell[0]->face(solver_data.periodic_face_pairs_level0[i].face_idx[0])->boundary_id());
      periodic_boundary_ids.insert(solver_data.periodic_face_pairs_level0[i].cell[1]->face(solver_data.periodic_face_pairs_level0[i].face_idx[1])->boundary_id());
    }

  // Compute penalty parameter for each cell
  array_penalty_parameter.resize(data->n_macro_cells()+data->n_macro_ghost_cells());
  QGauss<dim> quadrature(fe_degree+1);
  FEValues<dim> fe_values(mapping,
                          data->get_dof_handler(solver_data.poisson_dof_index).get_fe(),
                          quadrature, update_JxW_values);
  QGauss<dim-1> face_quadrature(fe_degree+1);
  FEFaceValues<dim> fe_face_values(mapping, data->get_dof_handler(solver_data.poisson_dof_index).get_fe(), face_quadrature, update_JxW_values);

  for (unsigned int i=0; i<data->n_macro_cells()+data->n_macro_ghost_cells(); ++i)
    for (unsigned int v=0; v<data->n_components_filled(i); ++v)
      {
        typename DoFHandler<dim>::cell_iterator cell = data->get_cell_iterator(i,v,solver_data.poisson_dof_index);
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
void LaplaceOperator<dim,Number>::disable_mean_value_constraint()
{
  this->apply_mean_value_constraint_in_matvec = false;
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
  const parallel::distributed::Vector<Number> *actual_src = &src;
  if(apply_mean_value_constraint_in_matvec)
    {
      tmp_projection_vector = src;
      apply_nullspace_projection(tmp_projection_vector);
      actual_src = &tmp_projection_vector;
    }

  // For continuous elements: set zero Dirichlet values on the input vector
  // (and remember the src and dst values because we need to reset them at the
  // end). Note that we should only have edge constrained indices for non-pure
  // Neumann problems.
  for (unsigned int i=0; i<edge_constrained_indices.size(); ++i)
    {
      Assert(!apply_mean_value_constraint_in_matvec, ExcInternalError());

      edge_constrained_values[i] =
        std::pair<Number,Number>(src.local_element(edge_constrained_indices[i]),
                                 dst.local_element(edge_constrained_indices[i]));
      const_cast<parallel::distributed::Vector<Number>&>(src).local_element(edge_constrained_indices[i]) = 0.;
    }

  run_vmult_loop(dst, *actual_src);

  // Apply Dirichlet boundary conditions in the continuous case by simulating
  // a one in the diagonal (note that the ConstraintMatrix passed to the
  // MatrixFree object takes care of Dirichlet conditions on outer
  // (non-refinement edge) boundaries)
  const std::vector<unsigned int> &
    constrained_dofs = data->get_constrained_dofs(solver_data.poisson_dof_index);
  for (unsigned int i=0; i<constrained_dofs.size(); ++i)
    dst.local_element(constrained_dofs[i]) += src.local_element(constrained_dofs[i]);

  // reset edge constrained values, multiply by unit matrix and add into
  // destination
  for (unsigned int i=0; i<edge_constrained_indices.size(); ++i)
    {
      const_cast<parallel::distributed::Vector<Number>&>(src).local_element(edge_constrained_indices[i]) = edge_constrained_values[i].first;
      dst.local_element(edge_constrained_indices[i]) = edge_constrained_values[i].second + edge_constrained_values[i].first;
    }

  if (apply_mean_value_constraint_in_matvec)
    apply_nullspace_projection(dst);
}



template <int dim, typename Number>
void LaplaceOperator<dim,Number>::run_vmult_loop(parallel::distributed::Vector<Number> &dst,
                                                 const parallel::distributed::Vector<Number> &src) const
{
  Assert(src.partitioners_are_globally_compatible(*data->get_dof_info(solver_data.poisson_dof_index).vector_partitioner), ExcInternalError());
  Assert(dst.partitioners_are_globally_compatible(*data->get_dof_info(solver_data.poisson_dof_index).vector_partitioner), ExcInternalError());

  switch (fe_degree)
    {
    case 0:
      data->loop (&LaplaceOperator::template local_apply<0>,
                  &LaplaceOperator::template local_apply_face<0>,
                  &LaplaceOperator::template local_apply_boundary<0>,
                  this, dst, src);
      break;
    case 1:
      data->loop (&LaplaceOperator::template local_apply<1>,
                  &LaplaceOperator::template local_apply_face<1>,
                  &LaplaceOperator::template local_apply_boundary<1>,
                  this, dst, src);
      break;
    case 2:
      data->loop (&LaplaceOperator::template local_apply<2>,
                  &LaplaceOperator::template local_apply_face<2>,
                  &LaplaceOperator::template local_apply_boundary<2>,
                  this, dst, src);
      break;
    case 3:
      data->loop (&LaplaceOperator::template local_apply<3>,
                  &LaplaceOperator::template local_apply_face<3>,
                  &LaplaceOperator::template local_apply_boundary<3>,
                  this, dst, src);
      break;
    case 4:
      data->loop (&LaplaceOperator::template local_apply<4>,
                  &LaplaceOperator::template local_apply_face<4>,
                  &LaplaceOperator::template local_apply_boundary<4>,
                  this, dst, src);
      break;
    case 5:
      data->loop (&LaplaceOperator::template local_apply<5>,
                  &LaplaceOperator::template local_apply_face<5>,
                  &LaplaceOperator::template local_apply_boundary<5>,
                  this, dst, src);
      break;
    case 6:
      data->loop (&LaplaceOperator::template local_apply<6>,
                  &LaplaceOperator::template local_apply_face<6>,
                  &LaplaceOperator::template local_apply_boundary<6>,
                  this, dst, src);
      break;
    case 7:
      data->loop (&LaplaceOperator::template local_apply<7>,
                  &LaplaceOperator::template local_apply_face<7>,
                  &LaplaceOperator::template local_apply_boundary<7>,
                  this, dst, src);
      break;
    case 8:
      data->loop (&LaplaceOperator::template local_apply<8>,
                  &LaplaceOperator::template local_apply_face<8>,
                  &LaplaceOperator::template local_apply_boundary<8>,
                  this, dst, src);
      break;
    case 9:
      data->loop (&LaplaceOperator::template local_apply<9>,
                  &LaplaceOperator::template local_apply_face<9>,
                  &LaplaceOperator::template local_apply_boundary<9>,
                  this, dst, src);
      break;
    case 10:
      data->loop (&LaplaceOperator::template local_apply<10>,
                  &LaplaceOperator::template local_apply_face<10>,
                  &LaplaceOperator::template local_apply_boundary<10>,
                  this, dst, src);
      break;
    default:
      AssertThrow(false, ExcMessage("Only polynomial degrees 0 up to 10 instantiated"));
    }
}



template <int dim, typename Number>
void LaplaceOperator<dim,Number>
::vmult_interface_down(parallel::distributed::Vector<Number> &dst,
                       const parallel::distributed::Vector<Number> &src) const
{
  dst = 0;

  if (!have_interface_matrices)
    return;

  // set zero Dirichlet values on the input vector (and remember the src and
  // dst values because we need to reset them at the end)
  for (unsigned int i=0; i<edge_constrained_indices.size(); ++i)
    {
      const double src_val = src.local_element(edge_constrained_indices[i]);
      const_cast<parallel::distributed::Vector<Number>&>(src).local_element(edge_constrained_indices[i]) = 0.;
      edge_constrained_values[i] = std::pair<Number,Number>(src_val,
                                                            dst.local_element(edge_constrained_indices[i]));
    }

  run_vmult_loop(dst, src);

  unsigned int c=0;
  for (unsigned int i=0; i<edge_constrained_indices.size(); ++i)
    {
      for ( ; c<edge_constrained_indices[i]; ++c)
        dst.local_element(c) = 0.;
      ++c;

      // reset the src values
      const_cast<parallel::distributed::Vector<Number>&>(src).local_element(edge_constrained_indices[i]) = edge_constrained_values[i].first;
    }
  for ( ; c<dst.local_size(); ++c)
    dst.local_element(c) = 0.;
}



template <int dim, typename Number>
void LaplaceOperator<dim,Number>
::vmult_interface_up(parallel::distributed::Vector<Number> &dst,
                     const parallel::distributed::Vector<Number> &src) const
{
  dst = 0;

  if (!have_interface_matrices)
    return;

  parallel::distributed::Vector<Number> src_cpy (src);
  unsigned int c=0;
  for (unsigned int i=0; i<edge_constrained_indices.size(); ++i)
    {
      for ( ; c<edge_constrained_indices[i]; ++c)
        src_cpy.local_element(c) = 0.;
      ++c;
    }
  for ( ; c<src_cpy.local_size(); ++c)
    src_cpy.local_element(c) = 0.;

  run_vmult_loop (dst, src_cpy);

  for (unsigned int i=0; i<edge_constrained_indices.size(); ++i)
    {
      dst.local_element(edge_constrained_indices[i]) = 0.;
    }
}



template <int dim, typename Number>
void LaplaceOperator<dim,Number>::apply_nullspace_projection(parallel::distributed::Vector<Number> &vec) const
{
  if (needs_mean_value_constraint)
    {
      const Number mean_val = vec.mean_value();
      vec.add(-mean_val);
    }
}



template <int dim, typename Number>
types::global_dof_index LaplaceOperator<dim,Number>::m() const
{
  return data->get_vector_partitioner(solver_data.poisson_dof_index)->size();
}



template <int dim, typename Number>
types::global_dof_index LaplaceOperator<dim,Number>::n() const
{
  return data->get_vector_partitioner(solver_data.poisson_dof_index)->size();
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
  if (!vector.partitioners_are_compatible(*data->get_dof_info(solver_data.poisson_dof_index).vector_partitioner))
    data->initialize_dof_vector(vector, solver_data.poisson_dof_index);
}



template <int dim, typename Number>
void LaplaceOperator<dim,Number>
::compute_inverse_diagonal (parallel::distributed::Vector<Number> &inverse_diagonal_entries)
{
  data->initialize_dof_vector(inverse_diagonal_entries, solver_data.poisson_dof_index);
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

  if(apply_mean_value_constraint_in_matvec)
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

  const std::vector<unsigned int> &
    constrained_dofs = data->get_constrained_dofs();
  for (unsigned int i=0; i<constrained_dofs.size(); ++i)
    inverse_diagonal_entries.local_element(constrained_dofs[i]) = 1.;
  for (unsigned int i=0; i<edge_constrained_indices.size(); ++i)
    {
      inverse_diagonal_entries.local_element(edge_constrained_indices[i]) = 1.;
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
                                                  solver_data.poisson_dof_index,
                                                  solver_data.poisson_quad_index);

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
  // Nothing to do for continuous elements
  if (data.get_dof_handler(solver_data.poisson_dof_index).get_fe().dofs_per_vertex > 0)
    return;

  FEFaceEvaluation<dim,degree,degree+1,1,Number> fe_eval(data,true,
                                                         solver_data.poisson_dof_index,
                                                         solver_data.poisson_quad_index);
  FEFaceEvaluation<dim,degree,degree+1,1,Number> fe_eval_neighbor(data,false,
                                                                  solver_data.poisson_dof_index,
                                                                  solver_data.poisson_quad_index);

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
  // Nothing to do for continuous elements
  if (data.get_dof_handler(solver_data.poisson_dof_index).get_fe().dofs_per_vertex > 0)
    return;

  FEFaceEvaluation<dim,degree,degree+1,1,Number> fe_eval(data, true,
                                                         solver_data.poisson_dof_index,
                                                         solver_data.poisson_quad_index);
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
                                                  solver_data.poisson_dof_index,
                                                  solver_data.poisson_quad_index);

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
  // Nothing to do for continuous elements
  if (data.get_dof_handler(solver_data.poisson_dof_index).get_fe().dofs_per_vertex > 0)
    return;

  FEFaceEvaluation<dim,degree,degree+1,1,Number> phi(data,true,
                                                     solver_data.poisson_dof_index,
                                                     solver_data.poisson_quad_index);
  FEFaceEvaluation<dim,degree,degree+1,1,Number> phi_outer(data,false,
                                                           solver_data.poisson_dof_index,
                                                           solver_data.poisson_quad_index);

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
  // Nothing to do for continuous elements
  if (data.get_dof_handler(solver_data.poisson_dof_index).get_fe().dofs_per_vertex > 0)
    return;

  FEFaceEvaluation<dim,degree,degree+1,1,Number> phi (data, true,
                                                      solver_data.poisson_dof_index,
                                                      solver_data.poisson_quad_index);

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
    //IterationNumberControl solver_control (10, 1e-15);

    SolverCG<parallel::distributed::Vector<typename Operator::value_type> >
      solver_coarse (solver_control, solver_memory);
    typename VectorMemory<parallel::distributed::Vector<typename Operator::value_type> >::Pointer r(solver_memory);
    *r = src;
    coarse_matrix.apply_nullspace_projection(*r);
    if (use_jacobi)
      {
        JacobiPreconditioner<typename Operator::value_type> preconditioner(inverse_diagonal);
        solver_coarse.solve (coarse_matrix, dst, *r, preconditioner);
      }
    else
      solver_coarse.solve (coarse_matrix, dst, *r, PreconditionIdentity());
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



// This is a modification of MGSmootherPrecondition from deal.II but
// re-written in order to avoid global communication
template<typename MatrixType, typename PreconditionerType, typename VectorType>
class MGSmootherPreconditionMF : public MGSmoother<VectorType>
{
public:
  /**
   * Constructor. Sets smoothing parameters.
   */
  MGSmootherPreconditionMF(const MGLevelObject<MatrixType> &matrices,
                           const MGLevelObject<PreconditionerType> &preconditioners,
                           const bool is_pre_smoother,
                           const unsigned int steps = 1,
                           const bool variable = false,
                           const bool symmetric = false,
                           const bool transpose = false)
    :
    MGSmoother<VectorType>(steps, variable, symmetric, transpose),
    matrices (&matrices),
    smoothers (&preconditioners),
    is_pre_smoother (is_pre_smoother)
  {}

  virtual void clear()
  {
    matrices = 0;
    smoothers = 0;
  }

  /**
   * The actual smoothing method.
   */
  virtual void smooth (const unsigned int level,
                       VectorType         &u,
                       const VectorType   &rhs) const
  {
    unsigned int maxlevel = matrices->max_level();
    unsigned int steps2 = this->steps;

    if (this->variable)
      steps2 *= (1<<(maxlevel-level));

    typename VectorMemory<VectorType>::Pointer r(this->vector_memory);
    typename VectorMemory<VectorType>::Pointer d(this->vector_memory);

    if (!is_pre_smoother || steps2 > 1)
      {
        r->reinit(u,true);
        d->reinit(u,true);
      }

    bool T = this->transpose;
    if (this->symmetric && (steps2 % 2 == 0))
      T = false;

    for (unsigned int i=0; i<steps2; ++i)
      {
        if (T)
          {
            if (steps2 == 1 && is_pre_smoother)
              (*smoothers)[level].Tvmult(u, rhs);
            else
              {
                (*matrices)[level].Tvmult(*r,u);
                r->sadd(-1.,1.,rhs);
                (*smoothers)[level].Tvmult(*d, *r);
              }
          }
        else
          {
            if (steps2 == 1 && is_pre_smoother)
              (*smoothers)[level].vmult(u, rhs);
            else
              {
                (*matrices)[level].vmult(*r,u);
                r->sadd(-1.,rhs);
                (*smoothers)[level].vmult(*d, *r);
              }
          }
        if (steps2 > 1 || !is_pre_smoother)
          u += *d;
        if (this->symmetric)
          T = !T;
      }
  }

private:
  /**
   * Pointer to the matrices.
   */
  const MGLevelObject<MatrixType> *matrices;

  /**
   * Object containing relaxation methods.
   */
  const MGLevelObject<PreconditionerType> *smoothers;

  /**
   * Sets whether we are pre- or post-smoothing. In pre-smoothing with only
   * one step, we will have a zero input vector and not need to compute the
   * initial matrix-vector product.
   */
  const bool is_pre_smoother;
};



template <int dim>
void PoissonSolver<dim>::initialize (const Mapping<dim> &mapping,
                                     const MatrixFree<dim,double> &matrix_free,
                                     const PoissonSolverData<dim> &solver_data)
{
  global_matrix.reinit(matrix_free, mapping, solver_data);

  const DoFHandler<dim> &dof_handler = matrix_free.get_dof_handler(solver_data.poisson_dof_index);
  const parallel::Triangulation<dim> *tria =
    dynamic_cast<const parallel::Triangulation<dim> *>(&dof_handler.get_triangulation());
  AssertThrow(tria != 0, ExcMessage("Only works for distributed triangulations"));
  mpi_communicator = tria->get_communicator();

  mg_constrained_dofs.clear();
  ZeroFunction<dim> zero_function;
  typename FunctionMap<dim>::type dirichlet_boundary;
  for (std::set<types::boundary_id>::const_iterator it =
         solver_data.dirichlet_boundaries.begin();
       it != solver_data.dirichlet_boundaries.end(); ++it)
    dirichlet_boundary[*it] = &zero_function;
  mg_constrained_dofs.initialize(dof_handler, dirichlet_boundary);

  mg_matrices.resize(0, tria->n_global_levels()-1);
  mg_interface_matrices.resize(0, tria->n_global_levels()-1);
  chebyshev_smoothers.resize(0, tria->n_global_levels()-1);

  typename SMOOTHER::AdditionalData smoother_data_l0;
  for (unsigned int level = 0; level<tria->n_global_levels(); ++level)
    {
      mg_matrices[level].reinit(dof_handler, mapping, solver_data,
                                mg_constrained_dofs, level);
      mg_interface_matrices[level].initialize(mg_matrices[level]);
      typename SMOOTHER::AdditionalData smoother_data;

      // we do not need the mean value constraint for smoothers on the
      // multigrid levels, so we can disable it
      mg_matrices[level].disable_mean_value_constraint();
      if (level > 0)
        {
          smoother_data.smoothing_range = solver_data.smoother_smoothing_range;
          smoother_data.degree = solver_data.smoother_poly_degree;
          smoother_data.eig_cg_n_iterations = 20;
        }
      else
        {
          smoother_data.smoothing_range = 0.;
          if (solver_data.coarse_solver != PoissonSolverData<dim>::coarse_chebyshev_smoother)
          {
            smoother_data.eig_cg_n_iterations = 0;
          }
          // TODO: here we would like to have an adaptive choice...
          else if (dof_handler.n_dofs(0) > 2000)
            {
              smoother_data.degree = 100;
              smoother_data.eig_cg_n_iterations = 200;
            }
          else
            {
              smoother_data.degree = 40;
              smoother_data.eig_cg_n_iterations = 100;
            }
        }
      mg_matrices[level].compute_inverse_diagonal(smoother_data.matrix_diagonal_inverse);
      chebyshev_smoothers[level].initialize(mg_matrices[level], smoother_data);
      if (level == 0)
        smoother_data_l0.matrix_diagonal_inverse = smoother_data.matrix_diagonal_inverse;
    }

  mg_pre_smoother.reset(new MGSmootherPreconditionMF<LevelMatrixType,SMOOTHER,parallel::distributed::Vector<Number> >(mg_matrices, chebyshev_smoothers, true));
  mg_post_smoother.reset(new MGSmootherPreconditionMF<LevelMatrixType,SMOOTHER,parallel::distributed::Vector<Number> >(mg_matrices, chebyshev_smoothers, false));

  switch (solver_data.coarse_solver)
    {
    case PoissonSolverData<dim>::coarse_chebyshev_smoother:
      {
        mg_coarse.reset(new MGCoarseFromSmoother<parallel::distributed::Vector<Number> >(*mg_pre_smoother));
        break;
      }
    case PoissonSolverData<dim>::coarse_iterative_noprec:
      {
        mg_coarse.reset(new MGCoarseIterative<LevelMatrixType>(mg_matrices[0], 0));
        break;
      }
    case PoissonSolverData<dim>::coarse_iterative_jacobi:
      {
        mg_coarse.reset(new MGCoarseIterative<LevelMatrixType>(mg_matrices[0], &smoother_data_l0.matrix_diagonal_inverse));
        break;
      }
    default:
      AssertThrow(false, ExcMessage("Unknown coarse-grid solver given"));
    }

  mg_transfer.set_laplace_operator(mg_matrices);
  mg_transfer.initialize_constraints(mg_constrained_dofs);
  mg_transfer.add_periodicity(solver_data.periodic_face_pairs_level0);
  mg_transfer.build(dof_handler);

  mg_matrix.reset(new mg::Matrix<parallel::distributed::Vector<Number> > (mg_matrices));
  mg_interface.reset(new  mg::Matrix<parallel::distributed::Vector<Number> >(mg_interface_matrices));

  mg.reset(new Multigrid<parallel::distributed::Vector<Number> > (dof_handler,
                                                                  *mg_matrix,
                                                                  *mg_coarse,
                                                                  mg_transfer,
                                                                  *mg_pre_smoother,
                                                                  *mg_post_smoother));
  mg->set_edge_matrices(*mg_interface, *mg_interface);

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
void
PoissonSolver<dim>::apply_precondition (parallel::distributed::Vector<double> &dst,
                                        const parallel::distributed::Vector<double> &src) const
{
  Assert(preconditioner.get() != 0,
         ExcNotInitialized());
  preconditioner->vmult(dst, src);
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
  AssertThrow(std::isfinite(solver_control.last_value()),
              ExcMessage("Poisson solver contained NaN of Inf values"));
  return solver_control.last_step();
}


// explicit instantiations
template class LaplaceOperator<2,double>;
template class LaplaceOperator<3,double>;
template class LaplaceOperator<2,float>;
template class LaplaceOperator<3,float>;
template class PoissonSolver<2>;
template class PoissonSolver<3>;
