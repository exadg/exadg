#ifndef INCLUDE_MOVING_MESH_H_
#define INCLUDE_MOVING_MESH_H_

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/mapping_fe_field.h>
#include <deal.II/fe/mapping_q_cache.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>

#include "mesh.h"

#include "../poisson/spatial_discretization/operator.h"

using namespace dealii;

template<int dim, typename Number>
class MovingMeshBase : public Mesh<dim>
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  MovingMeshBase(parallel::TriangulationBase<dim> const & triangulation_in,
                 unsigned int const                       mapping_degree_static_in,
                 unsigned int const                       mapping_degree_moving_in,
                 MPI_Comm const &                         mpi_comm_in)
    : Mesh<dim>(mapping_degree_static_in), triangulation(triangulation_in), mpi_comm(mpi_comm_in)
  {
    mapping_ale.reset(new MappingQCache<dim>(mapping_degree_moving_in));
  }

  virtual ~MovingMeshBase()
  {
  }

  virtual void
  move_mesh(double const time) = 0;

  /*
   * This function implements the interface of base class Mesh<dim>
   */
  Mapping<dim> const &
  get_mapping() const override
  {
    if(mapping_ale.get() == 0)
      return *this->mapping;
    else
      return *mapping_ale;
  }

  /*
   * This function extracts the grid coordinates of the current mesh configuration, i.e.,
   * a mapping describing the mesh displacement has to be used here.
   */
  void
  fill_grid_coordinates_vector(VectorType & vector, DoFHandler<dim> const & dof_handler)
  {
    Mapping<dim> const & mapping = *mapping_ale;

    IndexSet relevant_dofs_grid;
    DoFTools::extract_locally_relevant_dofs(dof_handler, relevant_dofs_grid);

    vector.reinit(dof_handler.locally_owned_dofs(), relevant_dofs_grid, mpi_comm);

    FiniteElement<dim> const & fe = dof_handler.get_fe();

    FEValues<dim> fe_values(mapping,
                            fe,
                            Quadrature<dim>(fe.get_unit_support_points()),
                            update_quadrature_points);

    std::vector<types::global_dof_index> dof_indices(fe.dofs_per_cell);
    for(const auto & cell : dof_handler.active_cell_iterators())
    {
      if(!cell->is_artificial())
      {
        fe_values.reinit(cell);
        cell->get_dof_indices(dof_indices);
        for(unsigned int i = 0; i < dof_indices.size(); ++i)
        {
          unsigned int const d     = fe.system_to_component_index(i).first;
          Point<dim> const   point = fe_values.quadrature_point(i);
          vector(dof_indices[i])   = point[d];
        }
      }
    }

    vector.update_ghost_values();
  }

protected:
  // needed for re-initialization of MappingQCache
  parallel::TriangulationBase<dim> const & triangulation;

  std::shared_ptr<MappingQCache<dim>> mapping_ale;

private:
  // MPI communicator
  MPI_Comm const & mpi_comm;
};

template<int dim, typename Number>
class MovingMeshAnalytical : public MovingMeshBase<dim, Number>
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  MovingMeshAnalytical(parallel::TriangulationBase<dim> const & triangulation_in,
                       unsigned int const                       mapping_degree_static_in,
                       unsigned int const                       mapping_degree_moving_in,
                       MPI_Comm const &                         mpi_comm_in,
                       std::shared_ptr<Function<dim>> const     mesh_movement_function_in,
                       double const                             start_time)
    : MovingMeshBase<dim, Number>(triangulation_in,
                                  mapping_degree_static_in,
                                  mapping_degree_moving_in,
                                  mpi_comm_in),
      mesh_movement_function(mesh_movement_function_in),
      fe(new FESystem<dim>(FE_Q<dim>(mapping_degree_moving_in), dim)),
      dof_handler(triangulation_in)
  {
    dof_handler.distribute_dofs(*fe);
    dof_handler.distribute_mg_dofs();

    move_mesh(start_time);
  }

  /*
   * This function is formulated w.r.t. reference coordinates, i.e., the mapping describing
   * the initial mesh position has to be used for this function.
   */
  void
  move_mesh(double const time)
  {
    mesh_movement_function->set_time(time);

    initialize_mapping_q_cache(this->triangulation,
                               *this->mapping,
                               dof_handler,
                               mesh_movement_function);
  }

private:
  void
  initialize_mapping_q_cache(parallel::TriangulationBase<dim> const & triangulation,
                             Mapping<dim> const &                     mapping,
                             DoFHandler<dim> const &                  dof_handler,
                             std::shared_ptr<Function<dim>>           displacement_function)
  {
    FEValues<dim> fe_values(mapping,
                            fe->base_element(0),
                            Quadrature<dim>(fe->base_element(0).get_unit_support_points()),
                            update_quadrature_points);
    AssertThrow(MultithreadInfo::n_threads() == 1, ExcNotImplemented());

    this->mapping_ale->initialize(
      triangulation,
      [&](const typename Triangulation<dim>::cell_iterator & cell) -> std::vector<Point<dim>> {
        FiniteElement<dim> const & fe = dof_handler.get_fe();

        fe_values.reinit(cell);

        // compute displacement and add to original position
        unsigned int const      scalar_dofs_per_cell = fe.base_element(0).dofs_per_cell;
        std::vector<Point<dim>> points_moved(scalar_dofs_per_cell);
        for(unsigned int i = 0; i < scalar_dofs_per_cell; ++i)
        {
          Point<dim> const point = fe_values.quadrature_point(i);
          Point<dim>       displacement;
          for(unsigned int d = 0; d < dim; ++d)
            displacement[d] = displacement_function->value(point, d);

          points_moved[i] = point + displacement;
        }

        return points_moved;
      });
  }


  // An analytical function that describes the mesh motion
  std::shared_ptr<Function<dim>> mesh_movement_function;

  // Finite Element (use a continuous finite element space to describe the mesh movement)
  std::shared_ptr<FESystem<dim>> fe;

  // DoFHandler
  DoFHandler<dim> dof_handler;
};

template<int dim, typename Number>
class MovingMeshPoisson : public MovingMeshBase<dim, Number>
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  MovingMeshPoisson(parallel::TriangulationBase<dim> const &             triangulation,
                    unsigned int const                                   mapping_degree_static,
                    MPI_Comm const &                                     mpi_comm,
                    std::shared_ptr<Poisson::Operator<dim, Number, dim>> poisson_operator,
                    double const &                                       start_time)
    : MovingMeshBase<dim, Number>(triangulation,
                                  mapping_degree_static,
                                  poisson_operator->get_dof_handler().get_fe().degree,
                                  mpi_comm),
      poisson(poisson_operator)
  {
    // make sure that the mapping is initialized
    move_mesh(start_time);
  }

  void
  move_mesh(double const time)
  {
    VectorType displacement_fine, rhs;
    poisson->initialize_dof_vector(displacement_fine);
    poisson->initialize_dof_vector(rhs);

    // compute rhs and solve mesh deformation problem
    poisson->rhs(rhs, time);
    poisson->solve(displacement_fine, rhs);

    initialize_mapping_q_cache(this->triangulation,
                               *this->mapping,
                               poisson->get_dof_handler(),
                               displacement_fine);
  }

private:
  void
  initialize_mapping_q_cache(parallel::TriangulationBase<dim> const & triangulation,
                             Mapping<dim> const &                     mapping,
                             DoFHandler<dim> const &                  dof_handler,
                             VectorType const &                       dof_vector)
  {
    // we have to project the solution onto all coarse levels of the triangulation
    // (required for initialization of MappingQCache)
    MGLevelObject<VectorType> dof_vector_all_levels;
    unsigned int const        n_levels = triangulation.n_global_levels();
    dof_vector_all_levels.resize(0, n_levels - 1);

    MGTransferMatrixFree<dim, Number> transfer;
    transfer.build(dof_handler);
    transfer.interpolate_to_mg(dof_handler, dof_vector_all_levels, dof_vector);
    for(unsigned int level = 0; level < n_levels; level++)
      dof_vector_all_levels[level].update_ghost_values();

    FiniteElement<dim> const & fe = dof_handler.get_fe();
    AssertThrow(fe.element_multiplicity(0) == dim,
                ExcMessage("Expected finite element with dim components."));

    FE_Q<dim>     fe_scalar(fe.degree);
    FEValues<dim> fe_values(mapping,
                            fe,
                            Quadrature<dim>(fe_scalar.get_unit_support_points()),
                            update_quadrature_points);

    // update mapping according to mesh deformation computed above
    this->mapping_ale->initialize(
      triangulation,
      [&](const typename Triangulation<dim>::cell_iterator & cell_tria) -> std::vector<Point<dim>> {
        unsigned int const                      level = cell_tria->level();
        typename DoFHandler<dim>::cell_iterator cell(&triangulation,
                                                     level,
                                                     cell_tria->index(),
                                                     &dof_handler);

        unsigned int const        scalar_dofs_per_cell = fe_scalar.dofs_per_cell;
        std::vector<unsigned int> hierarchic_to_lexicographic(fe_scalar.dofs_per_cell);
        FETools::hierarchic_to_lexicographic_numbering<dim>(fe_scalar.degree,
                                                            hierarchic_to_lexicographic);
        std::vector<unsigned int> lexicographic_to_hierarchic =
          Utilities::invert_permutation(hierarchic_to_lexicographic);

        std::vector<Point<dim>> points_moved(scalar_dofs_per_cell);

        if(cell->level_subdomain_id() != numbers::artificial_subdomain_id)
        {
          fe_values.reinit(cell);
          std::vector<types::global_dof_index> dof_indices(fe.dofs_per_cell);
          cell->get_mg_dof_indices(dof_indices);

          // extract displacement and add to original position
          for(unsigned int i = 0; i < scalar_dofs_per_cell; ++i)
          {
            points_moved[i] = fe_values.quadrature_point(i);
          }

          for(unsigned int i = 0; i < dof_indices.size(); ++i)
          {
            std::pair<unsigned int, unsigned int> const id = fe.system_to_component_index(i);

            if(fe.dofs_per_vertex > 0) // FE_Q
              points_moved[id.second][id.first] += dof_vector_all_levels[level](dof_indices[i]);
            else // FE_DGQ
              points_moved[lexicographic_to_hierarchic[id.second]][id.first] +=
                dof_vector_all_levels[level](dof_indices[i]);
          }
        }

        return points_moved;
      });
  }

  std::shared_ptr<Poisson::Operator<dim, Number, dim>> poisson;
};

#endif /*INCLUDE_MOVING_MESH_H_*/
