#include "moving_mesh.h"

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/mapping_q_cache.h>
#include <deal.II/numerics/vector_tools.h>

template<int dim, typename Number>
MovingMesh<dim, Number>::MovingMesh(unsigned int const                       mapping_degree_in,
                                    parallel::TriangulationBase<dim> const & triangulation_in,
                                    unsigned int const                       polynomial_degree_in,
                                    std::shared_ptr<Function<dim>> const     function_in,
                                    double const                             start_time,
                                    MPI_Comm const &                         mpi_comm_in)
  : Mesh<dim>(mapping_degree_in),
    triangulation(triangulation_in),
    polynomial_degree(polynomial_degree_in),
    mesh_movement_function(function_in),
    mpi_comm(mpi_comm_in),
    fe(new FESystem<dim>(FE_Q<dim>(polynomial_degree), dim)),
#ifndef MAPPING_Q_CACHE
    grid_coordinates(triangulation_in.n_global_levels()),
#endif
    dof_handler(triangulation_in)
{
  dof_handler.distribute_dofs(*fe);
  dof_handler.distribute_mg_dofs();

  initialize_mapping_ale(start_time);
}

template<int dim, typename Number>
Mapping<dim> const &
MovingMesh<dim, Number>::get_mapping() const
{
  if(mapping_ale.get() == 0)
    return *this->mapping;
  else
    return *mapping_ale;
}

template<int dim, typename Number>
void
MovingMesh<dim, Number>::initialize_mapping_ale(double const time)
{
#ifdef MAPPING_Q_CACHE
  mapping_ale.reset(new MappingQCache<dim>(polynomial_degree));
#else
  mapping_ale.reset(new MappingFEField<dim, dim, VectorType>(dof_handler, grid_coordinates));
#endif

  move_mesh(time);
}

template<int dim, typename Number>
void
MovingMesh<dim, Number>::move_mesh(double const time)
{
  mesh_movement_function->set_time(time);

#ifdef MAPPING_Q_CACHE
  AssertThrow(mapping_ale.get() != 0, ExcMessage("mapping_ale is uninitialized."));

  std::shared_ptr<MappingQCache<dim>> mapping_q_cache =
    std::dynamic_pointer_cast<MappingQCache<dim>>(mapping_ale);
  AssertThrow(mapping_q_cache.get() != 0, ExcMessage("Could not cast to MappingQCache<dim>."));

  mapping_q_cache->initialize(
    triangulation,
    [&](const typename Triangulation<dim>::cell_iterator & cell_tria) -> std::vector<Point<dim>> {
      FiniteElement<dim> const & fe = dof_handler.get_fe();

      FEValues<dim> fe_values(*this->mapping,
                              fe,
                              Quadrature<dim>(fe.base_element(0).get_unit_support_points()),
                              update_quadrature_points);

      typename DoFHandler<dim>::cell_iterator cell(&triangulation,
                                                   cell_tria->level(),
                                                   cell_tria->index(),
                                                   &dof_handler);
      fe_values.reinit(cell);

      // compute displacement and add to original position
      std::vector<Point<dim>> points_moved(fe.base_element(0).dofs_per_cell);
      for(unsigned int i = 0; i < fe.base_element(0).dofs_per_cell; ++i)
      {
        Point<dim> const point = fe_values.quadrature_point(i);
        Point<dim>       displacement;
        for(unsigned int d = 0; d < dim; ++d)
          displacement[d] = mesh_movement_function->value(point, d);

        points_moved[i] = point + displacement;
      }

      return points_moved;
    });
#else
  (void)mapping_ale;
  unsigned int nlevel = dof_handler.get_triangulation().n_global_levels();
  for(unsigned int level = 0; level < nlevel; ++level)
  {
    VectorType initial_position;
    VectorType displacement;

    IndexSet relevant_dofs_grid;
    DoFTools::extract_locally_relevant_level_dofs(dof_handler, level, relevant_dofs_grid);

    initial_position.reinit(dof_handler.locally_owned_mg_dofs(level), relevant_dofs_grid, mpi_comm);
    displacement.reinit(dof_handler.locally_owned_mg_dofs(level), relevant_dofs_grid, mpi_comm);

    FEValues<dim> fe_values(*this->mapping,
                            *fe,
                            Quadrature<dim>(fe->get_unit_support_points()),
                            update_quadrature_points);

    std::vector<types::global_dof_index> dof_indices(fe->dofs_per_cell);
    for(const auto & cell : dof_handler.mg_cell_iterators_on_level(level))
    {
      if(cell->level_subdomain_id() != numbers::artificial_subdomain_id)
      {
        fe_values.reinit(cell);
        cell->get_active_or_mg_dof_indices(dof_indices);

        for(unsigned int i = 0; i < dof_indices.size(); ++i)
        {
          unsigned int const d     = fe->system_to_component_index(i).first;
          Point<dim> const   point = fe_values.quadrature_point(i);

          initial_position(dof_indices[i]) = point[d];
          displacement(dof_indices[i])     = mesh_movement_function->value(point, d);
        }
      }
    }

    grid_coordinates[level] = initial_position;
    grid_coordinates[level] += displacement;
    grid_coordinates[level].update_ghost_values();
  }
#endif
}

template<int dim, typename Number>
void
MovingMesh<dim, Number>::fill_grid_coordinates_vector(VectorType &            vector,
                                                      DoFHandler<dim> const & dof_handler)
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

template class MovingMesh<2, float>;
template class MovingMesh<2, double>;

template class MovingMesh<3, float>;
template class MovingMesh<3, double>;
