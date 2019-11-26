#include "moving_mesh.h"

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/mapping_q_cache.h>
#include <deal.II/numerics/vector_tools.h>

template<int dim, typename Number>
MovingMesh<dim, Number>::MovingMesh(parallel::TriangulationBase<dim> const & triangulation,
                                    unsigned int const                       polynomial_degree,
                                    std::shared_ptr<Function<dim>> const     function)
  : polynomial_degree(polynomial_degree),
    mesh_movement_function(function),
    fe(new FESystem<dim>(FE_Q<dim>(polynomial_degree), dim)),
    dof_handler(triangulation),
    grid_coordinates(triangulation.n_global_levels())
{
  dof_handler.distribute_dofs(*fe);
  dof_handler.distribute_mg_dofs();
}

template<int dim, typename Number>
std::shared_ptr<Mapping<dim>>
MovingMesh<dim, Number>::initialize_mapping_ale(double const time, Mapping<dim> const & mapping)
{
  std::shared_ptr<Mapping<dim>> mapping_ale;

  if(true)
    mapping_ale.reset(new MappingFEField<dim, dim, VectorType>(dof_handler, grid_coordinates));
  else
    mapping_ale.reset(new MappingQCache<dim>(polynomial_degree));

  move_mesh_analytical(time, mapping, *mapping_ale);

  return mapping_ale;
}

template<int dim, typename Number>
void
MovingMesh<dim, Number>::move_mesh_analytical(double const         time,
                                              Mapping<dim> const & mapping,
                                              Mapping<dim> &       mapping_ale)
{
  (void)mapping_ale;

  mesh_movement_function->set_time(time);

  unsigned int nlevel = dof_handler.get_triangulation().n_global_levels();
  for(unsigned int level = 0; level < nlevel; ++level)
  {
    VectorType initial_position;
    VectorType displacement;

    IndexSet relevant_dofs_grid;
    DoFTools::extract_locally_relevant_level_dofs(dof_handler, level, relevant_dofs_grid);

    initial_position.reinit(dof_handler.locally_owned_mg_dofs(level),
                            relevant_dofs_grid,
                            MPI_COMM_WORLD);
    displacement.reinit(dof_handler.locally_owned_mg_dofs(level),
                        relevant_dofs_grid,
                        MPI_COMM_WORLD);

    FEValues<dim> fe_values(mapping,
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
}

template<int dim, typename Number>
void
MovingMesh<dim, Number>::fill_grid_coordinates_vector(VectorType &            vector,
                                                      DoFHandler<dim> const & dof_handler,
                                                      Mapping<dim> const &    mapping)
{
  IndexSet relevant_dofs_grid;
  DoFTools::extract_locally_relevant_dofs(dof_handler, relevant_dofs_grid);

  vector.reinit(dof_handler.locally_owned_dofs(), relevant_dofs_grid, MPI_COMM_WORLD);

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
