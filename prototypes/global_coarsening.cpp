#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/numerics/data_out.h>

template<int dim>
void
print_mesh(const dealii::Triangulation<dim> & tria)
{
  dealii::DataOut<dim> data_out;
  data_out.attach_triangulation(tria);
  data_out.build_patches(1);
  data_out.write_vtu_in_parallel("output/mesh_active.vtu", tria.get_communicator());
}

template<int dim>
void
print_mesh(const dealii::Triangulation<dim> &                                      tria,
           const std::vector<dealii::LinearAlgebra::distributed::Vector<double>> & refinement_state)
{
  dealii::DataOut<dim> data_out;
  data_out.attach_triangulation(tria);

  const auto next_cell = [&](const auto & tria, const auto & cell_in) {
    auto cell = cell_in;

    while(true)
    {
      cell++;

      if(cell == tria.end())
        break;

      if(cell->is_locally_owned_on_level() and
         static_cast<bool>(refinement_state[cell->level()][cell->global_level_cell_index()]))
        return cell;
    }

    return tria.end();
  };

  // output mesh
  const auto first_cell = [&](const auto & tria) {
    const auto cell = tria.begin();

    if(cell == tria.end())
      return cell;

    if(cell->is_locally_owned_on_level() and
       static_cast<bool>(refinement_state[cell->level()][cell->global_level_cell_index()]))
      return cell;

    return next_cell(tria, cell);
  };

  data_out.set_cell_selection(first_cell, next_cell);

  data_out.build_patches(1);

  unsigned int level = 0;

  for(unsigned int l = 1; l < refinement_state.size(); ++l)
    for(const auto i : refinement_state[l])
      if(i == 1.0)
        level = l;

  level = dealii::Utilities::MPI::max(level, tria.get_communicator());

  data_out.write_vtu_in_parallel("output/mesh_level_" + std::to_string(level) + ".vtu",
                                 MPI_COMM_WORLD);
}

template<int dim>
void
test(const unsigned int n_local_refinements)
{
  const MPI_Comm comm = MPI_COMM_WORLD;

  // create locally refined mesh
  dealii::parallel::distributed::Triangulation<dim> tria(
    comm,
    dealii::Triangulation<dim>::none,
    dealii::parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy);

  dealii::GridGenerator::subdivided_hyper_cube(tria, 2, -1.0, 1.0);

  for(unsigned int i = 0; i < n_local_refinements; ++i)
  {
    for(auto cell : tria.active_cell_iterators())
      if(cell->is_locally_owned())
      {
        bool flag = true;
        for(int d = 0; d < dim; d++)
          if(cell->center()[d] > 0.0)
            flag = false;
        if(flag)
          cell->set_refine_flag();
      }
    tria.execute_coarsening_and_refinement();
  }

  // write it out normally
  print_mesh(tria);

  // data structure to track if a cell is "active" or not
  std::vector<dealii::LinearAlgebra::distributed::Vector<double>> refinement_state(
    tria.n_global_levels());
  for(unsigned int l = 0; l < tria.n_global_levels(); ++l)
    refinement_state[l].reinit(tria.global_level_cell_index_partitioner(l).lock());

  // ... copy the state of the actually acive cells
  for(const auto & cell : tria.active_cell_iterators())
    if(cell->is_locally_owned())
      refinement_state[cell->level()][cell->global_level_cell_index()] = 1.0;

  // print active mesh with alternative function
  print_mesh(tria, refinement_state);

  // perform global coarsening by modifying the values in the vectors
  for(unsigned int ll = 0; ll < refinement_state.size() - 1; ++ll)
  {
    auto refinement_state_temp = refinement_state;

    for(unsigned int l = 0; l < refinement_state.size() - ll; ++l)
    {
      refinement_state[l].update_ghost_values();
      refinement_state_temp[l] = 0.0;
    }

    for(unsigned int l = 0; l < refinement_state.size() - 1 - ll; ++l)
      for(const auto & parent_cell : tria.cell_iterators_on_level(l))
        if(parent_cell->is_locally_owned_on_level() == true and parent_cell->has_children())
        {
          // cell has been visited already -> nothing to do
          if(refinement_state_temp[parent_cell->level()][parent_cell->global_level_cell_index()] !=
             0.0)
            continue;

          // check if all chilren are active
          bool can_become_active = true;

          for(const auto & cell : parent_cell->child_iterators())
            if(refinement_state[cell->level()][cell->global_level_cell_index()] == 0)
              can_become_active = false;

          if(can_become_active)
          {
            // if yes: make children inactive and make this cell active
            refinement_state_temp[parent_cell->level()][parent_cell->global_level_cell_index()] =
              1.0;
            for(const auto & cell : parent_cell->child_iterators())
              refinement_state_temp[cell->level()][cell->global_level_cell_index()] =
                -1.0; // indicate that cell has been
                      // visitied and its state has
                      // changed
          }
          else
          {
            // if no: copy old state
            refinement_state_temp[parent_cell->level()][parent_cell->global_level_cell_index()] =
              refinement_state[parent_cell->level()][parent_cell->global_level_cell_index()];
            for(const auto & cell : parent_cell->child_iterators())
              refinement_state_temp[cell->level()][cell->global_level_cell_index()] =
                refinement_state[cell->level()][cell->global_level_cell_index()];
          }
        }
        else if(parent_cell->is_locally_owned_on_level() == true)
        {
          // cell has been visited already -> nothing to do
          if(refinement_state_temp[parent_cell->level()][parent_cell->global_level_cell_index()] !=
             0.0)
            continue;

          // if no: copy old state
          refinement_state_temp[parent_cell->level()][parent_cell->global_level_cell_index()] =
            refinement_state[parent_cell->level()][parent_cell->global_level_cell_index()];
        }

    for(unsigned int l = 0; l < refinement_state.size() - ll; ++l)
    {
      for(auto & i : refinement_state_temp[l])
        if(i == -1.0)
          i = 0.0; // not needed any more

      // notify owning cells about changing state
      refinement_state_temp[l].compress(dealii::VectorOperation::add);

      // copy new state
      refinement_state[l].zero_out_ghost_values();
      refinement_state[l].copy_locally_owned_data_from(refinement_state_temp[l]);
    }

    // print level mesh
    print_mesh(tria, refinement_state);
  }
}

int
main(int argc, char * argv[])
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  test<2>(7);
}
