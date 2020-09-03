
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/fe/mapping_q_generic.h>
#include <deal.II/fe/mapping_fe_field.h>

using namespace dealii;

void do_test()
{
  const int dim = 2;
  FESystem fe(FE_Q<dim>(/*degree=*/7), dim);
  Triangulation<dim> tria;
  GridGenerator::hyper_ball(tria);
  tria.refine_global(2);

  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  IndexSet relevant_dofs;
  DoFTools::extract_locally_relevant_dofs(dof_handler, relevant_dofs);

  LinearAlgebra::distributed::Vector<double> position;
  LinearAlgebra::distributed::Vector<double> displacement;
  LinearAlgebra::distributed::Vector<double> vector;
  position.reinit(dof_handler.locally_owned_dofs(), relevant_dofs, MPI_COMM_SELF);
  displacement.reinit(dof_handler.locally_owned_dofs(), relevant_dofs, MPI_COMM_SELF);
  vector.reinit(dof_handler.locally_owned_dofs(), relevant_dofs, MPI_COMM_SELF);

  MappingQGeneric<dim> mapping_original(fe.degree);
  {
    FEValues<dim> fe_values(mapping_original, fe,
                            Quadrature<dim>(fe.get_unit_support_points()),
                            update_quadrature_points);
    std::vector<types::global_dof_index> dof_indices(fe.dofs_per_cell);
    for (const auto & cell : dof_handler.active_cell_iterators())
      if (!cell->is_artificial())
        {
          fe_values.reinit(cell);
          cell->get_dof_indices(dof_indices);
          for (unsigned int i=0; i<fe.dofs_per_cell; ++i)
            {
              const unsigned int coordinate_direction =
                fe.system_to_component_index(i).first;
              const Point<dim> point = fe_values.quadrature_point(i);
              double sinval = coordinate_direction == 0 ? 0.25 : 0.1;
              for (unsigned int d=0; d<dim; ++d)
                sinval *= std::sin(2*numbers::PI*(point(d)+1)/(2));
              position(dof_indices[i]) = point[coordinate_direction];
              displacement(dof_indices[i]) = sinval;
            }
        }
  }

  DataOut<dim> data_out;
  data_out.add_data_vector(dof_handler, displacement, "displacement");

  DataOutBase::VtkFlags flags;
  flags.write_higher_order_cells = true;
  data_out.set_flags(flags);

  data_out.build_patches(mapping_original, fe.degree, DataOut<dim>::curved_inner_cells);

  std::ofstream file("grid-0.vtu");
  data_out.write_vtu(file);

  vector = position;
  vector += displacement;

  MappingFEField<dim,dim,LinearAlgebra::distributed::Vector<double>> mapping(dof_handler, vector);

  data_out.build_patches(mapping, fe.degree, DataOut<dim>::curved_inner_cells);
  std::ofstream file1("grid-1.vtu");
  data_out.write_vtu(file1);


  {
    FEValues<dim> fe_values(mapping_original, fe,
                            Quadrature<dim>(fe.get_unit_support_points()),
                            update_quadrature_points);
    std::vector<types::global_dof_index> dof_indices(fe.dofs_per_cell);
    for (const auto & cell : dof_handler.active_cell_iterators())
      {
        fe_values.reinit(cell);
        cell->get_dof_indices(dof_indices);
        for (unsigned int i=0; i<fe.dofs_per_cell; ++i)
          {
            const unsigned int coordinate_direction =
              fe.system_to_component_index(i).first;
            const Point<dim> point = fe_values.quadrature_point(i);
            double sinval = coordinate_direction == 0 ? 0.25 : 0.1;
            for (unsigned int d=0; d<dim; ++d)
              sinval *= std::sin(2*numbers::PI*(point(d)+1)/(2));
            displacement(dof_indices[i]) = -sinval;
          }
      }
  }

  if(false)
  {
    vector = position;
    vector += displacement;
    vector *= 0.4;
  }
  else
  {
    vector += displacement;
  }

  data_out.build_patches(mapping, fe.degree, DataOut<dim>::curved_inner_cells);
  std::ofstream file2("grid-2.vtu");
  data_out.write_vtu(file2);


}


int main()
{
  do_test();
}
