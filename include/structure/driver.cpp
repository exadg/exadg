/*
 * driver.cpp
 *
 *  Created on: 21.03.2020
 *      Author: fehn
 */

#include "driver.h"

namespace Structure
{
template<int dim, typename Number>
Driver<dim, Number>::Driver(MPI_Comm const & comm)
  : mpi_comm(comm),
    pcout(std::cout, Utilities::MPI::this_mpi_process(comm) == 0),
    overall_time(0.0),
    setup_time(0.0)
{
}

template<int dim, typename Number>
void
Driver<dim, Number>::print_header() const
{
  // clang-format off
  pcout << std::endl << std::endl << std::endl
  << "_________________________________________________________________________________" << std::endl
  << "                                                                                 " << std::endl
  << "                    High-order matrix-free elasticity solver                     " << std::endl
  << "_________________________________________________________________________________" << std::endl
  << std::endl;
  // clang-format on
}

template<int dim, typename Number>
void
Driver<dim, Number>::setup(std::shared_ptr<ApplicationBase<dim, Number>> app,
                           unsigned int const &                          degree,
                           unsigned int const &                          refine_space)
{
  timer.restart();

  print_header();
  print_dealii_info<Number>(pcout);
  print_MPI_info(pcout, mpi_comm);

  application = app;

  application->set_input_parameters(param);

  param.check_input_parameters();
  param.print(pcout, "List of input parameters:");

  // triangulation
  if(param.triangulation_type == TriangulationType::Distributed)
  {
    triangulation.reset(new parallel::distributed::Triangulation<dim>(
      mpi_comm,
      dealii::Triangulation<dim>::none,
      parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy));
  }
  else if(param.triangulation_type == TriangulationType::FullyDistributed)
  {
    triangulation.reset(new parallel::fullydistributed::Triangulation<dim>(mpi_comm));
  }
  else
  {
    AssertThrow(false, ExcMessage("Invalid parameter triangulation_type."));
  }

  application->create_grid(triangulation, refine_space, periodic_faces);

  print_grid_data(pcout, refine_space, *triangulation);

  // boundary conditions
  boundary_descriptor.reset(new BoundaryDescriptor<dim>());
  application->set_boundary_conditions(boundary_descriptor);

  // material_descriptor
  material_descriptor.reset(new MaterialDescriptor);
  application->set_material(*material_descriptor);

  // field functions and boundary conditions
  field_functions.reset(new FieldFunctions<dim>());
  application->set_field_functions(field_functions);

  // mapping
  unsigned int const mapping_degree = get_mapping_degree(param.mapping, degree);
  mesh.reset(new Mesh<dim>(mapping_degree));

  // setup spatial operator
  pde_operator.reset(new PDEOperator(*triangulation,
                                     mesh->get_mapping(),
                                     degree,
                                     periodic_faces,
                                     boundary_descriptor,
                                     field_functions,
                                     material_descriptor,
                                     param,
                                     mpi_comm));

  pde_operator->setup();

  // initialize postprocessor
  postprocessor = application->construct_postprocessor(param, mpi_comm);
  postprocessor->setup(pde_operator->get_dof_handler(), pde_operator->get_mapping());

  // initialize driver
  if(param.problem_type == ProblemType::Unsteady)
  {
    AssertThrow(false, ExcMessage("Unsteady solver has not been implemented!"));
  }
  else if(param.problem_type == ProblemType::Steady)
  {
    driver_steady.reset(
      new DriverSteady<dim, Number>(pde_operator, postprocessor, param, mpi_comm));
    driver_steady->setup();
  }
  else if(param.problem_type == ProblemType::QuasiStatic)
  {
    driver_quasi_static.reset(
      new DriverQuasiStatic<dim, Number>(pde_operator, postprocessor, param, mpi_comm));
    driver_quasi_static->setup();
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

  pde_operator->setup_solver();

  setup_time = timer.wall_time();
}

template<int dim, typename Number>
void
Driver<dim, Number>::solve() const
{
  if(param.problem_type == ProblemType::Unsteady)
  {
    AssertThrow(false, ExcMessage("Unsteady solver not implemented yet."));
  }
  else if(param.problem_type == ProblemType::Steady)
  {
    driver_steady->solve_problem();
  }
  else if(param.problem_type == ProblemType::QuasiStatic)
  {
    driver_quasi_static->solve_problem();
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

  overall_time += this->timer.wall_time();
}

template<int dim, typename Number>
void
Driver<dim, Number>::analyze_computing_times() const
{
  // TODO

  std::vector<std::string> name;
  std::vector<double>      wall_time;

  if(driver_quasi_static.get() != 0)
    driver_quasi_static->get_wall_times(name, wall_time);
}

template class Driver<2, float>;
template class Driver<3, float>;

template class Driver<2, double>;
template class Driver<3, double>;

} // namespace Structure
