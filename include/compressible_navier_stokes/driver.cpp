/*
 * driver.cpp
 *
 *  Created on: 26.03.2020
 *      Author: fehn
 */

#include "driver.h"
#include "../utilities/print_throughput.h"

namespace ExaDG
{
namespace CompNS
{
using namespace dealii;

template<int dim, typename Number>
Driver<dim, Number>::Driver(MPI_Comm const & comm)
  : mpi_comm(comm), pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_comm) == 0)
{
}

template<int dim, typename Number>
void
Driver<dim, Number>::print_header()
{
  // clang-format off
  pcout << std::endl << std::endl << std::endl
  << "_________________________________________________________________________________" << std::endl
  << "                                                                                 " << std::endl
  << "                High-order discontinuous Galerkin solver for the                 " << std::endl
  << "                 unsteady, compressible Navier-Stokes equations                  " << std::endl
  << "_________________________________________________________________________________" << std::endl
  << std::endl;
  // clang-format on
}

template<int dim, typename Number>
void
Driver<dim, Number>::setup(std::shared_ptr<ApplicationBase<dim, Number>> app,
                           unsigned int const &                          degree,
                           unsigned int const &                          refine_space,
                           unsigned int const &                          refine_time,
                           bool const &                                  is_throughput_study)
{
  Timer timer;
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

  boundary_descriptor_density.reset(new BoundaryDescriptor<dim>());
  boundary_descriptor_velocity.reset(new BoundaryDescriptor<dim>());
  boundary_descriptor_pressure.reset(new BoundaryDescriptor<dim>());
  boundary_descriptor_energy.reset(new BoundaryDescriptorEnergy<dim>());

  application->set_boundary_conditions(boundary_descriptor_density,
                                       boundary_descriptor_velocity,
                                       boundary_descriptor_pressure,
                                       boundary_descriptor_energy);

  verify_boundary_conditions(*boundary_descriptor_density, *triangulation, periodic_faces);
  verify_boundary_conditions(*boundary_descriptor_velocity, *triangulation, periodic_faces);
  verify_boundary_conditions(*boundary_descriptor_pressure, *triangulation, periodic_faces);
  verify_boundary_conditions(*boundary_descriptor_energy, *triangulation, periodic_faces);

  field_functions.reset(new FieldFunctions<dim>());
  application->set_field_functions(field_functions);

  // Mapping
  unsigned int const mapping_degree = get_mapping_degree(param.mapping, degree);
  mesh.reset(new Mesh<dim>(mapping_degree));

  // initialize compressible Navier-Stokes operator
  comp_navier_stokes_operator.reset(new DGOperator<dim, Number>(*triangulation,
                                                                mesh->get_mapping(),
                                                                degree,
                                                                boundary_descriptor_density,
                                                                boundary_descriptor_velocity,
                                                                boundary_descriptor_pressure,
                                                                boundary_descriptor_energy,
                                                                field_functions,
                                                                param,
                                                                "fluid",
                                                                mpi_comm));

  // initialize matrix_free
  matrix_free_data.reset(new MatrixFreeData<dim, Number>());
  matrix_free_data->data.tasks_parallel_scheme =
    MatrixFree<dim, Number>::AdditionalData::partition_partition;
  comp_navier_stokes_operator->fill_matrix_free_data(*matrix_free_data);

  matrix_free.reset(new MatrixFree<dim, Number>());
  matrix_free->reinit(mesh->get_mapping(),
                      matrix_free_data->get_dof_handler_vector(),
                      matrix_free_data->get_constraint_vector(),
                      matrix_free_data->get_quadrature_vector(),
                      matrix_free_data->data);

  // setup compressible Navier-Stokes operator
  comp_navier_stokes_operator->setup(matrix_free, matrix_free_data);

  // initialize postprocessor
  if(!is_throughput_study)
  {
    postprocessor = application->construct_postprocessor(degree, mpi_comm);
    postprocessor->setup(*comp_navier_stokes_operator);

    // initialize time integrator
    time_integrator.reset(new TimeIntExplRK<Number>(
      comp_navier_stokes_operator, param, refine_time, mpi_comm, postprocessor));
    time_integrator->setup(param.restarted_simulation);
  }

  timer_tree.insert({"Compressible flow", "Setup"}, timer.wall_time());
}

template<int dim, typename Number>
void
Driver<dim, Number>::solve()
{
  time_integrator->timeloop();
}

template<int dim, typename Number>
void
Driver<dim, Number>::print_statistics(double const total_time) const
{
  this->pcout << std::endl
              << "_________________________________________________________________________________"
              << std::endl
              << std::endl;

  this->pcout << "Performance results for compressible Navier-Stokes solver:" << std::endl;

  // Wall times
  timer_tree.insert({"Compressible flow"}, total_time);

  timer_tree.insert({"Compressible flow"}, time_integrator->get_timings());

  pcout << std::endl << "Timings for level 1:" << std::endl;
  timer_tree.print_level(pcout, 1);

  pcout << std::endl << "Timings for level 2:" << std::endl;
  timer_tree.print_level(pcout, 2);

  // Throughput in DoFs/s per time step per core
  types::global_dof_index const DoFs            = comp_navier_stokes_operator->get_number_of_dofs();
  unsigned int const            N_mpi_processes = Utilities::MPI::n_mpi_processes(mpi_comm);
  unsigned int const            N_time_steps    = time_integrator->get_number_of_time_steps();

  Utilities::MPI::MinMaxAvg overall_time_data = Utilities::MPI::min_max_avg(total_time, mpi_comm);
  double const              overall_time_avg  = overall_time_data.avg;

  print_throughput_unsteady(pcout, DoFs, overall_time_avg, N_time_steps, N_mpi_processes);

  // computational costs in CPUh
  print_costs(pcout, overall_time_avg, N_mpi_processes);

  this->pcout << "_________________________________________________________________________________"
              << std::endl
              << std::endl;
}

template<int dim, typename Number>
std::tuple<unsigned int, types::global_dof_index, double>
Driver<dim, Number>::apply_operator(std::string const & operator_type_string,
                                    unsigned int const  n_repetitions_inner,
                                    unsigned int const  n_repetitions_outer) const
{
  pcout << std::endl << "Computing matrix-vector product ..." << std::endl;

  Operator operator_type;
  string_to_enum(operator_type, operator_type_string);

  // Vectors
  VectorType dst, src;

  // initialize vectors
  comp_navier_stokes_operator->initialize_dof_vector(src);
  comp_navier_stokes_operator->initialize_dof_vector(dst);
  src = 1.0;
  dst = 1.0;

  // Timer and wall times
  Timer  timer;
  double wall_time = std::numeric_limits<double>::max();

  for(unsigned int i_outer = 0; i_outer < n_repetitions_outer; ++i_outer)
  {
    double current_wall_time = 0.0;

    // apply matrix-vector product several times
    for(unsigned int i = 0; i < n_repetitions_inner; ++i)
    {
      timer.restart();

#ifdef LIKWID_PERFMON
      LIKWID_MARKER_START(("compressible_deg_" + std::to_string(degree)).c_str());
#endif

      if(operator_type == Operator::ConvectiveTerm)
        comp_navier_stokes_operator->evaluate_convective(dst, src, 0.0);
      else if(operator_type == Operator::ViscousTerm)
        comp_navier_stokes_operator->evaluate_viscous(dst, src, 0.0);
      else if(operator_type == Operator::ViscousAndConvectiveTerms)
        comp_navier_stokes_operator->evaluate_convective_and_viscous(dst, src, 0.0);
      else if(operator_type == Operator::InverseMassMatrix)
        comp_navier_stokes_operator->apply_inverse_mass(dst, src);
      else if(operator_type == Operator::InverseMassMatrixDstDst)
        comp_navier_stokes_operator->apply_inverse_mass(dst, dst);
      else if(operator_type == Operator::VectorUpdate)
        dst.sadd(2.0, 1.0, src);
      else if(operator_type == Operator::EvaluateOperatorExplicit)
        comp_navier_stokes_operator->evaluate(dst, src, 0.0);
      else
        AssertThrow(false, ExcMessage("Specified operator type not implemented"));

#ifdef LIKWID_PERFMON
      LIKWID_MARKER_STOP(("compressible_deg_" + std::to_string(degree)).c_str());
#endif

      Utilities::MPI::MinMaxAvg wall_time =
        Utilities::MPI::min_max_avg(timer.wall_time(), mpi_comm);

      current_wall_time += wall_time.avg;
    }

    // compute average wall time
    current_wall_time /= (double)n_repetitions_inner;

    wall_time = std::min(wall_time, current_wall_time);
  }

  if(wall_time * n_repetitions_inner * n_repetitions_outer < 1.0 /*wall time in seconds*/)
  {
    this->pcout
      << std::endl
      << "WARNING: One should use a larger number of matrix-vector products to obtain reproducable results."
      << std::endl;
  }

  types::global_dof_index const dofs = comp_navier_stokes_operator->get_number_of_dofs();

  double dofs_per_walltime = (double)dofs / wall_time;

  unsigned int N_mpi_processes = Utilities::MPI::n_mpi_processes(mpi_comm);

  // clang-format off
  pcout << std::endl
        << std::scientific << std::setprecision(4)
        << "DoFs/sec:        " << dofs_per_walltime << std::endl
        << "DoFs/(sec*core): " << dofs_per_walltime/(double)N_mpi_processes << std::endl;
  // clang-format on

  pcout << std::endl << " ... done." << std::endl << std::endl;

  return std::tuple<unsigned int, types::global_dof_index, double>(
    comp_navier_stokes_operator->get_polynomial_degree(), dofs, dofs_per_walltime);
}

template class Driver<2, float>;
template class Driver<3, float>;

template class Driver<2, double>;
template class Driver<3, double>;

} // namespace CompNS
} // namespace ExaDG
