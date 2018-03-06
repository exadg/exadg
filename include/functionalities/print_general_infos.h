/*
 * print_general_infos.h
 *
 *  Created on: Feb 21, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_FUNCTIONALITIES_PRINT_GENERAL_INFOS_H_
#define INCLUDE_FUNCTIONALITIES_PRINT_GENERAL_INFOS_H_


// print MPI info
void print_MPI_info(ConditionalOStream const &pcout)
{
  pcout << std::endl << "MPI info:" << std::endl << std::endl;
  print_parameter(pcout,"Number of processes",Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD));
}

// print grid info
template<int dim>
void print_grid_data(ConditionalOStream const                        &pcout,
                     unsigned int const                              n_refine_space,
                     parallel::distributed::Triangulation<dim> const &triangulation)
{
  pcout << std::endl
        << "Generating grid for " << dim << "-dimensional problem:" << std::endl
        << std::endl;

  print_parameter(pcout,"Number of refinements",n_refine_space);
  print_parameter(pcout,"Number of cells",triangulation.n_global_active_cells());
  print_parameter(pcout,"Number of faces",triangulation.n_active_faces());
  print_parameter(pcout,"Number of vertices",triangulation.n_vertices());
}

template<int dim>
void print_grid_data(ConditionalOStream const                        &pcout,
                     unsigned int const                              n_refine_space_1,
                     parallel::distributed::Triangulation<dim> const &triangulation_1,
                     unsigned int const                              n_refine_space_2,
                     parallel::distributed::Triangulation<dim> const &triangulation_2)
{
  pcout << std::endl
        << "Generating grid for DOMAIN 1 for " << dim << "-dimensional problem:" << std::endl
        << std::endl;

  print_parameter(pcout,"Number of refinements",n_refine_space_1);
  print_parameter(pcout,"Number of cells",triangulation_1.n_global_active_cells());
  print_parameter(pcout,"Number of faces",triangulation_1.n_active_faces());
  print_parameter(pcout,"Number of vertices",triangulation_1.n_vertices());

  pcout << std::endl
        << "Generating grid for DOMAIN 2 for " << dim << "-dimensional problem:" << std::endl
        << std::endl;

  print_parameter(pcout,"Number of refinements",n_refine_space_2);
  print_parameter(pcout,"Number of cells",triangulation_2.n_global_active_cells());
  print_parameter(pcout,"Number of faces",triangulation_2.n_active_faces());
  print_parameter(pcout,"Number of vertices",triangulation_2.n_vertices());
}

#endif /* INCLUDE_FUNCTIONALITIES_PRINT_GENERAL_INFOS_H_ */
