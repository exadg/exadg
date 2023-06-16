/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2022 by the ExaDG authors
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

#ifndef INCLUDE_EXADG_FLUID_STRUCTURE_INTERACTION_PRECICE_PRECICE_ADAPTER_H_
#define INCLUDE_EXADG_FLUID_STRUCTURE_INTERACTION_PRECICE_PRECICE_ADAPTER_H_

// deal.II
#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/mapping_q_generic.h>

#include <deal.II/matrix_free/matrix_free.h>

// ExaDG
#include <exadg/fluid_structure_interaction/precice/dof_coupling.h>
#include <exadg/fluid_structure_interaction/precice/exadg_coupling.h>
#include <exadg/fluid_structure_interaction/precice/quad_coupling.h>
#include <exadg/utilities/tensor_utilities.h>

// preCICE
#ifdef EXADG_WITH_PRECICE
#  include <precice/SolverInterface.hpp>
#endif

#include <ostream>

namespace ExaDG
{
namespace preCICE
{
/**
 * The Adapter class keeps together with the CouplingInterfaes all
 * functionalities to couple deal.II to other solvers with preCICE i.e. data
 * structures are set up, necessary information is passed to preCICE etc.
 */
template<int dim,
         int data_dim,
         typename VectorType,
         typename VectorizedArrayType = dealii::VectorizedArray<double>>
class Adapter
{
public:
  static unsigned int const rank = n_components_to_rank<data_dim, dim>();

  using value_type = typename CouplingBase<dim, data_dim, VectorizedArrayType>::value_type;
  /**
   * @brief      Constructor, which sets up the precice Solverinterface
   *
   * @tparam     data_dim Dimension of the coupling data. Equivalent to n_components
   *             in the deal.II documentation
   *
   * @param[in]  parameters Parameter class, which hold the data specified
   *             in the parameters.prm file
   * @param[in]  dealii_boundary_surface_id Boundary ID of the
   *             triangulation, which is associated with the coupling
   *             surface.
   * @param[in]  data The applied matrix-free object
   * @param[in]  dof_index Index of the relevant dof_handler in the
   *             corresponding MatrixFree object
   * @param[in]  read_quad_index Index of the quadrature formula in the
   *             corresponding MatrixFree object which should be used for data
   *             reading
   * @param[in]  is_dirichlet Boolean to distinguish between Dirichlet type
   *             solver (using the DoFs for data reading) and Neumann type
   *             solver (using quadrature points for reading)
   */
  template<typename ParameterClass>
  Adapter(ParameterClass const & parameters, MPI_Comm mpi_comm);


  /**
   * @brief      Initializes preCICE and passes all relevant data to preCICE
   *
   * @param[in]  dealii_to_precice Data, which should be given to preCICE and
   *             exchanged with other participants. Wether this data is
   *             required already in the beginning depends on your
   *             individual configuration and preCICE determines it
   *             automatically. In many cases, this data will just represent
   *             your initial condition.
   */
  void
  initialize_precice(VectorType const & dealii_to_precice);

  void
  add_write_surface(
    dealii::types::boundary_id const                                            surface_id,
    std::string const &                                                         mesh_name,
    std::vector<std::string> const &                                            write_data_names,
    WriteDataType const                                                         write_data_type,
    std::shared_ptr<dealii::MatrixFree<dim, double, VectorizedArrayType> const> data,
    unsigned int const                                                          dof_index,
    unsigned int const                                                          write_quad_index);


  void
  add_read_surface(std::shared_ptr<dealii::MatrixFree<dim, double, VectorizedArrayType> const> data,
                   std::shared_ptr<ContainerInterfaceData<rank, dim, double>> interface_data,
                   std::string const &                                        mesh_name,
                   std::vector<std::string> const &                           read_data_name);

  /**
   * @brief      Advances preCICE after every timestep
   */
  void
  advance(double const computed_timestep_length);

  /**
   * @brief      Saves current state of time dependent variables in case of an
   *             implicit coupling
   *
   * @param[in]  state_variables Vector containing all variables to store as
   *             reference
   *
   * @note       This function only makes sense, if it is used with
   *             @p reload_old_state_if_required. Therefore, the order, in which the
   *             variables are passed into the vector must be the same for
   *             both functions.
   */
  void
  save_current_state_if_required(std::function<void()> const & save_state);

  /**
   * @brief      Reloads the previously stored variables in case of an implicit
   *             coupling. The current implementation supports subcycling,
   *             i.e. previously refers o the last time
   *             @p save_current_state_if_required() has been called.
   *
   * @param[out] state_variables Vector containing all variables to reload
   *             as reference
   *
   * @note       This function only makes sense, if the state variables have been
   *             stored by calling @p save_current_state_if_required. Therefore,
   *             the order, in which the variables are passed into the
   *             vector must be the same for both functions.
   */
  void
  reload_old_state_if_required(std::function<void()> const & reload_old_state);


  void
  write_data(std::string const & write_mesh_name,
             std::string const & write_data_name,
             VectorType const &  write_data,
             double const        computed_timestep_length);

  void
  read_block_data(std::string const & mesh_name, const std::string & data_name) const;

  /**
   * @brief is_coupling_ongoing Calls the preCICE API function isCouplingOnGoing
   *
   * @return returns true if the coupling has not yet been finished
   */
  bool
  is_coupling_ongoing() const;

  /**
   * @brief is_time_window_complete Calls the preCICE API function isTimeWindowComplete
   *
   * @return returns true if the coupling time window has been completed in the current
   *         iteration
   */
  bool
  is_time_window_complete() const;


private:
  // public precice solverinterface, needed in order to steer the time loop
  // inside the solver.
#ifdef EXADG_WITH_PRECICE
  std::shared_ptr<precice::SolverInterface> precice;
#endif

  /// The objects handling reading and writing data
  std::map<std::string, std::shared_ptr<CouplingBase<dim, data_dim, VectorizedArrayType>>> writer;

  // We restrict the reader to be of type ExaDGCoupling for the moment, as all other choices don't
  // make sense
  std::map<std::string, std::shared_ptr<ExaDGCoupling<dim, data_dim, VectorizedArrayType>>> reader;

  // Container to store time dependent data in case of an implicit coupling
  std::vector<VectorType> old_state_data;
};



template<int dim, int data_dim, typename VectorType, typename VectorizedArrayType>
template<typename ParameterClass>
Adapter<dim, data_dim, VectorType, VectorizedArrayType>::Adapter(ParameterClass const & parameters,
                                                                 MPI_Comm               mpi_comm)

{
#ifdef EXADG_WITH_PRECICE
  precice =
    std::make_shared<precice::SolverInterface>(parameters.participant_name,
                                               parameters.config_file,
                                               dealii::Utilities::MPI::this_mpi_process(mpi_comm),
                                               dealii::Utilities::MPI::n_mpi_processes(mpi_comm));

  AssertThrow(dim == precice->getDimensions(), dealii::ExcInternalError());
#else
  (void)parameters;
  (void)mpi_comm;
  AssertThrow(false,
              dealii::ExcMessage("EXADG_WITH_PRECICE has to be activated to use this code."));
#endif

  AssertThrow(dim > 1, dealii::ExcNotImplemented());
}



template<int dim, int data_dim, typename VectorType, typename VectorizedArrayType>
void
Adapter<dim, data_dim, VectorType, VectorizedArrayType>::add_write_surface(
  dealii::types::boundary_id const dealii_boundary_surface_id,
  std::string const &              mesh_name,
  std::vector<std::string> const & write_data_names,
  WriteDataType const              write_data_type,
  std::shared_ptr<dealii::MatrixFree<dim, double, VectorizedArrayType> const> data,
  unsigned int const                                                          dof_index,
  unsigned int const                                                          quad_index)
{
  // Check, if we already have such an interface
  auto const found_reader = reader.find(mesh_name);

  if(found_reader != reader.end())
  {
    // Duplicate the shared pointer
    writer.insert({mesh_name, found_reader->second});
  }
  else if(write_data_type == WriteDataType::values_on_dofs)
  {
    writer.insert(
      {mesh_name,
       std::make_shared<DoFCoupling<dim, data_dim, VectorizedArrayType>>(data,
#ifdef EXADG_WITH_PRECICE
                                                                         precice,
#endif
                                                                         mesh_name,
                                                                         dealii_boundary_surface_id,
                                                                         dof_index)});
  }
  else
  {
    Assert(write_data_type == WriteDataType::values_on_q_points or
             write_data_type == WriteDataType::normal_gradients_on_q_points,
           dealii::ExcNotImplemented());
    writer.insert({mesh_name,
                   std::make_shared<QuadCoupling<dim, data_dim, VectorizedArrayType>>(
                     data,
#ifdef EXADG_WITH_PRECICE
                     precice,
#endif
                     mesh_name,
                     dealii_boundary_surface_id,
                     dof_index,
                     quad_index)});
  }

  // Register the write data
  std::vector<dealii::Point<dim>> const points;
  for(auto const & data_name : write_data_names)
    writer.at(mesh_name)->add_write_data(data_name);

  writer.at(mesh_name)->set_write_data_type(write_data_type);
  // Finally initialize the surface
  writer.at(mesh_name)->define_coupling_mesh();
}



template<int dim, int data_dim, typename VectorType, typename VectorizedArrayType>
void
Adapter<dim, data_dim, VectorType, VectorizedArrayType>::add_read_surface(
  std::shared_ptr<dealii::MatrixFree<dim, double, VectorizedArrayType> const> data,
  std::shared_ptr<ContainerInterfaceData<rank, dim, double>>                  interface_data,
  std::string const &                                                         mesh_name,
  std::vector<std::string> const &                                            read_data_names)
{
  reader.insert(
    {mesh_name,
     std::make_shared<ExaDGCoupling<dim, data_dim, VectorizedArrayType>>(data,
#ifdef EXADG_WITH_PRECICE
                                                                         precice,
#endif
                                                                         mesh_name,
                                                                         interface_data)});

  for(auto const & data_name : read_data_names)
    reader.at(mesh_name)->add_read_data(data_name);
  reader.at(mesh_name)->define_coupling_mesh();
}



template<int dim, int data_dim, typename VectorType, typename VectorizedArrayType>
void
Adapter<dim, data_dim, VectorType, VectorizedArrayType>::initialize_precice(
  VectorType const & dealii_to_precice)
{
#ifdef EXADG_WITH_PRECICE
  // if(not dealii_to_precice.has_ghost_elements())
  //   dealii_to_precice.update_ghost_values();

  // Initialize preCICE internally
  precice->initialize();

  // Only the writer needs potentially to process the coupling mesh, if the
  // mapping is carried out in the solver
  // writer->process_coupling_mesh();

  // write initial writeData to preCICE if required
  if(precice->isActionRequired(precice::constants::actionWriteInitialData()))
  {
    writer[0]->write_data(dealii_to_precice, "");

    precice->markActionFulfilled(precice::constants::actionWriteInitialData());
  }
  precice->initializeData();

  // Maybe, read block-wise and work with an AlignedVector since the read data
  // (forces) is multiple times required during the Newton iteration
  //    if (shared_memory_parallel and precice->isReadDataAvailable())
  //      precice->readBlockVectorData(read_data_id,
  //                                   read_nodes_ids.size(),
  //                                   read_nodes_ids.data(),
  //                                   read_data.data());
#else
  (void)dealii_to_precice;
#endif
}

template<int dim, int data_dim, typename VectorType, typename VectorizedArrayType>
void
Adapter<dim, data_dim, VectorType, VectorizedArrayType>::write_data(
  std::string const & write_mesh_name,
  std::string const & write_data_name,
  VectorType const &  dealii_to_precice,
  double const        computed_timestep_length)
{
#ifdef EXADG_WITH_PRECICE
  if(precice->isWriteDataRequired(computed_timestep_length))
    writer.at(write_mesh_name)->write_data(dealii_to_precice, write_data_name);
#else
  (void)write_mesh_name;
  (void)write_data_name;
  (void)dealii_to_precice;
  (void)computed_timestep_length;
#endif
}

template<int dim, int data_dim, typename VectorType, typename VectorizedArrayType>
void
Adapter<dim, data_dim, VectorType, VectorizedArrayType>::advance(
  double const computed_timestep_length)
{
#ifdef EXADG_WITH_PRECICE
  // Here, we need to specify the computed time step length and pass it to
  // preCICE
  // TODO: The function returns the available time-step size which is required for time-step sync
  precice->advance(computed_timestep_length);
#else
  (void)computed_timestep_length;
#endif
}



template<int dim, int data_dim, typename VectorType, typename VectorizedArrayType>
void
Adapter<dim, data_dim, VectorType, VectorizedArrayType>::read_block_data(
  std::string const & mesh_name,
  std::string const & data_name) const
{
  reader.at(mesh_name)->read_block_data(data_name);
}



template<int dim, int data_dim, typename VectorType, typename VectorizedArrayType>
inline void
Adapter<dim, data_dim, VectorType, VectorizedArrayType>::save_current_state_if_required(
  std::function<void()> const & save_state)
{
#ifdef EXADG_WITH_PRECICE
  // First, we let preCICE check, whether we need to store the variables.
  // Then, the data is stored in the class
  if(precice->isActionRequired(precice::constants::actionWriteIterationCheckpoint()))
  {
    save_state();
    precice->markActionFulfilled(precice::constants::actionWriteIterationCheckpoint());
  }
#else
  (void)save_state;
#endif
}



template<int dim, int data_dim, typename VectorType, typename VectorizedArrayType>
inline void
Adapter<dim, data_dim, VectorType, VectorizedArrayType>::reload_old_state_if_required(
  std::function<void()> const & reload_old_state)
{
#ifdef EXADG_WITH_PRECICE
  // In case we need to reload a state, we just take the internally stored
  // data vectors and write then in to the input data
  if(precice->isActionRequired(precice::constants::actionReadIterationCheckpoint()))
  {
    reload_old_state();
    precice->markActionFulfilled(precice::constants::actionReadIterationCheckpoint());
  }
#else
  (void)reload_old_state;
#endif
}



template<int dim, int data_dim, typename VectorType, typename VectorizedArrayType>
inline bool
Adapter<dim, data_dim, VectorType, VectorizedArrayType>::is_coupling_ongoing() const
{
#ifdef EXADG_WITH_PRECICE
  return precice->isCouplingOngoing();
#else
  return true;
#endif
}



template<int dim, int data_dim, typename VectorType, typename VectorizedArrayType>
inline bool
Adapter<dim, data_dim, VectorType, VectorizedArrayType>::is_time_window_complete() const
{
#ifdef EXADG_WITH_PRECICE
  return precice->isTimeWindowComplete();
#else
  return true;
#endif
}


} // namespace preCICE
} // namespace ExaDG

#endif
