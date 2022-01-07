#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/mapping_q_generic.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <exadg/fluid_structure_interaction_precice/dof_interface.h>
#include <exadg/fluid_structure_interaction_precice/exadg_interface.h>
#include <exadg/fluid_structure_interaction_precice/quad_interface.h>
#include <precice/SolverInterface.hpp>

#include <ostream>

namespace Adapter
{
using namespace dealii;

/**
 * The Adapter class keeps together with the CouplingInterfaes all
 * functionalities to couple deal.II to other solvers with preCICE i.e. data
 * structures are set up, necessary information is passed to preCICE etc.
 */
template<int dim,
         int data_dim,
         typename VectorType,
         typename VectorizedArrayType = VectorizedArray<double>>
class Adapter
{
public:
  using value_type = typename CouplingInterface<dim, data_dim, VectorizedArrayType>::value_type;
  /**
   * @brief      Constructor, which sets up the precice Solverinterface
   *
   * @tparam     data_dim Dimension of the coupling data. Equivalent to n_components
   *             in the deal.II documentation
   *
   * @param[in]  parameters Parameter class, which hold the data specified
   *             in the parameters.prm file
   * @param[in]  dealii_boundary_interface_id Boundary ID of the
   *             triangulation, which is associated with the coupling
   *             interface.
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
  Adapter(const ParameterClass & parameters,
          const unsigned int     dealii_boundary_interface_id,
          std::shared_ptr<const MatrixFree<dim, double, VectorizedArrayType>> data,
          const unsigned int                                                  dof_index       = 0,
          const unsigned int                                                  read_quad_index = 0,
          const bool                                                          is_dirichlet = false);

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
  initialize(const VectorType & dealii_to_precice, const std::vector<Point<dim>> & points);

  /**
   * @brief      Advances preCICE after every timestep, converts data formats
   *             between preCICE and dealii
   *
   * @param[in]  dealii_to_precice Same data as in @p initialize_precice() i.e.
   *             data, which should be given to preCICE after each time step
   *             and exchanged with other participants.
   * @param[in]  computed_timestep_length Length of the timestep used by
   *             the solver.
   */
  void
  advance(const double computed_timestep_length);

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
  save_current_state_if_required(const std::function<void()> & save_state);

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
  reload_old_state_if_required(const std::function<void()> & reload_old_state);

  /**
   * @brief Public API adapter method, which calls the respective implementation
   *        in derived classes of the CouplingInterface. Have a look at the
   *        documentation there.
   */
  value_type
  read_on_quadrature_point(const unsigned int id_number, const unsigned int active_faces) const;


  void
  write_data(const VectorType & write_data, const double computed_timestep_length);

  std::vector<Tensor<1, dim>>
  read_block_data() const;

  /**
   * @brief Public API adapter method, which calls the respective implementation
   *        in derived classes of the CouplingInterface. Have a look at the
   *        documentation there.
   */
  void
  apply_dirichlet_bcs(AffineConstraints<double> & constraints) const;

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


  /// Boundary ID of the deal.II mesh, associated with the coupling
  /// interface. The variable is public and should be used during grid
  /// generation, but is also involved during system assembly. The only thing,
  /// one needs to make sure is, that this ID is not given to another part of
  /// the boundary e.g. clamped one.
  const unsigned int dealii_boundary_interface_id;


private:
  // public precice solverinterface, needed in order to steer the time loop
  // inside the solver.
  std::shared_ptr<precice::SolverInterface> precice;
  /// The objects handling reading and writing data
  std::shared_ptr<CouplingInterface<dim, data_dim, VectorizedArrayType>> writer;
  std::shared_ptr<CouplingInterface<dim, data_dim, VectorizedArrayType>> reader;

  // Container to store time dependent data in case of an implicit coupling
  std::vector<VectorType> old_state_data;
  double                  old_time_value = 0;
};



template<int dim, int data_dim, typename VectorType, typename VectorizedArrayType>
template<typename ParameterClass>
Adapter<dim, data_dim, VectorType, VectorizedArrayType>::Adapter(
  const ParameterClass &                                              parameters,
  const unsigned int                                                  dealii_boundary_interface_id,
  std::shared_ptr<const MatrixFree<dim, double, VectorizedArrayType>> data,
  const unsigned int                                                  dof_index,
  const unsigned int /* read_quad_index*/,
  const bool /*is_dirichlet*/)
  : dealii_boundary_interface_id(dealii_boundary_interface_id)
{
  precice =
    std::make_shared<precice::SolverInterface>(parameters.participant_name,
                                               parameters.config_file,
                                               Utilities::MPI::this_mpi_process(MPI_COMM_WORLD),
                                               Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD));

  AssertThrow(dim == precice->getDimensions(), ExcInternalError());
  AssertThrow(dim > 1, ExcNotImplemented());

  reader = std::make_shared<ExaDGInterface<dim, data_dim, VectorizedArrayType>>(
    data, precice, parameters.read_mesh_name, dealii_boundary_interface_id);

  // 2. Set the writer, which is defined in the parameter file
  if(parameters.write_mesh_name == parameters.read_mesh_name)
    writer = reader;
  else if(parameters.write_data_specification == "values_on_dofs")
    writer = std::make_shared<DoFInterface<dim, data_dim, VectorizedArrayType>>(
      data, precice, parameters.write_mesh_name, dealii_boundary_interface_id, dof_index);
  else
  {
    Assert(parameters.write_data_specification == "values_on_quads" ||
             parameters.write_data_specification == "normal_gradients_on_quads",
           ExcNotImplemented());
    writer = std::make_shared<QuadInterface<dim, data_dim, VectorizedArrayType>>(
      data,
      precice,
      parameters.write_mesh_name,
      dealii_boundary_interface_id,
      dof_index,
      parameters.write_quad_index);
  }

  reader->add_read_data(parameters.read_data_name);
  writer->add_write_data(parameters.write_data_name, parameters.write_data_specification);

  Assert(reader.get() != nullptr, ExcInternalError());
  Assert(writer.get() != nullptr, ExcInternalError());
}



template<int dim, int data_dim, typename VectorType, typename VectorizedArrayType>
void
Adapter<dim, data_dim, VectorType, VectorizedArrayType>::initialize(
  const VectorType &              dealii_to_precice,
  const std::vector<Point<dim>> & points)
{
  if(!dealii_to_precice.has_ghost_elements())
    dealii_to_precice.update_ghost_values();

  reader->define_coupling_mesh(points);
  writer->define_coupling_mesh(points);

  // Initialize preCICE internally
  precice->initialize();

  // Only the writer needs potentially to process the coupling mesh, if the
  // mapping is carried out in the solver
  writer->process_coupling_mesh();

  // write initial writeData to preCICE if required
  if(precice->isActionRequired(precice::constants::actionWriteInitialData()))
  {
    writer->write_data(dealii_to_precice);

    precice->markActionFulfilled(precice::constants::actionWriteInitialData());
  }
  precice->initializeData();

  // Maybe, read block-wise and work with an AlignedVector since the read data
  // (forces) is multiple times required during the Newton iteration
  //    if (shared_memory_parallel && precice->isReadDataAvailable())
  //      precice->readBlockVectorData(read_data_id,
  //                                   read_nodes_ids.size(),
  //                                   read_nodes_ids.data(),
  //                                   read_data.data());
}

template<int dim, int data_dim, typename VectorType, typename VectorizedArrayType>
void
Adapter<dim, data_dim, VectorType, VectorizedArrayType>::write_data(
  const VectorType & dealii_to_precice,
  const double       computed_timestep_length)
{
  if(precice->isWriteDataRequired(computed_timestep_length))
    writer->write_data(dealii_to_precice);
}

template<int dim, int data_dim, typename VectorType, typename VectorizedArrayType>
void
Adapter<dim, data_dim, VectorType, VectorizedArrayType>::advance(
  const double computed_timestep_length)
{
  // Here, we need to specify the computed time step length and pass it to
  // preCICE
  precice->advance(computed_timestep_length);

  //    if (shared_memory_parallel && precice->isReadDataAvailable())
  //      precice->readBlockVectorData(read_data_id,
  //                                   read_nodes_ids.size(),
  //                                   read_nodes_ids.data(),
  //                                   read_data.data());
}


template<int dim, int data_dim, typename VectorType, typename VectorizedArrayType>
inline typename Adapter<dim, data_dim, VectorType, VectorizedArrayType>::value_type
Adapter<dim, data_dim, VectorType, VectorizedArrayType>::read_on_quadrature_point(
  const unsigned int id_number,
  const unsigned int active_faces) const
{
  return reader->read_on_quadrature_point(id_number, active_faces);
}



template<int dim, int data_dim, typename VectorType, typename VectorizedArrayType>
std::vector<Tensor<1, dim>>
Adapter<dim, data_dim, VectorType, VectorizedArrayType>::read_block_data() const
{
  return reader->read_block_data();
}



template<int dim, int data_dim, typename VectorType, typename VectorizedArrayType>
void
Adapter<dim, data_dim, VectorType, VectorizedArrayType>::apply_dirichlet_bcs(
  AffineConstraints<double> & constraints) const
{
  reader->apply_Dirichlet_bcs(constraints);
}



template<int dim, int data_dim, typename VectorType, typename VectorizedArrayType>
inline void
Adapter<dim, data_dim, VectorType, VectorizedArrayType>::save_current_state_if_required(
  const std::function<void()> & save_state)
{
  // First, we let preCICE check, whether we need to store the variables.
  // Then, the data is stored in the class
  if(precice->isActionRequired(precice::constants::actionWriteIterationCheckpoint()))
  {
    save_state();
    precice->markActionFulfilled(precice::constants::actionWriteIterationCheckpoint());
  }
}



template<int dim, int data_dim, typename VectorType, typename VectorizedArrayType>
inline void
Adapter<dim, data_dim, VectorType, VectorizedArrayType>::reload_old_state_if_required(
  const std::function<void()> & reload_old_state)
{
  // In case we need to reload a state, we just take the internally stored
  // data vectors and write then in to the input data
  if(precice->isActionRequired(precice::constants::actionReadIterationCheckpoint()))
  {
    reload_old_state();
    precice->markActionFulfilled(precice::constants::actionReadIterationCheckpoint());
  }
}



template<int dim, int data_dim, typename VectorType, typename VectorizedArrayType>
inline bool
Adapter<dim, data_dim, VectorType, VectorizedArrayType>::is_coupling_ongoing() const
{
  return precice->isCouplingOngoing();
}



template<int dim, int data_dim, typename VectorType, typename VectorizedArrayType>
inline bool
Adapter<dim, data_dim, VectorType, VectorizedArrayType>::is_time_window_complete() const
{
  return precice->isTimeWindowComplete();
}
} // namespace Adapter
