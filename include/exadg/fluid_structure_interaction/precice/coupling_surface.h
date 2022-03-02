#pragma once

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <exadg/matrix_free/integrators.h>
#include <precice/SolverInterface.hpp>

namespace ExaDG
{
namespace preCICE
{
/**
 * Enum to handle all implemented data write methods one can use
 */
enum class WriteDataType
{
  undefined,
  values_on_dofs,
  values_on_other_mesh,
  gradients_on_other_mesh,
  values_on_q_points,
  normal_gradients_on_q_points
};

/**
 * A pure abstract base class, which defines the methods for the functions
 * used in the main Adapter class. Each instance of all derived classes are
 * always dedicated to a specific coupling mesh, i.e., the vertices used for
 * the coupling. The instantiated objects provide functions on how to read
 * and write data on this mesh and how to define the mesh by means of its
 * spatial coordinates.
 */
template<int dim, int data_dim, typename VectorizedArrayType>
class CouplingSurface
{
public:
  CouplingSurface(std::shared_ptr<const dealii::MatrixFree<dim, double, VectorizedArrayType>> data,
                  std::shared_ptr<precice::SolverInterface> precice,
                  const std::string                         mesh_name,
                  const dealii::types::boundary_id          surface_id);

  virtual ~CouplingSurface() = default;

  /// Alias for the face integrator
  using FEFaceIntegrator = FaceIntegrator<dim, data_dim, double, VectorizedArrayType>;
  using value_type       = typename FEFaceIntegrator::value_type;
  /**
   * @brief define_coupling_mesh Define the coupling mesh associated to the
   *        data points
   */
  virtual void
  define_coupling_mesh(const std::vector<dealii::Point<dim>> & vec) = 0;

  /**
   * @brief process_coupling_mesh (optional) Handle post-preCICE-initialization
   *        steps, e.g. do computations on received partitions or create
   *        communication patterns. This function just returns in the base
   *        class implementation.
   */
  virtual void
  process_coupling_mesh();

  /**
   * @brief write_data Write the data associated to the defined vertices
   *        to preCICE
   *
   * @param data_vector Vector holding the global solution to be passed to
   *        preCICE.
   */
  virtual void
  write_data(const dealii::LinearAlgebra::distributed::Vector<double> & data_vector,
             const std::string &                                        data_name) = 0;


  virtual void
  read_block_data(const std::string & data_name) const;

  /**
   * @brief Queries data IDs from preCICE for the given read data name
   * @param read_data_name
   */
  void
  add_read_data(const std::string & read_data_name);

  /**
   * @brief Queries data IDs from preCICE for the given write data name
   * @param write_data_name
   */
  void
  add_write_data(const std::string & write_data_name);

  /**
   * @brief
   *
   */
  void
  set_write_data_type(WriteDataType write_data_specification);

protected:
  /**
   * @brief Print information of the current setup
   *
   * @param[in] reader Boolean in order to decide if we want read or write
   *            data information
   * @param[in] local_size The number of element the local process works on
   */
  void
  print_info(const bool reader, const unsigned int local_size) const;

  /// The dealii::MatrixFree object (preCICE can only handle double precision)
  std::shared_ptr<const dealii::MatrixFree<dim, double, VectorizedArrayType>> mf_data;

  /// public precice solverinterface
  std::shared_ptr<precice::SolverInterface> precice;

  /// Configuration parameters
  const std::string mesh_name;
  int               mesh_id;
  // Map between data ID (preCICE) and the data name
  std::map<std::string, int> read_data_map;
  std::map<std::string, int> write_data_map;

  const types::boundary_id dealii_boundary_surface_id;

  WriteDataType write_data_type;

  virtual std::string
  get_surface_type() const = 0;
};



template<int dim, int data_dim, typename VectorizedArrayType>
CouplingSurface<dim, data_dim, VectorizedArrayType>::CouplingSurface(
  std::shared_ptr<const dealii::MatrixFree<dim, double, VectorizedArrayType>> data,
  std::shared_ptr<precice::SolverInterface>                                   precice,
  const std::string                                                           mesh_name,
  const dealii::types::boundary_id                                            surface_id)
  : mf_data(data),
    precice(precice),
    mesh_name(mesh_name),
    dealii_boundary_surface_id(surface_id),
    write_data_type(WriteDataType::undefined)
{
  Assert(data.get() != nullptr, dealii::ExcNotInitialized());
  Assert(precice.get() != nullptr, dealii::ExcNotInitialized());

  // Ask preCICE already in the constructor for the IDs
  mesh_id = precice->getMeshID(mesh_name);
}


template<int dim, int data_dim, typename VectorizedArrayType>
void
CouplingSurface<dim, data_dim, VectorizedArrayType>::add_read_data(
  const std::string & read_data_name)
{
  Assert(mesh_id != -1, dealii::ExcNotInitialized());
  const int read_data_id = precice->getDataID(read_data_name, mesh_id);
  read_data_map.insert({read_data_name, read_data_id});
}



template<int dim, int data_dim, typename VectorizedArrayType>
void
CouplingSurface<dim, data_dim, VectorizedArrayType>::add_write_data(
  const std::string & write_data_name)
{
  Assert(mesh_id != -1, dealii::ExcNotInitialized());
  const int write_data_id = precice->getDataID(write_data_name, mesh_id);
  write_data_map.insert({write_data_name, write_data_id});
}


template<int dim, int data_dim, typename VectorizedArrayType>
void
CouplingSurface<dim, data_dim, VectorizedArrayType>::set_write_data_type(
  WriteDataType write_data_specification)
{
  write_data_type = write_data_specification;
}



template<int dim, int data_dim, typename VectorizedArrayType>
void
CouplingSurface<dim, data_dim, VectorizedArrayType>::process_coupling_mesh()
{
  return;
}



template<int dim, int data_dim, typename VectorizedArrayType>
void
CouplingSurface<dim, data_dim, VectorizedArrayType>::read_block_data(const std::string &) const
{
  AssertThrow(false, dealii::ExcNotImplemented());
}


template<int dim, int data_dim, typename VectorizedArrayType>
void
CouplingSurface<dim, data_dim, VectorizedArrayType>::print_info(const bool         reader,
                                                                const unsigned int local_size) const
{
  Assert(mf_data.get() != 0, dealii::ExcNotInitialized());
  ConditionalOStream pcout(std::cout,
                           Utilities::MPI::this_mpi_process(
                             mf_data->get_dof_handler().get_communicator()) == 0);
  const auto         map = (reader ? read_data_map : write_data_map);

  auto        names      = map.begin();
  std::string data_names = names->first;
  ++names;
  for(; names != map.end(); ++names)
    data_names += std::string(", ") + names->first;

  pcout << "--     Data " << (reader ? "reading" : "writing") << ":\n"
        << "--     . data name(s): " << data_names << "\n"
        << "--     . associated mesh: " << mesh_name << "\n"
        << "--     . Number of coupling nodes: "
        << Utilities::MPI::sum(local_size, mf_data->get_dof_handler().get_communicator()) << "\n"
        << "--     . Node location: " << get_surface_type() << "\n"
        << std::endl;
}

} // namespace preCICE
} // namespace ExaDG