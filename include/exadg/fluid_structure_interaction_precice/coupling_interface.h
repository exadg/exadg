#pragma once

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <exadg/matrix_free/integrators.h>
#include <precice/SolverInterface.hpp>

namespace Adapter
{
using namespace dealii;

/**
 * Enum to handle all implemented data write methods one can use
 */
enum class WriteDataType
{
  undefined,
  values_on_dofs,
  values_on_other_mesh,
  gradients_on_other_mesh,
  values_on_quads,
  normal_gradients_on_quads
};

/**
 * A pure abstract base class, which defines the interface for the functions
 * used in the main Adapter class. Each instance of all derived classes are
 * always dedicated to a specific coupling mesh and may provide functions on
 * how to read and write data on this mesh and how to define the mesh.
 */
template<int dim, int data_dim, typename VectorizedArrayType>
class CouplingInterface
{
public:
  CouplingInterface(std::shared_ptr<const MatrixFree<dim, double, VectorizedArrayType>> data,
                    std::shared_ptr<precice::SolverInterface>                           precice,
                    std::string                                                         mesh_name,
                    types::boundary_id interface_id);

  virtual ~CouplingInterface() = default;

  /// Alias for the face integrator
  using FEFaceIntegrator = FaceIntegrator<dim, data_dim, double, VectorizedArrayType>;
  using value_type       = typename FEFaceIntegrator::value_type;
  /**
   * @brief define_coupling_mesh Define the coupling mesh associated to the
   *        data points
   */
  virtual void
  define_coupling_mesh(const std::vector<Point<dim>> & vec) = 0;

  /**
   * @brief process_coupling_mesh (optional) Handle post-preCICE-initialization
   *        steps, e.g. do computations on recieved partitions or create
   *        communication patterns. This function just returns in the base
   *        class implementation.
   */
  virtual void
  process_coupling_mesh();

  /**
   * @brief write_data Write the data associated to the defined vertice
   *        to preCICE
   *
   * @param data_vector Vector holding the global solution to be passed to
   *        preCICE.
   */
  virtual void
  write_data(const LinearAlgebra::distributed::Vector<double> & data_vector) = 0;


  virtual std::vector<Tensor<1, dim>>
  read_block_data() const;

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
  add_write_data(const std::string & write_data_name, const std::string & write_data_specification);

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

  /// The MatrixFree object (preCICE can only handle double precision)
  std::shared_ptr<const MatrixFree<dim, double, VectorizedArrayType>> mf_data;

  /// public precice solverinterface
  std::shared_ptr<precice::SolverInterface> precice;

  /// Configuration parameters
  const std::string mesh_name;
  std::string       read_data_name  = "";
  std::string       write_data_name = "";
  int               mesh_id         = -1;
  int               read_data_id    = -1;
  int               write_data_id   = -1;

  const types::boundary_id dealii_boundary_interface_id;

  WriteDataType write_data_type = WriteDataType::undefined;

  virtual std::string
  get_interface_type() const = 0;
};



template<int dim, int data_dim, typename VectorizedArrayType>
CouplingInterface<dim, data_dim, VectorizedArrayType>::CouplingInterface(
  std::shared_ptr<const MatrixFree<dim, double, VectorizedArrayType>> data,
  std::shared_ptr<precice::SolverInterface>                           precice,
  std::string                                                         mesh_name,
  const types::boundary_id                                            interface_id)
  : mf_data(data),
    precice(precice),
    mesh_name(mesh_name),
    dealii_boundary_interface_id(interface_id)
{
  Assert(data.get() != nullptr, ExcNotInitialized());
  Assert(precice.get() != nullptr, ExcNotInitialized());

  // Ask preCICE already in the constructor for the IDs
  mesh_id = precice->getMeshID(mesh_name);
}


template<int dim, int data_dim, typename VectorizedArrayType>
void
CouplingInterface<dim, data_dim, VectorizedArrayType>::add_read_data(
  const std::string & read_data_name_)
{
  Assert(mesh_id != -1, ExcNotInitialized());
  read_data_name = read_data_name_;
  read_data_id   = precice->getDataID(read_data_name, mesh_id);
}



template<int dim, int data_dim, typename VectorizedArrayType>
void
CouplingInterface<dim, data_dim, VectorizedArrayType>::add_write_data(
  const std::string & write_data_name_,
  const std::string & write_data_specification)
{
  Assert(mesh_id != -1, ExcNotInitialized());
  write_data_name = write_data_name_;
  write_data_id   = precice->getDataID(write_data_name, mesh_id);

  if(write_data_specification == "values_on_dofs")
    write_data_type = WriteDataType::values_on_dofs;
  else if(write_data_specification == "values_on_other_mesh")
    write_data_type = WriteDataType::values_on_other_mesh;
  else if(write_data_specification == "gradients_on_other_mesh")
    write_data_type = WriteDataType::gradients_on_other_mesh;
  else if(write_data_specification == "values_on_quads")
    write_data_type = WriteDataType::values_on_quads;
  else if(write_data_specification == "normal_gradients_on_quads")
    write_data_type = WriteDataType::normal_gradients_on_quads;
  else
    AssertThrow(false, ExcMessage("Unknwon write data type."));
}



template<int dim, int data_dim, typename VectorizedArrayType>
void
CouplingInterface<dim, data_dim, VectorizedArrayType>::process_coupling_mesh()
{
  return;
}



template<int dim, int data_dim, typename VectorizedArrayType>
std::vector<Tensor<1, dim>>
CouplingInterface<dim, data_dim, VectorizedArrayType>::read_block_data() const
{
  AssertThrow(false, ExcNotImplemented());
}


template<int dim, int data_dim, typename VectorizedArrayType>
void
CouplingInterface<dim, data_dim, VectorizedArrayType>::print_info(
  const bool         reader,
  const unsigned int local_size) const
{
  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

  pcout << "--     Data " << (reader ? "reading" : "writing") << ":\n"
        << "--     . data name: " << (reader ? read_data_name : write_data_name) << "\n"
        << "--     . associated mesh: " << mesh_name << "\n"
        << "--     . Number of interface nodes: " << Utilities::MPI::sum(local_size, MPI_COMM_WORLD)
        << "\n"
        << "--     . Node location: " << get_interface_type() << "\n"
        << std::endl;
}
} // namespace Adapter
