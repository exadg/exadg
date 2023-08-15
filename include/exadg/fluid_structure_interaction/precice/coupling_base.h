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

#ifndef INCLUDE_EXADG_FLUID_STRUCTURE_INTERACTION_PRECICE_COUPLING_BASE_H_
#define INCLUDE_EXADG_FLUID_STRUCTURE_INTERACTION_PRECICE_COUPLING_BASE_H_

// deal.II
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/matrix_free/matrix_free.h>

// ExaDG
#include <exadg/matrix_free/integrators.h>

// preCICE
#ifdef EXADG_WITH_PRECICE
#  include <precice/SolverInterface.hpp>
#endif

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
class CouplingBase
{
public:
  CouplingBase(dealii::MatrixFree<dim, double, VectorizedArrayType> const & data,
#ifdef EXADG_WITH_PRECICE
               std::shared_ptr<precice::SolverInterface> precice,
#endif
               std::string const                mesh_name,
               dealii::types::boundary_id const surface_id);

  virtual ~CouplingBase() = default;

  /// Alias for the face integrator
  using FEFaceIntegrator = FaceIntegrator<dim, data_dim, double, VectorizedArrayType>;
  using value_type       = typename FEFaceIntegrator::value_type;
  /**
   * @brief define_coupling_mesh Define the coupling mesh associated to the
   *        data points
   */
  virtual void
  define_coupling_mesh() = 0;

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
  write_data(dealii::LinearAlgebra::distributed::Vector<double> const & data_vector,
             std::string const &                                        data_name) = 0;


  virtual void
  read_block_data(std::string const & data_name) const;

  /**
   * @brief Queries data IDs from preCICE for the given read data name
   * @param read_data_name
   */
  void
  add_read_data(std::string const & read_data_name);

  /**
   * @brief Queries data IDs from preCICE for the given write data name
   * @param write_data_name
   */
  void
  add_write_data(std::string const & write_data_name);

  /**
   * @brief Set the WriteDataType in this class which determines the location of
   * the write data (e.g. DoFs)
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
  print_info(bool const reader, unsigned int const local_size) const;

  /// The dealii::MatrixFree object (preCICE can only handle double precision)
  dealii::MatrixFree<dim, double, VectorizedArrayType> const & matrix_free;

  /// public precice solverinterface
#ifdef EXADG_WITH_PRECICE
  std::shared_ptr<precice::SolverInterface> precice;
#endif

  /// Configuration parameters
  std::string const mesh_name;
  int               mesh_id;
  // Map between data ID (preCICE) and the data name
  std::map<std::string, int> read_data_map;
  std::map<std::string, int> write_data_map;

  dealii::types::boundary_id const dealii_boundary_surface_id;

  WriteDataType write_data_type;

  virtual std::string
  get_surface_type() const = 0;
};



template<int dim, int data_dim, typename VectorizedArrayType>
CouplingBase<dim, data_dim, VectorizedArrayType>::CouplingBase(
  dealii::MatrixFree<dim, double, VectorizedArrayType> const & matrix_free_,
#ifdef EXADG_WITH_PRECICE
  std::shared_ptr<precice::SolverInterface> precice,
#endif
  std::string const                mesh_name,
  dealii::types::boundary_id const surface_id)
  : matrix_free(matrix_free_),
#ifdef EXADG_WITH_PRECICE
    precice(precice),
#endif
    mesh_name(mesh_name),
    dealii_boundary_surface_id(surface_id),
    write_data_type(WriteDataType::undefined)
{
#ifdef EXADG_WITH_PRECICE
  Assert(precice.get() != nullptr, dealii::ExcNotInitialized());

  // Ask preCICE already in the constructor for the IDs
  mesh_id = precice->getMeshID(mesh_name);
#else
  AssertThrow(false,
              dealii::ExcMessage("EXADG_WITH_PRECICE has to be activated to use this code."));
  mesh_id                 = 0;
#endif
}


template<int dim, int data_dim, typename VectorizedArrayType>
void
CouplingBase<dim, data_dim, VectorizedArrayType>::add_read_data(std::string const & read_data_name)
{
  Assert(mesh_id != -1, dealii::ExcNotInitialized());
#ifdef EXADG_WITH_PRECICE
  int const read_data_id = precice->getDataID(read_data_name, mesh_id);
#else
  int const read_data_id  = 0;
#endif
  read_data_map.insert({read_data_name, read_data_id});
}



template<int dim, int data_dim, typename VectorizedArrayType>
void
CouplingBase<dim, data_dim, VectorizedArrayType>::add_write_data(
  std::string const & write_data_name)
{
  Assert(mesh_id != -1, dealii::ExcNotInitialized());
#ifdef EXADG_WITH_PRECICE
  int const write_data_id = precice->getDataID(write_data_name, mesh_id);
#else
  int const write_data_id = 0;
#endif
  write_data_map.insert({write_data_name, write_data_id});
}


template<int dim, int data_dim, typename VectorizedArrayType>
void
CouplingBase<dim, data_dim, VectorizedArrayType>::set_write_data_type(
  WriteDataType write_data_specification)
{
  write_data_type = write_data_specification;
}



template<int dim, int data_dim, typename VectorizedArrayType>
void
CouplingBase<dim, data_dim, VectorizedArrayType>::process_coupling_mesh()
{
  return;
}



template<int dim, int data_dim, typename VectorizedArrayType>
void
CouplingBase<dim, data_dim, VectorizedArrayType>::read_block_data(std::string const &) const
{
  AssertThrow(false, dealii::ExcNotImplemented());
}


template<int dim, int data_dim, typename VectorizedArrayType>
void
CouplingBase<dim, data_dim, VectorizedArrayType>::print_info(bool const         reader,
                                                             unsigned int const local_size) const
{
  Assert(matrix_free.get() != 0, dealii::ExcNotInitialized());
  dealii::ConditionalOStream pcout(std::cout,
                                   dealii::Utilities::MPI::this_mpi_process(
                                     matrix_free->get_dof_handler().get_communicator()) == 0);
  auto const                 map = (reader ? read_data_map : write_data_map);

  auto        names      = map.begin();
  std::string data_names = names->first;
  ++names;
  for(; names != map.end(); ++names)
    data_names += std::string(", ") + names->first;

  pcout << "--     Data " << (reader ? "reading" : "writing") << ":\n"
        << "--     . data name(s): " << data_names << "\n"
        << "--     . associated mesh: " << mesh_name << "\n"
        << "--     . Number of coupling nodes: "
        << dealii::Utilities::MPI::sum(local_size,
                                       matrix_free->get_dof_handler().get_communicator())
        << "\n"
        << "--     . Node location: " << get_surface_type() << "\n"
        << std::endl;
}

} // namespace preCICE
} // namespace ExaDG

#endif
