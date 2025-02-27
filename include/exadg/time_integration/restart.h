/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2021 by the ExaDG authors
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

#ifndef INCLUDE_EXADG_TIME_INTEGRATION_RESTART_H_
#define INCLUDE_EXADG_TIME_INTEGRATION_RESTART_H_

// C/C++
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <algorithm>
#include <fstream>
#include <sstream>

// deal.II
#include <deal.II/base/mpi.h>
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/manifold.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/matrix_free/fe_point_evaluation.h>
#include <deal.II/numerics/vector_tools.h>

// ExaDG
#include <exadg/grid/grid_data.h>
#include <exadg/grid/grid_utilities.h>
#include <exadg/grid/mapping_dof_vector.h>
#include <exadg/matrix_free/matrix_free_data.h>
#include <exadg/operators/mapping_flags.h>
#include <exadg/operators/mass_operator.h>
#include <exadg/operators/quadrature.h>
#include <exadg/solvers_and_preconditioners/preconditioners/jacobi_preconditioner.h>

namespace ExaDG
{
inline std::string
restart_filename(std::string const & name, MPI_Comm const & mpi_comm)
{
  std::string const rank =
    dealii::Utilities::int_to_string(dealii::Utilities::MPI::this_mpi_process(mpi_comm));

  std::string const filename = name + "." + rank + ".restart";

  return filename;
}

inline void
rename_restart_files(std::string const & filename)
{
  // backup: rename current restart file into restart.old in case something fails while writing
  std::string const from = filename;
  std::string const to   = filename + ".old";

  std::ifstream ifile(from.c_str());
  if((bool)ifile) // rename only if file already exists
  {
    int const error = rename(from.c_str(), to.c_str());

    AssertThrow(error == 0, dealii::ExcMessage("Can not rename file: " + from + " -> " + to));
  }
}

inline void
write_restart_file(std::ostringstream & oss, std::string const & filename)
{
  std::ofstream stream(filename.c_str());

  stream << oss.str() << std::endl;
}

template<typename VectorType>
inline void
print_vector_l2_norm(VectorType const & vector)
{
  MPI_Comm const & mpi_comm = vector.get_mpi_communicator();
  double const     l2_norm  = vector.l2_norm();
  if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
  {
    std::cout << "    vector global l2 norm: " << std::scientific << std::setprecision(8)
              << std::setw(20) << l2_norm << "\n";
  }
}

/**
 * Utility function to read or write the local entries of a
 * dealii::LinearAlgebra::distributed::(Block)Vector
 * from/to a boost archive per block and entry.
 * Using the `&` operator, loading from or writing to the
 * archive is determined from the type.
 */
template<typename VectorType, typename BoostArchiveType>
inline void
read_write_distributed_vector(VectorType & vector, BoostArchiveType & archive)
{
  // Print vector norm here only *before* writing.
  if(std::is_same<BoostArchiveType, boost::archive::text_oarchive>::value or
     std::is_same<BoostArchiveType, boost::archive::binary_oarchive>::value)
  {
    print_vector_l2_norm(vector);
  }

  // Depending on VectorType, we have to loop over the blocks to
  // access the local entries via vector.local_element(i).
  using Number = typename VectorType::value_type;
  if constexpr(std::is_same<std::remove_cv_t<VectorType>,
                            dealii::LinearAlgebra::distributed::Vector<Number>>::value)
  {
    for(unsigned int i = 0; i < vector.locally_owned_size(); ++i)
    {
      archive & vector.local_element(i);
    }
  }
  else if constexpr(std::is_same<std::remove_cv_t<VectorType>,
                                 dealii::LinearAlgebra::distributed::BlockVector<Number>>::value)
  {
    for(unsigned int i = 0; i < vector.n_blocks(); ++i)
    {
      for(unsigned int j = 0; j < vector.block(i).locally_owned_size(); ++j)
      {
        archive & vector.block(i).local_element(j);
      }
    }
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("Reading into this VectorType not supported."));
  }

  // Print vector norm here only *after* reading.
  if(std::is_same<BoostArchiveType, boost::archive::text_iarchive>::value or
     std::is_same<BoostArchiveType, boost::archive::binary_iarchive>::value)
  {
    print_vector_l2_norm(vector);
  }
}

/** Utility function to convert a vector of block vector pointers into a
 * vector of vectors of VectorType pointers, where all vectors from each
 * individual block are summarized in a std::vector.
 * This is useful for solution transfer and serialization.
 */
template<typename VectorType, typename BlockVectorType>
std::vector<std::vector<VectorType *>>
get_vectors_per_block(std::vector<BlockVectorType *> const & block_vectors)
{
  unsigned int const n_blocks = block_vectors.at(0)->n_blocks();
  for(unsigned int i = 0; i < block_vectors.size(); ++i)
  {
    AssertThrow(block_vectors[i]->n_blocks() == n_blocks,
                dealii::ExcMessage("Provided number of blocks per "
                                   "BlockVector must be equal."));
  }

  std::vector<std::vector<VectorType *>> vectors_per_block;
  for(unsigned int i = 0; i < n_blocks; ++i)
  {
    std::vector<VectorType *> vectors;
    for(unsigned int j = 0; j < block_vectors.size(); ++j)
    {
      vectors.push_back(&block_vectors[j]->block(i));
    }
    vectors_per_block.push_back(vectors);
  }

  return vectors_per_block;
}

/** Utility function to setup a BlockVector given a vector
 * of DoFHandlers only containing owned DoFs. This can be used
 * in combination with `get_vectors_per_block()` to obtain vectors
 * of VectorType pointers as required for `dealii::SolutionTransfer`.
 */
template<int dim, typename BlockVectorType>
std::vector<BlockVectorType>
get_block_vectors_from_dof_handlers(
  unsigned int const                                        n_vectors,
  std::vector<dealii::DoFHandler<dim, dim> const *> const & dof_handlers)
{
  unsigned int const n_blocks = dof_handlers.size();

  // Setup first BlockVector
  BlockVectorType block_vector(n_blocks);
  for(unsigned int i = 0; i < n_blocks; ++i)
  {
    block_vector.block(i).reinit(dof_handlers[i]->locally_owned_dofs(),
                                 dof_handlers[i]->get_communicator());
  }
  block_vector.collect_sizes();

  std::vector<BlockVectorType> block_vectors(n_vectors, block_vector);

  return block_vectors;
}

template<typename VectorType>
std::vector<bool>
get_ghost_state(std::vector<VectorType *> const & vectors)
{
  std::vector<bool> has_ghost_elements(vectors.size());
  for(unsigned int i = 0; i < has_ghost_elements.size(); ++i)
  {
    has_ghost_elements[i] = vectors[i]->has_ghost_elements();
  }
  return has_ghost_elements;
}

template<typename VectorType>
void
set_ghost_state(std::vector<VectorType *> const & vectors,
                std::vector<bool> const &         had_ghost_elements)
{
  AssertThrow(vectors.size() == had_ghost_elements.size(),
              dealii::ExcMessage("Vector sizes do not match."));

  for(unsigned int i = 0; i < had_ghost_elements.size(); ++i)
  {
    if(had_ghost_elements[i])
    {
      vectors[i]->update_ghost_values();
    }
    else
    {
      vectors[i]->zero_out_ghost_values();
    }
  }
}

/**
 * Utility function to serialize a `dealii::Triangulation`. This is only implemented for the
 * serial case since we only require the coarsest triangulation to be read from file when
 * deserializing into a `dealii::parallel::distributed::Triangulation`.
 */
template<int dim, typename TriangulationType>
inline void
save_coarse_triangulation(std::string const &       filename_base,
                          TriangulationType const & triangulation)
{
  if constexpr(std::is_same<std::remove_cv_t<TriangulationType>,
                            dealii::parallel::distributed::Triangulation<dim, dim>>::value or
               std::is_same<std::remove_cv_t<TriangulationType>,
                            dealii::parallel::fullydistributed::Triangulation<dim, dim>>::value)
  {
    AssertThrow(false,
                dealii::ExcMessage("Only TriangulationType::Serial, i.e., "
                                   "dealii::Triangulation<dim> supported."));
  }

  std::string const filename = filename_base + ".coarse_triangulation";
  if(dealii::Utilities::MPI::this_mpi_process(triangulation.get_communicator()) == 0)
  {
    // Serialization only creates a single file, move with one process only.
    rename_restart_files(filename + ".info");
    rename_restart_files(filename + "_triangulation.data");

    // For `dealii::Triangulation` the triangulation is the same for all processes.
    triangulation.save(filename);
  }
}

/**
 * Utility function to store a std::vector<VectorType> in a triangulation and serialize.
 * We assume that the Triangulation(s) linked to the DoFHandlers are all identical.
 * Note also that the sequence of vectors and DoFHandlers here and in
 * deserialize_triangulation_and_load_vectors() *must* be identical.
 * This function does not consider a mapping to be stored, if it is
 * not provided within the `dof_handlers` (and hence treated like all other vectors).
 */
template<int dim, typename VectorType>
inline void
store_vectors_in_triangulation_and_serialize(
  std::string const &                                       filename_base,
  std::vector<std::vector<VectorType const *>> const &      vectors_per_dof_handler,
  std::vector<dealii::DoFHandler<dim, dim> const *> const & dof_handlers)
{
  AssertThrow(vectors_per_dof_handler.size() > 0,
              dealii::ExcMessage("No vectors to store in triangulation."));
  AssertThrow(vectors_per_dof_handler.size() == dof_handlers.size(),
              dealii::ExcMessage("Number of vectors of vectors and DoFHandlers do not match."));
  auto const & triangulation = dof_handlers.at(0)->get_triangulation();

  // Check if all the Triangulation(s) associated with the DoFHandlers point to the same object.
  for(unsigned int i = 1; i < dof_handlers.size(); ++i)
  {
    AssertThrow(&dof_handlers[i]->get_triangulation() == &triangulation,
                dealii::ExcMessage("Triangulations of DoFHandlers are not identical."));
  }

  // Loop over the DoFHandlers and store the vectors in the triangulation.
  std::vector<std::shared_ptr<dealii::parallel::distributed::SolutionTransfer<dim, VectorType>>>
    solution_transfers;
  for(unsigned int i = 0; i < dof_handlers.size(); ++i)
  {
    for(unsigned int j = 0; j < vectors_per_dof_handler[i].size(); ++j)
    {
      print_vector_l2_norm(*vectors_per_dof_handler[i][j]);
    }
    solution_transfers.push_back(
      std::make_shared<dealii::parallel::distributed::SolutionTransfer<dim, VectorType>>(
        *dof_handlers[i]));
    solution_transfers[i]->prepare_for_serialization(vectors_per_dof_handler[i]);
  }

  // Serialize the triangulation keeping a maximum of two snapshots.
  std::string const filename = filename_base + ".triangulation";
  if(dealii::Utilities::MPI::this_mpi_process(dof_handlers[0]->get_communicator()) == 0)
  {
    // Serialization only creates a single file, move with one process only.
    rename_restart_files(filename);
    rename_restart_files(filename + ".info");
    rename_restart_files(filename + "_fixed.data");
    rename_restart_files(filename + "_triangulation.data");
  }

  // Collective call for serialization.
  triangulation.save(filename);
}

/**
 * Same as the function above, but the mapping is stored for tensor-product elements
 * as one of the vectors, while for any other element type, we ignore the mapping in
 * the projection when deserializing.
 */
template<int dim, typename VectorType>
inline void
store_vectors_in_triangulation_and_serialize(
  std::string const &                                       filename_base,
  std::vector<std::vector<VectorType const *>> const &      vectors_per_dof_handler,
  std::vector<dealii::DoFHandler<dim, dim> const *> const & dof_handlers,
  dealii::Mapping<dim> const &                              mapping,
  dealii::DoFHandler<dim> const *                           dof_handler_mapping,
  unsigned int const                                        mapping_degree)
{
  AssertThrow(vectors_per_dof_handler.size() > 0,
              dealii::ExcMessage("No vectors to store in triangulation."));
  AssertThrow(vectors_per_dof_handler.size() == dof_handlers.size(),
              dealii::ExcMessage("Number of vectors of vectors and DoFHandlers do not match."));

  auto const & triangulation = dof_handlers.at(0)->get_triangulation();

  // Check if all the Triangulation(s) associated with the DoFHandlers point to the same object.
  for(unsigned int i = 1; i < dof_handlers.size(); ++i)
  {
    AssertThrow(&dof_handlers[i]->get_triangulation() == &triangulation,
                dealii::ExcMessage("Triangulations of DoFHandlers are not identical."));
  }

  AssertThrow(triangulation.all_reference_cells_are_hyper_cube(),
              dealii::ExcMessage("Serialization including mapping not "
                                 "supported for non-hypercube cell types."));

  // Initialize vector to hold grid coordinates.
  bool       vector_initialized = false;
  VectorType vector_grid_coordinates;
  for(unsigned int i = 0; i < dof_handlers.size(); ++i)
  {
    if(dof_handlers[i] == dof_handler_mapping and not vector_initialized)
    {
      // Cheaper setup if we already have a vector given in the input arguments.
      vector_grid_coordinates.reinit(*vectors_per_dof_handler[i][0],
                                     true /* omit_zeroing_entries */);
      vector_initialized = true;
      break;
    }
  }

  if(not vector_initialized)
  {
    // More expensive setup extracting the `dealii::IndexSet`.
    dealii::IndexSet const & locally_owned_dofs = dof_handler_mapping->locally_owned_dofs();
    dealii::IndexSet const   locally_relevant_dofs =
      dealii::DoFTools::extract_locally_relevant_dofs(*dof_handler_mapping);
    vector_grid_coordinates.reinit(locally_owned_dofs,
                                   locally_relevant_dofs,
                                   dof_handler_mapping->get_communicator());
  }

  // Fill vector with mapping.
  MappingDoFVector<dim, typename VectorType::value_type> mapping_dof_vector(mapping_degree);
  mapping_dof_vector.fill_grid_coordinates_vector(mapping,
                                                  vector_grid_coordinates,
                                                  *dof_handler_mapping);

  // Attach vector holding mapping and corresponding `dof_handler_mapping`.
  std::vector<std::vector<VectorType const *>> vectors_per_dof_handler_extended =
    vectors_per_dof_handler;
  std::vector<VectorType const *> tmp = {&vector_grid_coordinates};
  vectors_per_dof_handler_extended.push_back(tmp);

  std::vector<dealii::DoFHandler<dim, dim> const *> dof_handlers_extended = dof_handlers;
  dof_handlers_extended.push_back(dof_handler_mapping);

  // Use utility function that ignores the mapping.
  store_vectors_in_triangulation_and_serialize(filename_base,
                                               vectors_per_dof_handler_extended,
                                               dof_handlers_extended);
}

/**
 * Utility function to deserialize the stored triangulation.
 */
template<int dim>
inline std::shared_ptr<dealii::Triangulation<dim>>
deserialize_triangulation(std::string const &     filename_base,
                          TriangulationType const triangulation_type,
                          MPI_Comm const &        mpi_communicator)
{
  std::shared_ptr<dealii::Triangulation<dim>> triangulation_old;

  // Deserialize the checkpointed triangulation,
  if(triangulation_type == TriangulationType::Serial)
  {
    triangulation_old = std::make_shared<dealii::Triangulation<dim>>();
    triangulation_old->load(filename_base + ".triangulation");
  }
  else if(triangulation_type == TriangulationType::Distributed)
  {
    // Deserialize the coarse triangulation to be stored by the user
    // during `create_grid` in the respective application.
    dealii::Triangulation<dim, dim> coarse_triangulation;
    try
    {
      coarse_triangulation.load(filename_base + ".coarse_triangulation");
    }
    catch(...)
    {
      AssertThrow(false,
                  dealii::ExcMessage(
                    "Deserializing coarse triangulation expected in\n" + filename_base +
                    ".coarse_triangulation\n"
                    "make sure to store the coarse grid during `create_grid`\n"
                    "in the respective application.h using TriangulationType::Serial."));
    }

    std::shared_ptr<dealii::parallel::distributed::Triangulation<dim>> tmp =
      std::make_shared<dealii::parallel::distributed::Triangulation<dim>>(mpi_communicator);

    tmp->copy_triangulation(coarse_triangulation);
    coarse_triangulation.clear();

    // We do not need manifolds when applying the refinements, since we recover the mapping
    // separately.
    tmp->reset_all_manifolds();
    tmp->load(filename_base + ".triangulation");

    triangulation_old = std::dynamic_pointer_cast<dealii::Triangulation<dim>>(tmp);
  }
  else if(triangulation_type == TriangulationType::FullyDistributed)
  {
    // Note that the number of MPI processes the triangulation was
    // saved with cannot change and hence autopartitioning is disabled.
    std::shared_ptr<dealii::parallel::fullydistributed::Triangulation<dim>> tmp =
      std::make_shared<dealii::parallel::fullydistributed::Triangulation<dim>>(mpi_communicator);
    tmp->load(filename_base + ".triangulation");

    triangulation_old = std::dynamic_pointer_cast<dealii::Triangulation<dim>>(tmp);
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("TriangulationType not supported."));
  }

  return triangulation_old;
}

/**
 * Utility function to load vectors via `dealii::SolutionTransfer`
 * assuming the `Triangulation` the `DoFHandler` was initialized with
 * actually stores the related data.
 */
template<int dim, typename VectorType>
inline void
load_vectors(std::vector<std::vector<VectorType *>> &                  vectors_per_dof_handler,
             std::vector<dealii::DoFHandler<dim, dim> const *> const & dof_handlers)
{
  // The DoFHandlers and vectors are already initialized and
  // ``vectors_per_dof_handler`` contain only owned DoFs.
  AssertThrow(vectors_per_dof_handler.size() > 0,
              dealii::ExcMessage("No vectors to load into from triangulation."));
  AssertThrow(vectors_per_dof_handler.size() == dof_handlers.size(),
              dealii::ExcMessage("Number of vectors of vectors and DoFHandlers do not match."));

  // Loop over the DoFHandlers and load the vectors stored in
  // the triangulation the DoFHandlers were initialized with.
  for(unsigned int i = 0; i < dof_handlers.size(); ++i)
  {
    dealii::parallel::distributed::SolutionTransfer<dim, VectorType> solution_transfer(
      *dof_handlers[i]);
    solution_transfer.deserialize(vectors_per_dof_handler[i]);

    for(unsigned int j = 0; j < vectors_per_dof_handler[i].size(); ++j)
    {
      print_vector_l2_norm(*vectors_per_dof_handler[i][j]);
    }
  }
}

/**
 * Same as the above function, but consider for a mapping added as an additional vector
 * added during `store_vectors_in_triangulation_and_serialize()`.
 */
template<int dim, typename VectorType>
inline std::shared_ptr<dealii::Mapping<dim>>
load_vectors(std::vector<std::vector<VectorType *>> &                  vectors_per_dof_handler,
             std::vector<dealii::DoFHandler<dim, dim> const *> const & dof_handlers,
             dealii::DoFHandler<dim> const *                           dof_handler_mapping,
             unsigned int const                                        mapping_degree)
{
  // We need a collective call to `SolutionTransfer::deserialize()` with all vectors in a
  // single container. Hence, create a mapping vector and add a pointer to the input argument.
  dealii::IndexSet const & locally_owned_dofs = dof_handler_mapping->locally_owned_dofs();
  dealii::IndexSet const & locally_relevant_dofs =
    dealii::DoFTools::extract_locally_relevant_dofs(*dof_handler_mapping);
  VectorType vector_grid_coordinates(locally_owned_dofs,
                                     locally_relevant_dofs,
                                     dof_handler_mapping->get_communicator());

  // Standard utility function, sequence as in `store_vectors_in_triangulation_and_serialize()`.
  std::vector<std::vector<VectorType *>> vectors_per_dof_handler_extended = vectors_per_dof_handler;
  std::vector<VectorType *>              tmp = {&vector_grid_coordinates};
  vectors_per_dof_handler_extended.push_back(tmp);
  std::vector<dealii::DoFHandler<dim, dim> const *> dof_handlers_extended = dof_handlers;
  dof_handlers_extended.push_back(dof_handler_mapping);

  load_vectors(vectors_per_dof_handler_extended, dof_handlers_extended);

  // Reconstruct the mapping given the deserialized grid coordinate vector.
  std::shared_ptr<dealii::Mapping<dim>> mapping;
  GridUtilities::create_mapping(mapping,
                                get_element_type(dof_handler_mapping->get_triangulation()),
                                mapping_degree);
  MappingDoFVector<dim, typename VectorType::value_type> mapping_dof_vector(mapping_degree);
  mapping_dof_vector.fill_grid_coordinates_vector(*mapping,
                                                  vector_grid_coordinates,
                                                  *dof_handler_mapping);

  return mapping;
}

/**
 * Utility function to collect integration points via `dealii::FEEvaluation`.
 */
template<int dim, int n_components, typename Number>
inline std::vector<dealii::Point<dim>>
collect_integration_points(
  dealii::MatrixFree<dim, Number, dealii::VectorizedArray<Number>> const & matrix_free,
  unsigned int const                                                       dof_index,
  unsigned int const                                                       quad_index)
{
  CellIntegrator<dim, n_components, Number> fe_eval(matrix_free, dof_index, quad_index);

  // Conservative estimate for the number of points.
  std::vector<dealii::Point<dim>> integration_points;
  integration_points.reserve(
    matrix_free.get_dof_handler(dof_index).get_triangulation().n_active_cells() *
    fe_eval.n_q_points);

  for(unsigned int cell_batch_idx = 0; cell_batch_idx < matrix_free.n_cell_batches();
      ++cell_batch_idx)
  {
    fe_eval.reinit(cell_batch_idx);
    for(const unsigned int q : fe_eval.quadrature_point_indices())
    {
      dealii::Point<dim, dealii::VectorizedArray<Number>> const cell_batch_points =
        fe_eval.quadrature_point(q);
      for(unsigned int i = 0; i < dealii::VectorizedArray<Number>::size(); ++i)
      {
        dealii::Point<dim> p;
        for(unsigned int d = 0; d < dim; ++d)
        {
          p[d] = cell_batch_points[d][i];
        }
        integration_points.push_back(p);
      }
    }
  }

  return integration_points;
}

/**
 * Utility function to compute the right hand side of a projection with values given in integration
 * points obtained via `collect_integration_points()`.
 */
template<int dim, int n_components, typename Number, typename VectorType>
inline VectorType
assemble_projection_rhs(
  dealii::MatrixFree<dim, Number, dealii::VectorizedArray<Number>> const & matrix_free,
  CellIntegrator<dim, n_components, Number> &                              fe_eval,
  std::vector<
    typename dealii::FEPointEvaluation<n_components, dim, dim, Number>::value_type> const &
                     values_source_in_q_points_target,
  unsigned int const dof_index)
{
  VectorType system_rhs;
  matrix_free.initialize_dof_vector(system_rhs, dof_index);

  unsigned int idx_q_point = 0;

  for(unsigned int cell_batch_idx = 0; cell_batch_idx < matrix_free.n_cell_batches();
      ++cell_batch_idx)
  {
    fe_eval.reinit(cell_batch_idx);
    for(unsigned int const q : fe_eval.quadrature_point_indices())
    {
      dealii::Tensor<1, n_components, dealii::VectorizedArray<Number>> tmp;

      for(unsigned int i = 0; i < dealii::VectorizedArray<Number>::size(); ++i)
      {
        typename dealii::FEPointEvaluation<n_components, dim, dim, Number>::value_type const
          values = values_source_in_q_points_target[idx_q_point];

        // Increment index into `values_source_in_q_points_target` which is dictated
        // by `integration_points_target`, i.e., `collect_integration_points()`.
        ++idx_q_point;

        if constexpr(n_components == 1)
        {
          tmp[0][i] = values;
        }
        else
        {
          for(unsigned int c = 0; c < n_components; ++c)
          {
            tmp[c][i] = values[c];
          }
        }
      }

      fe_eval.submit_value(tmp, q);
    }
    fe_eval.integrate(dealii::EvaluationFlags::values);
    fe_eval.distribute_local_to_global(system_rhs);
  }
  system_rhs.compress(dealii::VectorOperation::add);

  return system_rhs;
}

/**
 * Utilitiy function to project vectors from a source to a target triangulation
 * via `dealii::RemotePointEvaluation`, matrix-free mass operator evaluation
 * and a Jacobi-preconditioned CG solver.
 */
template<int dim, typename Number, int n_components, typename VectorType>
inline void
project_vectors(
  std::vector<VectorType *> const &                                        source_vectors,
  dealii::DoFHandler<dim> const &                                          source_dof_handler,
  std::shared_ptr<dealii::Mapping<dim>> const &                            source_mapping,
  std::vector<VectorType *> const &                                        target_vectors,
  dealii::DoFHandler<dim> const &                                          target_dof_handler,
  dealii::MatrixFree<dim, Number, dealii::VectorizedArray<Number>> const & target_matrix_free,
  dealii::AffineConstraints<Number> const &                                constraints,
  unsigned int const                                                       dof_index,
  unsigned int const                                                       quad_index,
  double const &                                                           rpe_tolerance_unit_cell,
  bool const rpe_enforce_unique_mapping)
{
  // Setup operator and preconditioner outside of the loop since the operator remains unchanged.
  MassOperatorData<dim> mass_operator_data;
  mass_operator_data.dof_index  = dof_index;
  mass_operator_data.quad_index = quad_index;

  MassOperator<dim, n_components, Number> mass_operator;
  mass_operator.initialize(target_matrix_free, constraints, mass_operator_data);

  JacobiPreconditioner<MassOperator<dim, n_components, Number>> jacobi_preconditioner(
    mass_operator, true /* initialize_preconditioner */);

  // Setup RemotePointEvaluation since the `source_vectors` all live on the same triangulation.
  typename dealii::Utilities::MPI::RemotePointEvaluation<dim>::AdditionalData rpe_data(
    rpe_tolerance_unit_cell,
    rpe_enforce_unique_mapping,
    0 /* rtree_level */,
    {} /* marked_vertices */);

  dealii::Utilities::MPI::RemotePointEvaluation<dim> rpe_source(rpe_data);

  // The sequence of integration points follows from the sequence of points as encountered during
  // cell batch loop.
  std::vector<dealii::Point<dim>> integration_points_target =
    collect_integration_points<dim, n_components, Number>(target_matrix_free,
                                                          dof_index,
                                                          quad_index);

  rpe_source.reinit(integration_points_target,
                    source_dof_handler.get_triangulation(),
                    *source_mapping);
  AssertThrow(rpe_source.all_points_found(),
              dealii::ExcMessage("Could not interpolate source grid vector in target grid."));

  CellIntegrator<dim, n_components, Number> fe_eval(target_matrix_free, dof_index, quad_index);

  // Loop over vectors and project.
  for(unsigned int i = 0; i < target_vectors.size(); ++i)
  {
    // Evaluate the source vector at the target integration points.
    VectorType const & source_vector = *source_vectors.at(i);
    source_vector.update_ghost_values();

    std::vector<
      typename dealii::FEPointEvaluation<n_components, dim, dim, Number>::value_type> const
      values_source_in_q_points_target = dealii::VectorTools::point_values<n_components>(
        rpe_source, source_dof_handler, source_vector, dealii::VectorTools::EvaluationFlags::avg);

    // Assemble right hand side vector for the projection.
    VectorType system_rhs = assemble_projection_rhs<dim, n_components, Number, VectorType>(
      target_matrix_free, fe_eval, values_source_in_q_points_target, dof_index);

    // CG solver for global projection.
    unsigned int constexpr max_iter = 10000;
    double const abs_tol            = 1e-16 * system_rhs.l2_norm();
    double constexpr rel_tol        = 1e-12;

    dealii::ReductionControl     reduction_control(max_iter, abs_tol, rel_tol);
    dealii::SolverCG<VectorType> solver_cg(reduction_control);

    VectorType sol;
    sol.reinit(system_rhs, false /* omit_zeroing_entries */);

    solver_cg.solve(mass_operator, sol, system_rhs, jacobi_preconditioner);

    *target_vectors[i] = sol;

    if(dealii::Utilities::MPI::this_mpi_process(target_dof_handler.get_communicator()) == 0)
    {
      std::cout << "    global projection required " << reduction_control.last_step()
                << " CG iterations.\n";
    }
  }
}

/**
 * Utility function to perform grid-to-grid projection using `dealii::RemotePointEvaluation`. We
 * assume we only have a single `dealii::FiniteElement` per `dealii::DoFHandler`. The VectorType
 * template argument is assumed not to be of `BlockVector` type. Note that this function initializes
 * `dealii::MatrixFree` and `dealii::RemotePointEvaluation` object and hence should be used with
 * caution.
 */
template<int dim, typename VectorType>
inline void
grid_to_grid_projection(
  std::vector<std::vector<VectorType *>> const &       source_vectors_per_dof_handler,
  std::vector<dealii::DoFHandler<dim> const *> const & source_dof_handlers,
  std::shared_ptr<dealii::Mapping<dim>> const &        source_mapping,
  std::vector<std::vector<VectorType *>> &             target_vectors_per_dof_handler,
  std::vector<dealii::DoFHandler<dim> const *> const & target_dof_handlers,
  std::shared_ptr<dealii::Mapping<dim> const> const &  target_mapping,
  double const &                                       rpe_tolerance_unit_cell,
  bool const                                           rpe_enforce_unique_mapping)
{
  // Check input dimensions.
  AssertThrow(source_vectors_per_dof_handler.size() == source_dof_handlers.size(),
              dealii::ExcMessage("First dimension of source vector of vectors "
                                 "has to match source DoFHandler count."));
  AssertThrow(target_vectors_per_dof_handler.size() == target_dof_handlers.size(),
              dealii::ExcMessage("First dimension of target vector of vectors "
                                 "has to match target DoFHandler count."));
  AssertThrow(source_dof_handlers.size() == target_dof_handlers.size(),
              dealii::ExcMessage("Target and source DoFHandler counts have to match"));
  AssertThrow(source_vectors_per_dof_handler.size() > 0,
              dealii::ExcMessage("Vector of source vectors empty."));
  for(unsigned int i = 0; i < source_vectors_per_dof_handler.size(); ++i)
  {
    AssertThrow(source_vectors_per_dof_handler[i].size() ==
                  target_vectors_per_dof_handler.at(i).size(),
                dealii::ExcMessage("Vectors of source and target vectors need to have same size."));
  }

  // Setup `dealii::MatrixFree` object with multiple `dealii::DoFHandler`s.
  using Number = typename VectorType::value_type;
  MatrixFreeData<dim, Number> matrix_free_data;

  MappingFlags mapping_flags;
  mapping_flags.cells =
    dealii::update_quadrature_points | dealii::update_values | dealii::update_JxW_values;
  matrix_free_data.append_mapping_flags(mapping_flags);

  dealii::AffineConstraints<Number> empty_constraints;
  empty_constraints.clear();
  empty_constraints.close();
  for(unsigned int i = 0; i < target_dof_handlers.size(); ++i)
  {
    matrix_free_data.insert_dof_handler(target_dof_handlers[i], std::to_string(i));
    matrix_free_data.insert_constraint(&empty_constraints, std::to_string(i));

    ElementType element_type = get_element_type(target_dof_handlers[i]->get_triangulation());

    std::shared_ptr<dealii::Quadrature<dim>> quadrature =
      create_quadrature<dim>(element_type, target_dof_handlers[i]->get_fe().degree + 2);

    matrix_free_data.insert_quadrature(*quadrature, std::to_string(i));
  }

  dealii::MatrixFree<dim, Number, dealii::VectorizedArray<Number>> matrix_free;
  matrix_free.reinit(*target_mapping,
                     matrix_free_data.get_dof_handler_vector(),
                     matrix_free_data.get_constraint_vector(),
                     matrix_free_data.get_quadrature_vector(),
                     matrix_free_data.data);

  // Project vectors per `dealii::DoFHandler`.
  for(unsigned int i = 0; i < target_dof_handlers.size(); ++i)
  {
    unsigned int const n_components = target_dof_handlers[i]->get_fe().n_components();
    if(n_components == 1)
    {
      project_vectors<dim, Number, 1 /* n_components */, VectorType>(
        source_vectors_per_dof_handler.at(i),
        *source_dof_handlers.at(i),
        source_mapping,
        target_vectors_per_dof_handler.at(i),
        *target_dof_handlers.at(i),
        matrix_free,
        empty_constraints,
        i /* dof_index */,
        i /* quad_index */,
        rpe_tolerance_unit_cell,
        rpe_enforce_unique_mapping);
    }
    else if(n_components == dim)
    {
      project_vectors<dim, Number, dim /* n_components */, VectorType>(
        source_vectors_per_dof_handler.at(i),
        *source_dof_handlers.at(i),
        source_mapping,
        target_vectors_per_dof_handler.at(i),
        *target_dof_handlers.at(i),
        matrix_free,
        empty_constraints,
        i /* dof_index */,
        i /* quad_index */,
        rpe_tolerance_unit_cell,
        rpe_enforce_unique_mapping);
    }
    else if(n_components == dim + 2)
    {
      project_vectors<dim, Number, dim + 2 /* n_components */, VectorType>(
        source_vectors_per_dof_handler.at(i),
        *source_dof_handlers.at(i),
        source_mapping,
        target_vectors_per_dof_handler.at(i),
        *target_dof_handlers.at(i),
        matrix_free,
        empty_constraints,
        i /* dof_index */,
        i /* quad_index */,
        rpe_tolerance_unit_cell,
        rpe_enforce_unique_mapping);
    }
    else
    {
      AssertThrow(n_components == 1 or n_components == dim,
                  dealii::ExcMessage("The requested number of components is not"
                                     "supported in `grid_to_grid_projection()`."));
    }
  }
}

} // namespace ExaDG

#endif /* INCLUDE_EXADG_TIME_INTEGRATION_RESTART_H_ */
