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

#ifndef EXADG_TIME_INTEGRATION_RESTART_H_
#define EXADG_TIME_INTEGRATION_RESTART_H_

// C/C++
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <fstream>

// deal.II
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/numerics/solution_transfer.h>

// ExaDG
#include <exadg/grid/grid_utilities.h>
#include <exadg/grid/mapping_dof_vector.h>
#include <exadg/time_integration/restart_data.h>
#include <exadg/utilities/create_directories.h>

namespace ExaDG
{
inline std::string
generate_restart_filename(std::string const & name)
{
  // Filename does not incorporate rank information, files in-/output with single rank only.
  std::string const filename = name + ".restart";

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

/** Utility function to convert a vector of block vector pointers into a
 * vector of vectors of `VectorType` pointers, where all vectors from each
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

/** Utility function to setup a `BlockVector` given a vector
 * of `DoFHandlers` only containing owned DoFs. This can be used
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

  // Setup first `BlockVector`
  BlockVectorType block_vector(n_blocks);
  for(unsigned int i = 0; i < n_blocks; ++i)
  {
    block_vector.block(i).reinit(dof_handlers[i]->locally_owned_dofs(),
                                 dof_handlers[i]->get_mpi_communicator());
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
 * Utility function to write the parameters a discretization is serialized with. This is to recover
 * the parameters when deserializing.
 */
inline void
write_deserialization_parameters(MPI_Comm const &                  mpi_comm,
                                 std::string const &               directory,
                                 std::string const &               filename_base,
                                 DeserializationParameters const & parameters)
{
  // Create folder if not existent.
  create_directories(directory, mpi_comm);

  // Filename for deserialization parameters has to match `read_deserialization_parameters()`.
  std::string const filename = directory + filename_base + ".deserialization_parameters";

  // Write the parameters with a single processor.
  if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
  {
    // Serialization only creates a single file, move with one process only.
    rename_restart_files(filename);

    // Write deserialization parameters.
    std::ofstream stream(filename);
    AssertThrow(stream, dealii::ExcMessage("Could not write deserialization parameters to file."));

    // Text archive type for debugging purposes.
    // boost::archive::text_oarchive output_archive(stream);
    boost::archive::binary_oarchive output_archive(stream);

    // Sequence has to match `read_deserialization_parameters()`.
    output_archive & parameters.degree;
    output_archive & parameters.degree_u;
    output_archive & parameters.degree_p;
    output_archive & parameters.mapping_degree;
    output_archive & parameters.consider_mapping_write;
    output_archive & parameters.triangulation_type;
    output_archive & parameters.spatial_discretization;
  }
}

/**
 * Utility function to read the parameters a discretization is serialized with. This is to recover
 * the parameters when deserializing.
 */
inline DeserializationParameters
read_deserialization_parameters(MPI_Comm const &    mpi_comm,
                                std::string const & directory,
                                std::string const & filename_base)
{
  DeserializationParameters parameters;

  // Filename for deserialization parameters has to match `write_deserialization_parameters()`.
  std::string const filename = directory + filename_base + ".deserialization_parameters";

  // Read the parameters with a single processor.
  if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
  {
    // Read deserialization parameters.
    std::ifstream stream(filename);
    AssertThrow(stream, dealii::ExcMessage("Could not read deserialization parameters from file."));

    // Text archive type for debugging purposes.
    // boost::archive::text_iarchive input_archive(stream);
    boost::archive::binary_iarchive input_archive(stream);

    // Sequence has to match `write_deserialization_parameters()`.
    input_archive & parameters.degree;
    input_archive & parameters.degree_u;
    input_archive & parameters.degree_p;
    input_archive & parameters.mapping_degree;
    input_archive & parameters.consider_mapping_write;
    input_archive & parameters.triangulation_type;
    input_archive & parameters.spatial_discretization;
  }

  // Broadcast parameters to all processes.
  parameters = dealii::Utilities::MPI::broadcast(mpi_comm, parameters, 0);

  return parameters;
}

/*
 * Utility function to check if mapping is correctly treated in de-/serialization. This is to
 * provide easier to interpret error messages on ExaDG level in case the number of DoF vectors
 * mismatches. This might be due to the mapping being described as a displacement vector, which
 * might also be de-/serialized, while we have to deserialize exactly what we serialized.
 */
inline void
check_mapping_deserialization(bool const consider_mapping_read_source,
                              bool const consider_mapping_write_as_serialized)
{
  if(consider_mapping_read_source)
  {
    if(consider_mapping_write_as_serialized == false)
    {
      AssertThrow(false,
                  dealii::ExcMessage(
                    "Mapping was not considered when writing the restart data, but shall "
                    "be considered when reading the restart data. This is not supported."));
    }
  }
  else
  {
    if(consider_mapping_write_as_serialized == true)
    {
      AssertThrow(false,
                  dealii::ExcMessage(
                    "Mapping was considered when writing the restart data, but shall "
                    "not be considered when reading the restart data. This is not supported."));
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
save_coarse_triangulation(std::string const &       directory,
                          std::string const &       filename_base,
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

  MPI_Comm const & mpi_comm = triangulation.get_mpi_communicator();

  // Create folder if not existent.
  create_directories(directory, mpi_comm);

  std::string const filename = directory + filename_base + ".coarse_triangulation";
  if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
  {
    // Serialization only creates a single file, move with one process only.
    rename_restart_files(filename + ".info");
    rename_restart_files(filename + "_triangulation.data");

    // For `dealii::Triangulation` the triangulation is the same for all processes.
    triangulation.save(filename);
  }
}

/**
 * Utility function to store a `std::vector<VectorType>` in a triangulation and serialize.
 * We assume that the `Triangulation(s)` linked to the `DoFHandlers` are all identical.
 * Note also that the sequence of vectors and `DoFHandlers` here and in
 * `deserialize_triangulation_and_load_vectors()` *must* be identical.
 * This function does not consider a mapping to be stored, which has to be
 * provided within the `dof_handlers`, hence treated like all other vectors in serialization), and
 * re-applied after deserialization.
 */
template<int dim, typename VectorType>
inline void
store_vectors_in_triangulation_and_serialize(
  std::string const &                                       directory,
  std::string const &                                       filename_base,
  std::vector<dealii::DoFHandler<dim, dim> const *> const & dof_handlers,
  std::vector<std::vector<VectorType const *>> const &      vectors_per_dof_handler)
{
  AssertThrow(vectors_per_dof_handler.size() > 0,
              dealii::ExcMessage("No vectors to store in triangulation."));
  AssertThrow(vectors_per_dof_handler.size() == dof_handlers.size(),
              dealii::ExcMessage("Number of vectors of vectors and DoFHandlers do not match."));
  auto const & triangulation = dof_handlers.at(0)->get_triangulation();

  // Check if all the `Triangulation(s)` associated with the DoFHandlers point to the same object.
  for(unsigned int i = 1; i < dof_handlers.size(); ++i)
  {
    AssertThrow(&dof_handlers[i]->get_triangulation() == &triangulation,
                dealii::ExcMessage("Triangulations of DoFHandlers are not identical."));
  }

  // Loop over the `DoFHandlers` and store the vectors in the triangulation.
  std::vector<std::shared_ptr<dealii::SolutionTransfer<dim, VectorType>>> solution_transfers;
  std::vector<std::vector<bool>> has_ghost_elements_per_dof_handler;
  for(unsigned int i = 0; i < dof_handlers.size(); ++i)
  {
    // Store ghost state.
    std::vector<bool> has_ghost_elements = get_ghost_state(vectors_per_dof_handler[i]);
    has_ghost_elements_per_dof_handler.push_back(has_ghost_elements);
    for(unsigned int j = 0; j < vectors_per_dof_handler[i].size(); ++j)
    {
      vectors_per_dof_handler[i][j]->update_ghost_values();
      print_vector_l2_norm(*vectors_per_dof_handler[i][j]);
    }

    solution_transfers.push_back(
      std::make_shared<dealii::SolutionTransfer<dim, VectorType>>(*dof_handlers[i]));
    solution_transfers[i]->prepare_for_serialization(vectors_per_dof_handler[i]);
  }

  // Serialize the triangulation keeping a maximum of two snapshots.
  std::string const filename = directory + filename_base + ".triangulation";
  MPI_Comm const &  mpi_comm = dof_handlers[0]->get_mpi_communicator();
  if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
  {
    // Serialization only creates a single file, move with one process only.
    rename_restart_files(filename);
    rename_restart_files(filename + ".info");
    rename_restart_files(filename + "_fixed.data");
    rename_restart_files(filename + "_triangulation.data");
  }

  // Collective call for serialization, general case requires ghosted vectors.
  triangulation.save(filename);

  // Recover ghost state.
  for(unsigned int i = 0; i < dof_handlers.size(); ++i)
  {
    set_ghost_state(vectors_per_dof_handler[i], has_ghost_elements_per_dof_handler[i]);
  }
}

/**
 * Same as the function above, but the mapping is stored for tensor-product elements
 * as one of the vectors, while for any other element type, we ignore the mapping in
 * the projection when deserializing.
 */
template<int dim, typename VectorType>
inline void
store_vectors_in_triangulation_and_serialize(
  std::string const &                                       directory,
  std::string const &                                       filename_base,
  std::vector<dealii::DoFHandler<dim, dim> const *> const & dof_handlers,
  std::vector<std::vector<VectorType const *>> const &      vectors_per_dof_handler,
  dealii::Mapping<dim> const &                              mapping,
  dealii::DoFHandler<dim> const *                           dof_handler_mapping,
  unsigned int const                                        mapping_degree)
{
  AssertThrow(vectors_per_dof_handler.size() > 0,
              dealii::ExcMessage("No vectors to store in triangulation."));
  AssertThrow(vectors_per_dof_handler.size() == dof_handlers.size(),
              dealii::ExcMessage("Number of vectors of vectors and DoFHandlers do not match."));

  auto const & triangulation = dof_handlers.at(0)->get_triangulation();

  // Check if all the `Triangulation(s)` associated with the `DoFHandlers` point to the same object.
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
                                   dof_handler_mapping->get_mpi_communicator());
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
  store_vectors_in_triangulation_and_serialize(directory,
                                               filename_base,
                                               dof_handlers_extended,
                                               vectors_per_dof_handler_extended);
}

/**
 * Utility function to deserialize the stored triangulation.
 */
template<int dim>
inline std::shared_ptr<dealii::Triangulation<dim>>
deserialize_triangulation(std::string const &     directory,
                          std::string const &     filename_base,
                          TriangulationType const triangulation_type,
                          MPI_Comm const &        mpi_communicator)
{
  std::shared_ptr<dealii::Triangulation<dim>> triangulation_old;

  std::string const filename = directory + filename_base;

  // Deserialize the checkpointed triangulation,
  if(triangulation_type == TriangulationType::Serial)
  {
    triangulation_old = std::make_shared<dealii::Triangulation<dim>>();
    triangulation_old->load(filename + ".triangulation");
  }
  else if(triangulation_type == TriangulationType::Distributed)
  {
    // Deserialize the coarse triangulation to be stored by the user
    // during `create_grid` in the respective application.
    dealii::Triangulation<dim, dim> coarse_triangulation;
    try
    {
      coarse_triangulation.load(filename + ".coarse_triangulation");
    }
    catch(...)
    {
      AssertThrow(false,
                  dealii::ExcMessage(
                    "Deserializing coarse triangulation expected in\n" + filename +
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
    tmp->set_all_manifold_ids(dealii::numbers::flat_manifold_id);
    tmp->load(filename + ".triangulation");

    triangulation_old = std::dynamic_pointer_cast<dealii::Triangulation<dim>>(tmp);
  }
  else if(triangulation_type == TriangulationType::FullyDistributed)
  {
    // Note that the number of MPI processes the triangulation was
    // saved with cannot change and hence autopartitioning is disabled.
    std::shared_ptr<dealii::parallel::fullydistributed::Triangulation<dim>> tmp =
      std::make_shared<dealii::parallel::fullydistributed::Triangulation<dim>>(mpi_communicator);
    tmp->load(filename + ".triangulation");

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
  // The `dof_handlers` and `vectors_per_dof_handler` are already initialized and
  // `vectors_per_dof_handler` contain only owned DoFs.
  AssertThrow(vectors_per_dof_handler.size() > 0,
              dealii::ExcMessage("No vectors to load into from triangulation."));
  AssertThrow(vectors_per_dof_handler.size() == dof_handlers.size(),
              dealii::ExcMessage("Number of vectors of vectors and DoFHandlers do not match."));

  // Loop over the DoFHandlers and load the vectors stored in
  // the triangulation the DoFHandlers were initialized with.
  for(unsigned int i = 0; i < dof_handlers.size(); ++i)
  {
    dealii::SolutionTransfer<dim, VectorType> solution_transfer(*dof_handlers[i]);

    // Reinit vectors that do not already have ghost entries.
    bool all_ghosted = false;
    for(unsigned int j = 0; j < vectors_per_dof_handler[i].size(); ++j)
    {
      if(not vectors_per_dof_handler[i][j]->has_ghost_elements())
      {
        all_ghosted = false;
        break;
      }
    }
    if(not all_ghosted)
    {
      dealii::IndexSet const & locally_owned_dofs = dof_handlers[i]->locally_owned_dofs();
      dealii::IndexSet const   locally_relevant_dofs =
        dealii::DoFTools::extract_locally_relevant_dofs(*dof_handlers[i]);
      for(unsigned int j = 0; j < vectors_per_dof_handler[i].size(); ++j)
      {
        if(not vectors_per_dof_handler[i][j]->has_ghost_elements())
        {
          vectors_per_dof_handler[i][j]->reinit(locally_owned_dofs,
                                                locally_relevant_dofs,
                                                dof_handlers[i]->get_mpi_communicator());
        }
      }
    }

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
inline std::shared_ptr<MappingDoFVector<dim, typename VectorType::value_type>>
load_vectors(std::vector<std::vector<VectorType *>> &                  vectors_per_dof_handler,
             std::vector<dealii::DoFHandler<dim, dim> const *> const & dof_handlers,
             dealii::DoFHandler<dim> const *                           dof_handler_mapping,
             unsigned int const                                        mapping_degree)
{
  // We need a collective call to `SolutionTransfer::deserialize()` with all vectors in a
  // single container. Hence, create a mapping vector and add a pointer to the input argument.
  dealii::IndexSet const & locally_owned_dofs = dof_handler_mapping->locally_owned_dofs();
  VectorType               vector_grid_coordinates(locally_owned_dofs,
                                     dof_handler_mapping->get_mpi_communicator());

  // Standard utility function, sequence as in `store_vectors_in_triangulation_and_serialize()`.
  std::vector<std::vector<VectorType *>> vectors_per_dof_handler_extended = vectors_per_dof_handler;
  std::vector<VectorType *>              tmp = {&vector_grid_coordinates};
  vectors_per_dof_handler_extended.push_back(tmp);
  std::vector<dealii::DoFHandler<dim, dim> const *> dof_handlers_extended = dof_handlers;
  dof_handlers_extended.push_back(dof_handler_mapping);

  load_vectors(vectors_per_dof_handler_extended, dof_handlers_extended);

  // Reconstruct the mapping given the deserialized grid coordinate vector.
  std::shared_ptr<MappingDoFVector<dim, typename VectorType::value_type>> mapping_dof_vector =
    std::make_shared<MappingDoFVector<dim, typename VectorType::value_type>>(mapping_degree);
  mapping_dof_vector->initialize_mapping_from_dof_vector(nullptr /* mapping */,
                                                         vector_grid_coordinates,
                                                         *dof_handler_mapping);
  return mapping_dof_vector;
}

} // namespace ExaDG

#endif /* EXADG_TIME_INTEGRATION_RESTART_H_ */
