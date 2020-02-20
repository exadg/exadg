/*
 * matrix_free_wrapper.h
 *
 *  Created on: Feb 14, 2020
 *      Author: fehn
 */

#ifndef INCLUDE_FUNCTIONALITIES_MATRIX_FREE_WRAPPER_H_
#define INCLUDE_FUNCTIONALITIES_MATRIX_FREE_WRAPPER_H_

#include <deal.II/distributed/tria.h>
#include <deal.II/matrix_free/matrix_free.h>

#include "../operators/mapping_flags.h"
#include "categorization.h"

using namespace dealii;

template<int dim, typename Number>
struct MatrixFreeWrapper
{
public:
  MatrixFreeWrapper(Mapping<dim> const & mapping_in) : mapping(mapping_in)
  {
    matrix_free.reset(new MatrixFree<dim, Number>());
  }

  /*
   * Returns pointer to actual matrix-free object.
   */
  std::shared_ptr<MatrixFree<dim, Number>>
  get_matrix_free()
  {
    return matrix_free;
  }

  /*
   * This function performs a "complete" reinit() of MatrixFree<dim, Number>.
   */
  void
  reinit(bool const                                        use_cell_based_face_loops,
         std::shared_ptr<parallel::TriangulationBase<dim>> triangulation)
  {
    // cell-based face loops
    if(use_cell_based_face_loops)
    {
      auto tria =
        std::dynamic_pointer_cast<parallel::distributed::Triangulation<dim> const>(triangulation);
      Categorization::do_cell_based_loops(*tria, data);
    }

    data.tasks_parallel_scheme = MatrixFree<dim, Number>::AdditionalData::partition_partition;

    matrix_free->reinit(mapping, dof_handler_vec, constraint_vec, quadrature_vec, data);
  }

  /*
   * This function only updates geometry terms and is called in combination with
   * moving mesh (ALE) methods.
   *
   * TODO: MatrixFree<dim, Number> should provide a separate function
   *
   *  matrix_free->update_geometry()
   *
   * for this purpose.
   */
  void
  update_geometry()
  {
    // use a separate additional_data object since we only want to update what is really necessary,
    // so that the update is computationally efficient
    typename MatrixFree<dim, Number>::AdditionalData data_update_geometry;

    data_update_geometry = this->data;
    // connectivity of elements stays the same
    data_update_geometry.initialize_indices = false;
    data_update_geometry.initialize_mapping = true;

    // TODO: problems occur if the mesh is not deformed (displacement = 0)
    matrix_free->reinit(
      mapping, dof_handler_vec, constraint_vec, quadrature_vec, data_update_geometry);
  }

  template<typename Operator>
  void
  append_data_structures(Operator const & pde_operator)
  {
    pde_operator.append_data_structures(*this);
  }

  void
  append_mapping_flags(MappingFlags const & flags_other)
  {
    MappingFlags flags;

    flags.cells          = this->data.mapping_update_flags;
    flags.inner_faces    = this->data.mapping_update_flags_inner_faces;
    flags.boundary_faces = this->data.mapping_update_flags_boundary_faces;

    // append
    flags = flags || flags_other;

    this->data.mapping_update_flags                = flags.cells;
    this->data.mapping_update_flags_inner_faces    = flags.inner_faces;
    this->data.mapping_update_flags_boundary_faces = flags.boundary_faces;
  }

  void
  insert_dof_handler(DoFHandler<dim> const * dof_handler, std::string const & name)
  {
    insert_element(dof_handler_vec, dof_index_map, dof_handler, name);
  }

  void
  insert_constraint(AffineConstraints<double> const * constraint, std::string const & name)
  {
    insert_element(constraint_vec, constraint_index_map, constraint, name);
  }

  void insert_quadrature(Quadrature<1> const & quadrature, std::string const & name)
  {
    insert_element(quadrature_vec, quad_index_map, quadrature, name);
  }

  unsigned int
  get_dof_index(std::string const & name)
  {
    return get_index(dof_index_map, name);
  }

  unsigned int
  get_constraint_index(std::string const & name)
  {
    return get_index(constraint_index_map, name);
  }

  unsigned int
  get_quad_index(std::string const & name)
  {
    return get_index(quad_index_map, name);
  }

private:
  template<typename T>
  void
  insert_element(std::vector<T> &                      vector,
                 std::map<std::string, unsigned int> & map,
                 T const &                             element,
                 std::string const &                   name)
  {
    unsigned int index = vector.size();

    auto it = map.find(name);

    // make sure that this element does not already exist
    if(it == map.end())
    {
      map.insert(std::pair<std::string, unsigned int>(name, index));
    }
    else
    {
      AssertThrow(it == map.end(), ExcMessage("Element already exists. Aborting."));
    }

    vector.resize(index + 1);
    vector.at(index) = element;
  }

  unsigned int
  get_index(std::map<std::string, unsigned int> const & map, std::string const & name)
  {
    auto it = map.find(name);

    unsigned int index = numbers::invalid_unsigned_int;

    if(it != map.end())
    {
      index = it->second;
    }
    else
    {
      AssertThrow(it != map.end(), ExcMessage("Could not find element. Aborting."));
    }

    return index;
  }

  // the actual MatrixFree object
  std::shared_ptr<MatrixFree<dim, Number>> matrix_free;

  // maps between names and indices for DoFHandler, Constraint, Quadrature
  std::map<std::string, unsigned int> dof_index_map;
  std::map<std::string, unsigned int> constraint_index_map;
  std::map<std::string, unsigned int> quad_index_map;

  // collection of data structures required for initialization and update of matrix_free

  // mesh containing mapping information
  Mapping<dim> const & mapping;

  // additional data
  typename MatrixFree<dim, Number>::AdditionalData data;

  // DoFHandler, Constraint, Quadrature vectors
  std::vector<DoFHandler<dim> const *>           dof_handler_vec;
  std::vector<AffineConstraints<double> const *> constraint_vec;
  std::vector<Quadrature<1>>                     quadrature_vec;
};


#endif /* INCLUDE_FUNCTIONALITIES_MATRIX_FREE_WRAPPER_H_ */
