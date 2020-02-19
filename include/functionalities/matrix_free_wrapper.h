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

#include "categorization.h"

using namespace dealii;

template<int dim, typename Number>
struct MatrixFreeWrapper
{
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
    pde_operator->append_data_structures(*this);
  }

  // the actual MatrixFree object
  std::shared_ptr<MatrixFree<dim, Number>> matrix_free;

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
