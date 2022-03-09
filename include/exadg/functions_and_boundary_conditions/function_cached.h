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

#ifndef INCLUDE_FUNCTIONALITIES_FUNCTION_INTERPOLATION_H_
#define INCLUDE_FUNCTIONALITIES_FUNCTION_INTERPOLATION_H_

namespace ExaDG
{
/*
 * Note:
 * The default argument "double" could be removed but this implies that all BoundaryDescriptors
 * that use FunctionCached require another template parameter "Number", which requires
 * changes of major parts of the code.
 */
template<int rank, int dim, typename Number = double>
class FunctionCached
{
public:
  typedef dealii::Tensor<rank, dim, Number> value_type;

private:
  using Id = std::tuple<unsigned int /*face*/, unsigned int /*q*/, unsigned int /*v*/>;

  using MapVectorIndex      = std::map<Id, dealii::types::global_dof_index>;
  using ArraySolutionValues = std::vector<value_type>;

public:
  FunctionCached() : map_map_vector_index(nullptr), map_array_solution(nullptr)
  {
  }

  value_type
  tensor_value(unsigned int const face,
               unsigned int const q,
               unsigned int const v,
               unsigned int const quad_index) const
  {
    Assert(map_map_vector_index != nullptr,
           dealii::ExcMessage("Pointer global_map_vector_index is not initialized."));
    Assert(map_map_vector_index->find(quad_index) != map_map_vector_index->end(),
           dealii::ExcMessage("Specified quad_index does not exist in global_map_vector_index."));

    Assert(map_array_solution != nullptr,
           dealii::ExcMessage("Pointer map_array_solution is not initialized."));
    Assert(map_array_solution->find(quad_index) != map_array_solution->end(),
           dealii::ExcMessage("Specified quad_index does not exist in map_array_solution."));

    MapVectorIndex const &      map_vector_index = map_map_vector_index->find(quad_index)->second;
    ArraySolutionValues const & array_solution   = map_array_solution->find(quad_index)->second;

    Id                              id    = std::make_tuple(face, q, v);
    dealii::types::global_dof_index index = map_vector_index.find(id)->second;

    Assert(index < array_solution.size(),
           dealii::ExcMessage("Index exceeds dimensions of vector."));

    return array_solution[index];
  }

  void
  set_data_pointer(std::map<unsigned int, MapVectorIndex> const &      map_map_vector_index_,
                   std::map<unsigned int, ArraySolutionValues> const & map_array_solution_)
  {
    map_map_vector_index = &map_map_vector_index_;
    map_array_solution   = &map_array_solution_;
  }

private:
  std::map<unsigned int, MapVectorIndex> const *      map_map_vector_index;
  std::map<unsigned int, ArraySolutionValues> const * map_array_solution;
};

} // namespace ExaDG

#endif /* INCLUDE_FUNCTIONALITIES_FUNCTION_INTERPOLATION_H_ */
