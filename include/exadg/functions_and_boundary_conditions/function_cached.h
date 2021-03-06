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
using namespace dealii;

/*
 * Note:
 * The default argument "double" could be removed but this implies that all BoundaryDescriptors
 * that use FunctionCached require another template parameter "Number", which requires
 * changes of major parts of the code.
 */
template<int rank, int dim, typename Number = double>
class FunctionCached
{
private:
  typedef std::tuple<unsigned int /*face*/, unsigned int /*q*/, unsigned int /*v*/> Id;
  typedef std::map<Id, types::global_dof_index>                                     MapVectorIndex;
  typedef std::vector<std::vector<Tensor<rank, dim, Number>>> ArrayVectorTensor;

public:
  FunctionCached() : global_map_vector_index(nullptr), map_solution(nullptr)
  {
  }

  Tensor<rank, dim, Number>
  tensor_value(unsigned int const face,
               unsigned int const q,
               unsigned int const v,
               unsigned int const quad_index) const
  {
    Assert(global_map_vector_index != nullptr,
           ExcMessage("Pointer global_map_vector_index is not initialized."));
    Assert(global_map_vector_index->find(quad_index) != global_map_vector_index->end(),
           ExcMessage("Specified quad_index does not exist in global_map_vector_index."));

    Assert(map_solution != nullptr, ExcMessage("Pointer map_solution is not initialized."));
    Assert(map_solution->find(quad_index) != map_solution->end(),
           ExcMessage("Specified quad_index does not exist in map_solution."));

    MapVectorIndex const &    map_vector_index = global_map_vector_index->find(quad_index)->second;
    ArrayVectorTensor const & array_solution   = map_solution->find(quad_index)->second;

    Id                      id    = std::make_tuple(face, q, v);
    types::global_dof_index index = map_vector_index.find(id)->second;

    Assert(index < array_solution.size(), ExcMessage("Index exceeds dimensions of vector."));

    std::vector<Tensor<rank, dim, double>> const & vector_solution = array_solution[index];

    // average
    unsigned int              counter = 0;
    Tensor<rank, dim, double> solution;
    for(unsigned int i = 0; i < vector_solution.size(); ++i)
    {
      solution += vector_solution[i];
      ++counter;
    }

    Assert(counter > 0, ExcMessage("Vector must contain at least one value."));

    solution *= 1.0 / (double)counter;

    return solution;
  }

  void
  set_data_pointer(std::map<unsigned int, MapVectorIndex> const &    global_map_vector_index_in,
                   std::map<unsigned int, ArrayVectorTensor> const & map_solution_in)
  {
    global_map_vector_index = &global_map_vector_index_in;
    map_solution            = &map_solution_in;
  }

private:
  std::map<unsigned int, MapVectorIndex> const *    global_map_vector_index;
  std::map<unsigned int, ArrayVectorTensor> const * map_solution;
};

} // namespace ExaDG

#endif /* INCLUDE_FUNCTIONALITIES_FUNCTION_INTERPOLATION_H_ */
