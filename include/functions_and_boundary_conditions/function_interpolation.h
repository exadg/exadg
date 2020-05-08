/*
 * function_interpolation.h
 *
 *  Created on: Mar 5, 2020
 *      Author: fehn
 */

#ifndef INCLUDE_FUNCTIONALITIES_FUNCTION_INTERPOLATION_H_
#define INCLUDE_FUNCTIONALITIES_FUNCTION_INTERPOLATION_H_

using namespace dealii;

/*
 * Note:
 * The default argument "double" could be removed but this implies that all BoundaryDescriptor's
 * that use FunctionInterpolation require another template parameter "Number", which requires
 * changes of major parts of the code.
 */
template<int rank, int dim, typename Number = double>
class FunctionInterpolation
{
private:
  typedef std::tuple<unsigned int /*face*/, unsigned int /*q*/, unsigned int /*v*/> Id;
  typedef std::map<Id, std::vector<Tensor<rank, dim, Number>>>                      ArraySolution;

public:
  FunctionInterpolation() : map_solution(nullptr)
  {
  }

  Tensor<rank, dim, Number>
  tensor_value(unsigned int const face,
               unsigned int const q,
               unsigned int const v,
               unsigned int const quad_index) const
  {
    Assert(map_solution != nullptr, ExcMessage("Pointer map_solution is not initialized."));
    Assert(map_solution->find(quad_index) != map_solution->end(),
           ExcMessage("Specified quad_index does not exist in map_solution."));

    ArraySolution const & array_solution = map_solution->find(quad_index)->second;

    Id id = std::make_tuple(face, q, v);

    std::vector<Tensor<rank, dim, Number>> const & vector_solution =
      array_solution.find(id)->second;

    Tensor<rank, dim, Number> result;
    for(auto solution = vector_solution.begin(); solution != vector_solution.end(); ++solution)
    {
      result += (*solution);
    }

    if(vector_solution.size() > 0)
      result *= 1.0 / Number(vector_solution.size());

    return result;
  }

  void
  set_data_pointer(std::map<unsigned int, ArraySolution> const & map_solution_in)
  {
    map_solution = &map_solution_in;
  }

private:
  std::map<unsigned int, ArraySolution> const * map_solution;
};

#endif /* INCLUDE_FUNCTIONALITIES_FUNCTION_INTERPOLATION_H_ */
