/*
 * function_interpolation.h
 *
 *  Created on: Mar 5, 2020
 *      Author: fehn
 */

#ifndef INCLUDE_FUNCTIONALITIES_FUNCTION_INTERPOLATION_H_
#define INCLUDE_FUNCTIONALITIES_FUNCTION_INTERPOLATION_H_

using namespace dealii;

template<int rank, int dim, typename Number>
class FunctionInterpolation : public Function<dim, Number>
{
private:
  typedef std::tuple<unsigned int /*face*/, unsigned int /*q*/, unsigned int /*v*/> Id;
  typedef std::map<Id, std::vector<Tensor<rank, dim, Number>>>                      ArraySolution;

public:
  Tensor<rank, dim, Number>
  value(unsigned int const face,
        unsigned int const q,
        unsigned int const v,
        unsigned int const quad_index) const
  {
    Assert(map_solution != nullptr, ExcMessage("Pointer map_solution is not initialized."));

    Id id = std::make_tuple(face, q, v);

    ArraySolution & array_solution = map_solution->find(quad_index);

    std::vector<Tensor<rank, dim, Number>> & vector_solution = array_solution.find(id);

    Tensor<rank, dim, Number> result;
    for(auto solution = vector_solution.begin(); solution != vector_solution.end(); ++solution)
    {
      result += (*solution);
    }

    result *= 1.0 / Number(vector_solution.size());

    return result;
  }

  void
  set_data_pointer(std::map<unsigned int, ArraySolution> const & map_solution_in)
  {
    map_solution = &map_solution_in;
  }

private:
  std::map<unsigned int, ArraySolution> * map_solution;
};

#endif /* INCLUDE_FUNCTIONALITIES_FUNCTION_INTERPOLATION_H_ */
