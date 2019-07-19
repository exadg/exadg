/*
 * solution_field.h
 *
 *  Created on: Mar 9, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_POSTPROCESSOR_SOLUTION_FIELD_H_
#define INCLUDE_POSTPROCESSOR_SOLUTION_FIELD_H_

#include <deal.II/lac/la_parallel_vector.h>

using namespace dealii;

enum class SolutionFieldType
{
  scalar,
  vector
};

template<int dim, typename Number>
class SolutionField
{
public:
  SolutionField()
    : type(SolutionFieldType::scalar), name("solution"), dof_handler(nullptr), vector(nullptr)
  {
  }

  SolutionFieldType type;

  std::string name;

  DoFHandler<dim> const * dof_handler;

  LinearAlgebra::distributed::Vector<Number> const * vector;
};


#endif /* INCLUDE_POSTPROCESSOR_SOLUTION_FIELD_H_ */
