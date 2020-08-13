/*
 * parameter_study.h
 *
 *  Created on: 24.03.2020
 *      Author: fehn
 */

#ifndef INCLUDE_UTILITIES_PARAMETER_STUDY_H_
#define INCLUDE_UTILITIES_PARAMETER_STUDY_H_

// deal.II
#include <deal.II/base/parameter_handler.h>

namespace ExaDG
{
/*
 * study throughput as a function of polynomial degree or problem size
 */
enum class RunType
{
  RefineHAndP, // run simulation for a specified range of mesh refinements and polynomial degrees
  FixedProblemSize,     // increase polynomial degree and keep problem size approximately constant
  IncreasingProblemSize // run at fixed polynomial degree
};

void
string_to_enum(RunType & enum_type, std::string const string_type)
{
  // clang-format off
  if     (string_type == "RefineHAndP")           enum_type = RunType::RefineHAndP;
  else if(string_type == "FixedProblemSize")      enum_type = RunType::FixedProblemSize;
  else if(string_type == "IncreasingProblemSize") enum_type = RunType::IncreasingProblemSize;
  else AssertThrow(false, ExcMessage("Not implemented."));
  // clang-format on
}

/*
 * Determines mesh refinement level l and number of subdivisions in 1d of hyper_cube mesh for a
 * given number of minimum and maximum number of unknowns and a given RunType.
 */
inline void
fill_resolutions_vector(
  std::vector<
    std::tuple<unsigned int /*k*/, unsigned int /*l*/, unsigned int /*subdivisions hypercube*/>> &
                                resolutions,
  unsigned int const            dim,
  unsigned int const            degree,
  unsigned int const            dofs_per_element,
  types::global_dof_index const n_dofs_min,
  types::global_dof_index const n_dofs_max,
  RunType const &               run_type)
{
  unsigned int l = 0, n_subdivisions_1d = 1;

  double n_cells_min = (double)n_dofs_min / dofs_per_element;
  double n_cells_max = (double)n_dofs_max / dofs_per_element;

  int    refine_level = 0;
  double n_cells      = 1.0;

  while(n_cells <= std::pow(2, dim) * n_cells_max)
  {
    // We want to increase the problem size approximately by a factor of two, which is
    // realized by using a coarse grid with {3,4}^dim elements in 2D and {3,4,5}^dim elements
    // in 3D.

    // coarse grid with 3^dim cells, and refine_level-2 uniform refinements
    if(refine_level >= 2)
    {
      n_subdivisions_1d = 3;
      n_cells           = std::pow(n_subdivisions_1d, dim) * std::pow(2., (refine_level - 2) * dim);

      if(n_cells >= n_cells_min && n_cells <= n_cells_max)
      {
        // set mesh refinement
        l = refine_level - 2;

        if(run_type == RunType::FixedProblemSize)
          break;
        else if(run_type == RunType::IncreasingProblemSize)
          resolutions.push_back(
            std::tuple<unsigned int, unsigned int, unsigned int>(degree, l, n_subdivisions_1d));
        else
          AssertThrow(false, ExcMessage("Not implemented:"));
      }
    }

    // coarse grid with only a single cell, and refine_level uniform refinements
    {
      n_subdivisions_1d = 1;
      n_cells           = std::pow(2., refine_level * dim);

      if(n_cells >= n_cells_min && n_cells <= n_cells_max)
      {
        // set mesh refinement
        l = refine_level;

        if(run_type == RunType::FixedProblemSize)
          break;
        else if(run_type == RunType::IncreasingProblemSize)
          resolutions.push_back(
            std::tuple<unsigned int, unsigned int, unsigned int>(degree, l, n_subdivisions_1d));
        else
          AssertThrow(false, ExcMessage("Not implemented:"));
      }
    }

    // coarse grid with 5^dim cells, and refine_level-2 uniform refinements
    if(dim == 3 && refine_level >= 2)
    {
      n_subdivisions_1d = 5;
      n_cells           = std::pow(n_subdivisions_1d, dim) * std::pow(2., (refine_level - 2) * dim);

      if(n_cells >= n_cells_min && n_cells <= n_cells_max)
      {
        // set mesh refinement
        l = refine_level - 2;

        if(run_type == RunType::FixedProblemSize)
          break;
        else if(run_type == RunType::IncreasingProblemSize)
          resolutions.push_back(
            std::tuple<unsigned int, unsigned int, unsigned int>(degree, l, n_subdivisions_1d));
        else
          AssertThrow(false, ExcMessage("Not implemented:"));
      }
    }

    // perform one global refinement
    ++refine_level;
    n_cells = std::pow(2., refine_level * dim);
  }

  if(run_type == RunType::FixedProblemSize)
  {
    AssertThrow((n_cells >= n_cells_min && n_cells <= n_cells_max),
                ExcMessage("No mesh found that meets the requirements regarding problem size. "
                           "Make sure that maximum number of dofs is sufficiently larger than "
                           "minimum number of dofs."));

    resolutions.push_back(
      std::tuple<unsigned int, unsigned int, unsigned int>(degree, l, n_subdivisions_1d));
  }
  else
  {
    AssertThrow(run_type == RunType::IncreasingProblemSize, ExcMessage("Not implemented."));
  }
}

struct ParameterStudy
{
  ParameterStudy()
  {
  }

  ParameterStudy(const std::string & input_file)
  {
    dealii::ParameterHandler prm;
    add_parameters(prm);
    prm.parse_input(input_file, "", true, true);

    string_to_enum(run_type, run_type_string);

    verify_parameters();
  }

  void
  add_parameters(dealii::ParameterHandler & prm)
  {
    // clang-format off
    prm.enter_subsection("General");
      prm.add_parameter("Precision",
                        precision,
                        "Floating point precision.",
                        Patterns::Selection("float|double"),
                        false);
      prm.add_parameter("Dim",
                        dim,
                        "Number of space dimension.",
                        Patterns::Integer(2,3),
                        true);
      prm.add_parameter("RunType",
                        run_type_string,
                        "Type of throughput study.",
                        Patterns::Selection("RefineHAndP|FixedProblemSize|IncreasingProblemSize"),
                        true);
      prm.add_parameter("DegreeMin",
                        degree_min,
                        "Minimal polynomial degree of shape functions.",
                        Patterns::Integer(1,15),
                        true);
      prm.add_parameter("DegreeMax",
                        degree_max,
                        "Maximal polynomial degree of shape functions.",
                        Patterns::Integer(1,15),
                        true);
      prm.add_parameter("RefineSpaceMin",
                        refine_space_min,
                        "Minimal number of mesh refinements.",
                        Patterns::Integer(0,20),
                        true);
      prm.add_parameter("RefineSpaceMax",
                        refine_space_max,
                        "Maximal number of mesh refinements.",
                        Patterns::Integer(0,20),
                        true);
      prm.add_parameter("DofsMin",
                        n_dofs_min,
                        "Minimal number of degrees of freedom.",
                        Patterns::Integer(1),
                        true);
      prm.add_parameter("DofsMax",
                        n_dofs_max,
                        "Maximal number of degrees of freedom.",
                        Patterns::Integer(1),
                        true);
    prm.leave_subsection();
    // clang-format on
  }

  void
  verify_parameters()
  {
    if(run_type == RunType::RefineHAndP)
    {
      AssertThrow(degree_max >= degree_min, ExcMessage("Invalid parameters."));
      AssertThrow(refine_space_max >= refine_space_min, ExcMessage("Invalid parameters."));
    }
    else if(run_type == RunType::FixedProblemSize)
    {
      AssertThrow(degree_max >= degree_min, ExcMessage("Invalid parameters."));
      AssertThrow(n_dofs_max >= n_dofs_min, ExcMessage("Invalid parameters."));
    }
    else if(run_type == RunType::IncreasingProblemSize)
    {
      AssertThrow(
        degree_min == degree_max,
        ExcMessage(
          "Only a single polynomial degree can be considered for RunType::IncreasingProblemSize"));

      AssertThrow(n_dofs_max >= n_dofs_min, ExcMessage("Invalid parameters."));
    }
    else
    {
      AssertThrow(false, ExcMessage("not implemented."));
    }
  }

  void
  fill_resolution_vector(
    std::function<unsigned int(std::string, unsigned int, unsigned int)> const &
                get_dofs_per_element,
    std::string operator_type)
  {
    if(run_type == RunType::RefineHAndP)
    {
      unsigned int const n_cells_1d = 1;

      // k-refinement
      for(unsigned int degree = degree_min; degree <= degree_max; ++degree)
      {
        // h-refinement
        for(unsigned int refine_space = refine_space_min; refine_space <= refine_space_max;
            ++refine_space)
        {
          resolutions.push_back(
            std::tuple<unsigned int, unsigned int, unsigned int>(degree, refine_space, n_cells_1d));
        }
      }
    }
    else if(run_type == RunType::FixedProblemSize || run_type == RunType::IncreasingProblemSize)
    {
      for(unsigned int degree = degree_min; degree <= degree_max; ++degree)
      {
        unsigned int dofs_per_element = get_dofs_per_element(operator_type, dim, degree);
        fill_resolutions_vector(
          resolutions, dim, degree, dofs_per_element, n_dofs_min, n_dofs_max, run_type);
      }
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }
  }

  std::string precision = "double";

  unsigned int dim = 2;

  std::string run_type_string = "RefineHAndP";
  RunType     run_type        = RunType::RefineHAndP;

  unsigned int degree_min = 3; // minimal polynomial degree
  unsigned int degree_max = 3; // maximal polynomial degree

  unsigned int refine_space_min = 0; // minimal number of global refinements
  unsigned int refine_space_max = 0; // maximal number of global refinements

  types::global_dof_index n_dofs_min = 1e4; // minimal number of unknowns
  types::global_dof_index n_dofs_max = 3e4; // maximal number of unknowns

  // a vector storing tuples of the form (degree k, refine level l, n_subdivisions_1d)
  std::vector<std::tuple<unsigned int, unsigned int, unsigned int>> resolutions;
};
} // namespace ExaDG


#endif /* INCLUDE_UTILITIES_PARAMETER_STUDY_H_ */
