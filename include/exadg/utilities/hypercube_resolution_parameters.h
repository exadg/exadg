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

#ifndef INCLUDE_EXADG_UTILITIES_HYPERCUBE_RESOLUTION_PARAMETERS_H_
#define INCLUDE_EXADG_UTILITIES_HYPERCUBE_RESOLUTION_PARAMETERS_H_

// deal.II
#include <deal.II/base/parameter_handler.h>

// ExaDG
#include <exadg/configuration/config.h>

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
  else AssertThrow(false, dealii::ExcMessage("Not implemented."));
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
  unsigned int const                    dim,
  unsigned int const                    degree,
  unsigned int const                    dofs_per_element,
  dealii::types::global_dof_index const n_dofs_min,
  dealii::types::global_dof_index const n_dofs_max,
  RunType const &                       run_type)
{
  unsigned int l = 0, n_subdivisions_1d = 1;

  dealii::types::global_dof_index n_cells_min =
    (n_dofs_min + dofs_per_element - 1) / dofs_per_element;
  dealii::types::global_dof_index n_cells_max = n_dofs_max / dofs_per_element;

  int                             refine_level = 0;
  dealii::types::global_dof_index n_cells      = 1;

  while(n_cells <= dealii::Utilities::pow(2ULL, dim) * n_cells_max)
  {
    // We want to increase the problem size approximately by a factor of two, which is
    // realized by using a coarse grid with {3,4}^dim elements in 2D and {3,4,5}^dim elements
    // in 3D.

    // coarse grid with 3^dim cells, and refine_level-2 uniform refinements
    if(refine_level >= 2)
    {
      n_subdivisions_1d = 3;
      n_cells           = dealii::Utilities::pow(n_subdivisions_1d, dim) *
                dealii::Utilities::pow(2ULL, (refine_level - 2) * dim);

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
          AssertThrow(false, dealii::ExcMessage("Not implemented:"));
      }
    }

    // coarse grid with only a single cell, and refine_level uniform refinements
    {
      n_subdivisions_1d = 1;
      n_cells           = dealii::Utilities::pow(2ULL, refine_level * dim);

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
          AssertThrow(false, dealii::ExcMessage("Not implemented:"));
      }
    }

    // coarse grid with 5^dim cells, and refine_level-2 uniform refinements
    if(dim == 3 && refine_level >= 2)
    {
      n_subdivisions_1d = 5;
      n_cells           = dealii::Utilities::pow(n_subdivisions_1d, dim) *
                dealii::Utilities::pow(2ULL, (refine_level - 2) * dim);

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
          AssertThrow(false, dealii::ExcMessage("Not implemented:"));
      }
    }

    // perform one global refinement
    ++refine_level;
    n_cells = dealii::Utilities::pow(2ULL, refine_level * dim);
  }

  if(run_type == RunType::FixedProblemSize)
  {
    AssertThrow((n_cells >= n_cells_min && n_cells <= n_cells_max),
                dealii::ExcMessage(
                  "No mesh found that meets the requirements regarding problem size. "
                  "Make sure that maximum number of dofs is sufficiently larger than "
                  "minimum number of dofs."));

    resolutions.push_back(
      std::tuple<unsigned int, unsigned int, unsigned int>(degree, l, n_subdivisions_1d));
  }
  else
  {
    AssertThrow(run_type == RunType::IncreasingProblemSize, dealii::ExcMessage("Not implemented."));
  }
}

struct HypercubeResolutionParameters
{
  HypercubeResolutionParameters()
  {
  }

  HypercubeResolutionParameters(const std::string & input_file, unsigned int const dim) : dim(dim)
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
    prm.enter_subsection("Resolution");
      prm.add_parameter("RunType",
                        run_type_string,
                        "Type of throughput study.",
                        dealii::Patterns::Selection("RefineHAndP|FixedProblemSize|IncreasingProblemSize"),
                        true);
      prm.add_parameter("DegreeMin",
                        degree_min,
                        "Minimal polynomial degree of shape functions.",
                        dealii::Patterns::Integer(1,EXADG_DEGREE_MAX),
                        true);
      prm.add_parameter("DegreeMax",
                        degree_max,
                        "Maximal polynomial degree of shape functions.",
                        dealii::Patterns::Integer(1,EXADG_DEGREE_MAX),
                        true);
      prm.add_parameter("RefineSpaceMin",
                        refine_space_min,
                        "Minimal number of mesh refinements.",
                        dealii::Patterns::Integer(0,20),
                        true);
      prm.add_parameter("RefineSpaceMax",
                        refine_space_max,
                        "Maximal number of mesh refinements.",
                        dealii::Patterns::Integer(0,20),
                        true);
      prm.add_parameter("DofsMin",
                        n_dofs_min,
                        "Minimal number of degrees of freedom.",
                        dealii::Patterns::Integer(1),
                        true);
      prm.add_parameter("DofsMax",
                        n_dofs_max,
                        "Maximal number of degrees of freedom.",
                        dealii::Patterns::Integer(1),
                        true);
    prm.leave_subsection();
    // clang-format on
  }

  void
  verify_parameters()
  {
    if(run_type == RunType::RefineHAndP)
    {
      AssertThrow(degree_max >= degree_min, dealii::ExcMessage("Invalid parameters."));
      AssertThrow(refine_space_max >= refine_space_min, dealii::ExcMessage("Invalid parameters."));
    }
    else if(run_type == RunType::FixedProblemSize)
    {
      AssertThrow(degree_max >= degree_min, dealii::ExcMessage("Invalid parameters."));
      AssertThrow(n_dofs_max >= n_dofs_min, dealii::ExcMessage("Invalid parameters."));
    }
    else if(run_type == RunType::IncreasingProblemSize)
    {
      AssertThrow(
        degree_min == degree_max,
        dealii::ExcMessage(
          "Only a single polynomial degree can be considered for RunType::IncreasingProblemSize"));

      AssertThrow(n_dofs_max >= n_dofs_min, dealii::ExcMessage("Invalid parameters."));
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("not implemented."));
    }
  }

  void
  fill_resolution_vector(
    std::function<unsigned int(std::string, unsigned int, unsigned int)> const &
                get_dofs_per_element,
    std::string input_file)
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
        unsigned int dofs_per_element = get_dofs_per_element(input_file, dim, degree);
        fill_resolutions_vector(
          resolutions, dim, degree, dofs_per_element, n_dofs_min, n_dofs_max, run_type);
      }
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
    }
  }

  unsigned int dim = 2; // number of space dimensions

  std::string run_type_string = "RefineHAndP";
  RunType     run_type        = RunType::RefineHAndP;

  unsigned int degree_min = 3; // minimal polynomial degree
  unsigned int degree_max = 3; // maximal polynomial degree

  unsigned int refine_space_min = 0; // minimal number of global refinements
  unsigned int refine_space_max = 0; // maximal number of global refinements

  dealii::types::global_dof_index n_dofs_min = 1e4; // minimal number of unknowns
  dealii::types::global_dof_index n_dofs_max = 3e4; // maximal number of unknowns

  // a vector storing tuples of the form (degree k, refine level l, n_subdivisions_1d)
  std::vector<std::tuple<unsigned int, unsigned int, unsigned int>> resolutions;
};
} // namespace ExaDG


#endif /* INCLUDE_EXADG_UTILITIES_HYPERCUBE_RESOLUTION_PARAMETERS_H_ */
