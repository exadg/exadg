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

#ifndef INCLUDE_EXADG_POISSON_OVERSET_GRIDS_USER_INTERFACE_APPLICATION_BASE_H_
#define INCLUDE_EXADG_POISSON_OVERSET_GRIDS_USER_INTERFACE_APPLICATION_BASE_H_

// ExaDG
#include <exadg/poisson/user_interface/application_base.h>

namespace ExaDG
{
namespace Poisson
{
template<int dim, int n_components, typename Number>
class ApplicationOversetGridsBase
{
public:
  ApplicationOversetGridsBase(std::string parameter_file) : parameter_file(parameter_file)
  {
  }

  void
  add_parameters(dealii::ParameterHandler & prm)
  {
    resolution1.add_parameters(prm, "ResolutionDomain1");
    resolution2.add_parameters(prm, "ResolutionDomain2");

    domain1->add_parameters(prm);
    domain2->add_parameters(prm);
  }

  virtual ~ApplicationOversetGridsBase()
  {
  }

  void
  setup()
  {
    // parse and set resolution parameters for both domains
    parse_resolution_parameters();
    domain1->set_parameters_refinement_study(resolution1.degree,
                                             resolution1.refine_space,
                                             0 /* not used */);
    domain2->set_parameters_refinement_study(resolution2.degree,
                                             resolution2.refine_space,
                                             0 /* not used */);

    domain1->setup();
    domain2->setup();
  }

  std::shared_ptr<ApplicationBase<dim, n_components, Number>> domain1, domain2;

private:
  /**
   * Here, parse only those parameters not covered by ApplicationBase implementations
   * (domain1 and domain2).
   */
  void
  parse_resolution_parameters()
  {
    dealii::ParameterHandler prm;

    resolution1.add_parameters(prm, "ResolutionDomain1");
    resolution2.add_parameters(prm, "ResolutionDomain2");

    prm.parse_input(parameter_file, "", true, true);
  }

  std::string parameter_file;

  ResolutionParameters resolution1, resolution2;
};

} // namespace Poisson

} // namespace ExaDG

#endif /* INCLUDE_EXADG_POISSON_OVERSET_GRIDS_USER_INTERFACE_APPLICATION_BASE_H_ */
