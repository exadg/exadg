/*
 * mesh.h
 *
 *  Created on: Feb 13, 2020
 *      Author: fehn
 */

#ifndef INCLUDE_FUNCTIONALITIES_MESH_H_
#define INCLUDE_FUNCTIONALITIES_MESH_H_

#include <deal.II/fe/mapping_q_generic.h>

namespace ExaDG
{
using namespace dealii;

/*
 * Base class, currently only defined by the public interface get_mapping()
 */
template<int dim>
class Mesh
{
public:
  Mesh(std::shared_ptr<Mapping<dim>> mapping_in) : mapping(mapping_in)
  {
  }

  virtual ~Mesh()
  {
  }

  virtual Mapping<dim> const &
  get_mapping() const
  {
    return *mapping;
  }

protected:
  std::shared_ptr<Mapping<dim>> mapping;
};

} // namespace ExaDG

#endif /* INCLUDE_FUNCTIONALITIES_MESH_H_ */
