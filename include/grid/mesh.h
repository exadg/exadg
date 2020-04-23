/*
 * mesh.h
 *
 *  Created on: Feb 13, 2020
 *      Author: fehn
 */

#ifndef INCLUDE_FUNCTIONALITIES_MESH_H_
#define INCLUDE_FUNCTIONALITIES_MESH_H_

#include <deal.II/fe/mapping_q_generic.h>

using namespace dealii;

/*
 * Base class, currently only defined by the public interface get_mapping()
 */
template<int dim>
class Mesh
{
public:
  Mesh(unsigned int const mapping_degree)
  {
    mapping.reset(new MappingQGeneric<dim>(mapping_degree));
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
  std::shared_ptr<MappingQGeneric<dim>> mapping;
};


#endif /* INCLUDE_FUNCTIONALITIES_MESH_H_ */
