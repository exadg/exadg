/*
 * dealii_extensions.h
 *
 *  Created on: May 27, 2019
 *      Author: fehn
 */

#ifndef APPLICATIONS_GRID_TOOLS_DEALII_EXTENSIONS_H_
#define APPLICATIONS_GRID_TOOLS_DEALII_EXTENSIONS_H_

namespace dealii
{
namespace GridTools
{
// In deal.II, rotate() is only implemented for dim = 2. Write specialization
// for dim = 3 explicitly to allow generic implementations of test cases.
void
rotate(const double angle, Triangulation<3> & triangulation)
{
  (void)angle;
  (void)triangulation;

  AssertThrow(false, ExcMessage("GridTools::rotate() is only available for dim = 2."));
}

} // namespace GridTools

namespace GridGenerator
{
// In deal.II, extrude_triangulation() is only implemented for dim = 3
// as output triangulation. Write specialization for dim = 2 explicitly
// to allow generic implementations of test cases.
void
extrude_triangulation(const Triangulation<2, 2> &             input,
                      const unsigned int                      n_slices,
                      const double                            height,
                      Triangulation<2, 2> &                   result,
                      const bool                              copy_manifold_ids   = false,
                      const std::vector<types::manifold_id> & manifold_priorities = {})
{
  (void)input;
  (void)n_slices;
  (void)height;
  (void)result;
  (void)copy_manifold_ids;
  (void)manifold_priorities;

  AssertThrow(false,
              ExcMessage("GridTools::extrude_triangulation() is only available "
                         "for Triangulation<3, 3> as output triangulation."));
}


} // namespace GridGenerator

} // namespace dealii

#endif /* APPLICATIONS_GRID_TOOLS_DEALII_EXTENSIONS_H_ */
