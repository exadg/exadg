/*
 * levels_hybrid_multigrid.h
 *
 *  Created on: Jun 28, 2019
 *      Author: fehn
 */

#ifndef INCLUDE_SOLVERS_AND_PRECONDITIONERS_MULTIGRID_LEVELS_HYBRID_MULTIGRID_H_
#define INCLUDE_SOLVERS_AND_PRECONDITIONERS_MULTIGRID_LEVELS_HYBRID_MULTIGRID_H_


struct MGDoFHandlerIdentifier
{
  MGDoFHandlerIdentifier(unsigned int degree, bool is_dg) : degree(degree), is_dg(is_dg)
  {
  }

  bool
  operator<(const MGDoFHandlerIdentifier & other) const
  {
    return !((degree >= other.degree) && (is_dg >= other.is_dg));
  }

  bool
  operator==(const MGDoFHandlerIdentifier & other) const
  {
    return (degree == other.degree) && (is_dg == other.is_dg);
  }

  unsigned int degree;
  bool         is_dg;
};

struct MGLevelInfo
{
  MGLevelInfo(unsigned int h_level, unsigned int degree, bool is_dg)
    : _h_level(h_level), _dof_handler_id(degree, is_dg)
  {
  }
  MGLevelInfo(unsigned int h_level, MGDoFHandlerIdentifier dof_handler_id)
    : _h_level(h_level), _dof_handler_id(dof_handler_id)
  {
  }

  unsigned int
  h_level() const
  {
    return _h_level;
  }

  unsigned int
  degree() const
  {
    return _dof_handler_id.degree;
  }

  bool
  is_dg() const
  {
    return _dof_handler_id.is_dg;
  }

  MGDoFHandlerIdentifier
  dof_handler_id() const
  {
    return _dof_handler_id;
  }

private:
  unsigned int           _h_level;
  MGDoFHandlerIdentifier _dof_handler_id;
};


#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_MULTIGRID_LEVELS_HYBRID_MULTIGRID_H_ */
