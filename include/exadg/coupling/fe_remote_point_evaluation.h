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

#ifndef INCLUDE_COUPLING_FE_REMOTE_POINT_EVALUATION_H_
#define INCLUDE_COUPLING_FE_REMOTE_POINT_EVALUATION_H_

#include <deal.II/matrix_free/fe_evaluation.h>

#include <exadg/coupling/fe_remote_point_evaluation_communicator.h>

namespace ExaDG
{
/**
 * A class to access the fields in FERemotePointEvaluationData.
 *
 * The function gather_evaluate() allow also to fill
 * FERemotePointEvaluationData but one could imagine that one
 * works on an external instance of FERemotePointEvaluationData
 * which is handled and filled outside by the user.
 */
template<int dim,
         int n_components,
         typename Number,
         typename VectorizedArrayType = dealii::VectorizedArray<Number>>
class FERemotePointEvaluation
{
public:
  using data_type     = FERemotePointEvaluationData<dim, n_components, Number, VectorizedArrayType>;
  using value_type    = typename data_type::value_type;
  using gradient_type = typename data_type::gradient_type;

  template<template<int> class MeshType>
  FERemotePointEvaluation(const FERemotePointEvaluationCommunicator<dim, Number> &    comm,
                          const MeshType<dim> &                                       mesh,
                          const dealii::VectorTools::EvaluationFlags::EvaluationFlags vt_flags =
                            dealii::VectorTools::EvaluationFlags::avg)
    : comm(&comm), vt_flags(vt_flags), cell(dealii::numbers::invalid_unsigned_int)
  {
    set_mesh(mesh);
  }

  FERemotePointEvaluation(const FERemotePointEvaluationCommunicator<dim, Number> &    comm_in,
                          const dealii::VectorTools::EvaluationFlags::EvaluationFlags vt_flags =
                            dealii::VectorTools::EvaluationFlags::avg)
    : comm(&comm_in), vt_flags(vt_flags), cell(dealii::numbers::invalid_unsigned_int)
  {
  }

  template<template<int> class MeshType>
  void
  setup_mesh_type(const MeshType<dim> & mesh_in)
  {
    AssertThrow((!tria) && (!dof_handler), dealii::ExcMessage("Mesh has already been set!"));

    set_mesh(mesh_in);
  }


  template<typename VectorType>
  void
  gather_evaluate(const VectorType & vector, const dealii::EvaluationFlags::EvaluationFlags flags)
  {
    if(tria)
    {
      AssertThrow(n_components == 1, dealii::ExcNotImplemented());
      comm->update_ghost_values(this->data, *tria, vector, flags, vt_flags);
    }
    else if(dof_handler)
      comm->update_ghost_values(this->data, *dof_handler, vector, flags, vt_flags);
    else
      AssertThrow(false, dealii::ExcNotImplemented());
  }

  void
  reinit(const unsigned int cell)
  {
    this->cell = cell;
  }

  value_type
  get_value(const unsigned int q) const
  {
    Assert(this->cell != dealii::numbers::invalid_unsigned_int, dealii::ExcInternalError());

    const unsigned int index = comm->get_shift(cell) + q;

    AssertIndexRange(index, data.values.size());

    return data.values[index];
  }

  gradient_type
  get_gradient(const unsigned int q) const
  {
    Assert(this->cell != dealii::numbers::invalid_unsigned_int, dealii::ExcInternalError());

    const unsigned int index = comm->get_shift(cell) + q;

    AssertIndexRange(index, data.gradients.size());

    return data.gradients[index];
  }

private:
  void
  set_mesh(const dealii::Triangulation<dim> & tria)
  {
    this->tria = &tria;
  }

  void
  set_mesh(const dealii::DoFHandler<dim> & dof_handler)
  {
    this->dof_handler = &dof_handler;
  }

  data_type data;

  dealii::SmartPointer<const FERemotePointEvaluationCommunicator<dim, Number>> comm;
  const dealii::VectorTools::EvaluationFlags::EvaluationFlags                  vt_flags;

  dealii::SmartPointer<const dealii::Triangulation<dim>> tria;
  dealii::SmartPointer<const dealii::DoFHandler<dim>>    dof_handler;

  unsigned int cell;
};
} // namespace ExaDG

#endif /*INCLUDE_COUPLING_FE_REMOTE_POINT_EVALUATION_H_*/
