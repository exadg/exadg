// ---------------------------------------------------------------------
//
// Copyright (C) 2020 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.
//
// ---------------------------------------------------------------------

#define FE_EVAL_FACTORY_DEGREE_MAX 15

#include <deal.II/matrix_free/evaluation_template_factory.templates.h>

DEAL_II_NAMESPACE_OPEN

template struct dealii::internal::FEEvaluationFactory<deal_II_dimension,
                                                      deal_II_scalar_vectorized::value_type,
                                                      deal_II_scalar_vectorized>;

template struct dealii::internal::FEFaceEvaluationFactory<deal_II_dimension,
                                                          deal_II_scalar_vectorized::value_type,
                                                          deal_II_scalar_vectorized>;

template struct dealii::internal::CellwiseInverseMassFactory<deal_II_dimension,
                                                             deal_II_scalar_vectorized::value_type,
                                                             deal_II_scalar_vectorized>;

DEAL_II_NAMESPACE_CLOSE
