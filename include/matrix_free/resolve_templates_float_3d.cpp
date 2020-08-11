/*
 * resolve_templates_float_3d.cpp
 *
 *  Created on: May 2, 2019
 *      Author: fehn
 */

#include "evaluation_template_factory.templates.h"

template struct dealii::internal::FEEvaluationFactory<3, 1, float>;
template struct dealii::internal::FEEvaluationFactory<3, 3, float>;

template struct dealii::internal::FEFaceEvaluationFactory<3, 1, float>;
template struct dealii::internal::FEFaceEvaluationFactory<3, 3, float>;

// inverse mass
template struct dealii::internal::CellwiseInverseMassFactory<3, 1, float>;
template struct dealii::internal::CellwiseInverseMassFactory<3, 3, float>;
template struct dealii::internal::CellwiseInverseMassFactory<3, 3 + 2, float>;
