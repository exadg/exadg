/*
 * EvaluateFunctions.h
 *
 *  Created on: Nov 30, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_EVALUATEFUNCTIONS_H_
#define INCLUDE_EVALUATEFUNCTIONS_H_

template<int dim, typename value_type>
inline void evaluate_scalar_function (VectorizedArray<value_type>                   &value,
                                      std_cxx11::shared_ptr<Function<dim> >         function,
                                      Point<dim,VectorizedArray<value_type> > const &q_points,
                                      double const                                  &eval_time)
{
  value_type array [VectorizedArray<value_type>::n_array_elements];
  for (unsigned int n=0; n<VectorizedArray<value_type>::n_array_elements; ++n)
  {
    Point<dim> q_point;
    for (unsigned int d=0; d<dim; ++d)
      q_point[d] = q_points[d][n];

    function->set_time(eval_time);
    array[n] = function->value(q_point);
  }
  value.load(&array[0]);
}

template<int dim, typename value_type>
inline void evaluate_vectorial_function(Tensor<1,dim,VectorizedArray<value_type> >    &value,
                                        std_cxx11::shared_ptr<Function<dim> >         function,
                                        Point<dim,VectorizedArray<value_type> > const &q_points,
                                        double const                                  &eval_time)
{
  for(unsigned int d=0;d<dim;++d)
  {
    value_type array [VectorizedArray<value_type>::n_array_elements];
    for (unsigned int n=0; n<VectorizedArray<value_type>::n_array_elements; ++n)
    {
      Point<dim> q_point;
      for (unsigned int d=0; d<dim; ++d)
        q_point[d] = q_points[d][n];

      function->set_time(eval_time);
      array[n] = function->value(q_point,d);
    }
    value[d].load(&array[0]);
  }
}


#endif /* INCLUDE_EVALUATEFUNCTIONS_H_ */
