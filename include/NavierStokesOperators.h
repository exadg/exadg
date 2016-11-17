/*
 * NavierStokesOperators.h
 *
 *  Created on: Jun 6, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_NAVIERSTOKESOPERATORS_H_
#define INCLUDE_NAVIERSTOKESOPERATORS_H_

#include "../include/BoundaryDescriptorNavierStokes.h"
#include "BaseOperator.h"

// forward declarations
template <int dim, int fe_degree, int fe_degree_xwall, int n_q_points_1d,
      int n_components_, typename Number, bool is_enriched> class FEEvaluationWrapper;

template <int dim, int fe_degree, int fe_degree_xwall, int n_q_points_1d,
      int n_components_, typename Number, bool is_enriched> class FEFaceEvaluationWrapper;

template <int dim, int fe_degree, int fe_degree_xwall, int n_q_points_1d,
      int n_components_, typename Number, bool is_enriched> class FEEvaluationWrapperPressure;

template <int dim, int fe_degree, int fe_degree_xwall, int n_q_points_1d,
      int n_components_, typename Number, bool is_enriched> class FEFaceEvaluationWrapperPressure;

template<int dim>
struct BodyForceOperatorData
{
  BodyForceOperatorData ()
    :
    dof_index(0)
  {}

  unsigned int dof_index;
  std_cxx11::shared_ptr<Function<dim> > rhs;
};

template <int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule, typename value_type>
class BodyForceOperator: public BaseOperator<dim>
{
public:
  BodyForceOperator()
    :
    data(nullptr),
    eval_time(0.0)
  {}

  static const bool is_xwall = (xwall_quad_rule>1) ? true : false;
  static const unsigned int n_actual_q_points_vel_linear = (is_xwall) ? xwall_quad_rule : fe_degree+1;
  typedef FEEvaluationWrapper<dim,fe_degree,fe_degree_xwall,n_actual_q_points_vel_linear,dim,value_type,is_xwall> FEEval_Velocity_Velocity_linear;

  void initialize(MatrixFree<dim,value_type> const &mf_data,
                  BodyForceOperatorData<dim> const &operator_data_in)
  {
    this->data = &mf_data;
    this->operator_data = operator_data_in;
  }

  void evaluate(parallel::distributed::Vector<value_type> &dst,
                const value_type                          evaluation_time) const
  {
    dst = 0;
    evaluate_add(dst,evaluation_time);
  }

  void evaluate_add(parallel::distributed::Vector<value_type> &dst,
                    value_type const                          evaluation_time) const
  {
    this->eval_time = evaluation_time;

    parallel::distributed::Vector<value_type> src;
    data->cell_loop(&BodyForceOperator<dim,fe_degree,fe_degree_xwall,xwall_quad_rule,value_type>::local_evaluate, this, dst, src);
  }

private:
  void local_evaluate (const MatrixFree<dim,value_type>                 &data,
                       parallel::distributed::Vector<value_type>        &dst,
                       const parallel::distributed::Vector<value_type>  &,
                       const std::pair<unsigned int,unsigned int>       &cell_range) const
  {
    FEEval_Velocity_Velocity_linear fe_eval_velocity(data,this->fe_param,operator_data.dof_index);

    // set correct evaluation time for the evaluation of the rhs-function
    operator_data.rhs->set_time(eval_time);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval_velocity.reinit (cell);

      for (unsigned int q=0; q<fe_eval_velocity.n_q_points; ++q)
      {
        Point<dim,VectorizedArray<value_type> > q_points = fe_eval_velocity.quadrature_point(q);
        Tensor<1,dim,VectorizedArray<value_type> > rhs;

        for(unsigned int d=0;d<dim;++d)
        {
          value_type array [VectorizedArray<value_type>::n_array_elements];
          for (unsigned int n=0; n<VectorizedArray<value_type>::n_array_elements; ++n)
          {
            Point<dim> q_point;
            for (unsigned int d=0; d<dim; ++d)
            q_point[d] = q_points[d][n];
            array[n] = operator_data.rhs->value(q_point,d);
          }
          rhs[d].load(&array[0]);
        }
        fe_eval_velocity.submit_value (rhs, q);
      }
      fe_eval_velocity.integrate (true,false);
      fe_eval_velocity.distribute_local_to_global(dst);
    }
  }

  MatrixFree<dim,value_type> const * data;
  BodyForceOperatorData<dim> operator_data;
  value_type mutable eval_time;
};

struct MassMatrixOperatorData
{
  MassMatrixOperatorData ()
    :
    dof_index(0)
  {}

  unsigned int dof_index;
};

template <int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule, typename value_type>
class MassMatrixOperator: public BaseOperator<dim>
{
public:
  MassMatrixOperator()
    :
    data(nullptr)
  {}

  static const bool is_xwall = (xwall_quad_rule>1) ? true : false;
  static const unsigned int n_actual_q_points_vel_linear = (is_xwall) ? xwall_quad_rule : fe_degree+1;
  typedef FEEvaluationWrapper<dim,fe_degree,fe_degree_xwall,n_actual_q_points_vel_linear,dim,value_type,is_xwall> FEEval_Velocity_Velocity_linear;

  void initialize(MatrixFree<dim,value_type> const &mf_data,
                  MassMatrixOperatorData const     &mass_matrix_operator_data_in)
  {
    this->data = &mf_data;
    this->mass_matrix_operator_data = mass_matrix_operator_data_in;
  }

  // apply matrix vector multiplication
  void apply (parallel::distributed::Vector<value_type>       &dst,
              const parallel::distributed::Vector<value_type> &src) const
  {
    dst = 0;
    apply_add(dst,src);
  }

  void apply_add (parallel::distributed::Vector<value_type>       &dst,
                  const parallel::distributed::Vector<value_type> &src) const
  {
    apply_mass_matrix(dst,src);
  }

  void calculate_diagonal(parallel::distributed::Vector<value_type> &diagonal) const
  {
    diagonal = 0;

    add_diagonal(diagonal);

//     verify_calculation_of_diagonal(diagonal);
  }

  void add_diagonal(parallel::distributed::Vector<value_type> &diagonal) const
  {
    parallel::distributed::Vector<value_type>  src_dummy(diagonal);

    data->cell_loop(&MassMatrixOperator<dim,fe_degree,fe_degree_xwall,xwall_quad_rule,value_type>::local_diagonal, this, diagonal, src_dummy);
  }

  void verify_calculation_of_diagonal(parallel::distributed::Vector<value_type> &diagonal) const
  {
    parallel::distributed::Vector<value_type>  diagonal2(diagonal);
    diagonal2 = 0.0;
    parallel::distributed::Vector<value_type>  src(diagonal2);
    parallel::distributed::Vector<value_type>  dst(diagonal2);
    for (unsigned int i=0;i<diagonal.local_size();++i)
    {
      src.local_element(i) = 1.0;
      apply(dst,src);
      diagonal2.local_element(i) = dst.local_element(i);
      src.local_element(i) = 0.0;
    }
    std::cout<<"L2 norm diagonal - Variant 1: "<<std::setprecision(10)<<diagonal.l2_norm()<<std::endl;
    std::cout<<"L2 norm diagonal - Variant 2: "<<std::setprecision(10)<<diagonal2.l2_norm()<<std::endl;
    diagonal2.add(-1.0,diagonal);
    std::cout<<"L2 error diagonal: "<<diagonal2.l2_norm()<<std::endl;
  }

  void invert_diagonal(parallel::distributed::Vector<value_type> &diagonal) const
  {
    for (unsigned int i=0;i<diagonal.local_size();++i)
    {
      if(std::abs(diagonal.local_element(i)) > 1.0e-10)
        diagonal.local_element(i) = 1.0/diagonal.local_element(i);
      else
        diagonal.local_element(i) = 1.0;
    }
  }

  void calculate_inverse_diagonal(parallel::distributed::Vector<value_type> &diagonal) const
  {
    calculate_diagonal(diagonal);

    invert_diagonal(diagonal);
  }

  MassMatrixOperatorData const & get_operator_data() const
  {
    return mass_matrix_operator_data;
  }

private:
  void apply_mass_matrix (parallel::distributed::Vector<value_type>        &dst,
                          const parallel::distributed::Vector<value_type>  &src) const
  {
    data->cell_loop(&MassMatrixOperator<dim,fe_degree,fe_degree_xwall,xwall_quad_rule, value_type>::local_apply, this, dst, src);
  }

  void local_apply (const MatrixFree<dim,value_type>                 &data,
                    parallel::distributed::Vector<value_type>        &dst,
                    const parallel::distributed::Vector<value_type>  &src,
                    const std::pair<unsigned int,unsigned int>       &cell_range) const
  {
    FEEval_Velocity_Velocity_linear fe_eval_velocity(data,this->fe_param,mass_matrix_operator_data.dof_index);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval_velocity.reinit(cell);
      fe_eval_velocity.read_dof_values(src);
      fe_eval_velocity.evaluate (true,false,false);

      for (unsigned int q=0; q<fe_eval_velocity.n_q_points; ++q)
      {
        fe_eval_velocity.submit_value (fe_eval_velocity.get_value(q), q);
      }
      fe_eval_velocity.integrate (true,false);
      fe_eval_velocity.distribute_local_to_global (dst);
    }
  }

  void local_diagonal (const MatrixFree<dim,value_type>                 &data,
                       parallel::distributed::Vector<value_type>        &dst,
                       const parallel::distributed::Vector<value_type>  &,
                       const std::pair<unsigned int,unsigned int>       &cell_range) const
  {
    FEEval_Velocity_Velocity_linear fe_eval_velocity(data,this->fe_param,mass_matrix_operator_data.dof_index);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval_velocity.reinit (cell);

      VectorizedArray<value_type> local_diagonal_vector[fe_eval_velocity.tensor_dofs_per_cell*dim];
      for (unsigned int j=0; j<fe_eval_velocity.dofs_per_cell*dim; ++j)
      {
        for (unsigned int i=0; i<fe_eval_velocity.dofs_per_cell*dim; ++i)
          fe_eval_velocity.write_cellwise_dof_value(i,make_vectorized_array<value_type>(0.));
        fe_eval_velocity.write_cellwise_dof_value(j,make_vectorized_array<value_type>(1.));

        // copied from local_apply //TODO
        fe_eval_velocity.evaluate (true,false,false);

        for (unsigned int q=0; q<fe_eval_velocity.n_q_points; ++q)
        {
          fe_eval_velocity.submit_value (fe_eval_velocity.get_value(q), q);
        }
        fe_eval_velocity.integrate (true,false);
        // copied from local_apply

        local_diagonal_vector[j] = fe_eval_velocity.read_cellwise_dof_value(j);
      }
      for (unsigned int j=0; j<fe_eval_velocity.dofs_per_cell*dim; ++j)
        fe_eval_velocity.write_cellwise_dof_value(j,local_diagonal_vector[j]);

      fe_eval_velocity.distribute_local_to_global (dst);
    }
  }

  MatrixFree<dim,value_type> const * data;
  MassMatrixOperatorData mass_matrix_operator_data;
};

template<int dim>
struct ViscousOperatorData
{
  ViscousOperatorData ()
    :
    formulation_viscous_term(FormulationViscousTerm::DivergenceFormulation),
    IP_formulation_viscous(InteriorPenaltyFormulation::SIPG),
    IP_factor_viscous(1.0),
    dof_index(0),
    viscosity(1.0)
  {}

  FormulationViscousTerm formulation_viscous_term;
  InteriorPenaltyFormulation IP_formulation_viscous;
  double IP_factor_viscous;
  unsigned int dof_index;

  std_cxx11::shared_ptr<BoundaryDescriptorNavierStokes<dim> > bc;

  /*
   * This variable 'viscosity' is only used when initializing the ViscousOperator.
   * In order to change/update this coefficient during the simulation (e.g., varying viscosity/turbulence)
   * use the element variable 'const_viscosity' of ViscousOperator and the corresponding setter
   * set_constant_viscosity().
   */
  double viscosity;
};

template <int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule, typename Number=double>
class ViscousOperator : public BaseOperator<dim>
{
public:
  typedef Number value_type;

  ViscousOperator()
    :
    data(nullptr),
    const_viscosity(-1.0),
    eval_time(0.0)
  {}

  static const bool is_xwall = (xwall_quad_rule>1) ? true : false;
  static const unsigned int n_actual_q_points_vel_linear = (is_xwall) ? xwall_quad_rule : fe_degree+1;
  typedef FEEvaluationWrapper<dim,fe_degree,fe_degree_xwall,n_actual_q_points_vel_linear,dim,Number,is_xwall> FEEval_Velocity_Velocity_linear;
  typedef FEFaceEvaluationWrapper<dim,fe_degree,fe_degree_xwall,n_actual_q_points_vel_linear,dim,Number,is_xwall> FEFaceEval_Velocity_Velocity_linear;

  void initialize(Mapping<dim> const              &mapping,
                  MatrixFree<dim,Number> const    &mf_data,
                  ViscousOperatorData<dim> const  &operator_data_in)
  {

    this->data = &mf_data;
    this->operator_data = operator_data_in;

    compute_array_penalty_parameter(mapping);

    const_viscosity = operator_data.viscosity;
  }

  void set_constant_viscosity(double const viscosity_in)
  {
    const_viscosity = viscosity_in;
  }

  // returns true if viscous_coefficient table has been filled with spatially varying viscosity values
  bool viscosity_is_variable() const
  {
    return viscous_coefficient_cell.n_elements()>0;
  }

  double get_const_viscosity() const
  {
    return const_viscosity;
  }

  void initialize_viscous_coefficients()
  {
    Assert(xwall_quad_rule > fe_degree +1, ExcMessage("this may cause a memory error"));
    this->viscous_coefficient_cell.reinit(this->data->n_macro_cells(), Utilities::fixed_int_power<FEEval_Velocity_Velocity_linear::n_q_points,dim>::value);
    this->viscous_coefficient_cell.fill(make_vectorized_array<Number>(const_viscosity));

    this->viscous_coefficient_face.reinit(this->data->n_macro_inner_faces()+this->data->n_macro_boundary_faces(), Utilities::fixed_int_power<FEFaceEval_Velocity_Velocity_linear::n_q_points,dim-1>::value);
    this->viscous_coefficient_face.fill(make_vectorized_array<Number>(const_viscosity));
    this->viscous_coefficient_face_neighbor.reinit(this->data->n_macro_inner_faces()+this->data->n_macro_boundary_faces(), Utilities::fixed_int_power<FEFaceEval_Velocity_Velocity_linear::n_q_points,dim-1>::value);
    this->viscous_coefficient_face_neighbor.fill(make_vectorized_array<Number>(const_viscosity));
  }

  void set_viscous_coefficient_cell(unsigned int const cell, unsigned int const q, VectorizedArray<Number> const  & value)
  {
    viscous_coefficient_cell[cell][q] = value;
  }

  void set_viscous_coefficient_face(unsigned int const face, unsigned int const q, VectorizedArray<Number> const  & value)
  {
    viscous_coefficient_face[face][q] = value;
  }

  void set_viscous_coefficient_face_neighbor(unsigned int const face, unsigned int const q, VectorizedArray<Number> const  & value)
  {
    viscous_coefficient_face_neighbor[face][q] = value;
  }

  Table<2,VectorizedArray<Number> > const & get_viscous_coefficient_face() const
  {
    return viscous_coefficient_face;
  }

  Table<2,VectorizedArray<Number> > const & get_viscous_coefficient_cell() const
  {
    return viscous_coefficient_cell;
  }

  // apply matrix vector multiplication
  void vmult (parallel::distributed::Vector<Number>       &dst,
              const parallel::distributed::Vector<Number> &src) const
  {
    apply(dst,src);
  }

  // apply matrix vector multiplication
  void apply (parallel::distributed::Vector<Number>       &dst,
              const parallel::distributed::Vector<Number> &src) const
  {
    dst = 0;
    apply_add(dst,src);
  }

  void apply_add (parallel::distributed::Vector<Number>       &dst,
                  const parallel::distributed::Vector<Number> &src) const
  {
    apply_viscous(dst,src);
  }

  void rhs (parallel::distributed::Vector<Number> &dst,
            Number const                          evaluation_time) const
  {
    dst = 0;
    rhs_add(dst,evaluation_time);
  }

  void rhs_add (parallel::distributed::Vector<Number> &dst,
                Number const                          evaluation_time) const
  {
    this->eval_time = evaluation_time;

    parallel::distributed::Vector<Number> src;
    data->loop(&ViscousOperator<dim,fe_degree,fe_degree_xwall,xwall_quad_rule,Number>::local_rhs_viscous,
               &ViscousOperator<dim,fe_degree,fe_degree_xwall,xwall_quad_rule,Number>::local_rhs_viscous_face,
               &ViscousOperator<dim,fe_degree,fe_degree_xwall,xwall_quad_rule,Number>::local_rhs_viscous_boundary_face,
               this, dst, src);
  }

  void evaluate (parallel::distributed::Vector<Number>       &dst,
                 const parallel::distributed::Vector<Number> &src,
                 Number const                                evaluation_time) const
  {
    dst = 0;
    evaluate_add(dst,src,evaluation_time);
  }

  void evaluate_add (parallel::distributed::Vector<Number>       &dst,
                     const parallel::distributed::Vector<Number> &src,
                     Number const                                evaluation_time) const
  {
    this->eval_time = evaluation_time;
    evaluate_viscous(dst,src);
  }

  void calculate_diagonal(parallel::distributed::Vector<Number> &diagonal) const
  {
    diagonal = 0;

    add_diagonal(diagonal);

    // verify_calculation_of_diagonal(diagonal);
  }

  void add_diagonal(parallel::distributed::Vector<Number> &diagonal) const
  {
    parallel::distributed::Vector<Number>  src_dummy(diagonal);

    data->loop(&ViscousOperator<dim,fe_degree,fe_degree_xwall,xwall_quad_rule,Number>::local_diagonal,
               &ViscousOperator<dim,fe_degree,fe_degree_xwall,xwall_quad_rule,Number>::local_diagonal_face,
               &ViscousOperator<dim,fe_degree,fe_degree_xwall,xwall_quad_rule,Number>::local_diagonal_boundary_face,
               this, diagonal, src_dummy);
  }

  void verify_calculation_of_diagonal(parallel::distributed::Vector<Number> &diagonal) const
  {
      parallel::distributed::Vector<Number>  diagonal2(diagonal);
      diagonal2 = 0.0;
      parallel::distributed::Vector<Number>  src(diagonal2);
      parallel::distributed::Vector<Number>  dst(diagonal2);
      for (unsigned int i=0;i<diagonal.local_size();++i)
      {
        src.local_element(i) = 1.0;
        apply(dst,src);
        diagonal2.local_element(i) = dst.local_element(i);
        src.local_element(i) = 0.0;
      }

      std::cout<<"L2 norm diagonal - Variant 1: "<<diagonal.l2_norm()<<std::endl;
      std::cout<<"L2 norm diagonal - Variant 2: "<<diagonal2.l2_norm()<<std::endl;
      diagonal2.add(-1.0,diagonal);
      std::cout<<"L2 error diagonal: "<<diagonal2.l2_norm()<<std::endl;
  }

  void invert_diagonal(parallel::distributed::Vector<Number> &diagonal) const
  {
    for (unsigned int i=0;i<diagonal.local_size();++i)
    {
      if( std::abs(diagonal.local_element(i)) > 1.0e-10 )
        diagonal.local_element(i) = 1.0/diagonal.local_element(i);
      else
        diagonal.local_element(i) = 1.0;
    }
  }

  void calculate_inverse_diagonal(parallel::distributed::Vector<Number> &diagonal) const
  {
    calculate_diagonal(diagonal);

    invert_diagonal(diagonal);
  }

  ViscousOperatorData<dim> const & get_operator_data() const
  {
    return operator_data;
  }

private:
  void compute_array_penalty_parameter(const Mapping<dim> &mapping)
  {
    // Compute penalty parameter for each cell
    array_penalty_parameter.resize(data->n_macro_cells()+data->n_macro_ghost_cells());
    QGauss<dim> quadrature(fe_degree+1);
    FEValues<dim> fe_values(mapping,data->get_dof_handler(operator_data.dof_index).get_fe(),quadrature, update_JxW_values);
    QGauss<dim-1> face_quadrature(fe_degree+1);
    FEFaceValues<dim> fe_face_values(mapping, data->get_dof_handler(operator_data.dof_index).get_fe(), face_quadrature, update_JxW_values);

    for (unsigned int i=0; i<data->n_macro_cells()+data->n_macro_ghost_cells(); ++i)
    {
      for (unsigned int v=0; v<data->n_components_filled(i); ++v)
      {
        typename DoFHandler<dim>::cell_iterator cell = data->get_cell_iterator(i,v,operator_data.dof_index);
        fe_values.reinit(cell);
        double volume = 0;
        for (unsigned int q=0; q<quadrature.size(); ++q)
          volume += fe_values.JxW(q);
        double surface_area = 0;
        for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
        {
          fe_face_values.reinit(cell, f);
          const double factor = (cell->at_boundary(f) && !cell->has_periodic_neighbor(f)) ? 1. : 0.5;
          for (unsigned int q=0; q<face_quadrature.size(); ++q)
            surface_area += fe_face_values.JxW(q) * factor;
        }
        array_penalty_parameter[i][v] = surface_area / volume;
      }
    }
  }

  Number get_penalty_factor() const
  {
    return operator_data.IP_factor_viscous * (fe_degree + 1.0) * (fe_degree + 1.0);
  }

  void apply_viscous (parallel::distributed::Vector<Number>        &dst,
                      const parallel::distributed::Vector<Number>  &src) const
  {
    data->loop(&ViscousOperator<dim,fe_degree,fe_degree_xwall,xwall_quad_rule,Number>::local_apply_viscous,
               &ViscousOperator<dim,fe_degree,fe_degree_xwall,xwall_quad_rule,Number>::local_apply_viscous_face,
               &ViscousOperator<dim,fe_degree,fe_degree_xwall,xwall_quad_rule,Number>::local_apply_viscous_boundary_face,
               this, dst, src);
  }

  void local_apply_viscous (const MatrixFree<dim,Number>                 &data,
                            parallel::distributed::Vector<Number>        &dst,
                            const parallel::distributed::Vector<Number>  &src,
                            const std::pair<unsigned int,unsigned int>       &cell_range) const
  {
    AssertThrow(const_viscosity>0.0,ExcMessage("Constant viscosity has not been set!"));

    FEEval_Velocity_Velocity_linear fe_eval_velocity(data,this->fe_param,operator_data.dof_index);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval_velocity.reinit(cell);
      fe_eval_velocity.read_dof_values(src);
      fe_eval_velocity.evaluate (false,true,false);

      for (unsigned int q=0; q<fe_eval_velocity.n_q_points; ++q)
      {
        VectorizedArray<Number> viscosity = make_vectorized_array<Number>(const_viscosity);
        if(viscosity_is_variable())
          viscosity = viscous_coefficient_cell[cell][q];

        if(operator_data.formulation_viscous_term == FormulationViscousTerm::DivergenceFormulation)
        {
          fe_eval_velocity.submit_gradient (viscosity*make_vectorized_array<Number>(2.)*fe_eval_velocity.get_symmetric_gradient(q), q);
        }
        else if(operator_data.formulation_viscous_term == FormulationViscousTerm::LaplaceFormulation)
        {
          fe_eval_velocity.submit_gradient (viscosity*fe_eval_velocity.get_gradient(q), q);
        }
        else
        {
          AssertThrow(false, ExcMessage("FORMULATION_VISCOUS_TERM is not specified - possibilities are DivergenceFormulation and LaplaceFormulation"));
        }
      }
      fe_eval_velocity.integrate (false,true);
      fe_eval_velocity.distribute_local_to_global (dst);
    }
  }

  void local_apply_viscous_face (const MatrixFree<dim,Number>                &data,
                                 parallel::distributed::Vector<Number>       &dst,
                                 const parallel::distributed::Vector<Number> &src,
                                 const std::pair<unsigned int,unsigned int>  &face_range) const
  {
    FEFaceEval_Velocity_Velocity_linear fe_eval_velocity(data,this->fe_param,true,operator_data.dof_index);
    FEFaceEval_Velocity_Velocity_linear fe_eval_velocity_neighbor(data,this->fe_param,false,operator_data.dof_index);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval_velocity.reinit (face);
      fe_eval_velocity_neighbor.reinit (face);

      fe_eval_velocity.read_dof_values(src);
      fe_eval_velocity.evaluate(true,true);
      fe_eval_velocity_neighbor.read_dof_values(src);
      fe_eval_velocity_neighbor.evaluate(true,true);

      VectorizedArray<Number> tau_IP = std::max(fe_eval_velocity.read_cell_data(array_penalty_parameter),fe_eval_velocity_neighbor.read_cell_data(array_penalty_parameter))
                                              * get_penalty_factor();

      for(unsigned int q=0;q<fe_eval_velocity.n_q_points;++q)
      {
        Tensor<1,dim,VectorizedArray<Number> > uM = fe_eval_velocity.get_value(q);
        Tensor<1,dim,VectorizedArray<Number> > uP = fe_eval_velocity_neighbor.get_value(q);
        VectorizedArray<Number> average_viscosity = make_vectorized_array<Number>(const_viscosity);
        VectorizedArray<Number> max_viscosity = make_vectorized_array<Number>(const_viscosity);
        if(viscosity_is_variable())
        {
          // harmonic weighting according to Schott and Rasthofer et al (2015)
          average_viscosity = 2. * viscous_coefficient_face[face][q] * viscous_coefficient_face_neighbor[face][q] /
                             (viscous_coefficient_face[face][q] + viscous_coefficient_face_neighbor[face][q]);
          max_viscosity = average_viscosity;
        }

        Tensor<1,dim,VectorizedArray<Number> > jump_value = uM - uP;

        Tensor<2,dim,VectorizedArray<Number> > average_gradient_tensor;
        if(operator_data.formulation_viscous_term == FormulationViscousTerm::DivergenceFormulation)
        {
          // {{F}} = (F⁻ + F⁺)/2 where F = 2 * nu * symmetric_gradient -> nu * (symmetric_gradient⁻ + symmetric_gradient⁺)
          average_gradient_tensor = ( fe_eval_velocity.get_symmetric_gradient(q) + fe_eval_velocity_neighbor.get_symmetric_gradient(q));
        }
        else if(operator_data.formulation_viscous_term == FormulationViscousTerm::LaplaceFormulation)
        {
          average_gradient_tensor = ( fe_eval_velocity.get_gradient(q) + fe_eval_velocity_neighbor.get_gradient(q)) * make_vectorized_array<Number>(0.5);
        }
        else
        {
          AssertThrow(false, ExcMessage("FORMULATION_VISCOUS_TERM is not specified - possibilities are DivergenceFormulation and LaplaceFormulation"));
        }
        Tensor<2,dim,VectorizedArray<Number> > jump_tensor =
            outer_product(jump_value,fe_eval_velocity.get_normal_vector(q));

        //we do not want to symmetrize the penalty part
        average_gradient_tensor = average_viscosity*average_gradient_tensor - max_viscosity * jump_tensor * tau_IP;
        Tensor<1,dim,VectorizedArray<Number> > average_gradient = average_gradient_tensor*fe_eval_velocity.get_normal_vector(q);

        if(operator_data.formulation_viscous_term == FormulationViscousTerm::DivergenceFormulation)
        {
          if(operator_data.IP_formulation_viscous == InteriorPenaltyFormulation::NIPG)
          {
            fe_eval_velocity.submit_gradient(fe_eval_velocity.make_symmetric(average_viscosity*jump_tensor),q);
            fe_eval_velocity_neighbor.submit_gradient(fe_eval_velocity.make_symmetric(average_viscosity*jump_tensor),q);
          }
          else if(operator_data.IP_formulation_viscous == InteriorPenaltyFormulation::SIPG)
          {
            fe_eval_velocity.submit_gradient(-fe_eval_velocity.make_symmetric(average_viscosity*jump_tensor),q);
            fe_eval_velocity_neighbor.submit_gradient(-fe_eval_velocity.make_symmetric(average_viscosity*jump_tensor),q);
          }
          else
            AssertThrow(false, ExcMessage("IP_FORMULATION_VISCOUS is not specified - possibilities are SIPG and NIPG"));
        }
        else if(operator_data.formulation_viscous_term == FormulationViscousTerm::LaplaceFormulation)
        {
          if(operator_data.IP_formulation_viscous == InteriorPenaltyFormulation::NIPG)
          {
            fe_eval_velocity.submit_gradient(0.5*average_viscosity*jump_tensor,q);
            fe_eval_velocity_neighbor.submit_gradient(0.5*average_viscosity*jump_tensor,q);
          }
          else if(operator_data.IP_formulation_viscous == InteriorPenaltyFormulation::SIPG)
          {
            fe_eval_velocity.submit_gradient(-0.5*average_viscosity*jump_tensor,q);
            fe_eval_velocity_neighbor.submit_gradient(-0.5*average_viscosity*jump_tensor,q);
          }
          else
            AssertThrow(false, ExcMessage("IP_FORMULATION_VISCOUS is not specified - possibilities are SIPG and NIPG"));
        }
        else
        {
          AssertThrow(false, ExcMessage("FORMULATION_VISCOUS_TERM is not specified - possibilities are DivergenceFormulation and LaplaceFormulation"));
        }
        fe_eval_velocity.submit_value(-average_gradient,q);
        fe_eval_velocity_neighbor.submit_value(average_gradient,q);
      }
      fe_eval_velocity.integrate(true,true);
      fe_eval_velocity.distribute_local_to_global(dst);
      fe_eval_velocity_neighbor.integrate(true,true);
      fe_eval_velocity_neighbor.distribute_local_to_global(dst);
    }
  }

  void local_apply_viscous_boundary_face (const MatrixFree<dim,Number>                 &data,
                                          parallel::distributed::Vector<Number>        &dst,
                                          const parallel::distributed::Vector<Number>  &src,
                                          const std::pair<unsigned int,unsigned int>   &face_range) const
  {
    FEFaceEval_Velocity_Velocity_linear fe_eval_velocity(data,this->fe_param,true,operator_data.dof_index);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval_velocity.reinit (face);
      fe_eval_velocity.read_dof_values(src);
      fe_eval_velocity.evaluate(true,true);

      VectorizedArray<Number> tau_IP = fe_eval_velocity.read_cell_data(array_penalty_parameter)
                                             * get_penalty_factor();

      typename std::map<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >::iterator it;
      types::boundary_id boundary_id = data.get_boundary_indicator(face);

      for(unsigned int q=0;q<fe_eval_velocity.n_q_points;++q)
      {
        VectorizedArray<Number> viscosity = make_vectorized_array<Number>(const_viscosity);
        if(viscosity_is_variable())
          viscosity = viscous_coefficient_face[face][q];

        it = operator_data.bc->dirichlet_bc.find(boundary_id);
        if(it != operator_data.bc->dirichlet_bc.end())
        {
          // applying inhomogeneous Dirichlet BC (value+ = - value- + 2g , grad+ = grad-)
          Tensor<1,dim,VectorizedArray<Number> > uM = fe_eval_velocity.get_value(q);
          Tensor<1,dim,VectorizedArray<Number> > uP = -uM;
          Tensor<1,dim,VectorizedArray<Number> > jump_value = uM - uP;

          Tensor<2,dim,VectorizedArray<Number> > average_gradient_tensor;
          if(operator_data.formulation_viscous_term == FormulationViscousTerm::DivergenceFormulation)
          {
            average_gradient_tensor = make_vectorized_array<Number>(2.) * fe_eval_velocity.get_symmetric_gradient(q);
          }
          else if(operator_data.formulation_viscous_term == FormulationViscousTerm::LaplaceFormulation)
          {
            average_gradient_tensor = fe_eval_velocity.get_gradient(q);
          }
          else
          {
            AssertThrow(false, ExcMessage("FORMULATION_VISCOUS_TERM is not specified - possibilities are DivergenceFormulation and LaplaceFormulation"));
          }
          Tensor<2,dim,VectorizedArray<Number> > jump_tensor
            = outer_product(jump_value,fe_eval_velocity.get_normal_vector(q));

          //we do not want to symmetrize the penalty part
          average_gradient_tensor = viscosity*(average_gradient_tensor - jump_tensor * tau_IP);

          Tensor<1,dim,VectorizedArray<Number> > average_gradient = average_gradient_tensor*fe_eval_velocity.get_normal_vector(q);

          if(operator_data.formulation_viscous_term == FormulationViscousTerm::DivergenceFormulation)
          {
            if(operator_data.IP_formulation_viscous == InteriorPenaltyFormulation::NIPG)
            {
              fe_eval_velocity.submit_gradient(fe_eval_velocity.make_symmetric(viscosity*jump_tensor),q);
            }
            else if(operator_data.IP_formulation_viscous == InteriorPenaltyFormulation::SIPG)
            {
              fe_eval_velocity.submit_gradient(-fe_eval_velocity.make_symmetric(viscosity*jump_tensor),q);
            }
            else
              AssertThrow(false, ExcMessage("IP_FORMULATION_VISCOUS is not specified - possibilities are SIPG and NIPG"));
          }
          else if(operator_data.formulation_viscous_term == FormulationViscousTerm::LaplaceFormulation)
          {
            if(operator_data.IP_formulation_viscous == InteriorPenaltyFormulation::NIPG)
            {
              fe_eval_velocity.submit_gradient(0.5*viscosity*jump_tensor,q);
            }
            else if(operator_data.IP_formulation_viscous == InteriorPenaltyFormulation::SIPG)
            {
              fe_eval_velocity.submit_gradient(-0.5*viscosity*jump_tensor,q);
            }
            else
              AssertThrow(false, ExcMessage("IP_FORMULATION_VISCOUS is not specified - possibilities are SIPG and NIPG"));
          }
          else
          {
            AssertThrow(false, ExcMessage("Formulation of viscous term not specified - possibilities are DIVERGENCE_FORMULATION_VISCOUS and LAPLACE_FORMULATION_VISCOUS"));
          }
          fe_eval_velocity.submit_value(-average_gradient,q);
        }

        it = operator_data.bc->neumann_bc.find(boundary_id);
        if(it != operator_data.bc->neumann_bc.end())
        {
          // applying inhomogeneous Neumann BC (value+ = value- , grad+ =  - grad- +2h)
          Tensor<1,dim,VectorizedArray<Number> > jump_value;
          Tensor<1,dim,VectorizedArray<Number> > average_gradient;
          Tensor<2,dim,VectorizedArray<Number> > jump_tensor
            = outer_product(jump_value,fe_eval_velocity.get_normal_vector(q));

          if(operator_data.formulation_viscous_term == FormulationViscousTerm::DivergenceFormulation)
          {
            if(operator_data.IP_formulation_viscous == InteriorPenaltyFormulation::NIPG)
            {
              fe_eval_velocity.submit_gradient(fe_eval_velocity.make_symmetric(viscosity*jump_tensor),q);
            }
            else if(operator_data.IP_formulation_viscous == InteriorPenaltyFormulation::SIPG)
            {
              fe_eval_velocity.submit_gradient(-fe_eval_velocity.make_symmetric(viscosity*jump_tensor),q);
            }
            else
              AssertThrow(false, ExcMessage("IP_FORMULATION_VISCOUS is not specified - possibilities are SIPG and NIPG"));
          }
          else if(operator_data.formulation_viscous_term == FormulationViscousTerm::LaplaceFormulation)
          {
            if(operator_data.IP_formulation_viscous == InteriorPenaltyFormulation::NIPG)
            {
              fe_eval_velocity.submit_gradient(0.5*viscosity*jump_tensor,q);
            }
            else if(operator_data.IP_formulation_viscous == InteriorPenaltyFormulation::SIPG)
            {
              fe_eval_velocity.submit_gradient(-0.5*viscosity*jump_tensor,q);
            }
            else
              AssertThrow(false, ExcMessage("IP_FORMULATION_VISCOUS is not specified - possibilities are SIPG and NIPG"));
          }
          else
          {
            AssertThrow(false, ExcMessage("Formulation of viscous term not specified - possibilities are DIVERGENCE_FORMULATION_VISCOUS and LAPLACE_FORMULATION_VISCOUS"));
          }
          fe_eval_velocity.submit_value(-average_gradient,q);
        }
      }
      fe_eval_velocity.integrate(true,true);
      fe_eval_velocity.distribute_local_to_global(dst);
    }
  }

  void local_diagonal (const MatrixFree<dim,Number>                 &data,
                       parallel::distributed::Vector<Number>        &dst,
                       const parallel::distributed::Vector<Number>  &,
                       const std::pair<unsigned int,unsigned int>   &cell_range) const
  {
    FEEval_Velocity_Velocity_linear fe_eval_velocity(data,this->fe_param,operator_data.dof_index);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval_velocity.reinit (cell);
      VectorizedArray<Number> local_diagonal_vector[fe_eval_velocity.tensor_dofs_per_cell*dim];
      for (unsigned int j=0; j<fe_eval_velocity.dofs_per_cell*dim; ++j)
      {
        for (unsigned int i=0; i<fe_eval_velocity.dofs_per_cell*dim; ++i)
          fe_eval_velocity.write_cellwise_dof_value(i,make_vectorized_array<Number>(0.));
        fe_eval_velocity.write_cellwise_dof_value(j,make_vectorized_array<Number>(1.));

        // copied from local_apply_viscous
        fe_eval_velocity.evaluate (false,true,false);

        for (unsigned int q=0; q<fe_eval_velocity.n_q_points; ++q)
        {
          VectorizedArray<Number> viscosity = make_vectorized_array<Number>(const_viscosity);
          if(viscosity_is_variable())
            viscosity = viscous_coefficient_cell[cell][q];

          if(operator_data.formulation_viscous_term == FormulationViscousTerm::DivergenceFormulation)
          {
            fe_eval_velocity.submit_gradient (viscosity*make_vectorized_array<Number>(2.)*fe_eval_velocity.get_symmetric_gradient(q), q);
          }
          else if(operator_data.formulation_viscous_term == FormulationViscousTerm::LaplaceFormulation)
          {
            fe_eval_velocity.submit_gradient (viscosity*fe_eval_velocity.get_gradient(q), q);
          }
          else
          {
            AssertThrow(false, ExcMessage("FORMULATION_VISCOUS_TERM is not specified - possibilities are DivergenceFormulation and LaplaceFormulation"));
          }
        }
        fe_eval_velocity.integrate (false,true);
        // copied from local_apply_viscous
        local_diagonal_vector[j] = fe_eval_velocity.read_cellwise_dof_value(j);
      }
      for (unsigned int j=0; j<fe_eval_velocity.dofs_per_cell*dim; ++j)
        fe_eval_velocity.write_cellwise_dof_value(j,local_diagonal_vector[j]);
      fe_eval_velocity.distribute_local_to_global (dst);
    }
  }

  void local_diagonal_face (const MatrixFree<dim,Number>                &data,
                            parallel::distributed::Vector<Number>       &dst,
                            const parallel::distributed::Vector<Number> &,
                            const std::pair<unsigned int,unsigned int>  &face_range) const
  {
    FEFaceEval_Velocity_Velocity_linear fe_eval_velocity(data,this->fe_param,true,operator_data.dof_index);
    FEFaceEval_Velocity_Velocity_linear fe_eval_velocity_neighbor(data,this->fe_param,false,operator_data.dof_index);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval_velocity.reinit (face);
      fe_eval_velocity_neighbor.reinit (face);

      VectorizedArray<Number> tau_IP = std::max(fe_eval_velocity.read_cell_data(array_penalty_parameter),fe_eval_velocity_neighbor.read_cell_data(array_penalty_parameter))
                                              * get_penalty_factor();

      // element-
      VectorizedArray<Number> local_diagonal_vector[fe_eval_velocity.tensor_dofs_per_cell*dim];
      for (unsigned int j=0; j<fe_eval_velocity.dofs_per_cell*dim; ++j)
      {
        // set dof value j of element- to 1 and all other dof values of element- to zero
        for (unsigned int i=0; i<fe_eval_velocity.dofs_per_cell*dim; ++i)
          fe_eval_velocity.write_cellwise_dof_value(i,make_vectorized_array<Number>(0.));
        fe_eval_velocity.write_cellwise_dof_value(j,make_vectorized_array<Number>(1.));
        // set all dof values of element+ to zero
        for (unsigned int i=0; i<fe_eval_velocity_neighbor.dofs_per_cell*dim; ++i)
          fe_eval_velocity_neighbor.write_cellwise_dof_value(i, make_vectorized_array<Number>(0.));

        // copied from local_apply_viscous_face (note that fe_eval_neighbor.submit... has to be removed) //TODO
        fe_eval_velocity.evaluate(true,true);
        fe_eval_velocity_neighbor.evaluate(true,true);

        for(unsigned int q=0;q<fe_eval_velocity.n_q_points;++q)
        {
          Tensor<1,dim,VectorizedArray<Number> > uM = fe_eval_velocity.get_value(q);
          Tensor<1,dim,VectorizedArray<Number> > uP = fe_eval_velocity_neighbor.get_value(q);
          VectorizedArray<Number> average_viscosity = make_vectorized_array<Number>(const_viscosity);
          VectorizedArray<Number> max_viscosity = make_vectorized_array<Number>(const_viscosity);
          if(viscosity_is_variable())
          {
            // harmonic weighting according to Schott and Rasthofer et al (2015)
            average_viscosity = 2. * viscous_coefficient_face[face][q] * viscous_coefficient_face_neighbor[face][q] /
                               (viscous_coefficient_face[face][q] + viscous_coefficient_face_neighbor[face][q]);
            max_viscosity = average_viscosity;
          }

          Tensor<1,dim,VectorizedArray<Number> > jump_value = uM - uP;

          Tensor<2,dim,VectorizedArray<Number> > average_gradient_tensor;
          if(operator_data.formulation_viscous_term == FormulationViscousTerm::DivergenceFormulation)
          {
            // {{F}} = (F⁻ + F⁺)/2 where F = 2 * nu * symmetric_gradient -> nu * (symmetric_gradient⁻ + symmetric_gradient⁺)
            average_gradient_tensor = ( fe_eval_velocity.get_symmetric_gradient(q) + fe_eval_velocity_neighbor.get_symmetric_gradient(q));
          }
          else if(operator_data.formulation_viscous_term == FormulationViscousTerm::LaplaceFormulation)
          {
            average_gradient_tensor = ( fe_eval_velocity.get_gradient(q) + fe_eval_velocity_neighbor.get_gradient(q)) * make_vectorized_array<Number>(0.5);
          }
          else
          {
            AssertThrow(false, ExcMessage("FORMULATION_VISCOUS_TERM is not specified - possibilities are DivergenceFormulation and LaplaceFormulation"));
          }
          Tensor<2,dim,VectorizedArray<Number> > jump_tensor =
              outer_product(jump_value,fe_eval_velocity.get_normal_vector(q));

          //we do not want to symmetrize the penalty part
          average_gradient_tensor = average_viscosity*average_gradient_tensor - max_viscosity * jump_tensor * tau_IP;
          Tensor<1,dim,VectorizedArray<Number> > average_gradient = average_gradient_tensor*fe_eval_velocity.get_normal_vector(q);

          if(operator_data.formulation_viscous_term == FormulationViscousTerm::DivergenceFormulation)
          {
            if(operator_data.IP_formulation_viscous == InteriorPenaltyFormulation::NIPG)
            {
              fe_eval_velocity.submit_gradient(fe_eval_velocity.make_symmetric(average_viscosity*jump_tensor),q);
            }
            else if(operator_data.IP_formulation_viscous == InteriorPenaltyFormulation::SIPG)
            {
              fe_eval_velocity.submit_gradient(-fe_eval_velocity.make_symmetric(average_viscosity*jump_tensor),q);
            }
            else
              AssertThrow(false, ExcMessage("IP_FORMULATION_VISCOUS is not specified - possibilities are SIPG and NIPG"));
          }
          else if(operator_data.formulation_viscous_term == FormulationViscousTerm::LaplaceFormulation)
          {
            if(operator_data.IP_formulation_viscous == InteriorPenaltyFormulation::NIPG)
            {
              fe_eval_velocity.submit_gradient(0.5*average_viscosity*jump_tensor,q);
            }
            else if(operator_data.IP_formulation_viscous == InteriorPenaltyFormulation::SIPG)
            {
              fe_eval_velocity.submit_gradient(-0.5*average_viscosity*jump_tensor,q);
            }
            else
              AssertThrow(false, ExcMessage("IP_FORMULATION_VISCOUS is not specified - possibilities are SIPG and NIPG"));
          }
          else
          {
            AssertThrow(false, ExcMessage("FORMULATION_VISCOUS_TERM is not specified - possibilities are DivergenceFormulation and LaplaceFormulation"));
          }
          fe_eval_velocity.submit_value(-average_gradient,q);
        }
        // integrate on element-
        fe_eval_velocity.integrate(true,true);
        // copied from local_apply_viscous_face (note that fe_eval_neighbor.submit... has to be removed)

        local_diagonal_vector[j] = fe_eval_velocity.read_cellwise_dof_value(j);
      }
      for (unsigned int j=0; j<fe_eval_velocity.dofs_per_cell*dim; ++j)
        fe_eval_velocity.write_cellwise_dof_value(j, local_diagonal_vector[j]);

      fe_eval_velocity.distribute_local_to_global(dst);

      // neighbor (element+)
      VectorizedArray<Number> local_diagonal_vector_neighbor[fe_eval_velocity_neighbor.tensor_dofs_per_cell*dim];
      for (unsigned int j=0; j<fe_eval_velocity_neighbor.dofs_per_cell*dim; ++j)
      {
        // set all dof values of element- to zero
        for (unsigned int i=0; i<fe_eval_velocity.dofs_per_cell*dim; ++i)
          fe_eval_velocity.write_cellwise_dof_value(i,make_vectorized_array<Number>(0.));
        // set dof value j of element+ to 1 and all other dof values of element+ to zero
        for (unsigned int i=0; i<fe_eval_velocity_neighbor.dofs_per_cell*dim; ++i)
          fe_eval_velocity_neighbor.write_cellwise_dof_value(i, make_vectorized_array<Number>(0.));
        fe_eval_velocity_neighbor.write_cellwise_dof_value(j,make_vectorized_array<Number>(1.));

        // copied from local_apply_viscous_face (note that fe_eval.submit... has to be removed)//TODO
        fe_eval_velocity.evaluate(true,true);
        fe_eval_velocity_neighbor.evaluate(true,true);

        for(unsigned int q=0;q<fe_eval_velocity.n_q_points;++q)
        {
          Tensor<1,dim,VectorizedArray<Number> > uM = fe_eval_velocity.get_value(q);
          Tensor<1,dim,VectorizedArray<Number> > uP = fe_eval_velocity_neighbor.get_value(q);
          VectorizedArray<Number> average_viscosity = make_vectorized_array<Number>(const_viscosity);
          VectorizedArray<Number> max_viscosity = make_vectorized_array<Number>(const_viscosity);
          if(viscosity_is_variable())
          {
            // harmonic weighting according to Schott and Rasthofer et al (2015)
            average_viscosity = 2. * viscous_coefficient_face[face][q] * viscous_coefficient_face_neighbor[face][q] /
                               (viscous_coefficient_face[face][q] + viscous_coefficient_face_neighbor[face][q]);
            max_viscosity = average_viscosity;
          }

          Tensor<1,dim,VectorizedArray<Number> > jump_value = uM - uP;

          Tensor<2,dim,VectorizedArray<Number> > average_gradient_tensor;
          if(operator_data.formulation_viscous_term == FormulationViscousTerm::DivergenceFormulation)
          {
            // {{F}} = (F⁻ + F⁺)/2 where F = 2 * nu * symmetric_gradient -> nu * (symmetric_gradient⁻ + symmetric_gradient⁺)
            average_gradient_tensor = ( fe_eval_velocity.get_symmetric_gradient(q) + fe_eval_velocity_neighbor.get_symmetric_gradient(q));
          }
          else if(operator_data.formulation_viscous_term == FormulationViscousTerm::LaplaceFormulation)
          {
            average_gradient_tensor = ( fe_eval_velocity.get_gradient(q) + fe_eval_velocity_neighbor.get_gradient(q)) * make_vectorized_array<Number>(0.5);
          }
          else
          {
            AssertThrow(false, ExcMessage("FORMULATION_VISCOUS_TERM is not specified - possibilities are DivergenceFormulation and LaplaceFormulation"));
          }
          Tensor<2,dim,VectorizedArray<Number> > jump_tensor =
              outer_product(jump_value,fe_eval_velocity.get_normal_vector(q));

          //we do not want to symmetrize the penalty part
          average_gradient_tensor = average_viscosity*average_gradient_tensor - max_viscosity * jump_tensor * tau_IP;
          Tensor<1,dim,VectorizedArray<Number> > average_gradient = average_gradient_tensor*fe_eval_velocity.get_normal_vector(q);

          if(operator_data.formulation_viscous_term == FormulationViscousTerm::DivergenceFormulation)
          {
            if(operator_data.IP_formulation_viscous == InteriorPenaltyFormulation::NIPG)
            {
              fe_eval_velocity_neighbor.submit_gradient(fe_eval_velocity.make_symmetric(average_viscosity*jump_tensor),q);
            }
            else if(operator_data.IP_formulation_viscous == InteriorPenaltyFormulation::SIPG)
            {
              fe_eval_velocity_neighbor.submit_gradient(-fe_eval_velocity.make_symmetric(average_viscosity*jump_tensor),q);
            }
            else
              AssertThrow(false, ExcMessage("IP_FORMULATION_VISCOUS is not specified - possibilities are SIPG and NIPG"));
          }
          else if(operator_data.formulation_viscous_term == FormulationViscousTerm::LaplaceFormulation)
          {
            if(operator_data.IP_formulation_viscous == InteriorPenaltyFormulation::NIPG)
            {
              fe_eval_velocity_neighbor.submit_gradient(0.5*average_viscosity*jump_tensor,q);
            }
            else if(operator_data.IP_formulation_viscous == InteriorPenaltyFormulation::SIPG)
            {
              fe_eval_velocity_neighbor.submit_gradient(-0.5*average_viscosity*jump_tensor,q);
            }
            else
              AssertThrow(false, ExcMessage("IP_FORMULATION_VISCOUS is not specified - possibilities are SIPG and NIPG"));
          }
          else
          {
            AssertThrow(false, ExcMessage("FORMULATION_VISCOUS_TERM is not specified - possibilities are DivergenceFormulation and LaplaceFormulation"));
          }
          fe_eval_velocity_neighbor.submit_value(average_gradient,q);
        }
        // integrate on element+
        fe_eval_velocity_neighbor.integrate(true,true);
        // copied from local_apply_viscous_face  (note that fe_eval.submit... has to be removed)

        local_diagonal_vector_neighbor[j] = fe_eval_velocity_neighbor.read_cellwise_dof_value(j);
      }
      for (unsigned int j=0; j<fe_eval_velocity_neighbor.dofs_per_cell*dim; ++j)
        fe_eval_velocity_neighbor.write_cellwise_dof_value(j, local_diagonal_vector_neighbor[j]);
      fe_eval_velocity_neighbor.distribute_local_to_global(dst);
    }
  }

  void local_diagonal_boundary_face (const MatrixFree<dim,Number>                 &data,
                                     parallel::distributed::Vector<Number>        &dst,
                                     const parallel::distributed::Vector<Number>  &,
                                     const std::pair<unsigned int,unsigned int>   &face_range) const
  {
    FEFaceEval_Velocity_Velocity_linear fe_eval_velocity(data,this->fe_param,true,operator_data.dof_index);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval_velocity.reinit (face);

      VectorizedArray<Number> tau_IP = fe_eval_velocity.read_cell_data(array_penalty_parameter)
                                             * get_penalty_factor();

      typename std::map<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >::iterator it;
      types::boundary_id boundary_id = data.get_boundary_indicator(face);

      VectorizedArray<Number> local_diagonal_vector[fe_eval_velocity.tensor_dofs_per_cell*dim];
      for (unsigned int j=0; j<fe_eval_velocity.dofs_per_cell*dim; ++j)
      {
        // set dof value j of element- to 1 and all other dof values of element- to zero
        for (unsigned int i=0; i<fe_eval_velocity.dofs_per_cell*dim; ++i)
          fe_eval_velocity.write_cellwise_dof_value(i, make_vectorized_array<Number>(0.));
        fe_eval_velocity.write_cellwise_dof_value(j, make_vectorized_array<Number>(1.));

        // copied from local_apply_viscous_boundary_face TODO
        fe_eval_velocity.evaluate(true,true);

        for(unsigned int q=0;q<fe_eval_velocity.n_q_points;++q)
        {
          VectorizedArray<Number> viscosity = make_vectorized_array<Number>(const_viscosity);
          if(viscosity_is_variable())
            viscosity = viscous_coefficient_face[face][q];

          it = operator_data.bc->dirichlet_bc.find(boundary_id);
          if(it != operator_data.bc->dirichlet_bc.end())
          {
            // applying inhomogeneous Dirichlet BC (value+ = - value- + 2g , grad+ = grad-)
            Tensor<1,dim,VectorizedArray<Number> > uM = fe_eval_velocity.get_value(q);
            Tensor<1,dim,VectorizedArray<Number> > uP = -uM;
            Tensor<1,dim,VectorizedArray<Number> > jump_value = uM - uP;

            Tensor<2,dim,VectorizedArray<Number> > average_gradient_tensor;
            if(operator_data.formulation_viscous_term == FormulationViscousTerm::DivergenceFormulation)
            {
              average_gradient_tensor = make_vectorized_array<Number>(2.) * fe_eval_velocity.get_symmetric_gradient(q);
            }
            else if(operator_data.formulation_viscous_term == FormulationViscousTerm::LaplaceFormulation)
            {
              average_gradient_tensor = fe_eval_velocity.get_gradient(q);
            }
            else
            {
              AssertThrow(false, ExcMessage("FORMULATION_VISCOUS_TERM is not specified - possibilities are DivergenceFormulation and LaplaceFormulation"));
            }
            Tensor<2,dim,VectorizedArray<Number> > jump_tensor
              = outer_product(jump_value,fe_eval_velocity.get_normal_vector(q));

            //we do not want to symmetrize the penalty part
            average_gradient_tensor = viscosity*(average_gradient_tensor - jump_tensor * tau_IP);

            Tensor<1,dim,VectorizedArray<Number> > average_gradient = average_gradient_tensor*fe_eval_velocity.get_normal_vector(q);

            if(operator_data.formulation_viscous_term == FormulationViscousTerm::DivergenceFormulation)
            {
              if(operator_data.IP_formulation_viscous == InteriorPenaltyFormulation::NIPG)
              {
                fe_eval_velocity.submit_gradient(fe_eval_velocity.make_symmetric(viscosity*jump_tensor),q);
              }
              else if(operator_data.IP_formulation_viscous == InteriorPenaltyFormulation::SIPG)
              {
                fe_eval_velocity.submit_gradient(-fe_eval_velocity.make_symmetric( viscosity*jump_tensor),q);
              }
              else
                AssertThrow(false, ExcMessage("IP_FORMULATION_VISCOUS is not specified - possibilities are SIPG and NIPG"));
            }
            else if(operator_data.formulation_viscous_term == FormulationViscousTerm::LaplaceFormulation)
            {
              if(operator_data.IP_formulation_viscous == InteriorPenaltyFormulation::NIPG)
              {
                fe_eval_velocity.submit_gradient(0.5*viscosity*jump_tensor,q);
              }
              else if(operator_data.IP_formulation_viscous == InteriorPenaltyFormulation::SIPG)
              {
                fe_eval_velocity.submit_gradient(-0.5*viscosity*jump_tensor,q);
              }
              else
                AssertThrow(false, ExcMessage("IP_FORMULATION_VISCOUS is not specified - possibilities are SIPG and NIPG"));
            }
            else
            {
              AssertThrow(false, ExcMessage("Formulation of viscous term not specified - possibilities are DIVERGENCE_FORMULATION_VISCOUS and LAPLACE_FORMULATION_VISCOUS"));
            }
            fe_eval_velocity.submit_value(-average_gradient,q);
          }

          it = operator_data.bc->neumann_bc.find(boundary_id);
          if(it != operator_data.bc->neumann_bc.end())
          {
            // applying inhomogeneous Neumann BC (value+ = value- , grad+ =  - grad- +2h)
            Tensor<1,dim,VectorizedArray<Number> > jump_value;
            Tensor<1,dim,VectorizedArray<Number> > average_gradient;
            Tensor<2,dim,VectorizedArray<Number> > jump_tensor
              = outer_product(jump_value,fe_eval_velocity.get_normal_vector(q));

            if(operator_data.formulation_viscous_term == FormulationViscousTerm::DivergenceFormulation)
            {
              if(operator_data.IP_formulation_viscous == InteriorPenaltyFormulation::NIPG)
              {
                fe_eval_velocity.submit_gradient(fe_eval_velocity.make_symmetric(viscosity*jump_tensor),q);
              }
              else if(operator_data.IP_formulation_viscous == InteriorPenaltyFormulation::SIPG)
              {
                fe_eval_velocity.submit_gradient(-fe_eval_velocity.make_symmetric(viscosity*jump_tensor),q);
              }
              else
                AssertThrow(false, ExcMessage("IP_FORMULATION_VISCOUS is not specified - possibilities are SIPG and NIPG"));
            }
            else if(operator_data.formulation_viscous_term == FormulationViscousTerm::LaplaceFormulation)
            {
              if(operator_data.IP_formulation_viscous == InteriorPenaltyFormulation::NIPG)
              {
                fe_eval_velocity.submit_gradient(0.5*viscosity*jump_tensor,q);
              }
              else if(operator_data.IP_formulation_viscous == InteriorPenaltyFormulation::SIPG)
              {
                fe_eval_velocity.submit_gradient(-0.5*viscosity*jump_tensor,q);
              }
              else
                AssertThrow(false, ExcMessage("IP_FORMULATION_VISCOUS is not specified - possibilities are SIPG and NIPG"));
            }
            else
            {
              AssertThrow(false, ExcMessage("Formulation of viscous term not specified - possibilities are DIVERGENCE_FORMULATION_VISCOUS and LAPLACE_FORMULATION_VISCOUS"));
            }
            fe_eval_velocity.submit_value(-average_gradient,q);
          }
        }
        fe_eval_velocity.integrate(true,true);
        // copied from local_apply_viscous__boundary_face

        local_diagonal_vector[j] = fe_eval_velocity.read_cellwise_dof_value(j);
      }
      for (unsigned int j=0; j<fe_eval_velocity.dofs_per_cell*dim; ++j)
        fe_eval_velocity.write_cellwise_dof_value(j, local_diagonal_vector[j]);
      fe_eval_velocity.distribute_local_to_global(dst);
    }
  }

  void evaluate_viscous (parallel::distributed::Vector<Number>        &dst,
                         const parallel::distributed::Vector<Number>  &src) const
  {
    data->loop(&ViscousOperator<dim,fe_degree,fe_degree_xwall,xwall_quad_rule,Number>::local_apply_viscous,
               &ViscousOperator<dim,fe_degree,fe_degree_xwall,xwall_quad_rule,Number>::local_apply_viscous_face,
               &ViscousOperator<dim,fe_degree,fe_degree_xwall,xwall_quad_rule,Number>::local_evaluate_viscous_boundary_face,
               this, dst, src);
  }

  void local_evaluate_viscous_boundary_face (const MatrixFree<dim,Number>                 &data,
                                             parallel::distributed::Vector<Number>        &dst,
                                             const parallel::distributed::Vector<Number>  &src,
                                             const std::pair<unsigned int,unsigned int>   &face_range) const
  {
    FEFaceEval_Velocity_Velocity_linear fe_eval_velocity(data,this->fe_param,true,operator_data.dof_index);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval_velocity.reinit (face);
      fe_eval_velocity.read_dof_values(src);
      fe_eval_velocity.evaluate(true,true);

      VectorizedArray<Number> tau_IP = fe_eval_velocity.read_cell_data(array_penalty_parameter)
                                             * get_penalty_factor();

      typename std::map<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >::iterator it;
      types::boundary_id boundary_id = data.get_boundary_indicator(face);

      for(unsigned int q=0;q<fe_eval_velocity.n_q_points;++q)
      {
        VectorizedArray<Number> viscosity = make_vectorized_array<Number>(const_viscosity);
        if(viscosity_is_variable())
          viscosity = viscous_coefficient_face[face][q];

        it = operator_data.bc->dirichlet_bc.find(boundary_id);
        if(it != operator_data.bc->dirichlet_bc.end())
        {
          // applying inhomogeneous Dirichlet BC (value+ = - value- + 2g , grad+ = grad-)
          Tensor<1,dim,VectorizedArray<Number> > uM = fe_eval_velocity.get_value(q);

          Point<dim,VectorizedArray<Number> > q_points = fe_eval_velocity.quadrature_point(q);
          Tensor<1,dim,VectorizedArray<Number> > g;
          //set correct time for the evaluation of boundary conditions
          it->second->set_time(eval_time);
          for(unsigned int d=0;d<dim;++d)
          {
            Number array [VectorizedArray<Number>::n_array_elements];
            for (unsigned int n=0; n<VectorizedArray<Number>::n_array_elements; ++n)
            {
              Point<dim> q_point;
              for (unsigned int d=0; d<dim; ++d)
                q_point[d] = q_points[d][n];
              array[n] = it->second->value(q_point,d);
            }
            g[d].load(&array[0]);
          }

          Tensor<1,dim,VectorizedArray<Number> > uP = -uM + make_vectorized_array<Number>(2.) * g;
          Tensor<1,dim,VectorizedArray<Number> > jump_value = uM - uP;

          Tensor<2,dim,VectorizedArray<Number> > average_gradient_tensor;
          if(operator_data.formulation_viscous_term == FormulationViscousTerm::DivergenceFormulation)
          {
            average_gradient_tensor = make_vectorized_array<Number>(2.) * fe_eval_velocity.get_symmetric_gradient(q);
          }
          else if(operator_data.formulation_viscous_term == FormulationViscousTerm::LaplaceFormulation)
          {
            average_gradient_tensor = fe_eval_velocity.get_gradient(q);
          }
          else
          {
            AssertThrow(false, ExcMessage("FORMULATION_VISCOUS_TERM is not specified - possibilities are DivergenceFormulation and LaplaceFormulation"));
          }
          Tensor<2,dim,VectorizedArray<Number> > jump_tensor
            = outer_product(jump_value,fe_eval_velocity.get_normal_vector(q));

          //we do not want to symmetrize the penalty part
          average_gradient_tensor = viscosity*(average_gradient_tensor - jump_tensor * tau_IP);

          Tensor<1,dim,VectorizedArray<Number> > average_gradient = average_gradient_tensor*fe_eval_velocity.get_normal_vector(q);

          if(operator_data.formulation_viscous_term == FormulationViscousTerm::DivergenceFormulation)
          {
            if(operator_data.IP_formulation_viscous == InteriorPenaltyFormulation::NIPG)
            {
              fe_eval_velocity.submit_gradient(fe_eval_velocity.make_symmetric(viscosity*jump_tensor),q);
            }
            else if(operator_data.IP_formulation_viscous == InteriorPenaltyFormulation::SIPG)
            {
              fe_eval_velocity.submit_gradient(-fe_eval_velocity.make_symmetric(viscosity*jump_tensor),q);
            }
            else
              AssertThrow(false, ExcMessage("IP_FORMULATION_VISCOUS is not specified - possibilities are SIPG and NIPG"));
          }
          else if(operator_data.formulation_viscous_term == FormulationViscousTerm::LaplaceFormulation)
          {
            if(operator_data.IP_formulation_viscous == InteriorPenaltyFormulation::NIPG)
            {
              fe_eval_velocity.submit_gradient(0.5*viscosity*jump_tensor,q);
            }
            else if(operator_data.IP_formulation_viscous == InteriorPenaltyFormulation::SIPG)
            {
              fe_eval_velocity.submit_gradient(-0.5*viscosity*jump_tensor,q);
            }
            else
              AssertThrow(false, ExcMessage("IP_FORMULATION_VISCOUS is not specified - possibilities are SIPG and NIPG"));
          }
          else
          {
            AssertThrow(false, ExcMessage("Formulation of viscous term not specified - possibilities are DIVERGENCE_FORMULATION_VISCOUS and LAPLACE_FORMULATION_VISCOUS"));
          }
          fe_eval_velocity.submit_value(-average_gradient,q);
        }

        it = operator_data.bc->neumann_bc.find(boundary_id);
        if(it != operator_data.bc->neumann_bc.end())
        {
          // applying inhomogeneous Neumann BC (value+ = value- , grad+ =  - grad- +2h)
          Tensor<1,dim,VectorizedArray<Number> > jump_value;

          Tensor<2,dim,VectorizedArray<Number> > jump_tensor
            = outer_product(jump_value,fe_eval_velocity.get_normal_vector(q));

          Point<dim,VectorizedArray<Number> > q_points = fe_eval_velocity.quadrature_point(q);
          Tensor<1,dim,VectorizedArray<Number> > h;
          // set correct time for the evaluation of boundary conditions
          it->second->set_time(eval_time);
          for(unsigned int d=0;d<dim;++d)
          {
            Number array [VectorizedArray<Number>::n_array_elements];
            for (unsigned int n=0; n<VectorizedArray<Number>::n_array_elements; ++n)
            {
              Point<dim> q_point;
              for (unsigned int d=0; d<dim; ++d)
                q_point[d] = q_points[d][n];
              array[n] = it->second->value(q_point,d);
            }
            h[d].load(&array[0]);
          }

          Tensor<1,dim,VectorizedArray<Number> > average_gradient = viscosity*h;

          if(operator_data.formulation_viscous_term == FormulationViscousTerm::DivergenceFormulation)
          {
            if(operator_data.IP_formulation_viscous == InteriorPenaltyFormulation::NIPG)
            {
              fe_eval_velocity.submit_gradient(fe_eval_velocity.make_symmetric(viscosity*jump_tensor),q);
            }
            else if(operator_data.IP_formulation_viscous == InteriorPenaltyFormulation::SIPG)
            {
              fe_eval_velocity.submit_gradient(-fe_eval_velocity.make_symmetric(viscosity*jump_tensor),q);
            }
            else
              AssertThrow(false, ExcMessage("IP_FORMULATION_VISCOUS is not specified - possibilities are SIPG and NIPG"));
          }
          else if(operator_data.formulation_viscous_term == FormulationViscousTerm::LaplaceFormulation)
          {
            if(operator_data.IP_formulation_viscous == InteriorPenaltyFormulation::NIPG)
            {
              fe_eval_velocity.submit_gradient(0.5*viscosity*jump_tensor,q);
            }
            else if(operator_data.IP_formulation_viscous == InteriorPenaltyFormulation::SIPG)
            {
              fe_eval_velocity.submit_gradient(-0.5*viscosity*jump_tensor,q);
            }
            else
              AssertThrow(false, ExcMessage("IP_FORMULATION_VISCOUS is not specified - possibilities are SIPG and NIPG"));
          }
          else
          {
            AssertThrow(false, ExcMessage("Formulation of viscous term not specified - possibilities are DIVERGENCE_FORMULATION_VISCOUS and LAPLACE_FORMULATION_VISCOUS"));
          }
          fe_eval_velocity.submit_value(-average_gradient,q);
        }
      }
      fe_eval_velocity.integrate(true,true);
      fe_eval_velocity.distribute_local_to_global(dst);
    }
  }

  void local_rhs_viscous (const MatrixFree<dim,Number>                 &,
                          parallel::distributed::Vector<Number>        &,
                          const parallel::distributed::Vector<Number>  &,
                          const std::pair<unsigned int,unsigned int>   &) const
  {

  }

  void local_rhs_viscous_face (const MatrixFree<dim,Number>                 &,
                               parallel::distributed::Vector<Number>        &,
                               const parallel::distributed::Vector<Number>  &,
                               const std::pair<unsigned int,unsigned int>   &) const
  {

  }

  void local_rhs_viscous_boundary_face (const MatrixFree<dim,Number>                &data,
                                        parallel::distributed::Vector<Number>       &dst,
                                        const parallel::distributed::Vector<Number> &,
                                        const std::pair<unsigned int,unsigned int>  &face_range) const
  {
    FEFaceEval_Velocity_Velocity_linear fe_eval_velocity(data,this->fe_param,true,operator_data.dof_index);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval_velocity.reinit (face);

      VectorizedArray<Number> tau_IP = fe_eval_velocity.read_cell_data(array_penalty_parameter) * get_penalty_factor();

      typename std::map<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >::iterator it;
      types::boundary_id boundary_id = data.get_boundary_indicator(face);

      for(unsigned int q=0;q<fe_eval_velocity.n_q_points;++q)
      {
        VectorizedArray<Number> viscosity = make_vectorized_array<Number>(const_viscosity);
        if(viscosity_is_variable())
          viscosity = viscous_coefficient_face[face][q];

        it = operator_data.bc->dirichlet_bc.find(boundary_id);
        if(it != operator_data.bc->dirichlet_bc.end())
        {
          // applying inhomogeneous Dirichlet BC (value+ = - value- + 2g , grad+ = grad-)
          Point<dim,VectorizedArray<Number> > q_points = fe_eval_velocity.quadrature_point(q);
          Tensor<1,dim,VectorizedArray<Number> > g;

          // set correct time for the evaluation of boundary conditions
          it->second->set_time(eval_time);
          for(unsigned int d=0;d<dim;++d)
          {
            Number array [VectorizedArray<Number>::n_array_elements];
            for (unsigned int n=0; n<VectorizedArray<Number>::n_array_elements; ++n)
            {
              Point<dim> q_point;
              for (unsigned int d=0; d<dim; ++d)
              q_point[d] = q_points[d][n];
              array[n] = it->second->value(q_point,d);
            }
            g[d].load(&array[0]);
          }

          Tensor<2,dim,VectorizedArray<Number> > jump_tensor
            = outer_product(2.*g,fe_eval_velocity.get_normal_vector(q));

          Tensor<2,dim,VectorizedArray<Number> > average_gradient_tensor = -viscosity * tau_IP * jump_tensor;
          Tensor<1,dim,VectorizedArray<Number> > average_gradient = average_gradient_tensor*fe_eval_velocity.get_normal_vector(q);

          if(operator_data.formulation_viscous_term == FormulationViscousTerm::DivergenceFormulation)
          {
            if(operator_data.IP_formulation_viscous == InteriorPenaltyFormulation::NIPG)
              fe_eval_velocity.submit_gradient(fe_eval_velocity.make_symmetric(viscosity*jump_tensor),q);
            else if(operator_data.IP_formulation_viscous == InteriorPenaltyFormulation::SIPG)
              fe_eval_velocity.submit_gradient(-fe_eval_velocity.make_symmetric(viscosity*jump_tensor),q);
            else
              AssertThrow(false, ExcMessage("IP_FORMULATION_VISCOUS is not specified - possibilities are SIPG and NIPG"));
          }
          else if(operator_data.formulation_viscous_term == FormulationViscousTerm::LaplaceFormulation)
          {
            if(operator_data.IP_formulation_viscous == InteriorPenaltyFormulation::NIPG)
              fe_eval_velocity.submit_gradient(0.5*viscosity*jump_tensor,q);
            else if(operator_data.IP_formulation_viscous == InteriorPenaltyFormulation::SIPG)
              fe_eval_velocity.submit_gradient(-0.5*viscosity*jump_tensor,q);
            else
              AssertThrow(false, ExcMessage("IP_FORMULATION_VISCOUS is not specified - possibilities are SIPG and NIPG"));
          }
          else
          {
            AssertThrow(false, ExcMessage("FORMULATION_VISCOUS_TERM is not specified - possibilities are DivergenceFormulation and LaplaceFormulation"));
          }
          fe_eval_velocity.submit_value(-average_gradient,q);

        }

        it = operator_data.bc->neumann_bc.find(boundary_id);
        if(it != operator_data.bc->neumann_bc.end())
        {
          // applying inhomogeneous Neumann BC (value+ = value- , grad+ = - grad- +2h)
          Point<dim,VectorizedArray<Number> > q_points = fe_eval_velocity.quadrature_point(q);
          Tensor<1,dim,VectorizedArray<Number> > h;

          // set correct time for the evaluation of boundary conditions
          it->second->set_time(eval_time);
          for(unsigned int d=0;d<dim;++d)
          {
            Number array [VectorizedArray<Number>::n_array_elements];
            for (unsigned int n=0; n<VectorizedArray<Number>::n_array_elements; ++n)
            {
              Point<dim> q_point;
              for (unsigned int d=0; d<dim; ++d)
                q_point[d] = q_points[d][n];
              array[n] = it->second->value(q_point,d);
            }
            h[d].load(&array[0]);
          }
          Tensor<1,dim,VectorizedArray<Number> > jump_value;

          Tensor<2,dim,VectorizedArray<Number> > jump_tensor
            = outer_product(jump_value,fe_eval_velocity.get_normal_vector(q));

          Tensor<1,dim,VectorizedArray<Number> > average_gradient = -viscosity*h;

          if(operator_data.formulation_viscous_term == FormulationViscousTerm::DivergenceFormulation)
          {
            if(operator_data.IP_formulation_viscous == InteriorPenaltyFormulation::NIPG)
              fe_eval_velocity.submit_gradient(fe_eval_velocity.make_symmetric(viscosity*jump_tensor),q);
            else if(operator_data.IP_formulation_viscous == InteriorPenaltyFormulation::SIPG)
              fe_eval_velocity.submit_gradient(-fe_eval_velocity.make_symmetric(viscosity*jump_tensor),q);
            else
              AssertThrow(false, ExcMessage("IP_FORMULATION_VISCOUS is not specified - possibilities are SIPG and NIPG"));
          }
          else if(operator_data.formulation_viscous_term == FormulationViscousTerm::LaplaceFormulation)
          {
            if(operator_data.IP_formulation_viscous == InteriorPenaltyFormulation::NIPG)
              fe_eval_velocity.submit_gradient(0.5*viscosity*jump_tensor,q);
            else if(operator_data.IP_formulation_viscous == InteriorPenaltyFormulation::SIPG)
              fe_eval_velocity.submit_gradient(-0.5*viscosity*jump_tensor,q);
            else
              AssertThrow(false, ExcMessage("IP_FORMULATION_VISCOUS is not specified - possibilities are SIPG and NIPG"));
          }
          else
          {
            AssertThrow(false, ExcMessage("FORMULATION_VISCOUS_TERM is not specified - possibilities are DivergenceFormulation and LaplaceFormulation"));
          }
          fe_eval_velocity.submit_value(-average_gradient,q);
        }
      }

      fe_eval_velocity.integrate(true,true);
      fe_eval_velocity.distribute_local_to_global(dst);
    }
  }
protected:
  MatrixFree<dim,Number> const * data;
  ViscousOperatorData<dim> operator_data;
private:
  AlignedVector<VectorizedArray<Number> > array_penalty_parameter;
  Number const_viscosity;
  Table<2,VectorizedArray<Number> > viscous_coefficient_cell;
  Table<2,VectorizedArray<Number> > viscous_coefficient_face;
  Table<2,VectorizedArray<Number> > viscous_coefficient_face_neighbor;
  Number mutable eval_time;
};

template<int dim>
struct GradientOperatorData
{
  GradientOperatorData ()
    :
    dof_index_velocity(0),
    dof_index_pressure(1),
    integration_by_parts_of_gradP(true),
    use_boundary_data(true)
  {}

  unsigned int dof_index_velocity;
  unsigned int dof_index_pressure;
  bool integration_by_parts_of_gradP;
  bool use_boundary_data;

  std_cxx11::shared_ptr<BoundaryDescriptorNavierStokes<dim> > bc;
};

template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename value_type>
class GradientOperator: public BaseOperator<dim>
{
public:
  GradientOperator()
    :
    data(nullptr),
    eval_time(0.0)
  {}

  static const bool is_xwall = (xwall_quad_rule>1) ? true : false;
  static const unsigned int n_actual_q_points_vel_linear = (is_xwall) ? xwall_quad_rule : fe_degree+1;
  typedef FEEvaluationWrapper<dim,fe_degree,fe_degree_xwall,n_actual_q_points_vel_linear,dim,value_type,is_xwall> FEEval_Velocity_Velocity_linear;
  typedef FEEvaluationWrapperPressure<dim,fe_degree_p,fe_degree_xwall,n_actual_q_points_vel_linear,1,value_type,is_xwall> FEEval_Pressure_Velocity_linear;

  typedef FEFaceEvaluationWrapper<dim,fe_degree,fe_degree_xwall,n_actual_q_points_vel_linear,dim,value_type,is_xwall> FEFaceEval_Velocity_Velocity_linear;
  typedef FEFaceEvaluationWrapperPressure<dim,fe_degree_p,fe_degree_xwall,n_actual_q_points_vel_linear,1,value_type,is_xwall> FEFaceEval_Pressure_Velocity_linear;

  void initialize(MatrixFree<dim,value_type> const &mf_data,
                  GradientOperatorData<dim> const  &operator_data_in)
  {
    this->data = &mf_data;
    this->operator_data = operator_data_in;
  }

  void apply (parallel::distributed::Vector<value_type>       &dst,
              const parallel::distributed::Vector<value_type> &src) const
  {
    dst = 0;
    apply_add(dst,src);
  }

  void apply_add (parallel::distributed::Vector<value_type>       &dst,
                  const parallel::distributed::Vector<value_type> &src) const
  {
    apply_gradient(dst,src);
  }

  void rhs (parallel::distributed::Vector<value_type> &dst,
            value_type const                          evaluation_time) const
  {
    dst = 0;
    rhs_add(dst,evaluation_time);
  }

  void rhs_add (parallel::distributed::Vector<value_type>  &dst,
                value_type const                           evaluation_time) const
  {
    this->eval_time = evaluation_time;
    rhs_gradient(dst);
  }

  void evaluate (parallel::distributed::Vector<value_type>        &dst,
                 const parallel::distributed::Vector<value_type>  &src,
                 value_type const                                 evaluation_time) const
  {
    dst = 0;
    evaluate_add(dst,src,evaluation_time);
  }

  void evaluate_add (parallel::distributed::Vector<value_type>       &dst,
                     const parallel::distributed::Vector<value_type> &src,
                     value_type const                                evaluation_time) const
  {
    this->eval_time = evaluation_time;
    evaluate_gradient(dst,src);
  }

private:
  void apply_gradient (parallel::distributed::Vector<value_type>        &dst,
                       const parallel::distributed::Vector<value_type>  &src) const
  {
    data->loop (&GradientOperator<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type>::local_apply_gradient,
                &GradientOperator<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type>::local_apply_gradient_face,
                &GradientOperator<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type>::local_apply_gradient_boundary_face,
                this, dst, src);
  }

  void rhs_gradient (parallel::distributed::Vector<value_type> &dst) const
  {
    parallel::distributed::Vector<value_type> src;
    data->loop (&GradientOperator<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type>::local_rhs_gradient,
                &GradientOperator<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type>::local_rhs_gradient_face,
                &GradientOperator<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type>::local_rhs_gradient_boundary_face,
                this, dst, src);
  }

  void evaluate_gradient (parallel::distributed::Vector<value_type>       &dst,
                          const parallel::distributed::Vector<value_type> &src) const
  {
    data->loop (&GradientOperator<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type>::local_apply_gradient,
                &GradientOperator<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type>::local_apply_gradient_face,
                &GradientOperator<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type>::local_evaluate_gradient_boundary_face,
                this, dst, src);
  }

  void local_apply_gradient (const MatrixFree<dim,value_type>                 &data,
                             parallel::distributed::Vector<value_type>        &dst,
                             const parallel::distributed::Vector<value_type>  &src,
                             const std::pair<unsigned int,unsigned int>       &cell_range) const
  {
    FEEval_Velocity_Velocity_linear fe_eval_velocity(data,this->fe_param,operator_data.dof_index_velocity);
    FEEval_Pressure_Velocity_linear fe_eval_pressure(data,this->fe_param,operator_data.dof_index_pressure);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval_velocity.reinit (cell);
      fe_eval_pressure.reinit (cell);
      fe_eval_pressure.read_dof_values(src);

      if(operator_data.integration_by_parts_of_gradP == true)
      {
        fe_eval_pressure.evaluate (true,false);
        for (unsigned int q=0; q<fe_eval_velocity.n_q_points; ++q)
        {
          Tensor<2,dim,VectorizedArray<value_type> > unit_times_p;
          VectorizedArray<value_type> p = fe_eval_pressure.get_value(q);
          for (unsigned int d=0;d<dim;++d)
            unit_times_p[d][d] = p;

          fe_eval_velocity.submit_gradient (-unit_times_p, q);
        }
        fe_eval_velocity.integrate (false,true);
      }
      else // integration_by_parts_of_gradP == false
      {
        fe_eval_pressure.evaluate (false,true);
        for (unsigned int q=0; q<fe_eval_velocity.n_q_points; ++q)
        {
          fe_eval_velocity.submit_value(fe_eval_pressure.get_gradient(q),q);
        }
        fe_eval_velocity.integrate (true,false);
      }
      fe_eval_velocity.distribute_local_to_global (dst);
    }
  }

  void local_apply_gradient_face (const MatrixFree<dim,value_type>                 &data,
                                  parallel::distributed::Vector<value_type>        &dst,
                                  const parallel::distributed::Vector<value_type>  &src,
                                  const std::pair<unsigned int,unsigned int>       &face_range) const
  {
    if(operator_data.integration_by_parts_of_gradP == true)
    {
      FEFaceEval_Velocity_Velocity_linear fe_eval_velocity(data,this->fe_param,true,operator_data.dof_index_velocity);
      FEFaceEval_Velocity_Velocity_linear fe_eval_velocity_neighbor(data,this->fe_param,false,operator_data.dof_index_velocity);

      FEFaceEval_Pressure_Velocity_linear fe_eval_pressure(data,this->fe_param,true,operator_data.dof_index_pressure);
      FEFaceEval_Pressure_Velocity_linear fe_eval_pressure_neighbor(data,this->fe_param,false,operator_data.dof_index_pressure);

      for(unsigned int face=face_range.first; face<face_range.second; face++)
      {
        fe_eval_velocity.reinit (face);
        fe_eval_velocity_neighbor.reinit (face);
        fe_eval_pressure.reinit (face);
        fe_eval_pressure_neighbor.reinit (face);
        fe_eval_pressure.read_dof_values(src);
        fe_eval_pressure_neighbor.read_dof_values(src);
        fe_eval_pressure.evaluate (true,false);
        fe_eval_pressure_neighbor.evaluate (true,false);
        for (unsigned int q=0; q<fe_eval_velocity.n_q_points; ++q)
        {
          Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval_pressure.get_normal_vector(q);
          VectorizedArray<value_type> meanpres = 0.5*(fe_eval_pressure.get_value(q)+fe_eval_pressure_neighbor.get_value(q));

          normal *= meanpres;

          fe_eval_velocity.submit_value (normal, q);
          fe_eval_velocity_neighbor.submit_value (-normal, q);
        }
        fe_eval_velocity.integrate (true,false);
        fe_eval_velocity_neighbor.integrate (true,false);
        fe_eval_velocity.distribute_local_to_global (dst);
        fe_eval_velocity_neighbor.distribute_local_to_global (dst);
      }
    }
  }

  void local_apply_gradient_boundary_face (const MatrixFree<dim,value_type>                &data,
                                           parallel::distributed::Vector<value_type>       &dst,
                                           const parallel::distributed::Vector<value_type> &src,
                                           const std::pair<unsigned int,unsigned int>      &face_range) const
  {
    if(operator_data.integration_by_parts_of_gradP == true)
    {
      FEFaceEval_Velocity_Velocity_linear fe_eval_velocity(data,this->fe_param,true,operator_data.dof_index_velocity);
      FEFaceEval_Pressure_Velocity_linear fe_eval_pressure(data,this->fe_param,true,operator_data.dof_index_pressure);

      for(unsigned int face=face_range.first; face<face_range.second; face++)
      {
        fe_eval_velocity.reinit (face);
        fe_eval_pressure.reinit (face);
        fe_eval_pressure.read_dof_values(src);
        fe_eval_pressure.evaluate (true,false);

        typename std::map<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >::iterator it;
        types::boundary_id boundary_id = data.get_boundary_indicator(face);

        for (unsigned int q=0; q<fe_eval_velocity.n_q_points; ++q)
        {
          if(operator_data.use_boundary_data == true)
          {
            it = operator_data.bc->dirichlet_bc.find(boundary_id);
            if(it != operator_data.bc->dirichlet_bc.end())
            {
              // on GammaD: p⁺ =  p⁻ -> {{p}} = p⁻
              // homogeneous part: p⁺ = p⁻ -> {{p}} = p⁻
              // inhomongenous part: p⁺ = 0 -> {{p}} = 0
              Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval_pressure.get_normal_vector(q);
              VectorizedArray<value_type> meanpres = fe_eval_pressure.get_value(q);
              normal *= meanpres;
              fe_eval_velocity.submit_value (normal, q);
            }

            it = operator_data.bc->neumann_bc.find(boundary_id);
            if(it != operator_data.bc->neumann_bc.end())
            {
              // on GammaN: p⁺ = - p⁻ + 2g -> {{p}} = g
              // homogeneous part: p⁺ = - p⁻ -> {{p}} = 0
              // inhomongenous part: p⁺ = 2g -> {{p}} = g
              Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval_pressure.get_normal_vector(q);
              VectorizedArray<value_type> meanpres = make_vectorized_array<value_type>(0.0);
              normal *= meanpres;
              fe_eval_velocity.submit_value (normal, q);
            }
          }
          else // use_boundary_data == false
          {
            // use p⁺ = p⁻ on all boundaries
            Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval_pressure.get_normal_vector(q);
            VectorizedArray<value_type> meanpres = fe_eval_pressure.get_value(q);
            normal *= meanpres;
            fe_eval_velocity.submit_value (normal, q);
          }
        }
        fe_eval_velocity.integrate (true,false);
        fe_eval_velocity.distribute_local_to_global (dst);
      }
    }
  }

  void local_evaluate_gradient_boundary_face (const MatrixFree<dim,value_type>                &data,
                                              parallel::distributed::Vector<value_type>       &dst,
                                              const parallel::distributed::Vector<value_type> &src,
                                              const std::pair<unsigned int,unsigned int>      &face_range) const
  {
    if(operator_data.integration_by_parts_of_gradP == true)
    {
      FEFaceEval_Velocity_Velocity_linear fe_eval_velocity(data,this->fe_param,true,operator_data.dof_index_velocity);
      FEFaceEval_Pressure_Velocity_linear fe_eval_pressure(data,this->fe_param,true,operator_data.dof_index_pressure);

      for(unsigned int face=face_range.first; face<face_range.second; face++)
      {
        fe_eval_velocity.reinit (face);
        fe_eval_pressure.reinit (face);
        fe_eval_pressure.read_dof_values(src);
        fe_eval_pressure.evaluate (true,false);

        typename std::map<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >::iterator it;
        types::boundary_id boundary_id = data.get_boundary_indicator(face);

        for (unsigned int q=0; q<fe_eval_velocity.n_q_points; ++q)
        {
          if(operator_data.use_boundary_data == true)
          {
            it = operator_data.bc->dirichlet_bc.find(boundary_id);
            if(it != operator_data.bc->dirichlet_bc.end())
            {
              // on GammaD: p⁺ =  p⁻ -> {{p}} = p⁻
              // homogeneous part: p⁺ = p⁻ -> {{p}} = p⁻
              // inhomongenous part: p⁺ = 0 -> {{p}} = 0
              Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval_pressure.get_normal_vector(q);
              VectorizedArray<value_type> meanpres = fe_eval_pressure.get_value(q);
              normal *= meanpres;
              fe_eval_velocity.submit_value (normal, q);
            }

            it = operator_data.bc->neumann_bc.find(boundary_id);
            if(it != operator_data.bc->neumann_bc.end())
            {
              // on GammaN: p⁺ = - p⁻ + 2g -> {{p}} = g
              // homogeneous part: p⁺ = - p⁻ -> {{p}} = 0
              // inhomongenous part: p⁺ = 2g -> {{p}} = g
              Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval_pressure.get_normal_vector(q);

              Point<dim,VectorizedArray<value_type> > q_points = fe_eval_pressure.quadrature_point(q);
              VectorizedArray<value_type> meanpres;
              // set correct time for the evaluation of boundary conditions
              it->second->set_time(eval_time);
              value_type array [VectorizedArray<value_type>::n_array_elements];
              for (unsigned int n=0; n<VectorizedArray<value_type>::n_array_elements; ++n)
              {
                Point<dim> q_point;
                for (unsigned int d=0; d<dim; ++d)
                  q_point[d] = q_points[d][n];
                array[n] = it->second->value(q_point);
              }
              meanpres.load(&array[0]);

              normal *= meanpres;
              fe_eval_velocity.submit_value (normal, q);
            }
          }
          else // use_boundary_data == false
          {
            // use p⁺ = p⁻ on all boundaries
            Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval_pressure.get_normal_vector(q);
            VectorizedArray<value_type> meanpres = fe_eval_pressure.get_value(q);
            normal *= meanpres;
            fe_eval_velocity.submit_value (normal, q);
          }
        }
        fe_eval_velocity.integrate (true,false);
        fe_eval_velocity.distribute_local_to_global (dst);
      }
    }
  }

  void local_rhs_gradient (const MatrixFree<dim,value_type>                 &,
                           parallel::distributed::Vector<value_type>        &,
                           const parallel::distributed::Vector<value_type>  &,
                           const std::pair<unsigned int,unsigned int>       &) const
  {

  }

  void local_rhs_gradient_face (const MatrixFree<dim,value_type>                 &,
                                parallel::distributed::Vector<value_type>        &,
                                const parallel::distributed::Vector<value_type>  &,
                                const std::pair<unsigned int,unsigned int>       &) const
  {

  }

  void local_rhs_gradient_boundary_face (const MatrixFree<dim,value_type>                &data,
                                         parallel::distributed::Vector<value_type>       &dst,
                                         const parallel::distributed::Vector<value_type> &,
                                         const std::pair<unsigned int,unsigned int>      &face_range) const
  {
    if(operator_data.integration_by_parts_of_gradP == true &&
       operator_data.use_boundary_data == true)
    {
      FEFaceEval_Velocity_Velocity_linear fe_eval_velocity(data,this->fe_param,true,operator_data.dof_index_velocity);

      for(unsigned int face=face_range.first; face<face_range.second; face++)
      {
        fe_eval_velocity.reinit (face);

        typename std::map<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >::iterator it;
        types::boundary_id boundary_id = data.get_boundary_indicator(face);

        for (unsigned int q=0; q<fe_eval_velocity.n_q_points; ++q)
        {
          it = operator_data.bc->dirichlet_bc.find(boundary_id);
          if(it != operator_data.bc->dirichlet_bc.end())
          {
            // on GammaD: p⁺ =  p⁻ -> {{p}} = p⁻
            // homogeneous part: p⁺ = p⁻ -> {{p}} = p⁻
            // inhomongenous part: p⁺ = 0 -> {{p}} = 0
            Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval_velocity.get_normal_vector(q);
            VectorizedArray<value_type> meanpres = make_vectorized_array<value_type>(0.0);
            normal *= meanpres;
            fe_eval_velocity.submit_value (-normal, q); // minus sign since this term occurs on the rhs of the equation
          }

          it = operator_data.bc->neumann_bc.find(boundary_id);
          if(it != operator_data.bc->neumann_bc.end())
          {
            // on GammaN: p⁺ = - p⁻ + 2g -> {{p}} = g
            // homogeneous part: p⁺ = - p⁻ -> {{p}} = 0
            // inhomongenous part: p⁺ = 2g -> {{p}} = g
            Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval_velocity.get_normal_vector(q);

            Point<dim,VectorizedArray<value_type> > q_points = fe_eval_velocity.quadrature_point(q);
            VectorizedArray<value_type> meanpres;
            // set correct time for the evaluation of boundary conditions
            it->second->set_time(eval_time);
            value_type array [VectorizedArray<value_type>::n_array_elements];
            for (unsigned int n=0; n<VectorizedArray<value_type>::n_array_elements; ++n)
            {
              Point<dim> q_point;
              for (unsigned int d=0; d<dim; ++d)
                q_point[d] = q_points[d][n];
              array[n] = it->second->value(q_point);
            }
            meanpres.load(&array[0]);

            normal *= meanpres;
            fe_eval_velocity.submit_value (-normal, q); // minus sign since this term occurs on the rhs of the equation
          }
        }
        fe_eval_velocity.integrate (true,false);
        fe_eval_velocity.distribute_local_to_global (dst);
      }
    }
  }

  MatrixFree<dim,value_type> const * data;
  GradientOperatorData<dim> operator_data;
  value_type mutable eval_time;
};

template<int dim>
struct DivergenceOperatorData
{
  DivergenceOperatorData ()
    :
    dof_index_velocity(0),
    dof_index_pressure(1),
    integration_by_parts_of_divU(true),
    use_boundary_data(true)
  {}

  unsigned int dof_index_velocity;
  unsigned int dof_index_pressure;
  bool integration_by_parts_of_divU;
  bool use_boundary_data;

  std_cxx11::shared_ptr<BoundaryDescriptorNavierStokes<dim> > bc;
};

template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename value_type>
class DivergenceOperator: public BaseOperator<dim>
{
public:
  DivergenceOperator()
    :
    data(nullptr),
    eval_time(0.0)
  {}

  static const bool is_xwall = (xwall_quad_rule>1) ? true : false;
  static const unsigned int n_actual_q_points_vel_linear = (is_xwall) ? xwall_quad_rule : fe_degree+1;

  typedef FEEvaluationWrapper<dim,fe_degree,fe_degree_xwall,n_actual_q_points_vel_linear,dim,value_type,is_xwall> FEEval_Velocity_Velocity_linear;
  typedef FEEvaluationWrapperPressure<dim,fe_degree_p,fe_degree_xwall,n_actual_q_points_vel_linear,1,value_type,is_xwall> FEEval_Pressure_Velocity_linear;

  typedef FEFaceEvaluationWrapper<dim,fe_degree,fe_degree_xwall,n_actual_q_points_vel_linear,dim,value_type,is_xwall> FEFaceEval_Velocity_Velocity_linear;
  typedef FEFaceEvaluationWrapperPressure<dim,fe_degree_p,fe_degree_xwall,n_actual_q_points_vel_linear,1,value_type,is_xwall> FEFaceEval_Pressure_Velocity_linear;

  void initialize(MatrixFree<dim,value_type> const  &mf_data,
                  DivergenceOperatorData<dim> const &operator_data_in)
  {
    this->data = &mf_data;
    this->operator_data = operator_data_in;
  }

  void apply (parallel::distributed::Vector<value_type>       &dst,
              const parallel::distributed::Vector<value_type> &src) const
  {
    dst = 0;
    apply_add(dst,src);
  }

  void apply_add (parallel::distributed::Vector<value_type>       &dst,
                  const parallel::distributed::Vector<value_type> &src) const
  {
    apply_divergence(dst,src);
  }

  void rhs (parallel::distributed::Vector<value_type> &dst,
            const value_type                          evaluation_time) const
  {
    dst = 0;
    rhs_add(dst,evaluation_time);
  }

  void rhs_add (parallel::distributed::Vector<value_type> &dst,
                const value_type                          evaluation_time) const
  {
    this->eval_time = evaluation_time;
    rhs_divergence(dst);
  }

  void evaluate (parallel::distributed::Vector<value_type>       &dst,
                 const parallel::distributed::Vector<value_type> &src,
                 const value_type                                evaluation_time) const
  {
    dst = 0;
    evaluate_add(dst,src,evaluation_time);
  }

  void evaluate_add (parallel::distributed::Vector<value_type>        &dst,
                     const parallel::distributed::Vector<value_type>  &src,
                     const value_type                                 evaluation_time) const
  {
    this->eval_time = evaluation_time;
    evaluate_divergence(dst,src);
  }

private:
  void apply_divergence (parallel::distributed::Vector<value_type>      &dst,
                        const parallel::distributed::Vector<value_type> &src) const
  {
    data->loop (&DivergenceOperator<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type>::local_apply_divergence,
                &DivergenceOperator<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type>::local_apply_divergence_face,
                &DivergenceOperator<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type>::local_apply_divergence_boundary_face,
                this, dst, src);
  }

  void rhs_divergence (parallel::distributed::Vector<value_type> &dst) const
  {
    parallel::distributed::Vector<value_type> src;
    data->loop (&DivergenceOperator<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type>::local_rhs_divergence,
                &DivergenceOperator<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type>::local_rhs_divergence_face,
                &DivergenceOperator<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type>::local_rhs_divergence_boundary_face,
                this, dst, src);
  }

  void evaluate_divergence (parallel::distributed::Vector<value_type>       &dst,
                            const parallel::distributed::Vector<value_type> &src) const
  {
    data->loop (&DivergenceOperator<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type>::local_apply_divergence,
                &DivergenceOperator<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type>::local_apply_divergence_face,
                &DivergenceOperator<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type>::local_evaluate_divergence_boundary_face,
                this, dst, src);
  }

  void local_apply_divergence (const MatrixFree<dim,value_type>                &data,
                               parallel::distributed::Vector<value_type>       &dst,
                               const parallel::distributed::Vector<value_type> &src,
                               const std::pair<unsigned int,unsigned int>      &cell_range) const
  {
    FEEval_Velocity_Velocity_linear fe_eval_velocity(data,this->fe_param,operator_data.dof_index_velocity);
    FEEval_Pressure_Velocity_linear fe_eval_pressure(data,this->fe_param,operator_data.dof_index_pressure);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval_pressure.reinit (cell);

      fe_eval_velocity.reinit (cell);
      fe_eval_velocity.read_dof_values(src);

      if(operator_data.integration_by_parts_of_divU == true)
      {
        fe_eval_velocity.evaluate (true,false,false);
        for (unsigned int q=0; q<fe_eval_velocity.n_q_points; ++q)
        {
          fe_eval_pressure.submit_gradient (-fe_eval_velocity.get_value(q), q); // minus sign due to integration by parts
        }
        fe_eval_pressure.integrate (false,true);
      }
      else // integration_by_parts_of_divU == false
      {
        fe_eval_velocity.evaluate (false,true,false);
        for (unsigned int q=0; q<fe_eval_velocity.n_q_points; ++q)
        {
            fe_eval_pressure.submit_value (fe_eval_velocity.get_divergence(q), q);
        }
        fe_eval_pressure.integrate (true,false);
      }
      fe_eval_pressure.distribute_local_to_global (dst);
    }
  }

  void local_apply_divergence_face (const MatrixFree<dim,value_type>                 &data,
                                    parallel::distributed::Vector<value_type>        &dst,
                                    const parallel::distributed::Vector<value_type>  &src,
                                    const std::pair<unsigned int,unsigned int>       &face_range) const
  {
    if(operator_data.integration_by_parts_of_divU == true)
    {
      FEFaceEval_Velocity_Velocity_linear fe_eval_velocity(data,this->fe_param,true,operator_data.dof_index_velocity);
      FEFaceEval_Velocity_Velocity_linear fe_eval_velocity_neighbor(data,this->fe_param,false,operator_data.dof_index_velocity);

      FEFaceEval_Pressure_Velocity_linear fe_eval_pressure(data,this->fe_param,true,operator_data.dof_index_pressure);
      FEFaceEval_Pressure_Velocity_linear fe_eval_pressure_neighbor(data,this->fe_param,false,operator_data.dof_index_pressure);

      for(unsigned int face=face_range.first; face<face_range.second; face++)
      {
        fe_eval_pressure.reinit (face);
        fe_eval_pressure_neighbor.reinit (face);

        fe_eval_velocity.reinit (face);
        fe_eval_velocity_neighbor.reinit (face);
        fe_eval_velocity.read_dof_values(src);
        fe_eval_velocity_neighbor.read_dof_values(src);
        fe_eval_velocity.evaluate (true,false);
        fe_eval_velocity_neighbor.evaluate (true,false);
        for (unsigned int q=0; q<fe_eval_velocity.n_q_points; ++q)
        {
          Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval_velocity.get_normal_vector(q);
          Tensor<1,dim,VectorizedArray<value_type> > meanvel = 0.5*(fe_eval_velocity.get_value(q)+fe_eval_velocity_neighbor.get_value(q));
          VectorizedArray<value_type> submitvalue = normal * meanvel;

          fe_eval_pressure.submit_value (submitvalue, q);
          fe_eval_pressure_neighbor.submit_value (-submitvalue, q);
        }
        fe_eval_pressure.integrate (true,false);
        fe_eval_pressure_neighbor.integrate (true,false);
        fe_eval_pressure.distribute_local_to_global (dst);
        fe_eval_pressure_neighbor.distribute_local_to_global (dst);
      }
    }
  }

  void local_apply_divergence_boundary_face (const MatrixFree<dim,value_type>                &data,
                                             parallel::distributed::Vector<value_type>       &dst,
                                             const parallel::distributed::Vector<value_type> &src,
                                             const std::pair<unsigned int,unsigned int>      &face_range) const
  {
    if(operator_data.integration_by_parts_of_divU == true)
    {
      FEFaceEval_Velocity_Velocity_linear fe_eval_velocity(data,this->fe_param,true,operator_data.dof_index_velocity);
      FEFaceEval_Pressure_Velocity_linear fe_eval_pressure(data,this->fe_param,true,operator_data.dof_index_pressure);

      for(unsigned int face=face_range.first; face<face_range.second; face++)
      {
        fe_eval_pressure.reinit (face);

        fe_eval_velocity.reinit(face);
        fe_eval_velocity.read_dof_values(src);
        fe_eval_velocity.evaluate (true,false);

        typename std::map<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >::iterator it;
        types::boundary_id boundary_id = data.get_boundary_indicator(face);

        for(unsigned int q=0;q<fe_eval_pressure.n_q_points;++q)
        {
          if(operator_data.use_boundary_data == true)
          {
            it = operator_data.bc->dirichlet_bc.find(boundary_id);
            if(it != operator_data.bc->dirichlet_bc.end())
            {
              // on GammaD: u⁺ = -u⁻ + 2g -> {{u}} = g
              // homogeneous part: u⁺ = -u⁻ -> {{u}} = 0
              // inhomongenous part: u⁺ = 2g -> {{u}} = g
              Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval_pressure.get_normal_vector(q);
              Tensor<1,dim,VectorizedArray<value_type> > meanvel;
              VectorizedArray<value_type> submitvalue = normal * meanvel;
              fe_eval_pressure.submit_value(submitvalue,q);
            }

            it = operator_data.bc->neumann_bc.find(boundary_id);
            if(it != operator_data.bc->neumann_bc.end())
            {
              // on GammaN: u⁺ = u⁻ -> {{u}} = u⁻
              // homogeneous part: u⁺ = u⁻ -> {{u}} = u⁻
              // inhomongenous part: u⁺ = 0 -> {{u}} = 0
              Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval_pressure.get_normal_vector(q);
              Tensor<1,dim,VectorizedArray<value_type> > meanvel = fe_eval_velocity.get_value(q);
              VectorizedArray<value_type> submitvalue = normal * meanvel;
              fe_eval_pressure.submit_value(submitvalue,q);
            }
          }
          else // use_boundary_data == false
          {
            // use u⁺ = u⁻ on all boundaries
            Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval_pressure.get_normal_vector(q);
            Tensor<1,dim,VectorizedArray<value_type> > meanvel = fe_eval_velocity.get_value(q);
            VectorizedArray<value_type> submitvalue = normal * meanvel;
            fe_eval_pressure.submit_value(submitvalue,q);
          }
        }
        fe_eval_pressure.integrate(true,false);
        fe_eval_pressure.distribute_local_to_global(dst);
      }
    }
  }

  void local_evaluate_divergence_boundary_face (const MatrixFree<dim,value_type>                &data,
                                                parallel::distributed::Vector<value_type>       &dst,
                                                const parallel::distributed::Vector<value_type> &src,
                                                const std::pair<unsigned int,unsigned int>      &face_range) const
  {
    if(operator_data.integration_by_parts_of_divU == true)
    {
      FEFaceEval_Velocity_Velocity_linear fe_eval_velocity(data,this->fe_param,true,operator_data.dof_index_velocity);
      FEFaceEval_Pressure_Velocity_linear fe_eval_pressure(data,this->fe_param,true,operator_data.dof_index_pressure);

      for(unsigned int face=face_range.first; face<face_range.second; face++)
      {
        fe_eval_pressure.reinit (face);

        fe_eval_velocity.reinit(face);
        fe_eval_velocity.read_dof_values(src);
        fe_eval_velocity.evaluate (true,false);

        typename std::map<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >::iterator it;
        types::boundary_id boundary_id = data.get_boundary_indicator(face);

        for(unsigned int q=0;q<fe_eval_pressure.n_q_points;++q)
        {
          if(operator_data.use_boundary_data == true)
          {
            it = operator_data.bc->dirichlet_bc.find(boundary_id);
            if(it != operator_data.bc->dirichlet_bc.end())
            {
              // on GammaD: u⁺ = -u⁻ + 2g -> {{u}} = g
              // homogeneous part: u⁺ = -u⁻ -> {{u}} = 0
              // inhomongenous part: u⁺ = 2g -> {{u}} = g
              Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval_pressure.get_normal_vector(q);
              Tensor<1,dim,VectorizedArray<value_type> > meanvel;

              Point<dim,VectorizedArray<value_type> > q_points = fe_eval_pressure.quadrature_point(q);
              // set correct time for the evaluation of boundary conditions
              it->second->set_time(eval_time);
              for(unsigned int d=0;d<dim;++d)
              {
                value_type array [VectorizedArray<value_type>::n_array_elements];
                for (unsigned int n=0; n<VectorizedArray<value_type>::n_array_elements; ++n)
                {
                  Point<dim> q_point;
                  for (unsigned int d=0; d<dim; ++d)
                    q_point[d] = q_points[d][n];
                  array[n] = it->second->value(q_point,d);
                }
                meanvel[d].load(&array[0]);
              }

              VectorizedArray<value_type> submitvalue = normal * meanvel;
              fe_eval_pressure.submit_value(submitvalue,q);
            }

            it = operator_data.bc->neumann_bc.find(boundary_id);
            if(it != operator_data.bc->neumann_bc.end())
            {
              // on GammaN: u⁺ = u⁻ -> {{u}} = u⁻
              // homogeneous part: u⁺ = u⁻ -> {{u}} = u⁻
              // inhomongenous part: u⁺ = 0 -> {{u}} = 0
              Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval_pressure.get_normal_vector(q);
              Tensor<1,dim,VectorizedArray<value_type> > meanvel = fe_eval_velocity.get_value(q);
              VectorizedArray<value_type> submitvalue = normal * meanvel;
              fe_eval_pressure.submit_value(submitvalue,q);
            }
          }
          else // use_boundary_data == false
          {
            // use u⁺ = u⁻ on all boundaries
            Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval_pressure.get_normal_vector(q);
            Tensor<1,dim,VectorizedArray<value_type> > meanvel = fe_eval_velocity.get_value(q);
            VectorizedArray<value_type> submitvalue = normal * meanvel;
            fe_eval_pressure.submit_value(submitvalue,q);
          }
        }
        fe_eval_pressure.integrate(true,false);
        fe_eval_pressure.distribute_local_to_global(dst);
      }
    }
  }

  void local_rhs_divergence (const MatrixFree<dim,value_type>              &,
                           parallel::distributed::Vector<value_type>       &,
                           const parallel::distributed::Vector<value_type> &,
                           const std::pair<unsigned int,unsigned int>      &) const
  {

  }

  void local_rhs_divergence_face (const MatrixFree<dim,value_type>                 &,
                                  parallel::distributed::Vector<value_type>        &,
                                  const parallel::distributed::Vector<value_type>  &,
                                  const std::pair<unsigned int,unsigned int>       &) const
  {

  }

  void local_rhs_divergence_boundary_face (const MatrixFree<dim,value_type>                &data,
                                           parallel::distributed::Vector<value_type>       &dst,
                                           const parallel::distributed::Vector<value_type> &,
                                           const std::pair<unsigned int,unsigned int>      &face_range) const
  {
    if(operator_data.integration_by_parts_of_divU == true &&
       operator_data.use_boundary_data == true)
    {
      FEFaceEval_Pressure_Velocity_linear fe_eval_pressure(data,this->fe_param,true,operator_data.dof_index_pressure);

      for(unsigned int face=face_range.first; face<face_range.second; face++)
      {
        fe_eval_pressure.reinit (face);

        typename std::map<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >::iterator it;
        types::boundary_id boundary_id = data.get_boundary_indicator(face);

        for(unsigned int q=0;q<fe_eval_pressure.n_q_points;++q)
        {
          it = operator_data.bc->dirichlet_bc.find(boundary_id);
          if(it != operator_data.bc->dirichlet_bc.end())
          {
            // on GammaD: u⁺ = -u⁻ + 2g -> {{u}} = g
            // homogeneous part: u⁺ = -u⁻ -> {{u}} = 0
            // inhomongenous part: u⁺ = 2g -> {{u}} = g
            Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval_pressure.get_normal_vector(q);
            Tensor<1,dim,VectorizedArray<value_type> > meanvel;

            Point<dim,VectorizedArray<value_type> > q_points = fe_eval_pressure.quadrature_point(q);
            // set correct time for the evaluation of boundary conditions
            it->second->set_time(eval_time);
            for(unsigned int d=0;d<dim;++d)
            {
              value_type array [VectorizedArray<value_type>::n_array_elements];
              for (unsigned int n=0; n<VectorizedArray<value_type>::n_array_elements; ++n)
              {
                Point<dim> q_point;
                for (unsigned int d=0; d<dim; ++d)
                  q_point[d] = q_points[d][n];
                array[n] = it->second->value(q_point,d);
              }
              meanvel[d].load(&array[0]);
            }

            VectorizedArray<value_type> submitvalue = normal * meanvel;
            fe_eval_pressure.submit_value(-submitvalue,q); // minus sign since this term occurs on the rhs of the equation
          }

          it = operator_data.bc->neumann_bc.find(boundary_id);
          if(it != operator_data.bc->neumann_bc.end())
          {
            // on GammaN: u⁺ = u⁻ -> {{u}} = u⁻
            // homogeneous part: u⁺ = u⁻ -> {{u}} = u⁻
            // inhomongenous part: u⁺ = 0 -> {{u}} = 0
            Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval_pressure.get_normal_vector(q);
            Tensor<1,dim,VectorizedArray<value_type> > meanvel;
            VectorizedArray<value_type> submitvalue = normal * meanvel;
            fe_eval_pressure.submit_value(-submitvalue,q); // minus sign since this term occurs on the rhs of the equation
          }
        }
        fe_eval_pressure.integrate(true,false);
        fe_eval_pressure.distribute_local_to_global(dst);
      }
    }
  }

  MatrixFree<dim,value_type> const * data;
  DivergenceOperatorData<dim> operator_data;
  value_type mutable eval_time;
};

template<int dim>
struct ConvectiveOperatorData
{
  ConvectiveOperatorData ()
    :
    dof_index(0)
  {}

  unsigned int dof_index;

  std_cxx11::shared_ptr<BoundaryDescriptorNavierStokes<dim> > bc;
};



template <int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule, typename value_type>
class ConvectiveOperator: public BaseOperator<dim>
{
public:
  ConvectiveOperator()
    :
    data(nullptr),
    eval_time(0.0),
    velocity_linearization(nullptr)
  {}

  static const bool is_xwall = (xwall_quad_rule>1) ? true : false;
  static const unsigned int n_actual_q_points_vel_nonlinear = (is_xwall) ? xwall_quad_rule : fe_degree+(fe_degree+2)/2;

  typedef FEEvaluationWrapper<dim,fe_degree,fe_degree_xwall,n_actual_q_points_vel_nonlinear,dim,value_type,is_xwall> FEEval_Velocity_Velocity_nonlinear;
  typedef FEFaceEvaluationWrapper<dim,fe_degree,fe_degree_xwall,n_actual_q_points_vel_nonlinear,dim,value_type,is_xwall> FEFaceEval_Velocity_Velocity_nonlinear;

  void initialize(MatrixFree<dim,value_type> const  &mf_data,
                  ConvectiveOperatorData<dim> const &operator_data_in)
  {
    this->data = &mf_data;
    this->operator_data = operator_data_in;
  }

  void evaluate (parallel::distributed::Vector<value_type>       &dst,
                 parallel::distributed::Vector<value_type> const &src,
                 value_type const                                evaluation_time) const
  {
    dst = 0;
    evaluate_add(dst,src,evaluation_time);
  }

  void evaluate_add (parallel::distributed::Vector<value_type>       &dst,
                     parallel::distributed::Vector<value_type> const &src,
                     value_type const                                evaluation_time) const
  {
    this->eval_time = evaluation_time;
    evaluate_convective_term(dst,src);
  }

  void apply_linearized (parallel::distributed::Vector<value_type>       &dst,
                         parallel::distributed::Vector<value_type> const &src,
                         parallel::distributed::Vector<value_type> const *vector_linearization,
                         value_type const                                evaluation_time) const
  {
    dst = 0;
    apply_linearized_add(dst,src,vector_linearization,evaluation_time);
  }

  void apply_linearized_add (parallel::distributed::Vector<value_type>       &dst,
                             parallel::distributed::Vector<value_type> const &src,
                             parallel::distributed::Vector<value_type> const *vector_linearization,
                             value_type const                                evaluation_time) const
  {
    this->eval_time = evaluation_time;
    velocity_linearization = vector_linearization;

    apply_linearized_convective_term(dst,src);

    velocity_linearization = nullptr;
  }

  void calculate_diagonal(parallel::distributed::Vector<value_type>       &diagonal,
                          parallel::distributed::Vector<value_type> const *vector_linearization,
                          value_type const                                evaluation_time) const
  {
    diagonal = 0;

    add_diagonal(diagonal,vector_linearization,evaluation_time);
  }

  void add_diagonal(parallel::distributed::Vector<value_type>       &diagonal,
                    parallel::distributed::Vector<value_type> const *vector_linearization,
                    value_type const                                evaluation_time) const
  {
    this->eval_time = evaluation_time;
    velocity_linearization = vector_linearization;

    parallel::distributed::Vector<value_type>  src_dummy(diagonal);

    data->loop(&ConvectiveOperator<dim,fe_degree,fe_degree_xwall,xwall_quad_rule,value_type>::local_diagonal,
               &ConvectiveOperator<dim,fe_degree,fe_degree_xwall,xwall_quad_rule,value_type>::local_diagonal_face,
               &ConvectiveOperator<dim,fe_degree,fe_degree_xwall,xwall_quad_rule,value_type>::local_diagonal_boundary_face,
               this, diagonal, src_dummy);
  }

  ConvectiveOperatorData<dim> const & get_operator_data() const
  {
    return operator_data;
  }


private:
  void evaluate_convective_term (parallel::distributed::Vector<value_type>       &dst,
                                 parallel::distributed::Vector<value_type> const &src) const
  {
    data->loop(&ConvectiveOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, value_type>::local_evaluate_convective_term,
               &ConvectiveOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, value_type>::local_evaluate_convective_term_face,
               &ConvectiveOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, value_type>::local_evaluate_convective_term_boundary_face,
               this, dst, src);
  }


  void local_evaluate_convective_term (const MatrixFree<dim,value_type>                 &data,
                                       parallel::distributed::Vector<value_type>        &dst,
                                       const parallel::distributed::Vector<value_type>  &src,
                                       const std::pair<unsigned int,unsigned int>       &cell_range) const
  {
    FEEval_Velocity_Velocity_nonlinear fe_eval_velocity(data,this->fe_param,operator_data.dof_index);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval_velocity.reinit(cell);
      fe_eval_velocity.read_dof_values(src);
      fe_eval_velocity.evaluate (true,false,false);

      for (unsigned int q=0; q<fe_eval_velocity.n_q_points; ++q)
      {
        // nonlinear convective flux F(u) = uu
        Tensor<1,dim,VectorizedArray<value_type> > u = fe_eval_velocity.get_value(q);
        Tensor<2,dim,VectorizedArray<value_type> > F = outer_product(u,u);
        fe_eval_velocity.submit_gradient (-F, q); // minus sign due to integration by parts
      }
      fe_eval_velocity.integrate (false,true);
      fe_eval_velocity.distribute_local_to_global(dst);
    }
  }

  void local_evaluate_convective_term_face (const MatrixFree<dim,value_type>                &data,
                                            parallel::distributed::Vector<value_type>       &dst,
                                            const parallel::distributed::Vector<value_type> &src,
                                            const std::pair<unsigned int,unsigned int>      &face_range) const
  {
    FEFaceEval_Velocity_Velocity_nonlinear fe_eval_velocity(data,this->fe_param,true,operator_data.dof_index);
    FEFaceEval_Velocity_Velocity_nonlinear fe_eval_velocity_neighbor(data,this->fe_param,false,operator_data.dof_index);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval_velocity.reinit(face);
      fe_eval_velocity.read_dof_values(src);
      fe_eval_velocity.evaluate(true, false);

      fe_eval_velocity_neighbor.reinit (face);
      fe_eval_velocity_neighbor.read_dof_values(src);
      fe_eval_velocity_neighbor.evaluate(true,false);

      for(unsigned int q=0;q<fe_eval_velocity.n_q_points;++q)
      {
        Tensor<1,dim,VectorizedArray<value_type> > uM = fe_eval_velocity.get_value(q);
        Tensor<1,dim,VectorizedArray<value_type> > uP = fe_eval_velocity_neighbor.get_value(q);
        Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval_velocity.get_normal_vector(q);

        const VectorizedArray<value_type> uM_n = uM*normal;
        const VectorizedArray<value_type> uP_n = uP*normal;

        // calculation of lambda according to Shahbazi et al., i.e.
        // lambda = max ( max |lambda(flux_jacobian_M)| , max |lambda(flux_jacobian_P)| )
        // where the maximum eigenvalue of the flux Jacobian is the
        // maximum eigenvalue of (u^T * normal) * I + u * normal^T, which is
        // abs(2*u^T*normal) (this can be verified by rank-1 matrix algebra)
        const VectorizedArray<value_type> lambda = 2.*std::max(std::abs(uM_n), std::abs(uP_n));

        Tensor<1,dim,VectorizedArray<value_type> > jump_value = uM - uP;
        Tensor<1,dim,VectorizedArray<value_type> > average_normal_flux = ( uM*uM_n + uP*uP_n) * make_vectorized_array<value_type>(0.5);
        Tensor<1,dim,VectorizedArray<value_type> > lf_flux = average_normal_flux + 0.5 * lambda * jump_value;

        fe_eval_velocity.submit_value(lf_flux,q);
        fe_eval_velocity_neighbor.submit_value(-lf_flux,q);
      }
      fe_eval_velocity.integrate(true,false);
      fe_eval_velocity.distribute_local_to_global(dst);
      fe_eval_velocity_neighbor.integrate(true,false);
      fe_eval_velocity_neighbor.distribute_local_to_global(dst);
    }
  }

  void local_evaluate_convective_term_boundary_face (const MatrixFree<dim,value_type>                &data,
                                                     parallel::distributed::Vector<value_type>       &dst,
                                                     const parallel::distributed::Vector<value_type> &src,
                                                     const std::pair<unsigned int,unsigned int>      &face_range) const
  {
    FEFaceEval_Velocity_Velocity_nonlinear fe_eval_velocity(data,this->fe_param,true,operator_data.dof_index);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval_velocity.reinit (face);
      fe_eval_velocity.read_dof_values(src);
      fe_eval_velocity.evaluate(true,false);

      typename std::map<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >::iterator it;
      types::boundary_id boundary_id = data.get_boundary_indicator(face);

      for(unsigned int q=0;q<fe_eval_velocity.n_q_points;++q)
      {
        it = operator_data.bc->dirichlet_bc.find(boundary_id);
        if(it != operator_data.bc->dirichlet_bc.end())
        {
          // on GammaD: u⁺ = -u⁻ + 2g
          Tensor<1,dim,VectorizedArray<value_type> > uM = fe_eval_velocity.get_value(q);

          Point<dim,VectorizedArray<value_type> > q_points = fe_eval_velocity.quadrature_point(q);
          Tensor<1,dim,VectorizedArray<value_type> > g;
          // set correct time for the evaluation of boundary conditions
          it->second->set_time(eval_time);
          for(unsigned int d=0;d<dim;++d)
          {
            value_type array [VectorizedArray<value_type>::n_array_elements];
            for (unsigned int n=0; n<VectorizedArray<value_type>::n_array_elements; ++n)
            {
              Point<dim> q_point;
              for (unsigned int d=0; d<dim; ++d)
                q_point[d] = q_points[d][n];
              array[n] = it->second->value(q_point,d);
            }
            g[d].load(&array[0]);
          }

          Tensor<1,dim,VectorizedArray<value_type> > uP = -uM + make_vectorized_array<value_type>(2.0)*g;
          Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval_velocity.get_normal_vector(q);
          const VectorizedArray<value_type> uM_n = uM*normal;
          const VectorizedArray<value_type> uP_n = uP*normal;

          const VectorizedArray<value_type> lambda = 2.*std::max(std::abs(uM_n), std::abs(uP_n));

          Tensor<1,dim,VectorizedArray<value_type> > jump_value = uM - uP;
          Tensor<1,dim,VectorizedArray<value_type> > average_normal_flux = ( uM*uM_n + uP*uP_n) * make_vectorized_array<value_type>(0.5);
          Tensor<1,dim,VectorizedArray<value_type> > lf_flux = average_normal_flux + 0.5 * lambda * jump_value;

          fe_eval_velocity.submit_value(lf_flux,q);
        }

        it = operator_data.bc->neumann_bc.find(boundary_id);
        if(it != operator_data.bc->neumann_bc.end())
        {
          // on GammaN: u⁺ = u⁻
          Tensor<1,dim,VectorizedArray<value_type> > uM = fe_eval_velocity.get_value(q);
          Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval_velocity.get_normal_vector(q);
          const VectorizedArray<value_type> uM_n = uM*normal;

          Tensor<1,dim,VectorizedArray<value_type> > average_normal_flux = uM*uM_n;
          Tensor<1,dim,VectorizedArray<value_type> > lf_flux = average_normal_flux;

          fe_eval_velocity.submit_value(lf_flux,q);
        }
      }

      fe_eval_velocity.integrate(true,false);
      fe_eval_velocity.distribute_local_to_global(dst);
    }
  }

  void apply_linearized_convective_term (parallel::distributed::Vector<value_type>       &dst,
                                         parallel::distributed::Vector<value_type> const &src) const
  {
    data->loop(&ConvectiveOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, value_type>::local_apply_linearized_convective_term,
               &ConvectiveOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, value_type>::local_apply_linearized_convective_term_face,
               &ConvectiveOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, value_type>::local_apply_linearized_convective_term_boundary_face,
               this, dst, src);
  }

  void local_apply_linearized_convective_term(const MatrixFree<dim,value_type>                &data,
                                              parallel::distributed::Vector<value_type>       &dst,
                                              const parallel::distributed::Vector<value_type> &src,
                                              const std::pair<unsigned int,unsigned int>      &cell_range) const
  {
    FEEval_Velocity_Velocity_nonlinear fe_eval(data,this->fe_param,operator_data.dof_index);
    FEEval_Velocity_Velocity_nonlinear fe_eval_linearization(data,this->fe_param,operator_data.dof_index);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);
      fe_eval.read_dof_values(src);
      fe_eval.evaluate (true,false,false);

      fe_eval_linearization.reinit(cell);
      fe_eval_linearization.read_dof_values(*velocity_linearization);
      fe_eval_linearization.evaluate (true,false,false);

      for (unsigned int q=0; q<fe_eval.n_q_points; ++q)
      {
        Tensor<1,dim,VectorizedArray<value_type> > delta_u = fe_eval.get_value(q);
        Tensor<1,dim,VectorizedArray<value_type> > u = fe_eval_linearization.get_value(q);
        Tensor<2,dim,VectorizedArray<value_type> > F = outer_product(u,delta_u);
        fe_eval.submit_gradient (-(F+transpose(F)), q);
      }
      fe_eval.integrate (false,true);
      fe_eval.distribute_local_to_global (dst);
    }
  }

  void local_apply_linearized_convective_term_face (const MatrixFree<dim,value_type>                &data,
                                                    parallel::distributed::Vector<value_type>       &dst,
                                                    const parallel::distributed::Vector<value_type> &src,
                                                    const std::pair<unsigned int,unsigned int>      &face_range) const
  {
    FEFaceEval_Velocity_Velocity_nonlinear fe_eval(data,this->fe_param,true,operator_data.dof_index);
    FEFaceEval_Velocity_Velocity_nonlinear fe_eval_neighbor(data,this->fe_param,false,operator_data.dof_index);

    FEFaceEval_Velocity_Velocity_nonlinear fe_eval_linearization(data,this->fe_param,true,operator_data.dof_index);
    FEFaceEval_Velocity_Velocity_nonlinear fe_eval_linearization_neighbor(data,this->fe_param,false,operator_data.dof_index);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval.reinit(face);
      fe_eval.read_dof_values(src);
      fe_eval.evaluate(true, false);

      fe_eval_neighbor.reinit (face);
      fe_eval_neighbor.read_dof_values(src);
      fe_eval_neighbor.evaluate(true, false);

      fe_eval_linearization.reinit(face);
      fe_eval_linearization.read_dof_values(*velocity_linearization);
      fe_eval_linearization.evaluate(true, false);

      fe_eval_linearization_neighbor.reinit (face);
      fe_eval_linearization_neighbor.read_dof_values(*velocity_linearization);
      fe_eval_linearization_neighbor.evaluate(true, false);

      for(unsigned int q=0;q<fe_eval.n_q_points;++q)
      {
        Tensor<1,dim,VectorizedArray<value_type> > uM = fe_eval_linearization.get_value(q);
        Tensor<1,dim,VectorizedArray<value_type> > uP = fe_eval_linearization_neighbor.get_value(q);
        Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval.get_normal_vector(q);

        const VectorizedArray<value_type> uM_n = uM*normal;
        const VectorizedArray<value_type> uP_n = uP*normal;

        const VectorizedArray<value_type> lambda = 2.*std::max(std::abs(uM_n), std::abs(uP_n));

        Tensor<1,dim,VectorizedArray<value_type> > delta_uM = fe_eval.get_value(q);
        Tensor<1,dim,VectorizedArray<value_type> > delta_uP = fe_eval_neighbor.get_value(q);

        const VectorizedArray<value_type> delta_uM_n = delta_uM*normal;
        const VectorizedArray<value_type> delta_uP_n = delta_uP*normal;

        Tensor<1,dim,VectorizedArray<value_type> > jump_value = delta_uM - delta_uP;
        Tensor<1,dim,VectorizedArray<value_type> > average_normal_flux = make_vectorized_array<value_type>(0.5)*
            (uM*delta_uM_n + delta_uM*uM_n + uP*delta_uP_n + delta_uP*uP_n);
        Tensor<1,dim,VectorizedArray<value_type> > lf_flux = average_normal_flux + 0.5 * lambda * jump_value;

        fe_eval.submit_value(lf_flux,q);
        fe_eval_neighbor.submit_value(-lf_flux,q);
      }
      fe_eval.integrate(true,false);
      fe_eval.distribute_local_to_global(dst);
      fe_eval_neighbor.integrate(true,false);
      fe_eval_neighbor.distribute_local_to_global(dst);
    }
  }

  void local_apply_linearized_convective_term_boundary_face(const MatrixFree<dim,value_type>                &data,
                                                            parallel::distributed::Vector<value_type>       &dst,
                                                            const parallel::distributed::Vector<value_type> &src,
                                                            const std::pair<unsigned int,unsigned int>      &face_range) const
  {
    FEFaceEval_Velocity_Velocity_nonlinear fe_eval(data,this->fe_param,true,operator_data.dof_index);
    FEFaceEval_Velocity_Velocity_nonlinear fe_eval_linearization(data,this->fe_param,true,operator_data.dof_index);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval.reinit (face);
      fe_eval.read_dof_values(src);
      fe_eval.evaluate(true,false);

      fe_eval_linearization.reinit (face);
      fe_eval_linearization.read_dof_values(*velocity_linearization);
      fe_eval_linearization.evaluate(true,false);

      typename std::map<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >::iterator it;
      types::boundary_id boundary_id = data.get_boundary_indicator(face);

      for(unsigned int q=0;q<fe_eval.n_q_points;++q)
      {
        it = operator_data.bc->dirichlet_bc.find(boundary_id);
        if(it != operator_data.bc->dirichlet_bc.end())
        {
          // on GammaD: u⁺ = -u⁻ + 2g
          Tensor<1,dim,VectorizedArray<value_type> > uM = fe_eval_linearization.get_value(q);

          Point<dim,VectorizedArray<value_type> > q_points = fe_eval.quadrature_point(q);
          Tensor<1,dim,VectorizedArray<value_type> > g;
          // set correct time for the evaluation of boundary conditions
          it->second->set_time(eval_time);
          for(unsigned int d=0;d<dim;++d)
          {
            value_type array [VectorizedArray<value_type>::n_array_elements];
            for (unsigned int n=0; n<VectorizedArray<value_type>::n_array_elements; ++n)
            {
              Point<dim> q_point;
              for (unsigned int d=0; d<dim; ++d)
              q_point[d] = q_points[d][n];
              array[n] = it->second->value(q_point,d);
            }
            g[d].load(&array[0]);
          }

          Tensor<1,dim,VectorizedArray<value_type> > uP = -uM + make_vectorized_array<value_type>(2.0)*g;
          Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval.get_normal_vector(q);
          const VectorizedArray<value_type> uM_n = uM*normal;
          const VectorizedArray<value_type> uP_n = uP*normal;

          const VectorizedArray<value_type> lambda = 2.*std::max(std::abs(uM_n), std::abs(uP_n));

          Tensor<1,dim,VectorizedArray<value_type> > delta_uM = fe_eval.get_value(q);
          Tensor<1,dim,VectorizedArray<value_type> > delta_uP = -delta_uM;

          const VectorizedArray<value_type> delta_uM_n = delta_uM*normal;
          const VectorizedArray<value_type> delta_uP_n = delta_uP*normal;

          Tensor<1,dim,VectorizedArray<value_type> > jump_value = delta_uM - delta_uP;
          Tensor<1,dim,VectorizedArray<value_type> > average_normal_flux = make_vectorized_array<value_type>(0.5)*
              (uM*delta_uM_n + delta_uM*uM_n + uP*delta_uP_n + delta_uP*uP_n);
          Tensor<1,dim,VectorizedArray<value_type> > lf_flux = average_normal_flux + 0.5 * lambda * jump_value;

          fe_eval.submit_value(lf_flux,q);
        }

        it = operator_data.bc->neumann_bc.find(boundary_id);
        if(it != operator_data.bc->neumann_bc.end())
        {
          // on GammaN: u⁺ = u⁻
          Tensor<1,dim,VectorizedArray<value_type> > uM = fe_eval_linearization.get_value(q);
          Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval.get_normal_vector(q);
          const VectorizedArray<value_type> uM_n = uM*normal;

          Tensor<1,dim,VectorizedArray<value_type> > delta_uM = fe_eval.get_value(q);
          const VectorizedArray<value_type> delta_uM_n = delta_uM*normal;

          Tensor<1,dim,VectorizedArray<value_type> > average_normal_flux = (uM*delta_uM_n + delta_uM*uM_n);
          Tensor<1,dim,VectorizedArray<value_type> > lf_flux = average_normal_flux;

          fe_eval.submit_value(lf_flux,q);
        }
      }
      fe_eval.integrate(true,false);
      fe_eval.distribute_local_to_global(dst);
    }
  }

  void local_diagonal (const MatrixFree<dim,value_type>                 &data,
                       parallel::distributed::Vector<value_type>        &dst,
                       const parallel::distributed::Vector<value_type>  &,
                       const std::pair<unsigned int,unsigned int>       &cell_range) const
  {
    FEEval_Velocity_Velocity_nonlinear fe_eval(data,this->fe_param,operator_data.dof_index);
    FEEval_Velocity_Velocity_nonlinear fe_eval_linearization(data,this->fe_param,operator_data.dof_index);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval_linearization.reinit(cell);
      fe_eval_linearization.read_dof_values(*velocity_linearization);
      fe_eval_linearization.evaluate (true,false,false);

      fe_eval.reinit(cell);

      VectorizedArray<value_type> local_diagonal_vector[fe_eval.tensor_dofs_per_cell*dim];

      for (unsigned int j=0; j<fe_eval.dofs_per_cell*dim; ++j)
      {
        for (unsigned int i=0; i<fe_eval.dofs_per_cell*dim; ++i)
          fe_eval.write_cellwise_dof_value(i,make_vectorized_array<value_type>(0.));
        fe_eval.write_cellwise_dof_value(j,make_vectorized_array<value_type>(1.));

        // copied for local_apply_linearized_convective_term
        fe_eval.evaluate (true,false,false);

        for (unsigned int q=0; q<fe_eval.n_q_points; ++q)
        {
          Tensor<1,dim,VectorizedArray<value_type> > delta_u = fe_eval.get_value(q);
          Tensor<1,dim,VectorizedArray<value_type> > u = fe_eval_linearization.get_value(q);
          Tensor<2,dim,VectorizedArray<value_type> > F = outer_product(u,delta_u);
          fe_eval.submit_gradient (-(F+transpose(F)), q);
        }
        fe_eval.integrate (false,true);
        // copied for local_apply_linearized_convective_term

        local_diagonal_vector[j] = fe_eval.read_cellwise_dof_value(j);
      }

      for (unsigned int j=0; j<fe_eval.dofs_per_cell*dim; ++j)
        fe_eval.write_cellwise_dof_value(j,local_diagonal_vector[j]);
      fe_eval.distribute_local_to_global (dst);
    }
  }

  void local_diagonal_face (const MatrixFree<dim,value_type>                &data,
                            parallel::distributed::Vector<value_type>       &dst,
                            const parallel::distributed::Vector<value_type> &,
                            const std::pair<unsigned int,unsigned int>      &face_range) const
  {
    FEFaceEval_Velocity_Velocity_nonlinear fe_eval(data,this->fe_param,true,operator_data.dof_index);
    FEFaceEval_Velocity_Velocity_nonlinear fe_eval_neighbor(data,this->fe_param,false,operator_data.dof_index);

    FEFaceEval_Velocity_Velocity_nonlinear fe_eval_linearization(data,this->fe_param,true,operator_data.dof_index);
    FEFaceEval_Velocity_Velocity_nonlinear fe_eval_linearization_neighbor(data,this->fe_param,false,operator_data.dof_index);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval_linearization.reinit(face);
      fe_eval_linearization.read_dof_values(*velocity_linearization);
      fe_eval_linearization.evaluate(true, false);

      fe_eval_linearization_neighbor.reinit (face);
      fe_eval_linearization_neighbor.read_dof_values(*velocity_linearization);
      fe_eval_linearization_neighbor.evaluate(true, false);

      fe_eval.reinit(face);
      fe_eval_neighbor.reinit (face);

      // element-
      VectorizedArray<value_type> local_diagonal_vector[fe_eval.tensor_dofs_per_cell*dim];
      for (unsigned int j=0; j<fe_eval.dofs_per_cell*dim; ++j)
      {
        // set dof value j of element- to 1 and all other dof values of element- to zero
        for (unsigned int i=0; i<fe_eval.dofs_per_cell*dim; ++i)
          fe_eval.write_cellwise_dof_value(i,make_vectorized_array<value_type>(0.));
        fe_eval.write_cellwise_dof_value(j,make_vectorized_array<value_type>(1.));
        // set all dof values of element+ to zero
        for (unsigned int i=0; i<fe_eval_neighbor.dofs_per_cell*dim; ++i)
          fe_eval_neighbor.write_cellwise_dof_value(i, make_vectorized_array<value_type>(0.));

        // copied from local_apply_linearized_convective_term_face  (note that fe_eval_neighbor.submit... has to be removed)
        fe_eval.evaluate(true, false);
        fe_eval_neighbor.evaluate(true, false);

        for(unsigned int q=0;q<fe_eval.n_q_points;++q)
        {
          Tensor<1,dim,VectorizedArray<value_type> > uM = fe_eval_linearization.get_value(q);
          Tensor<1,dim,VectorizedArray<value_type> > uP = fe_eval_linearization_neighbor.get_value(q);
          Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval.get_normal_vector(q);

          const VectorizedArray<value_type> uM_n = uM*normal;
          const VectorizedArray<value_type> uP_n = uP*normal;

          const VectorizedArray<value_type> lambda = 2.*std::max(std::abs(uM_n), std::abs(uP_n));

          Tensor<1,dim,VectorizedArray<value_type> > delta_uM = fe_eval.get_value(q);
          Tensor<1,dim,VectorizedArray<value_type> > delta_uP = fe_eval_neighbor.get_value(q);

          const VectorizedArray<value_type> delta_uM_n = delta_uM*normal;
          const VectorizedArray<value_type> delta_uP_n = delta_uP*normal;

          Tensor<1,dim,VectorizedArray<value_type> > jump_value = delta_uM - delta_uP;
          Tensor<1,dim,VectorizedArray<value_type> > average_normal_flux = make_vectorized_array<value_type>(0.5)*
              (uM*delta_uM_n + delta_uM*uM_n + uP*delta_uP_n + delta_uP*uP_n);
          Tensor<1,dim,VectorizedArray<value_type> > lf_flux = average_normal_flux + 0.5 * lambda * jump_value;

          fe_eval.submit_value(lf_flux,q);
        }
        // integrate on element-
        fe_eval.integrate(true,false);
        // copied from local_apply_linearized_convective_term_face  (note that fe_eval_neighbor.submit... has to be removed)

        local_diagonal_vector[j] = fe_eval.read_cellwise_dof_value(j);
      }
      for (unsigned int j=0; j<fe_eval.dofs_per_cell*dim; ++j)
        fe_eval.write_cellwise_dof_value(j, local_diagonal_vector[j]);

      fe_eval.distribute_local_to_global(dst);

      // neighbor (element+)
      VectorizedArray<value_type> local_diagonal_vector_neighbor[fe_eval_neighbor.tensor_dofs_per_cell*dim];
      for (unsigned int j=0; j<fe_eval_neighbor.dofs_per_cell*dim; ++j)
      {
        // set all dof values of element- to zero
        for (unsigned int i=0; i<fe_eval.dofs_per_cell*dim; ++i)
          fe_eval.write_cellwise_dof_value(i,make_vectorized_array<value_type>(0.));
        // set dof value j of element+ to 1 and all other dof values of element+ to zero
        for (unsigned int i=0; i<fe_eval_neighbor.dofs_per_cell*dim; ++i)
          fe_eval_neighbor.write_cellwise_dof_value(i, make_vectorized_array<value_type>(0.));
        fe_eval_neighbor.write_cellwise_dof_value(j,make_vectorized_array<value_type>(1.));

        // copied from local_apply_linearized_convective_term_face  (note that fe_eval.submit... has to be removed)
        fe_eval.evaluate(true, false);
        fe_eval_neighbor.evaluate(true, false);

        for(unsigned int q=0;q<fe_eval.n_q_points;++q)
        {
          Tensor<1,dim,VectorizedArray<value_type> > uM = fe_eval_linearization.get_value(q);
          Tensor<1,dim,VectorizedArray<value_type> > uP = fe_eval_linearization_neighbor.get_value(q);
          Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval.get_normal_vector(q);

          const VectorizedArray<value_type> uM_n = uM*normal;
          const VectorizedArray<value_type> uP_n = uP*normal;

          const VectorizedArray<value_type> lambda = 2.*std::max(std::abs(uM_n), std::abs(uP_n));

          Tensor<1,dim,VectorizedArray<value_type> > delta_uM = fe_eval.get_value(q);
          Tensor<1,dim,VectorizedArray<value_type> > delta_uP = fe_eval_neighbor.get_value(q);

          const VectorizedArray<value_type> delta_uM_n = delta_uM*normal;
          const VectorizedArray<value_type> delta_uP_n = delta_uP*normal;

          Tensor<1,dim,VectorizedArray<value_type> > jump_value = delta_uM - delta_uP;
          Tensor<1,dim,VectorizedArray<value_type> > average_normal_flux = make_vectorized_array<value_type>(0.5)*
              (uM*delta_uM_n + delta_uM*uM_n + uP*delta_uP_n + delta_uP*uP_n);
          Tensor<1,dim,VectorizedArray<value_type> > lf_flux = average_normal_flux + 0.5 * lambda * jump_value;

          fe_eval_neighbor.submit_value(-lf_flux,q);
        }
        // integrate on element+
        fe_eval_neighbor.integrate(true,false);
        // copied from local_apply_linearized_convective_term_face  (note that fe_eval.submit... has to be removed)

        local_diagonal_vector_neighbor[j] = fe_eval_neighbor.read_cellwise_dof_value(j);
      }
      for (unsigned int j=0; j<fe_eval_neighbor.dofs_per_cell*dim; ++j)
        fe_eval_neighbor.write_cellwise_dof_value(j, local_diagonal_vector_neighbor[j]);
      fe_eval_neighbor.distribute_local_to_global(dst);
    }
  }

  void local_diagonal_boundary_face (const MatrixFree<dim,value_type>                 &data,
                                     parallel::distributed::Vector<value_type>        &dst,
                                     const parallel::distributed::Vector<value_type>  &,
                                     const std::pair<unsigned int,unsigned int>       &face_range) const
  {
    FEFaceEval_Velocity_Velocity_nonlinear fe_eval(data,this->fe_param,true,operator_data.dof_index);
    FEFaceEval_Velocity_Velocity_nonlinear fe_eval_linearization(data,this->fe_param,true,operator_data.dof_index);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval_linearization.reinit (face);
      fe_eval_linearization.read_dof_values(*velocity_linearization);
      fe_eval_linearization.evaluate(true,false);

      fe_eval.reinit (face);

      // element-
      VectorizedArray<value_type> local_diagonal_vector[fe_eval.tensor_dofs_per_cell*dim];
      for (unsigned int j=0; j<fe_eval.dofs_per_cell*dim; ++j)
      {
        // set dof value j of element- to 1 and all other dof values of element- to zero
        for (unsigned int i=0; i<fe_eval.dofs_per_cell*dim; ++i)
          fe_eval.write_cellwise_dof_value(i,make_vectorized_array<value_type>(0.));
        fe_eval.write_cellwise_dof_value(j,make_vectorized_array<value_type>(1.));

        // copied from local_apply_linearized_convective_term_boundary_face
        fe_eval.evaluate(true,false);

        typename std::map<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >::iterator it;
        types::boundary_id boundary_id = data.get_boundary_indicator(face);

        for(unsigned int q=0;q<fe_eval.n_q_points;++q)
        {
          it = operator_data.bc->dirichlet_bc.find(boundary_id);
          if(it != operator_data.bc->dirichlet_bc.end())
          {
            // on GammaD: u⁺ = -u⁻ + 2g
            Tensor<1,dim,VectorizedArray<value_type> > uM = fe_eval_linearization.get_value(q);

            Point<dim,VectorizedArray<value_type> > q_points = fe_eval.quadrature_point(q);
            Tensor<1,dim,VectorizedArray<value_type> > g;
            // set correct time for the evaluation of boundary conditions
            it->second->set_time(eval_time);
            for(unsigned int d=0;d<dim;++d)
            {
              value_type array [VectorizedArray<value_type>::n_array_elements];
              for (unsigned int n=0; n<VectorizedArray<value_type>::n_array_elements; ++n)
              {
                Point<dim> q_point;
                for (unsigned int d=0; d<dim; ++d)
                  q_point[d] = q_points[d][n];
                array[n] = it->second->value(q_point,d);
              }
              g[d].load(&array[0]);
            }

            Tensor<1,dim,VectorizedArray<value_type> > uP = -uM + make_vectorized_array<value_type>(2.0)*g;
            Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval.get_normal_vector(q);
            const VectorizedArray<value_type> uM_n = uM*normal;
            const VectorizedArray<value_type> uP_n = uP*normal;

            const VectorizedArray<value_type> lambda = 2.*std::max(std::abs(uM_n), std::abs(uP_n));

            Tensor<1,dim,VectorizedArray<value_type> > delta_uM = fe_eval.get_value(q);
            Tensor<1,dim,VectorizedArray<value_type> > delta_uP = -delta_uM;

            const VectorizedArray<value_type> delta_uM_n = delta_uM*normal;
            const VectorizedArray<value_type> delta_uP_n = delta_uP*normal;

            Tensor<1,dim,VectorizedArray<value_type> > jump_value = delta_uM - delta_uP;
            Tensor<1,dim,VectorizedArray<value_type> > average_normal_flux = make_vectorized_array<value_type>(0.5)*
               (uM*delta_uM_n + delta_uM*uM_n + uP*delta_uP_n + delta_uP*uP_n);
            Tensor<1,dim,VectorizedArray<value_type> > lf_flux = average_normal_flux + 0.5 * lambda * jump_value;

            fe_eval.submit_value(lf_flux,q);
          }

          it = operator_data.bc->neumann_bc.find(boundary_id);
          if(it != operator_data.bc->neumann_bc.end())
          {
            // on GammaN: u⁺ = u⁻
            Tensor<1,dim,VectorizedArray<value_type> > uM = fe_eval_linearization.get_value(q);
            Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval.get_normal_vector(q);
            const VectorizedArray<value_type> uM_n = uM*normal;

            Tensor<1,dim,VectorizedArray<value_type> > delta_uM = fe_eval.get_value(q);
            const VectorizedArray<value_type> delta_uM_n = delta_uM*normal;

            Tensor<1,dim,VectorizedArray<value_type> > average_normal_flux = (uM*delta_uM_n + delta_uM*uM_n);
            Tensor<1,dim,VectorizedArray<value_type> > lf_flux = average_normal_flux;

            fe_eval.submit_value(lf_flux,q);
          }
        }
        fe_eval.integrate(true,false);
        // copied from local_apply_linearized_convective_term_boundary_face

        local_diagonal_vector[j] = fe_eval.read_cellwise_dof_value(j);
      }

      for (unsigned int j=0; j<fe_eval.dofs_per_cell*dim; ++j)
        fe_eval.write_cellwise_dof_value(j, local_diagonal_vector[j]);

      fe_eval.distribute_local_to_global(dst);
    }
  }

  MatrixFree<dim,value_type> const * data;
  ConvectiveOperatorData<dim> operator_data;
  mutable value_type eval_time;
  mutable parallel::distributed::Vector<value_type> const * velocity_linearization;
};

#endif /* INCLUDE_NAVIERSTOKESOPERATORS_H_ */
