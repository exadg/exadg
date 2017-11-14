/*
 * NavierStokesOperators.h
 *
 *  Created on: Jun 6, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_NAVIER_STOKES_OPERATORS_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_NAVIER_STOKES_OPERATORS_H_

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/lac/parallel_vector.h>

#include "../../incompressible_navier_stokes/user_interface/boundary_descriptor.h"
#include "operators/base_operator.h"
#include "../include/functionalities/evaluate_functions.h"

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
  std::shared_ptr<Function<dim> > rhs;
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
  typedef FEEvaluationWrapper<dim,fe_degree,fe_degree_xwall,n_actual_q_points_vel_linear,
                              dim,value_type,is_xwall> FEEval_Velocity_Velocity_linear;

  typedef BodyForceOperator<dim,fe_degree,fe_degree_xwall,xwall_quad_rule,value_type> This;

  void initialize(MatrixFree<dim,value_type> const &mf_data,
                  BodyForceOperatorData<dim> const &operator_data_in)
  {
    this->data = &mf_data;
    this->operator_data = operator_data_in;
  }

  void evaluate(parallel::distributed::Vector<value_type> &dst,
                double const                              evaluation_time) const
  {
    dst = 0;
    evaluate_add(dst,evaluation_time);
  }

  void evaluate_add(parallel::distributed::Vector<value_type> &dst,
                    double const                              evaluation_time) const
  {
    this->eval_time = evaluation_time;

    parallel::distributed::Vector<value_type> src;
    data->cell_loop(&This::cell_loop, this, dst, src);
  }

private:
  void cell_loop (const MatrixFree<dim,value_type>                &data,
                  parallel::distributed::Vector<value_type>       &dst,
                  const parallel::distributed::Vector<value_type> &,
                  const std::pair<unsigned int,unsigned int>      &cell_range) const
  {
    FEEval_Velocity_Velocity_linear fe_eval(data,this->fe_param,operator_data.dof_index);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval.reinit (cell);

      for (unsigned int q=0; q<fe_eval.n_q_points; ++q)
      {
        Point<dim,VectorizedArray<value_type> > q_points = fe_eval.quadrature_point(q);
        Tensor<1,dim,VectorizedArray<value_type> > rhs;

        evaluate_vectorial_function(rhs,operator_data.rhs,q_points,eval_time);

        fe_eval.submit_value (rhs, q);
      }
      fe_eval.integrate (true,false);
      fe_eval.distribute_local_to_global(dst);
    }
  }

  MatrixFree<dim,value_type> const * data;
  BodyForceOperatorData<dim> operator_data;
  double mutable eval_time;
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
  typedef FEEvaluationWrapper<dim,fe_degree,fe_degree_xwall,n_actual_q_points_vel_linear,
                              dim,value_type,is_xwall> FEEval_Velocity_Velocity_linear;

  typedef MassMatrixOperator<dim,fe_degree,fe_degree_xwall,xwall_quad_rule,value_type> This;

  void initialize(MatrixFree<dim,value_type> const &mf_data,
                  MassMatrixOperatorData const     &operator_data_in)
  {
    this->data = &mf_data;
    this->operator_data = operator_data_in;
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
    data->cell_loop(&This::cell_loop, this, dst, src);
  }

  void calculate_diagonal(parallel::distributed::Vector<value_type> &diagonal) const
  {
    diagonal = 0;

    add_diagonal(diagonal);
  }

  void add_diagonal(parallel::distributed::Vector<value_type> &diagonal) const
  {
    parallel::distributed::Vector<value_type>  src_dummy(diagonal);
    data->cell_loop(&This::cell_loop_diagonal, this, diagonal, src_dummy);
  }

  void add_block_jacobi_matrices(std::vector<LAPACKFullMatrix<value_type> > &matrices) const
  {
    parallel::distributed::Vector<value_type>  src;

    data->cell_loop(&This::cell_loop_calculate_block_jacobi_matrices, this, matrices, src);
  }

  MassMatrixOperatorData const & get_operator_data() const
  {
    return operator_data;
  }

  MatrixFree<dim,value_type> const & get_data() const
  {
    return *data;
  }

  FEParameters<dim> * get_fe_param() const
  {
    return this->fe_param;
  }

private:
  template<typename FEEvaluation>
  inline void do_cell_integral(FEEvaluation &fe_eval) const
  {
    fe_eval.evaluate (true,false,false);

    for (unsigned int q=0; q<fe_eval.n_q_points; ++q)
    {
      fe_eval.submit_value (fe_eval.get_value(q), q);
    }
    fe_eval.integrate (true,false);
  }

  void cell_loop (MatrixFree<dim,value_type> const                &data,
                  parallel::distributed::Vector<value_type>       &dst,
                  parallel::distributed::Vector<value_type> const &src,
                  std::pair<unsigned int,unsigned int> const      &cell_range) const
  {
    FEEval_Velocity_Velocity_linear fe_eval(data,this->fe_param,operator_data.dof_index);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);
      fe_eval.read_dof_values(src);

      do_cell_integral(fe_eval);

      fe_eval.distribute_local_to_global (dst);
    }
  }

  void cell_loop_diagonal (MatrixFree<dim,value_type> const                &data,
                           parallel::distributed::Vector<value_type>       &dst,
                           parallel::distributed::Vector<value_type> const &,
                           std::pair<unsigned int,unsigned int> const      &cell_range) const
  {
    FEEval_Velocity_Velocity_linear fe_eval(data,this->fe_param,operator_data.dof_index);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval.reinit (cell);

      VectorizedArray<value_type> local_diagonal_vector[fe_eval.tensor_dofs_per_cell*dim]; //tensor_dofs_per_cell >= dofs_per_cell
      for (unsigned int j=0; j<fe_eval.dofs_per_cell*dim; ++j)
      {
        for (unsigned int i=0; i<fe_eval.dofs_per_cell*dim; ++i)
          fe_eval.write_cellwise_dof_value(i,make_vectorized_array<value_type>(0.));
        fe_eval.write_cellwise_dof_value(j,make_vectorized_array<value_type>(1.));

        do_cell_integral(fe_eval);

        local_diagonal_vector[j] = fe_eval.read_cellwise_dof_value(j);
      }
      for (unsigned int j=0; j<fe_eval.dofs_per_cell*dim; ++j)
        fe_eval.write_cellwise_dof_value(j,local_diagonal_vector[j]);

      fe_eval.distribute_local_to_global (dst);
    }
  }

  void cell_loop_calculate_block_jacobi_matrices (MatrixFree<dim,value_type> const                &data,
                                                  std::vector<LAPACKFullMatrix<value_type> >      &matrices,
                                                  parallel::distributed::Vector<value_type> const &,
                                                  std::pair<unsigned int,unsigned int> const      &cell_range) const
  {
    FEEval_Velocity_Velocity_linear fe_eval(data,this->fe_param,operator_data.dof_index);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);

      // Note that the velocity has dim components.
      unsigned int dofs_per_cell = fe_eval.dofs_per_cell*dim;

      for (unsigned int j=0; j<dofs_per_cell; ++j)
      {
        for (unsigned int i=0; i<dofs_per_cell; ++i)
          fe_eval.begin_dof_values()[i] = make_vectorized_array<value_type>(0.);
        fe_eval.begin_dof_values()[j] = make_vectorized_array<value_type>(1.);

        do_cell_integral(fe_eval);

        for(unsigned int i=0; i<dofs_per_cell; ++i)
          for (unsigned int v=0; v<VectorizedArray<value_type>::n_array_elements; ++v)
            matrices[cell*VectorizedArray<value_type>::n_array_elements+v](i,j) += fe_eval.begin_dof_values()[i][v];
      }
    }
  }

  MatrixFree<dim,value_type> const * data;
  MassMatrixOperatorData operator_data;
};

template<int dim>
struct ViscousOperatorData
{
  ViscousOperatorData ()
    :
    formulation_viscous_term(FormulationViscousTerm::DivergenceFormulation),
    penalty_term_div_formulation(PenaltyTermDivergenceFormulation::Symmetrized),
    IP_formulation_viscous(InteriorPenaltyFormulation::SIPG),
    IP_factor_viscous(1.0),
    dof_index(0),
    viscosity(1.0)
  {}

  FormulationViscousTerm formulation_viscous_term;
  PenaltyTermDivergenceFormulation penalty_term_div_formulation;
  InteriorPenaltyFormulation IP_formulation_viscous;
  double IP_factor_viscous;
  unsigned int dof_index;

  std::shared_ptr<BoundaryDescriptorNavierStokesU<dim> > bc;

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

  enum class OperatorType {
    full,
    homogeneous,
    inhomogeneous
  };

  ViscousOperator()
    :
    data(nullptr),
    const_viscosity(-1.0),
    eval_time(0.0)
  {}

  static const bool is_xwall = (xwall_quad_rule>1) ? true : false;
  static const unsigned int n_actual_q_points_vel_linear = (is_xwall) ? xwall_quad_rule : fe_degree+1;
  typedef FEEvaluationWrapper<dim,fe_degree,fe_degree_xwall,n_actual_q_points_vel_linear,dim,Number,is_xwall>
    FEEval_Velocity_Velocity_linear;
  typedef FEFaceEvaluationWrapper<dim,fe_degree,fe_degree_xwall,n_actual_q_points_vel_linear,dim,Number,is_xwall>
    FEFaceEval_Velocity_Velocity_linear;

  typedef ViscousOperator<dim,fe_degree,fe_degree_xwall,xwall_quad_rule,Number> This;

  void initialize(Mapping<dim> const             &mapping,
                  MatrixFree<dim,Number> const   &mf_data,
                  ViscousOperatorData<dim> const &operator_data_in)
  {

    this->data = &mf_data;
    this->operator_data = operator_data_in;

    compute_array_penalty_parameter(mapping);

    const_viscosity = operator_data.viscosity;
  }

  FEParameters<dim> * get_fe_param() const
  {
    return this->fe_param;
  }

  void set_constant_viscosity(double const viscosity_in)
  {
    const_viscosity = viscosity_in;
  }

  /*
   *  This function returns true if viscous_coefficient table has been filled
   *  with spatially varying viscosity values.
   */
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
    this->viscous_coefficient_cell.reinit(this->data->n_macro_cells(),
                                          Utilities::fixed_int_power<n_actual_q_points_vel_linear,dim>::value);
    this->viscous_coefficient_cell.fill(make_vectorized_array<Number>(const_viscosity));

    this->viscous_coefficient_face.reinit(this->data->n_macro_inner_faces()+this->data->n_macro_boundary_faces(),
                                          Utilities::fixed_int_power<n_actual_q_points_vel_linear,dim-1>::value);
    this->viscous_coefficient_face.fill(make_vectorized_array<Number>(const_viscosity));

    this->viscous_coefficient_face_neighbor.reinit(this->data->n_macro_inner_faces(),
                                                   Utilities::fixed_int_power<n_actual_q_points_vel_linear,dim-1>::value);
    this->viscous_coefficient_face_neighbor.fill(make_vectorized_array<Number>(const_viscosity));

    // TODO: currently, the viscosity dof vector is initialized here
    this->data->initialize_dof_vector(viscosity,operator_data.dof_index);
  }

  void set_viscous_coefficient_cell(unsigned int const            cell,
                                    unsigned int const            q,
                                    VectorizedArray<Number> const &value)
  {
    viscous_coefficient_cell[cell][q] = value;
  }

  void set_viscous_coefficient_face(unsigned int const            face,
                                    unsigned int const            q,
                                    VectorizedArray<Number> const &value)
  {
    viscous_coefficient_face[face][q] = value;
  }

  void set_viscous_coefficient_face_neighbor(unsigned int const            face,
                                             unsigned int const            q,
                                             VectorizedArray<Number> const &value)
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
    data->loop(&This::cell_loop,&This::face_loop,
               &This::boundary_face_loop_hom_operator,this, dst, src);
  }

  // apply matrix vector multiplication for block Jacobi operator
  void apply_block_jacobi (parallel::distributed::Vector<Number>       &dst,
                           const parallel::distributed::Vector<Number> &src) const
  {
    dst = 0;
    apply_block_jacobi_add(dst,src);
  }

  void apply_block_jacobi_add (parallel::distributed::Vector<Number>       &dst,
                               const parallel::distributed::Vector<Number> &src) const
  {
    data->loop(&This::cell_loop,&This::face_loop_block_jacobi,
               &This::boundary_face_loop_hom_operator,this, dst, src);
  }

  void rhs (parallel::distributed::Vector<Number> &dst,
            double const                          evaluation_time) const
  {
    dst = 0;
    rhs_add(dst,evaluation_time);
  }

  void rhs_add (parallel::distributed::Vector<Number> &dst,
                double const                          evaluation_time) const
  {
    this->eval_time = evaluation_time;

    parallel::distributed::Vector<Number> src;
    data->loop(&This::cell_loop_inhom_operator,&This::face_loop_inhom_operator,
               &This::boundary_face_loop_inhom_operator,this, dst, src);
  }

  void evaluate (parallel::distributed::Vector<Number>       &dst,
                 const parallel::distributed::Vector<Number> &src,
                 double const                                evaluation_time) const
  {
    dst = 0;
    evaluate_add(dst,src,evaluation_time);
  }

  void evaluate_add (parallel::distributed::Vector<Number>       &dst,
                     const parallel::distributed::Vector<Number> &src,
                     double const                                evaluation_time) const
  {
    this->eval_time = evaluation_time;

    data->loop(&This::cell_loop,&This::face_loop,
               &This::boundary_face_loop_full_operator,this, dst, src);
  }

  void calculate_diagonal(parallel::distributed::Vector<Number> &diagonal) const
  {
    diagonal = 0;

    add_diagonal(diagonal);
  }

  void add_diagonal(parallel::distributed::Vector<Number> &diagonal) const
  {
    parallel::distributed::Vector<Number>  src_dummy(diagonal);

    data->loop(&This::cell_loop_diagonal,&This::face_loop_diagonal,
               &This::boundary_face_loop_diagonal,this, diagonal, src_dummy);
  }

  void add_block_jacobi_matrices(std::vector<LAPACKFullMatrix<value_type> > &matrices) const
  {
    parallel::distributed::Vector<value_type>  src;

    data->loop(&This::cell_loop_calculate_block_jacobi_matrices,&This::face_loop_calculate_block_jacobi_matrices,
               &This::boundary_face_loop_calculate_block_jacobi_matrices, this, matrices, src);
  }

  ViscousOperatorData<dim> const & get_operator_data() const
  {
    return operator_data;
  }

  void extract_viscous_coefficient_from_dof_vector ()
  {
    parallel::distributed::Vector<Number>  dummy;

    data->loop(&This::cell_loop_extract_viscous_coeff,
               &This::face_loop_extract_viscous_coeff,
               &This::boundary_face_loop_extract_viscous_coeff,
               this, dummy, this->viscosity);
  }

  parallel::distributed::Vector<Number> & get_viscosity_dof_vector()
  {
    return this->viscosity;
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

  /*
   *  This function returns the penalty factor of the interior penalty method
   *  for quadrilateral/hexahedral elements.
   */
  Number get_penalty_factor() const
  {
    return operator_data.IP_factor_viscous * (fe_degree + 1.0) * (fe_degree + 1.0);
  }

  template<typename FEEvaluation>
  inline void do_cell_integral(FEEvaluation &fe_eval, unsigned int const cell) const
  {
    AssertThrow(const_viscosity > 0.0, ExcMessage("Constant viscosity has not been set!"));

    fe_eval.evaluate (false,true,false);

    for (unsigned int q=0; q<fe_eval.n_q_points; ++q)
    {
      VectorizedArray<Number> viscosity = make_vectorized_array<Number>(const_viscosity);
      if(viscosity_is_variable())
        viscosity = viscous_coefficient_cell[cell][q];

      if(operator_data.formulation_viscous_term == FormulationViscousTerm::DivergenceFormulation)
      {
        fe_eval.submit_gradient (viscosity*make_vectorized_array<Number>(2.)*fe_eval.get_symmetric_gradient(q), q);
      }
      else if(operator_data.formulation_viscous_term == FormulationViscousTerm::LaplaceFormulation)
      {
        fe_eval.submit_gradient (viscosity*fe_eval.get_gradient(q), q);
      }
      else
      {
        AssertThrow(operator_data.formulation_viscous_term == FormulationViscousTerm::DivergenceFormulation ||
                    operator_data.formulation_viscous_term == FormulationViscousTerm::LaplaceFormulation,
                    ExcMessage("Specified formulation of viscous term is not implemented."));
      }
    }
    fe_eval.integrate (false,true);
  }

  void cell_loop (const MatrixFree<dim,Number>                 &data,
                  parallel::distributed::Vector<Number>        &dst,
                  const parallel::distributed::Vector<Number>  &src,
                  const std::pair<unsigned int,unsigned int>   &cell_range) const
  {
    FEEval_Velocity_Velocity_linear fe_eval(data,this->fe_param,operator_data.dof_index);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);
      fe_eval.read_dof_values(src);

      do_cell_integral(fe_eval,cell);

      fe_eval.distribute_local_to_global (dst);
    }
  }

  /*
   *  This function calculates the average viscosity for interior faces.
   */
  inline void calculate_average_viscosity(VectorizedArray<Number> &average_viscosity,
                                          unsigned int const      face,
                                          unsigned int const      q) const
  {
    // harmonic mean (harmonic weighting according to Schott and Rasthofer et al. (2015))
    average_viscosity = 2.0 * viscous_coefficient_face[face][q] * viscous_coefficient_face_neighbor[face][q] /
                       (viscous_coefficient_face[face][q] + viscous_coefficient_face_neighbor[face][q]);

    // arithmetic mean
//    average_viscosity = 0.5 * (viscous_coefficient_face[face][q] + viscous_coefficient_face_neighbor[face][q]);

    // maximum value
//    average_viscosity = std::max(viscous_coefficient_face[face][q], viscous_coefficient_face_neighbor[face][q]);
  }


  /*
   *  Calculation of "value_flux".
   */
  template<typename FEEvaluation>
  inline void calculate_value_flux(Tensor<2,dim,VectorizedArray<Number> >       &value_flux,
                                   Tensor<1,dim,VectorizedArray<Number> > const &jump_value,
                                   Tensor<1,dim,VectorizedArray<Number> > const &normal,
                                   VectorizedArray<Number> const                &viscosity,
                                   FEEvaluation const                           &fe_eval) const
  {
    // Value flux
    Tensor<2,dim,VectorizedArray<Number> > jump_tensor = outer_product(jump_value,normal);

    if(operator_data.formulation_viscous_term == FormulationViscousTerm::DivergenceFormulation)
    {
      if(operator_data.IP_formulation_viscous == InteriorPenaltyFormulation::NIPG)
      {
        value_flux = viscosity * fe_eval.make_symmetric(jump_tensor);
      }
      else if(operator_data.IP_formulation_viscous == InteriorPenaltyFormulation::SIPG)
      {
        value_flux = - viscosity * fe_eval.make_symmetric(jump_tensor);
      }
      else
      {
        AssertThrow(operator_data.IP_formulation_viscous == InteriorPenaltyFormulation::NIPG ||
                    operator_data.IP_formulation_viscous == InteriorPenaltyFormulation::SIPG,
                    ExcMessage("Specified interior penalty formulation is not implemented."));
      }
    }
    else if(operator_data.formulation_viscous_term == FormulationViscousTerm::LaplaceFormulation)
    {
      if(operator_data.IP_formulation_viscous == InteriorPenaltyFormulation::NIPG)
      {
        value_flux = 0.5 * viscosity * jump_tensor;
      }
      else if(operator_data.IP_formulation_viscous == InteriorPenaltyFormulation::SIPG)
      {
        value_flux = -0.5 * viscosity * jump_tensor;
      }
      else
      {
        AssertThrow(operator_data.IP_formulation_viscous == InteriorPenaltyFormulation::NIPG ||
                    operator_data.IP_formulation_viscous == InteriorPenaltyFormulation::SIPG,
                    ExcMessage("Specified interior penalty formulation is not implemented."));
      }
    }
    else
    {
      AssertThrow(operator_data.formulation_viscous_term == FormulationViscousTerm::DivergenceFormulation ||
                  operator_data.formulation_viscous_term == FormulationViscousTerm::LaplaceFormulation,
                  ExcMessage("Specified formulation of viscous term is not implemented."));
    }
  }

  /*
   *  This function calculates the jump value = interior_value - exterior_value
   *  depending on the operator type, the type of the boundary face
   *  and the given boundary conditions. The jump_value has to be calculated in order
   *  to evaluate both the value_flux and the gradient_flux.
   *
   *                            +----------------------+--------------------+------------------------+
   *                            | Dirichlet boundaries | Neumann boundaries | symmetry boundaries    |
   *  +-------------------------+----------------------+--------------------+------------------------+
   *  | full operator           | u⁺ = -u⁻ + 2g        | u⁺ = u⁻            | u⁺ = u⁻ - 2(u⁻*n)n     |
   *  +-------------------------+----------------------+--------------------+------------------------+
   *  | homogeneous operator    | u⁺ = -u⁻             | u⁺ = u⁻            | u⁺ = u⁻ - 2(u⁻*n)n     |
   *  +-------------------------+----------------------+--------------------+------------------------+
   *  | inhomogeneous operator  | u⁻ = 0, u⁺ = 2g      | u⁻ = 0, u⁺ = 0     | u⁻ = 0, u+ = 0         |
   *  +-------------------------+----------------------+--------------------+------------------------+
   */
  template<typename FEEvaluation>
  inline void calculate_jump_value_boundary_face(
      Tensor<1,dim,VectorizedArray<Number> > &jump_value,
      unsigned int const                     q,
      FEEvaluation const                     &fe_eval,
      OperatorType const                     &operator_type,
      BoundaryTypeU const                    &boundary_type,
      types::boundary_id const               boundary_id = types::boundary_id()) const
  {
    // velocity on element e⁻
    Tensor<1,dim,VectorizedArray<Number> > velocity_m;

    if(operator_type == OperatorType::full || operator_type == OperatorType::homogeneous)
    {
      velocity_m = fe_eval.get_value(q);
    }
    else if(operator_type == OperatorType::inhomogeneous)
    {
      // do nothing, velocity_m is already initialized with zeros
    }
    else
    {
      AssertThrow(false, ExcMessage("Specified OperatorType is not implemented!"));
    }

    // velocity on element e⁺
    Tensor<1,dim,VectorizedArray<Number> > velocity_p;

    if(operator_type == OperatorType::full)
    {
      if(boundary_type == BoundaryTypeU::Dirichlet)
      {
        Tensor<1,dim,VectorizedArray<Number> > g;
        typename std::map<types::boundary_id,std::shared_ptr<Function<dim> > >::iterator it;
        it = operator_data.bc->dirichlet_bc.find(boundary_id);
        Point<dim,VectorizedArray<Number> > q_points = fe_eval.quadrature_point(q);
        evaluate_vectorial_function(g,it->second,q_points,eval_time);

        velocity_p = - velocity_m + 2.0*g;
      }
      else if(boundary_type == BoundaryTypeU::Neumann)
      {
        velocity_p = velocity_m;
      }
      else if(boundary_type == BoundaryTypeU::Symmetry)
      {
        Tensor<1,dim,VectorizedArray<Number> > normal_m = fe_eval.get_normal_vector(q);
        velocity_p = velocity_m - 2.0 * (velocity_m*normal_m) * normal_m;
      }
      else
      {
        AssertThrow(false,ExcMessage("Boundary type of face is invalid or not implemented."));
      }
    }
    else if(operator_type == OperatorType::homogeneous)
    {
      if(boundary_type == BoundaryTypeU::Dirichlet)
      {
        velocity_p = - velocity_m;
      }
      else if(boundary_type == BoundaryTypeU::Neumann)
      {
        velocity_p = velocity_m;
      }
      else if(boundary_type == BoundaryTypeU::Symmetry)
      {
        Tensor<1,dim,VectorizedArray<Number> > normal_m = fe_eval.get_normal_vector(q);
        velocity_p = velocity_m - 2.0 * (velocity_m*normal_m) * normal_m;
      }
      else
      {
        AssertThrow(false,ExcMessage("Boundary type of face is invalid or not implemented."));
      }
    }
    else if(operator_type == OperatorType::inhomogeneous)
    {
      if(boundary_type == BoundaryTypeU::Dirichlet)
      {
        Tensor<1,dim,VectorizedArray<Number> > g;
        typename std::map<types::boundary_id,std::shared_ptr<Function<dim> > >::iterator it;
        it = operator_data.bc->dirichlet_bc.find(boundary_id);
        Point<dim,VectorizedArray<Number> > q_points = fe_eval.quadrature_point(q);
        evaluate_vectorial_function(g,it->second,q_points,eval_time);

        velocity_p = 2.0*g;
      }
      else if(boundary_type == BoundaryTypeU::Neumann)
      {
        // do nothing since velocity_p is already initialized with zeros
      }
      else if(boundary_type == BoundaryTypeU::Symmetry)
      {
        // do nothing since velocity_p is already initialized with zeros
      }
      else
      {
        AssertThrow(false,ExcMessage("Boundary type of face is invalid or not implemented."));
      }
    }
    else
    {
      AssertThrow(false, ExcMessage("Specified OperatorType is not implemented!"));
    }

    jump_value = velocity_m - velocity_p;
  }

  /*
   *  This function calculates the gradient in normal direction on element e
   *  depending on the formulation of the viscous term.
   */
  template<typename FEEvaluation>
  inline void calculate_normal_gradient(Tensor<1,dim,VectorizedArray<Number> > &normal_gradient,
                                        unsigned int const                     q,
                                        FEEvaluation                           &fe_eval) const
  {
    Tensor<2,dim,VectorizedArray<Number> > gradient;

    if(operator_data.formulation_viscous_term == FormulationViscousTerm::DivergenceFormulation)
    {
      /*
       * F = 2 * nu * symmetric_gradient
       *   = 2.0 * nu * 1/2 (grad(u) + grad(u)^T)
       */
      gradient = make_vectorized_array<Number>(2.0) * fe_eval.get_symmetric_gradient(q);
    }
    else if(operator_data.formulation_viscous_term == FormulationViscousTerm::LaplaceFormulation)
    {
      /*
       *  F = nu * grad(u)
       */
      gradient = fe_eval.get_gradient(q);
    }
    else
    {
      AssertThrow(operator_data.formulation_viscous_term == FormulationViscousTerm::DivergenceFormulation ||
                  operator_data.formulation_viscous_term == FormulationViscousTerm::LaplaceFormulation,
                  ExcMessage("Specified formulation of viscous term is not implemented."));
    }

    normal_gradient = gradient * fe_eval.get_normal_vector(q);
  }

  /*
   *  This function calculates the average gradient in normal direction for
   *  interior faces depending on the formulation of the viscous term.
   *  The average normal gradient has to be calculated in order to evaluate
   *   the gradient flux.
   */
  template<typename FEEvaluation>
  inline void calculate_average_normal_gradient(Tensor<1,dim,VectorizedArray<Number> > &average_normal_gradient,
                                                unsigned int const                     q,
                                                FEEvaluation                           &fe_eval,
                                                FEEvaluation                           &fe_eval_neighbor) const
  {
    Tensor<2,dim,VectorizedArray<Number> > average_gradient;

    if(operator_data.formulation_viscous_term == FormulationViscousTerm::DivergenceFormulation)
    {
      /*
       * {{F}} = (F⁻ + F⁺)/2 where F = 2 * nu * symmetric_gradient
       *   -> {{F}} = nu * (symmetric_gradient⁻ + symmetric_gradient⁺)
       */
      average_gradient = fe_eval.get_symmetric_gradient(q) + fe_eval_neighbor.get_symmetric_gradient(q);
    }
    else if(operator_data.formulation_viscous_term == FormulationViscousTerm::LaplaceFormulation)
    {
      /*
       * {{F}} = (F⁻ + F⁺)/2 where F = nu * grad(u)
       *   -> {{F}} = 0.5 * nu * (grad(u⁻) + grad(u⁺))
       */
      average_gradient = make_vectorized_array<Number>(0.5) * (fe_eval.get_gradient(q) + fe_eval_neighbor.get_gradient(q));
    }
    else
    {
      AssertThrow(operator_data.formulation_viscous_term == FormulationViscousTerm::DivergenceFormulation ||
                  operator_data.formulation_viscous_term == FormulationViscousTerm::LaplaceFormulation,
                  ExcMessage("Specified formulation of viscous term is not implemented."));
    }

    average_normal_gradient = average_gradient * fe_eval.get_normal_vector(q);
  }

  /*
   *  Calculation of gradient flux. Strictly speaking, this value is not a numerical flux since
   *  the flux is multiplied by the normal vector, i.e., "gradient_flux" = numerical_flux * normal,
   *  where normal denotes the normal vector of element e⁻.
   */
  inline void calculate_gradient_flux(Tensor<1,dim,VectorizedArray<Number> >       &gradient_flux,
                                      Tensor<1,dim,VectorizedArray<Number> > const &average_normal_gradient,
                                      Tensor<1,dim,VectorizedArray<Number> > const &jump_value,
                                      Tensor<1,dim,VectorizedArray<Number> > const &normal,
                                      VectorizedArray<Number> const                &viscosity,
                                      VectorizedArray<Number> const                &penalty_parameter) const
  {
    if(operator_data.formulation_viscous_term == FormulationViscousTerm::DivergenceFormulation)
    {
      if(operator_data.penalty_term_div_formulation == PenaltyTermDivergenceFormulation::Symmetrized)
      {
        //Tensor<2,dim,VectorizedArray<Number> > jump_tensor = outer_product(jump_value,normal);
        //gradient_flux = viscosity * average_normal_gradient
        //                - viscosity * penalty_parameter * (jump_tensor + transpose(jump_tensor)) * normal;

        gradient_flux = viscosity * average_normal_gradient
                        - viscosity * penalty_parameter * (jump_value + (jump_value * normal) * normal);
      }
      else if(operator_data.penalty_term_div_formulation == PenaltyTermDivergenceFormulation::NotSymmetrized)
      {
        gradient_flux = viscosity * average_normal_gradient - viscosity * penalty_parameter * jump_value;
      }
      else
      {
        AssertThrow(operator_data.penalty_term_div_formulation == PenaltyTermDivergenceFormulation::Symmetrized ||
            operator_data.penalty_term_div_formulation == PenaltyTermDivergenceFormulation::NotSymmetrized,
                    ExcMessage("Specified formulation of viscous term is not implemented."));
      }
    }
    else if(operator_data.formulation_viscous_term == FormulationViscousTerm::LaplaceFormulation)
    {
      gradient_flux = viscosity * average_normal_gradient - viscosity * penalty_parameter * jump_value;
    }
    else
    {
      AssertThrow(operator_data.formulation_viscous_term == FormulationViscousTerm::DivergenceFormulation ||
                  operator_data.formulation_viscous_term == FormulationViscousTerm::LaplaceFormulation,
                  ExcMessage("Specified formulation of viscous term is not implemented."));
    }
  }


  /*
   *  This function calculates the average velocity gradient in normal
   *  direction depending on the operator type, the type of the boundary face
   *  and the given boundary conditions. The average normal gradient has to
   *  be calculated in order to evaluate the gradient flux.
   *
   *  Divergence formulation: F(u) = nu * ( grad(u) + grad(u)^T )
   *  Laplace formulation: F(u) = nu * grad(u)
   *
   *                            +---------------------------------+-----------------------------------+---------------------------------------+
   *                            | Dirichlet boundaries            | Neumann boundaries                | symmetry boundaries                   |
   *  +-------------------------+---------------------------------+-----------------------------------+---------------------------------------+
   *  | full operator           | F(u⁺)*n = F(u⁻)*n               | F(u⁺)*n = -F(u⁻)*n + 2h           | F(u⁺)*n = -F(u⁻)*n + 2*[(F(u⁻)*n)*n]n |
   *  +-------------------------+---------------------------------+-----------------------------------+---------------------------------------+
   *  | homogeneous operator    | F(u⁺)*n = F(u⁻)*n               | F(u⁺)*n = -F(u⁻)*n                | F(u⁺)*n = -F(u⁻)*n + 2*[(F(u⁻)*n)*n]n |
   *  +-------------------------+---------------------------------+-----------------------------------+---------------------------------------+
   *  | inhomogeneous operator  | F(u⁻)*n  = 0, F(u⁺)*n = 0       | F(u⁻)*n  = 0, F(u⁺)*n = 2h        | F(u⁻)*n  = 0, F(u⁺)*n = 0             |
   *  +-------------------------+---------------------------------+-----------------------------------+---------------------------------------+
   *
   *                            +---------------------------------+-----------------------------------+---------------------------------------+
   *                            | Dirichlet boundaries            | Neumann boundaries                | symmetry boundaries                   |
   *  +-------------------------+---------------------------------+-----------------------------------+---------------------------------------+
   *  | full operator           | {{F(u)}}*n = F(u⁻)*n            | {{F(u)}}*n = h                    | {{F(u)}}*n = 2*[(F(u⁻)*n)*n]n         |
   *  +-------------------------+---------------------------------+-----------------------------------+---------------------------------------+
   *  | homogeneous operator    | {{F(u)}}*n = F(u⁻)*n            | {{F(u)}}*n = 0                    | {{F(u)}}*n = 2*[(F(u⁻)*n)*n]n         |
   *  +-------------------------+---------------------------------+-----------------------------------+---------------------------------------+
   *  | inhomogeneous operator  | {{F(u)}}*n = 0                  | {{F(u)}}*n = h                    | {{F(u)}}*n = 0                        |
   *  +-------------------------+---------------------------------+-----------------------------------+---------------------------------------+
   */
  template<typename FEEvaluation>
  inline void calculate_average_normal_gradient_boundary_face(
      Tensor<1,dim,VectorizedArray<Number> > &average_normal_gradient,
      unsigned int const                     q,
      FEEvaluation const                     &fe_eval,
      OperatorType const                     &operator_type,
      BoundaryTypeU const                    &boundary_type,
      types::boundary_id const               boundary_id = types::boundary_id()) const
  {
    if(operator_type == OperatorType::full)
    {
      if(boundary_type == BoundaryTypeU::Dirichlet)
      {
        calculate_normal_gradient(average_normal_gradient,q,fe_eval);
      }
      else if(boundary_type == BoundaryTypeU::Neumann)
      {
        Tensor<1,dim,VectorizedArray<Number> > h;
        typename std::map<types::boundary_id,std::shared_ptr<Function<dim> > >::iterator it;
        it = operator_data.bc->neumann_bc.find(boundary_id);
        Point<dim,VectorizedArray<Number> > q_points = fe_eval.quadrature_point(q);
        evaluate_vectorial_function(h,it->second,q_points,eval_time);

        average_normal_gradient = h;
      }
      else if(boundary_type == BoundaryTypeU::Symmetry)
      {
        calculate_normal_gradient(average_normal_gradient,q,fe_eval);
        Tensor<1,dim,VectorizedArray<Number> > normal_m = fe_eval.get_normal_vector(q);
        average_normal_gradient = 2.0 * (average_normal_gradient*normal_m) * normal_m;
      }
      else
      {
        AssertThrow(false,ExcMessage("Boundary type of face is invalid or not implemented."));
      }
    }
    else if(operator_type == OperatorType::homogeneous)
    {
      if(boundary_type == BoundaryTypeU::Dirichlet)
      {
        calculate_normal_gradient(average_normal_gradient,q,fe_eval);
      }
      else if(boundary_type == BoundaryTypeU::Neumann)
      {
        // do nothing since average_normal_gradient is already initialized with zeros
      }
      else if(boundary_type == BoundaryTypeU::Symmetry)
      {
        calculate_normal_gradient(average_normal_gradient,q,fe_eval);
        Tensor<1,dim,VectorizedArray<Number> > normal_m = fe_eval.get_normal_vector(q);
        average_normal_gradient = 2.0 * (average_normal_gradient*normal_m) * normal_m;
      }
      else
      {
        AssertThrow(false,ExcMessage("Boundary type of face is invalid or not implemented."));
      }
    }
    else if(operator_type == OperatorType::inhomogeneous)
    {
      if(boundary_type == BoundaryTypeU::Dirichlet)
      {
        // do nothing since average_normal_gradient is already initialized with zeros
      }
      else if(boundary_type == BoundaryTypeU::Neumann)
      {
        Tensor<1,dim,VectorizedArray<Number> > h;
        typename std::map<types::boundary_id,std::shared_ptr<Function<dim> > >::iterator it;
        it = operator_data.bc->neumann_bc.find(boundary_id);
        Point<dim,VectorizedArray<Number> > q_points = fe_eval.quadrature_point(q);
        evaluate_vectorial_function(h,it->second,q_points,eval_time);

        average_normal_gradient = h;
      }
      else if(boundary_type == BoundaryTypeU::Symmetry)
      {
        // do nothing since average_normal_gradient is already initialized with zeros
      }
      else
      {
        AssertThrow(false,ExcMessage("Boundary type of face is invalid or not implemented."));
      }
    }
    else
    {
      AssertThrow(false, ExcMessage("Specified OperatorType is not implemented!"));
    }
  }

  void face_loop (const MatrixFree<dim,Number>                &data,
                  parallel::distributed::Vector<Number>       &dst,
                  const parallel::distributed::Vector<Number> &src,
                  const std::pair<unsigned int,unsigned int>  &face_range) const
  {
    FEFaceEval_Velocity_Velocity_linear fe_eval(data,this->fe_param,true,operator_data.dof_index);
    FEFaceEval_Velocity_Velocity_linear fe_eval_neighbor(data,this->fe_param,false,operator_data.dof_index);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval.reinit (face);
      fe_eval_neighbor.reinit (face);

      fe_eval.read_dof_values(src);
      fe_eval_neighbor.read_dof_values(src);

      fe_eval.evaluate(true,true);
      fe_eval_neighbor.evaluate(true,true);

      VectorizedArray<Number> penalty_parameter = get_penalty_factor() *
          std::max(fe_eval.read_cell_data(array_penalty_parameter),fe_eval_neighbor.read_cell_data(array_penalty_parameter));

      for(unsigned int q=0;q<fe_eval.n_q_points;++q)
      {
        VectorizedArray<Number> average_viscosity = make_vectorized_array<Number>(const_viscosity);
        if(viscosity_is_variable())
          calculate_average_viscosity(average_viscosity,face,q);

        Tensor<1,dim,VectorizedArray<Number> > jump_value = fe_eval.get_value(q) - fe_eval_neighbor.get_value(q);
        Tensor<1,dim,VectorizedArray<Number> > normal = fe_eval.get_normal_vector(q);

        Tensor<2,dim,VectorizedArray<Number> > value_flux;
        calculate_value_flux(value_flux,jump_value,normal,average_viscosity,fe_eval);

        Tensor<1,dim,VectorizedArray<Number> > average_normal_gradient;
        calculate_average_normal_gradient(average_normal_gradient,q,fe_eval,fe_eval_neighbor);

        Tensor<1,dim,VectorizedArray<Number> > gradient_flux;
        calculate_gradient_flux(gradient_flux,average_normal_gradient,jump_value,normal,average_viscosity,penalty_parameter);

        fe_eval.submit_gradient(value_flux,q);
        fe_eval_neighbor.submit_gradient(value_flux,q);

        fe_eval.submit_value(-gradient_flux,q);
        fe_eval_neighbor.submit_value(gradient_flux,q); // + sign since n⁺ = -n⁻
      }
      fe_eval.integrate(true,true);
      fe_eval_neighbor.integrate(true,true);

      fe_eval.distribute_local_to_global(dst);
      fe_eval_neighbor.distribute_local_to_global(dst);
    }
  }

  void boundary_face_loop_hom_operator (const MatrixFree<dim,Number>                 &data,
                                        parallel::distributed::Vector<Number>        &dst,
                                        const parallel::distributed::Vector<Number>  &src,
                                        const std::pair<unsigned int,unsigned int>   &face_range) const
  {
    FEFaceEval_Velocity_Velocity_linear fe_eval(data,this->fe_param,true,operator_data.dof_index);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      types::boundary_id boundary_id = data.get_boundary_id(face);
      BoundaryTypeU boundary_type = BoundaryTypeU::Undefined;

      if(operator_data.bc->dirichlet_bc.find(boundary_id) != operator_data.bc->dirichlet_bc.end())
        boundary_type = BoundaryTypeU::Dirichlet;
      else if(operator_data.bc->neumann_bc.find(boundary_id) != operator_data.bc->neumann_bc.end())
        boundary_type = BoundaryTypeU::Neumann;
      else if(operator_data.bc->symmetry_bc.find(boundary_id) != operator_data.bc->symmetry_bc.end())
        boundary_type = BoundaryTypeU::Symmetry;

      AssertThrow(boundary_type != BoundaryTypeU::Undefined,
          ExcMessage("Boundary type of face is invalid or not implemented."));

      fe_eval.reinit (face);
      fe_eval.read_dof_values(src);
      fe_eval.evaluate(true,true);

      VectorizedArray<Number> penalty_parameter = get_penalty_factor() * fe_eval.read_cell_data(array_penalty_parameter);

      for(unsigned int q=0;q<fe_eval.n_q_points;++q)
      {
        VectorizedArray<Number> viscosity = make_vectorized_array<Number>(const_viscosity);
        if(viscosity_is_variable())
          viscosity = viscous_coefficient_face[face][q];

        Tensor<1,dim,VectorizedArray<Number> > jump_value;
        calculate_jump_value_boundary_face(jump_value,q,fe_eval,OperatorType::homogeneous,boundary_type);
        Tensor<1,dim,VectorizedArray<Number> > normal = fe_eval.get_normal_vector(q);

        Tensor<2,dim,VectorizedArray<Number> > value_flux;
        calculate_value_flux(value_flux,jump_value,normal,viscosity,fe_eval);

        Tensor<1,dim,VectorizedArray<Number> > average_normal_gradient;
        calculate_average_normal_gradient_boundary_face(average_normal_gradient,q,fe_eval,OperatorType::homogeneous,boundary_type);

        Tensor<1,dim,VectorizedArray<Number> > gradient_flux;
        calculate_gradient_flux(gradient_flux,average_normal_gradient,jump_value,normal,viscosity,penalty_parameter);

        fe_eval.submit_gradient(value_flux,q);
        fe_eval.submit_value(-gradient_flux,q);
      }
      fe_eval.integrate(true,true);
      fe_eval.distribute_local_to_global(dst);
    }
  }

  void boundary_face_loop_full_operator (const MatrixFree<dim,Number>                 &data,
                                         parallel::distributed::Vector<Number>        &dst,
                                         const parallel::distributed::Vector<Number>  &src,
                                         const std::pair<unsigned int,unsigned int>   &face_range) const
  {
    FEFaceEval_Velocity_Velocity_linear fe_eval(data,this->fe_param,true,operator_data.dof_index);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      types::boundary_id boundary_id = data.get_boundary_id(face);
      BoundaryTypeU boundary_type = BoundaryTypeU::Undefined;

      if(operator_data.bc->dirichlet_bc.find(boundary_id) != operator_data.bc->dirichlet_bc.end())
        boundary_type = BoundaryTypeU::Dirichlet;
      else if(operator_data.bc->neumann_bc.find(boundary_id) != operator_data.bc->neumann_bc.end())
        boundary_type = BoundaryTypeU::Neumann;
      else if(operator_data.bc->symmetry_bc.find(boundary_id) != operator_data.bc->symmetry_bc.end())
        boundary_type = BoundaryTypeU::Symmetry;

      AssertThrow(boundary_type != BoundaryTypeU::Undefined,
          ExcMessage("Boundary type of face is invalid or not implemented."));

      fe_eval.reinit (face);
      fe_eval.read_dof_values(src);
      fe_eval.evaluate(true,true);

      VectorizedArray<Number> penalty_parameter = get_penalty_factor() * fe_eval.read_cell_data(array_penalty_parameter);

      for(unsigned int q=0;q<fe_eval.n_q_points;++q)
      {
        VectorizedArray<Number> viscosity = make_vectorized_array<Number>(const_viscosity);
        if(viscosity_is_variable())
          viscosity = viscous_coefficient_face[face][q];

        Tensor<1,dim,VectorizedArray<Number> > jump_value;
        calculate_jump_value_boundary_face(jump_value,q,fe_eval,OperatorType::full,boundary_type,boundary_id);
        Tensor<1,dim,VectorizedArray<Number> > normal = fe_eval.get_normal_vector(q);

        Tensor<2,dim,VectorizedArray<Number> > value_flux;
        calculate_value_flux(value_flux,jump_value,normal,viscosity,fe_eval);

        Tensor<1,dim,VectorizedArray<Number> > average_normal_gradient;
        calculate_average_normal_gradient_boundary_face(average_normal_gradient,q,fe_eval,OperatorType::full,boundary_type,boundary_id);

        Tensor<1,dim,VectorizedArray<Number> > gradient_flux;
        calculate_gradient_flux(gradient_flux,average_normal_gradient,jump_value,normal,viscosity,penalty_parameter);

        fe_eval.submit_gradient(value_flux,q);
        fe_eval.submit_value(-gradient_flux,q);
      }

      fe_eval.integrate(true,true);
      fe_eval.distribute_local_to_global(dst);
    }
  }

  void cell_loop_inhom_operator (const MatrixFree<dim,Number>                 &,
                                 parallel::distributed::Vector<Number>        &,
                                 const parallel::distributed::Vector<Number>  &,
                                 const std::pair<unsigned int,unsigned int>   &) const
  {

  }

  void face_loop_inhom_operator (const MatrixFree<dim,Number>                 &,
                                 parallel::distributed::Vector<Number>        &,
                                 const parallel::distributed::Vector<Number>  &,
                                 const std::pair<unsigned int,unsigned int>   &) const
  {

  }

  void boundary_face_loop_inhom_operator (const MatrixFree<dim,Number>                &data,
                                          parallel::distributed::Vector<Number>       &dst,
                                          const parallel::distributed::Vector<Number> &,
                                          const std::pair<unsigned int,unsigned int>  &face_range) const
  {
    FEFaceEval_Velocity_Velocity_linear fe_eval(data,this->fe_param,true,operator_data.dof_index);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      types::boundary_id boundary_id = data.get_boundary_id(face);
      BoundaryTypeU boundary_type = BoundaryTypeU::Undefined;

      if(operator_data.bc->dirichlet_bc.find(boundary_id) != operator_data.bc->dirichlet_bc.end())
        boundary_type = BoundaryTypeU::Dirichlet;
      else if(operator_data.bc->neumann_bc.find(boundary_id) != operator_data.bc->neumann_bc.end())
        boundary_type = BoundaryTypeU::Neumann;
      else if(operator_data.bc->symmetry_bc.find(boundary_id) != operator_data.bc->symmetry_bc.end())
        boundary_type = BoundaryTypeU::Symmetry;

      AssertThrow(boundary_type != BoundaryTypeU::Undefined,
          ExcMessage("Boundary type of face is invalid or not implemented."));

      fe_eval.reinit (face);

      VectorizedArray<Number> penalty_parameter = get_penalty_factor() * fe_eval.read_cell_data(array_penalty_parameter);

      for(unsigned int q=0;q<fe_eval.n_q_points;++q)
      {
        VectorizedArray<Number> viscosity = make_vectorized_array<Number>(const_viscosity);
        if(viscosity_is_variable())
          viscosity = viscous_coefficient_face[face][q];

        Tensor<1,dim,VectorizedArray<Number> > jump_value;
        calculate_jump_value_boundary_face(jump_value,q,fe_eval,OperatorType::inhomogeneous,boundary_type,boundary_id);
        Tensor<1,dim,VectorizedArray<Number> > normal = fe_eval.get_normal_vector(q);

        Tensor<2,dim,VectorizedArray<Number> > value_flux;
        calculate_value_flux(value_flux,jump_value,normal,viscosity,fe_eval);

        Tensor<1,dim,VectorizedArray<Number> > average_normal_gradient;
        calculate_average_normal_gradient_boundary_face(average_normal_gradient,q,fe_eval,OperatorType::inhomogeneous,boundary_type,boundary_id);

        Tensor<1,dim,VectorizedArray<Number> > gradient_flux;
        calculate_gradient_flux(gradient_flux,average_normal_gradient,jump_value,normal,viscosity,penalty_parameter);

        fe_eval.submit_gradient(-value_flux,q); // - sign since this term appears on the rhs of the equations
        fe_eval.submit_value(gradient_flux,q); // + sign since this term appears on the rhs of the equations
      }
      fe_eval.integrate(true,true);
      fe_eval.distribute_local_to_global(dst);
    }
  }


  /*
   *  Block-jacobi operator: re-implement face_loop; cell_loop and boundary_face_loop are
   *  identical to homogeneous operator.
   */
  void face_loop_block_jacobi (const MatrixFree<dim,Number>                &data,
                               parallel::distributed::Vector<Number>       &dst,
                               const parallel::distributed::Vector<Number> &src,
                               const std::pair<unsigned int,unsigned int>  &face_range) const
  {
    FEFaceEval_Velocity_Velocity_linear fe_eval(data,this->fe_param,true,operator_data.dof_index);
    FEFaceEval_Velocity_Velocity_linear fe_eval_neighbor(data,this->fe_param,false,operator_data.dof_index);

    // perform face integral for element e⁻
    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval.reinit (face);
      fe_eval.read_dof_values(src);
      fe_eval.evaluate(true,true);

      fe_eval_neighbor.reinit (face);

      VectorizedArray<Number> penalty_parameter = get_penalty_factor() *
          std::max(fe_eval.read_cell_data(array_penalty_parameter),fe_eval_neighbor.read_cell_data(array_penalty_parameter));

      // integrate over face for element e⁻
      for(unsigned int q=0;q<fe_eval.n_q_points;++q)
      {
        VectorizedArray<Number> average_viscosity = make_vectorized_array<Number>(const_viscosity);
        if(viscosity_is_variable())
          calculate_average_viscosity(average_viscosity,face,q);

        // set exterior values to zero
        Tensor<1,dim,VectorizedArray<Number> > jump_value = fe_eval.get_value(q);
        Tensor<1,dim,VectorizedArray<Number> > normal = fe_eval.get_normal_vector(q);

        Tensor<2,dim,VectorizedArray<Number> > value_flux;
        calculate_value_flux(value_flux,jump_value,normal,average_viscosity,fe_eval);

        Tensor<1,dim,VectorizedArray<Number> > average_normal_gradient;
        calculate_normal_gradient(average_normal_gradient,q,fe_eval);
        // set exterior values to zero
        average_normal_gradient = make_vectorized_array<Number>(0.5) * average_normal_gradient;

        Tensor<1,dim,VectorizedArray<Number> > gradient_flux;
        calculate_gradient_flux(gradient_flux,average_normal_gradient,jump_value,normal,average_viscosity,penalty_parameter);

        fe_eval.submit_gradient(value_flux,q);
        fe_eval.submit_value(-gradient_flux,q);
      }
      fe_eval.integrate(true,true);

      fe_eval.distribute_local_to_global(dst);
    }


    // TODO: This has to be removed as soon as the new infrastructure is used that
    // allows to perform face integrals over all faces of the current element.
    // perform face integral for element e⁺
    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval.reinit (face);

      fe_eval_neighbor.reinit (face);
      fe_eval_neighbor.read_dof_values(src);
      fe_eval_neighbor.evaluate(true,true);

      VectorizedArray<Number> penalty_parameter = get_penalty_factor() *
          std::max(fe_eval.read_cell_data(array_penalty_parameter),fe_eval_neighbor.read_cell_data(array_penalty_parameter));

      // integrate over face for element e⁺
      for(unsigned int q=0;q<fe_eval_neighbor.n_q_points;++q)
      {
        VectorizedArray<Number> average_viscosity = make_vectorized_array<Number>(const_viscosity);
        if(viscosity_is_variable())
          calculate_average_viscosity(average_viscosity,face,q);

        // set exterior values to zero
        Tensor<1,dim,VectorizedArray<Number> > jump_value = fe_eval_neighbor.get_value(q);
        // multiply by -1.0 to get the correct normal vector !!!
        Tensor<1,dim,VectorizedArray<Number> > normal = - fe_eval_neighbor.get_normal_vector(q);

        Tensor<2,dim,VectorizedArray<Number> > value_flux;
        calculate_value_flux(value_flux,jump_value,normal,average_viscosity,fe_eval_neighbor);

        Tensor<1,dim,VectorizedArray<Number> > average_normal_gradient;
        calculate_normal_gradient(average_normal_gradient,q,fe_eval_neighbor);
        // set exterior values to zero
        // and multiply by -1.0 since normal vector n⁺ = -n⁻ !!!
        average_normal_gradient = make_vectorized_array<Number>(-0.5) * average_normal_gradient;

        Tensor<1,dim,VectorizedArray<Number> > gradient_flux;
        calculate_gradient_flux(gradient_flux,average_normal_gradient,jump_value,normal,average_viscosity,penalty_parameter);

        fe_eval_neighbor.submit_gradient(value_flux,q);
        fe_eval_neighbor.submit_value(-gradient_flux,q);
      }
      fe_eval_neighbor.integrate(true,true);

      fe_eval_neighbor.distribute_local_to_global(dst);
    }
  }

  /*
   *  Calculation of diagonal.
   */
  void cell_loop_diagonal (const MatrixFree<dim,Number>                 &data,
                           parallel::distributed::Vector<Number>        &dst,
                           const parallel::distributed::Vector<Number>  &,
                           const std::pair<unsigned int,unsigned int>   &cell_range) const
  {
    FEEval_Velocity_Velocity_linear fe_eval(data,this->fe_param,operator_data.dof_index);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval.reinit (cell);

      VectorizedArray<Number> local_diagonal_vector[fe_eval.tensor_dofs_per_cell*dim];
      for (unsigned int j=0; j<fe_eval.dofs_per_cell*dim; ++j)
      {
        for (unsigned int i=0; i<fe_eval.dofs_per_cell*dim; ++i)
          fe_eval.write_cellwise_dof_value(i,make_vectorized_array<Number>(0.));
        fe_eval.write_cellwise_dof_value(j,make_vectorized_array<Number>(1.));

        do_cell_integral(fe_eval,cell);

        local_diagonal_vector[j] = fe_eval.read_cellwise_dof_value(j);
      }
      for (unsigned int j=0; j<fe_eval.dofs_per_cell*dim; ++j)
        fe_eval.write_cellwise_dof_value(j,local_diagonal_vector[j]);
      fe_eval.distribute_local_to_global (dst);
    }
  }

  void face_loop_diagonal (const MatrixFree<dim,Number>                &data,
                           parallel::distributed::Vector<Number>       &dst,
                           const parallel::distributed::Vector<Number> &,
                           const std::pair<unsigned int,unsigned int>  &face_range) const
  {
    FEFaceEval_Velocity_Velocity_linear fe_eval(data,this->fe_param,true,operator_data.dof_index);
    FEFaceEval_Velocity_Velocity_linear fe_eval_neighbor(data,this->fe_param,false,operator_data.dof_index);

    // perform face intergrals for element e⁻
    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval.reinit (face);
      fe_eval_neighbor.reinit (face);

      VectorizedArray<Number> penalty_parameter = get_penalty_factor() *
          std::max(fe_eval.read_cell_data(array_penalty_parameter),fe_eval_neighbor.read_cell_data(array_penalty_parameter));

      VectorizedArray<Number> local_diagonal_vector[fe_eval.tensor_dofs_per_cell*dim];
      for (unsigned int j=0; j<fe_eval.dofs_per_cell*dim; ++j)
      {
        // set dof value j of element- to 1 and all other dof values of element- to zero
        for (unsigned int i=0; i<fe_eval.dofs_per_cell*dim; ++i)
          fe_eval.write_cellwise_dof_value(i,make_vectorized_array<Number>(0.));
        fe_eval.write_cellwise_dof_value(j,make_vectorized_array<Number>(1.));

        fe_eval.evaluate(true,true);

        for(unsigned int q=0;q<fe_eval.n_q_points;++q)
        {
          VectorizedArray<Number> average_viscosity = make_vectorized_array<Number>(const_viscosity);
          if(viscosity_is_variable())
            calculate_average_viscosity(average_viscosity,face,q);

          // set exterior values to zero
          Tensor<1,dim,VectorizedArray<Number> > jump_value = fe_eval.get_value(q);
          Tensor<1,dim,VectorizedArray<Number> > normal = fe_eval.get_normal_vector(q);

          Tensor<2,dim,VectorizedArray<Number> > value_flux;
          calculate_value_flux(value_flux,jump_value,normal,average_viscosity,fe_eval);

          Tensor<1,dim,VectorizedArray<Number> > average_normal_gradient;
          calculate_normal_gradient(average_normal_gradient,q,fe_eval);
          // set exterior values to zero
          average_normal_gradient = make_vectorized_array<Number>(0.5) * average_normal_gradient;

          Tensor<1,dim,VectorizedArray<Number> > gradient_flux;
          calculate_gradient_flux(gradient_flux,average_normal_gradient,jump_value,normal,average_viscosity,penalty_parameter);

          fe_eval.submit_gradient(value_flux,q);
          fe_eval.submit_value(-gradient_flux,q);
        }
        fe_eval.integrate(true,true);

        local_diagonal_vector[j] = fe_eval.read_cellwise_dof_value(j);
      }
      for (unsigned int j=0; j<fe_eval.dofs_per_cell*dim; ++j)
        fe_eval.write_cellwise_dof_value(j, local_diagonal_vector[j]);

      fe_eval.distribute_local_to_global(dst);

    }

    // TODO: This has to be removed as soon as the new infrastructure is used that
    // allows to perform face integrals over all faces of the current element.
    // Perform face intergrals for element e⁺.
    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval.reinit (face);
      fe_eval_neighbor.reinit (face);

      VectorizedArray<Number> penalty_parameter = get_penalty_factor() *
          std::max(fe_eval.read_cell_data(array_penalty_parameter),fe_eval_neighbor.read_cell_data(array_penalty_parameter));

      VectorizedArray<Number> local_diagonal_vector_neighbor[fe_eval_neighbor.tensor_dofs_per_cell*dim];
      for (unsigned int j=0; j<fe_eval_neighbor.dofs_per_cell*dim; ++j)
      {
        // set dof value j of element+ to 1 and all other dof values of element+ to zero
        for (unsigned int i=0; i<fe_eval_neighbor.dofs_per_cell*dim; ++i)
          fe_eval_neighbor.write_cellwise_dof_value(i, make_vectorized_array<Number>(0.));
        fe_eval_neighbor.write_cellwise_dof_value(j,make_vectorized_array<Number>(1.));

        fe_eval_neighbor.evaluate(true,true);

        for(unsigned int q=0;q<fe_eval_neighbor.n_q_points;++q)
        {
          VectorizedArray<Number> average_viscosity = make_vectorized_array<Number>(const_viscosity);
          if(viscosity_is_variable())
            calculate_average_viscosity(average_viscosity,face,q);

          // set exterior values to zero
          Tensor<1,dim,VectorizedArray<Number> > jump_value = fe_eval_neighbor.get_value(q);
          // multiply by -1.0 to get the correct normal vector !!!
          Tensor<1,dim,VectorizedArray<Number> > normal = - fe_eval_neighbor.get_normal_vector(q);

          Tensor<2,dim,VectorizedArray<Number> > value_flux;
          calculate_value_flux(value_flux,jump_value,normal,average_viscosity,fe_eval_neighbor);

          Tensor<1,dim,VectorizedArray<Number> > average_normal_gradient;
          calculate_normal_gradient(average_normal_gradient,q,fe_eval_neighbor);
          // set exterior values to zero
          // and multiply by -1.0 since normal vector n⁺ = -n⁻ !!!
          average_normal_gradient = make_vectorized_array<Number>(-0.5) * average_normal_gradient;

          Tensor<1,dim,VectorizedArray<Number> > gradient_flux;
          calculate_gradient_flux(gradient_flux,average_normal_gradient,jump_value,normal,average_viscosity,penalty_parameter);

          fe_eval_neighbor.submit_gradient(value_flux,q);
          fe_eval_neighbor.submit_value(-gradient_flux,q);
        }
        fe_eval_neighbor.integrate(true,true);

        local_diagonal_vector_neighbor[j] = fe_eval_neighbor.read_cellwise_dof_value(j);
      }
      for (unsigned int j=0; j<fe_eval_neighbor.dofs_per_cell*dim; ++j)
        fe_eval_neighbor.write_cellwise_dof_value(j, local_diagonal_vector_neighbor[j]);

      fe_eval_neighbor.distribute_local_to_global(dst);
    }
  }

  // TODO: This function has to be removed as soon as the new infrastructure is used that
  // allows to perform face integrals over all faces of the current element.
  void boundary_face_loop_diagonal (const MatrixFree<dim,Number>                 &data,
                                    parallel::distributed::Vector<Number>        &dst,
                                    const parallel::distributed::Vector<Number>  &,
                                    const std::pair<unsigned int,unsigned int>   &face_range) const
  {
    FEFaceEval_Velocity_Velocity_linear fe_eval(data,this->fe_param,true,operator_data.dof_index);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      types::boundary_id boundary_id = data.get_boundary_id(face);
      BoundaryTypeU boundary_type = BoundaryTypeU::Undefined;

      if(operator_data.bc->dirichlet_bc.find(boundary_id) != operator_data.bc->dirichlet_bc.end())
        boundary_type = BoundaryTypeU::Dirichlet;
      else if(operator_data.bc->neumann_bc.find(boundary_id) != operator_data.bc->neumann_bc.end())
        boundary_type = BoundaryTypeU::Neumann;
      else if(operator_data.bc->symmetry_bc.find(boundary_id) != operator_data.bc->symmetry_bc.end())
        boundary_type = BoundaryTypeU::Symmetry;

      AssertThrow(boundary_type != BoundaryTypeU::Undefined,
          ExcMessage("Boundary type of face is invalid or not implemented."));

      fe_eval.reinit (face);

      VectorizedArray<Number> penalty_parameter = get_penalty_factor() * fe_eval.read_cell_data(array_penalty_parameter);

      VectorizedArray<Number> local_diagonal_vector[fe_eval.tensor_dofs_per_cell*dim];
      for (unsigned int j=0; j<fe_eval.dofs_per_cell*dim; ++j)
      {
        // set dof value j of element- to 1 and all other dof values of element- to zero
        for (unsigned int i=0; i<fe_eval.dofs_per_cell*dim; ++i)
          fe_eval.write_cellwise_dof_value(i, make_vectorized_array<Number>(0.));
        fe_eval.write_cellwise_dof_value(j, make_vectorized_array<Number>(1.));

        fe_eval.evaluate(true,true);

        for(unsigned int q=0;q<fe_eval.n_q_points;++q)
        {
          VectorizedArray<Number> viscosity = make_vectorized_array<Number>(const_viscosity);
          if(viscosity_is_variable())
            viscosity = viscous_coefficient_face[face][q];

          Tensor<1,dim,VectorizedArray<Number> > jump_value;
          calculate_jump_value_boundary_face(jump_value,q,fe_eval,OperatorType::homogeneous,boundary_type);
          Tensor<1,dim,VectorizedArray<Number> > normal = fe_eval.get_normal_vector(q);

          Tensor<2,dim,VectorizedArray<Number> > value_flux;
          calculate_value_flux(value_flux,jump_value,normal,viscosity,fe_eval);

          Tensor<1,dim,VectorizedArray<Number> > average_normal_gradient;
          calculate_average_normal_gradient_boundary_face(average_normal_gradient,q,fe_eval,OperatorType::homogeneous,boundary_type);

          Tensor<1,dim,VectorizedArray<Number> > gradient_flux;
          calculate_gradient_flux(gradient_flux,average_normal_gradient,jump_value,normal,viscosity,penalty_parameter);

          fe_eval.submit_gradient(value_flux,q);
          fe_eval.submit_value(-gradient_flux,q);
        }
        fe_eval.integrate(true,true);

        local_diagonal_vector[j] = fe_eval.read_cellwise_dof_value(j);
      }
      for (unsigned int j=0; j<fe_eval.dofs_per_cell*dim; ++j)
        fe_eval.write_cellwise_dof_value(j, local_diagonal_vector[j]);

      fe_eval.distribute_local_to_global(dst);
    }
  }

  void cell_loop_calculate_block_jacobi_matrices (const MatrixFree<dim,value_type>                 &data,
                                                  std::vector<LAPACKFullMatrix<value_type> >       &matrices,
                                                  const parallel::distributed::Vector<value_type>  &,
                                                  const std::pair<unsigned int,unsigned int>       &cell_range) const
  {
    FEEval_Velocity_Velocity_linear fe_eval(data,this->fe_param,operator_data.dof_index);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);

      // Note that the velocity has dim components.
      unsigned int dofs_per_cell = fe_eval.dofs_per_cell*dim;

      for (unsigned int j=0; j<dofs_per_cell; ++j)
      {
        for (unsigned int i=0; i<dofs_per_cell; ++i)
          fe_eval.begin_dof_values()[i] = make_vectorized_array<value_type>(0.);
        fe_eval.begin_dof_values()[j] = make_vectorized_array<value_type>(1.);

        do_cell_integral(fe_eval,cell);

        for(unsigned int i=0; i<dofs_per_cell; ++i)
          for (unsigned int v=0; v<VectorizedArray<value_type>::n_array_elements; ++v)
            matrices[cell*VectorizedArray<value_type>::n_array_elements+v](i,j) += fe_eval.begin_dof_values()[i][v];
      }
    }
  }

  void face_loop_calculate_block_jacobi_matrices (const MatrixFree<dim,value_type>                &data,
                                                  std::vector<LAPACKFullMatrix<value_type> >      &matrices,
                                                  const parallel::distributed::Vector<value_type> &,
                                                  const std::pair<unsigned int,unsigned int>      &face_range) const
  {
    FEFaceEval_Velocity_Velocity_linear fe_eval(data,this->fe_param,true,operator_data.dof_index);
    FEFaceEval_Velocity_Velocity_linear fe_eval_neighbor(data,this->fe_param,false,operator_data.dof_index);

    // Perform face intergrals for element e⁻.
    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval.reinit (face);
      fe_eval_neighbor.reinit (face);

      // Note that the velocity has dim components.
      unsigned int dofs_per_cell = fe_eval.dofs_per_cell*dim;

      VectorizedArray<Number> penalty_parameter = get_penalty_factor() *
          std::max(fe_eval.read_cell_data(array_penalty_parameter),fe_eval_neighbor.read_cell_data(array_penalty_parameter));

      for (unsigned int j=0; j<dofs_per_cell; ++j)
      {
        // set dof value j of element- to 1 and all other dof values of element- to zero
        for (unsigned int i=0; i<dofs_per_cell; ++i)
          fe_eval.begin_dof_values()[i] = make_vectorized_array<value_type>(0.);
        fe_eval.begin_dof_values()[j] = make_vectorized_array<value_type>(1.);

        fe_eval.evaluate(true,true);

        for(unsigned int q=0;q<fe_eval.n_q_points;++q)
        {
          VectorizedArray<Number> average_viscosity = make_vectorized_array<Number>(const_viscosity);
          if(viscosity_is_variable())
            calculate_average_viscosity(average_viscosity,face,q);

          // set exterior values to zero
          Tensor<1,dim,VectorizedArray<Number> > jump_value = fe_eval.get_value(q);
          Tensor<1,dim,VectorizedArray<Number> > normal = fe_eval.get_normal_vector(q);

          Tensor<2,dim,VectorizedArray<Number> > value_flux;
          calculate_value_flux(value_flux,jump_value,normal,average_viscosity,fe_eval);

          Tensor<1,dim,VectorizedArray<Number> > average_normal_gradient;
          calculate_normal_gradient(average_normal_gradient,q,fe_eval);
          // set exterior values to zero
          average_normal_gradient = make_vectorized_array<Number>(0.5) * average_normal_gradient;

          Tensor<1,dim,VectorizedArray<Number> > gradient_flux;
          calculate_gradient_flux(gradient_flux,average_normal_gradient,jump_value,normal,average_viscosity,penalty_parameter);

          fe_eval.submit_gradient(value_flux,q);
          fe_eval.submit_value(-gradient_flux,q);
        }
        fe_eval.integrate(true,true);

        for (unsigned int v=0; v<VectorizedArray<value_type>::n_array_elements; ++v)
        {
          const unsigned int cell_number = data.get_face_info(face).cells_minus[v];
          if (cell_number != numbers::invalid_unsigned_int)
            for(unsigned int i=0; i<dofs_per_cell; ++i)
              matrices[cell_number](i,j) += fe_eval.begin_dof_values()[i][v];
        }
      }
    }



    // TODO: This has to be removed as soon as the new infrastructure is used that
    // allows to perform face integrals over all faces of the current element.
    // Perform face intergrals for element e⁺.
    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval.reinit (face);
      fe_eval_neighbor.reinit (face);

      // Note that the velocity has dim components.
      unsigned int dofs_per_cell = fe_eval_neighbor.dofs_per_cell*dim;

      VectorizedArray<Number> penalty_parameter = get_penalty_factor() *
          std::max(fe_eval.read_cell_data(array_penalty_parameter),fe_eval_neighbor.read_cell_data(array_penalty_parameter));

      for (unsigned int j=0; j<dofs_per_cell; ++j)
      {
        // set dof value j of element+ to 1 and all other dof values of element+ to zero
        for (unsigned int i=0; i<dofs_per_cell; ++i)
          fe_eval_neighbor.begin_dof_values()[i] = make_vectorized_array<value_type>(0.);
        fe_eval_neighbor.begin_dof_values()[j] = make_vectorized_array<value_type>(1.);

        fe_eval_neighbor.evaluate(true,true);

        for(unsigned int q=0;q<fe_eval.n_q_points;++q)
        {
          VectorizedArray<Number> average_viscosity = make_vectorized_array<Number>(const_viscosity);
          if(viscosity_is_variable())
            calculate_average_viscosity(average_viscosity,face,q);

          // set exterior values to zero
          Tensor<1,dim,VectorizedArray<Number> > jump_value = fe_eval_neighbor.get_value(q);
          // multiply by -1.0 to get the correct normal vector !!!
          Tensor<1,dim,VectorizedArray<Number> > normal = - fe_eval_neighbor.get_normal_vector(q);

          Tensor<2,dim,VectorizedArray<Number> > value_flux;
          calculate_value_flux(value_flux,jump_value,normal,average_viscosity,fe_eval_neighbor);

          Tensor<1,dim,VectorizedArray<Number> > average_normal_gradient;
          calculate_normal_gradient(average_normal_gradient,q,fe_eval_neighbor);
          // set exterior values to zero
          // and multiply by -1.0 since normal vector n⁺ = -n⁻ !!!
          average_normal_gradient = make_vectorized_array<Number>(-0.5) * average_normal_gradient;

          Tensor<1,dim,VectorizedArray<Number> > gradient_flux;
          calculate_gradient_flux(gradient_flux,average_normal_gradient,jump_value,normal,average_viscosity,penalty_parameter);

          fe_eval_neighbor.submit_gradient(value_flux,q);
          fe_eval_neighbor.submit_value(-gradient_flux,q);
        }
        fe_eval_neighbor.integrate(true,true);

        for (unsigned int v=0; v<VectorizedArray<value_type>::n_array_elements; ++v)
        {
          const unsigned int cell_number = data.get_face_info(face).cells_plus[v];
          if (cell_number != numbers::invalid_unsigned_int)
            for(unsigned int i=0; i<dofs_per_cell; ++i)
              matrices[cell_number](i,j) += fe_eval_neighbor.begin_dof_values()[i][v];
        }
      }
    }
  }

  // TODO: This function has to be removed as soon as the new infrastructure is used that
  // allows to perform face integrals over all faces of the current element.
  void boundary_face_loop_calculate_block_jacobi_matrices (const MatrixFree<dim,value_type>                &data,
                                                           std::vector<LAPACKFullMatrix<value_type> >      &matrices,
                                                           const parallel::distributed::Vector<value_type> &,
                                                           const std::pair<unsigned int,unsigned int>      &face_range) const
  {
    FEFaceEval_Velocity_Velocity_linear fe_eval(data,this->fe_param,true,operator_data.dof_index);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      types::boundary_id boundary_id = data.get_boundary_id(face);
      BoundaryTypeU boundary_type = BoundaryTypeU::Undefined;

      if(operator_data.bc->dirichlet_bc.find(boundary_id) != operator_data.bc->dirichlet_bc.end())
        boundary_type = BoundaryTypeU::Dirichlet;
      else if(operator_data.bc->neumann_bc.find(boundary_id) != operator_data.bc->neumann_bc.end())
        boundary_type = BoundaryTypeU::Neumann;
      else if(operator_data.bc->symmetry_bc.find(boundary_id) != operator_data.bc->symmetry_bc.end())
        boundary_type = BoundaryTypeU::Symmetry;

      AssertThrow(boundary_type != BoundaryTypeU::Undefined,
          ExcMessage("Boundary type of face is invalid or not implemented."));

      fe_eval.reinit (face);

      // Note that the velocity has dim components.
      unsigned int dofs_per_cell = fe_eval.dofs_per_cell*dim;

      VectorizedArray<Number> penalty_parameter = get_penalty_factor() * fe_eval.read_cell_data(array_penalty_parameter);

      for (unsigned int j=0; j<dofs_per_cell; ++j)
      {
        // set dof value j of element- to 1 and all other dof values of element- to zero
        for (unsigned int i=0; i<dofs_per_cell; ++i)
          fe_eval.begin_dof_values()[i] = make_vectorized_array<value_type>(0.);
        fe_eval.begin_dof_values()[j] = make_vectorized_array<value_type>(1.);

        fe_eval.evaluate(true,true);

        for(unsigned int q=0;q<fe_eval.n_q_points;++q)
        {
          VectorizedArray<Number> viscosity = make_vectorized_array<Number>(const_viscosity);
          if(viscosity_is_variable())
            viscosity = viscous_coefficient_face[face][q];

          Tensor<1,dim,VectorizedArray<Number> > jump_value;
          calculate_jump_value_boundary_face(jump_value,q,fe_eval,OperatorType::homogeneous,boundary_type);
          Tensor<1,dim,VectorizedArray<Number> > normal = fe_eval.get_normal_vector(q);

          Tensor<2,dim,VectorizedArray<Number> > value_flux;
          calculate_value_flux(value_flux,jump_value,normal,viscosity,fe_eval);

          Tensor<1,dim,VectorizedArray<Number> > average_normal_gradient;
          calculate_average_normal_gradient_boundary_face(average_normal_gradient,q,fe_eval,OperatorType::homogeneous,boundary_type);

          Tensor<1,dim,VectorizedArray<Number> > gradient_flux;
          calculate_gradient_flux(gradient_flux,average_normal_gradient,jump_value,normal,viscosity,penalty_parameter);

          fe_eval.submit_gradient(value_flux,q);
          fe_eval.submit_value(-gradient_flux,q);
        }
        fe_eval.integrate(true,true);

        for (unsigned int v=0; v<VectorizedArray<value_type>::n_array_elements; ++v)
        {
          const unsigned int cell_number = data.get_face_info(face).cells_minus[v];
          if (cell_number != numbers::invalid_unsigned_int)
            for(unsigned int i=0; i<dofs_per_cell; ++i)
              matrices[cell_number](i,j) += fe_eval.begin_dof_values()[i][v];
        }
      }
    }
  }

  void cell_loop_extract_viscous_coeff (MatrixFree<dim,Number> const                &data,
                                        parallel::distributed::Vector<Number>       &,
                                        parallel::distributed::Vector<Number> const &src,
                                        std::pair<unsigned int,unsigned int> const  &cell_range)
  {
    FEEval_Velocity_Velocity_linear fe_eval(data,this->fe_param,operator_data.dof_index);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);
      fe_eval.read_dof_values(src);

      fe_eval.evaluate(true,false,false);

      for (unsigned int q=0; q<fe_eval.n_q_points; ++q)
      {
        VectorizedArray<Number> viscosity = make_vectorized_array<Number>(const_viscosity);
        if(viscosity_is_variable())
        {
          // The viscosity is stored in a vector with dim components (as the velocity field)
          // but we only use the first component (TODO).
          Tensor<1,dim,VectorizedArray<Number> > viscosity_vector = fe_eval.get_value(q);
          // make sure that the turbulent viscosity is not negative.
          for(unsigned int n=0; n<VectorizedArray<Number>::n_array_elements; ++n)
            viscosity[n] = std::max(get_const_viscosity(),viscosity_vector[0][n]);

          // set viscous coefficient
          set_viscous_coefficient_cell(cell,q,viscosity);
        }
      }
    }
  }

  void face_loop_extract_viscous_coeff(MatrixFree<dim,Number> const                &data,
                                       parallel::distributed::Vector<Number>       &,
                                       parallel::distributed::Vector<Number> const &src,
                                       std::pair<unsigned int,unsigned int> const  &face_range)
  {
    FEFaceEval_Velocity_Velocity_linear fe_eval(data,this->fe_param,true,operator_data.dof_index);
    FEFaceEval_Velocity_Velocity_linear fe_eval_neighbor(data,this->fe_param,false,operator_data.dof_index);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval.reinit (face);
      fe_eval_neighbor.reinit (face);

      fe_eval.read_dof_values(src);
      fe_eval_neighbor.read_dof_values(src);

      // we only want to evaluate the gradient
      fe_eval.evaluate(true,false);
      fe_eval_neighbor.evaluate(true,false);

      for(unsigned int q=0;q<fe_eval.n_q_points;++q)
      {
        VectorizedArray<Number> viscosity = make_vectorized_array<Number>(get_const_viscosity());
        VectorizedArray<Number> viscosity_neighbor = make_vectorized_array<Number>(get_const_viscosity());

        // The viscosity is stored in a vector with dim components (as the velocity field)
        // but we only use the first component (TODO).
        Tensor<1,dim,VectorizedArray<Number> > viscosity_vector = fe_eval.get_value(q);
        // make sure that the turbulent viscosity is not negative.
        for(unsigned int n=0; n<VectorizedArray<Number>::n_array_elements; ++n)
          viscosity[n] = std::max(get_const_viscosity(),viscosity_vector[0][n]);

        Tensor<1,dim,VectorizedArray<Number> > viscosity_vector_neighbor = fe_eval_neighbor.get_value(q);
        // make sure that the turbulent viscosity is not negative.
        for(unsigned int n=0; n<VectorizedArray<Number>::n_array_elements; ++n)
          viscosity_neighbor[n] = std::max(get_const_viscosity(),viscosity_vector_neighbor[0][n]);

        // set viscous coefficient
        set_viscous_coefficient_face(face,q,viscosity);
        set_viscous_coefficient_face_neighbor(face,q,viscosity_neighbor);
      }
    }
  }

  void boundary_face_loop_extract_viscous_coeff(MatrixFree<dim,Number> const                &data,
                                                parallel::distributed::Vector<Number>       &,
                                                parallel::distributed::Vector<Number> const &src,
                                                std::pair<unsigned int,unsigned int> const  &face_range)
  {
    FEFaceEval_Velocity_Velocity_linear fe_eval(data,this->fe_param,true,operator_data.dof_index);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval.reinit (face);
      fe_eval.read_dof_values(src);

      fe_eval.evaluate(true,false);

      for(unsigned int q=0;q<fe_eval.n_q_points;++q)
      {
        VectorizedArray<Number> viscosity = make_vectorized_array<Number>(get_const_viscosity());

        // The viscosity is stored in a vector with dim components (as the velocity field)
        // but we only use the first component (TODO).
        Tensor<1,dim,VectorizedArray<Number> > viscosity_vector = fe_eval.get_value(q);
        // make sure that the turbulent viscosity is not negative.
        for(unsigned int n=0; n<VectorizedArray<Number>::n_array_elements; ++n)
          viscosity[n] = std::max(get_const_viscosity(),viscosity_vector[0][n]);

        // set viscous coefficient
        set_viscous_coefficient_face(face,q,viscosity);
      }
    }
  }


protected:
  MatrixFree<dim,Number> const * data;
  ViscousOperatorData<dim> operator_data;

private:
  AlignedVector<VectorizedArray<Number> > array_penalty_parameter;
  Number const_viscosity;

  // TODO dof-vector for variable viscosity field
  parallel::distributed::Vector<Number>  viscosity;

  Table<2,VectorizedArray<Number> > viscous_coefficient_cell;
  Table<2,VectorizedArray<Number> > viscous_coefficient_face;
  Table<2,VectorizedArray<Number> > viscous_coefficient_face_neighbor;
  double mutable eval_time;
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

  std::shared_ptr<BoundaryDescriptorNavierStokesP<dim> > bc;
};

template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename value_type>
class GradientOperator: public BaseOperator<dim>
{
public:
  enum class OperatorType {
    full,
    homogeneous,
    inhomogeneous
  };

  GradientOperator()
    :
    data(nullptr),
    eval_time(0.0),
    inverse_scaling_factor_pressure(1.0)
  {}

  static const bool is_xwall = (xwall_quad_rule>1) ? true : false;
  static const unsigned int n_actual_q_points_vel_linear = (is_xwall) ? xwall_quad_rule : fe_degree+1;
  typedef FEEvaluationWrapper<dim,fe_degree,fe_degree_xwall,n_actual_q_points_vel_linear,dim,value_type,is_xwall> FEEval_Velocity_Velocity_linear;
  typedef FEEvaluationWrapperPressure<dim,fe_degree_p,fe_degree_xwall,n_actual_q_points_vel_linear,1,value_type,is_xwall> FEEval_Pressure_Velocity_linear;

  typedef FEFaceEvaluationWrapper<dim,fe_degree,fe_degree_xwall,n_actual_q_points_vel_linear,dim,value_type,is_xwall> FEFaceEval_Velocity_Velocity_linear;
  typedef FEFaceEvaluationWrapperPressure<dim,fe_degree_p,fe_degree_xwall,n_actual_q_points_vel_linear,1,value_type,is_xwall> FEFaceEval_Pressure_Velocity_linear;

  typedef GradientOperator<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type> This;

  void initialize(MatrixFree<dim,value_type> const &mf_data,
                  GradientOperatorData<dim> const  &operator_data_in)
  {
    this->data = &mf_data;
    this->operator_data = operator_data_in;
  }

  void set_scaling_factor_pressure(double const &scaling_factor)
  {
    inverse_scaling_factor_pressure = 1.0/scaling_factor;
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
    data->loop (&This::cell_loop,&This::face_loop,
                &This::boundary_face_loop_hom_operator,this, dst, src);
  }

  void rhs (parallel::distributed::Vector<value_type> &dst,
            double const                              evaluation_time) const
  {
    dst = 0;
    rhs_add(dst,evaluation_time);
  }

  void rhs_add (parallel::distributed::Vector<value_type> &dst,
                double const                              evaluation_time) const
  {
    this->eval_time = evaluation_time;

    parallel::distributed::Vector<value_type> src;
    data->loop (&This::cell_loop_inhom_operator,
                &This::face_loop_inhom_operator,
                &This::boundary_face_loop_inhom_operator,
                this, dst, src);
  }

  void evaluate (parallel::distributed::Vector<value_type>       &dst,
                 const parallel::distributed::Vector<value_type> &src,
                 double const                                    evaluation_time) const
  {
    dst = 0;
    evaluate_add(dst,src,evaluation_time);
  }

  void evaluate_add (parallel::distributed::Vector<value_type>       &dst,
                     const parallel::distributed::Vector<value_type> &src,
                     double const                                    evaluation_time) const
  {
    this->eval_time = evaluation_time;

    data->loop (&This::cell_loop,&This::face_loop,
                &This::boundary_face_loop_full_operator,this, dst, src);
  }

private:
  template<typename FEEvaluationPressure, typename FEEvaluationVelocity>
  inline void do_cell_integral(FEEvaluationPressure &fe_eval_pressure,
                               FEEvaluationVelocity &fe_eval_velocity) const
  {
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
  }

  /*
   *  This function calculates the numerical flux for interior faces
   *  which is simply the average value (central flux).
   */
  inline void calculate_flux (VectorizedArray<value_type>       &flux,
                              VectorizedArray<value_type> const &value_m,
                              VectorizedArray<value_type> const &value_p) const
  {
    flux = 0.5*(value_m + value_p);
  }

  /*
   *  This function calculates the numerical flux for boundary faces
   *  depending on the operator type, the type of the boundary face
   *  and the given boundary conditions.
   *
   *                            +------------------------------+------------------------------+
   *                            | Neumann boundaries           | Dirichlet boundaries         |
   *  +-------------------------+------------------------------+------------------------------+
   *  | full operator           | p⁺ = p⁻        -> {{p}} = p⁻ | p⁺ = - p⁻ + 2g  -> {{p}} = g |
   *  +-------------------------+------------------------------+------------------------------+
   *  | homogeneous operator    | p⁺ = p⁻        -> {{p}} = p⁻ | p⁺ = - p⁻       -> {{p}} = 0 |
   *  +-------------------------+------------------------------+------------------------------+
   *  | inhomogeneous operator  | p⁻ = 0, p⁺ = 0 -> {{p}} = 0  | p⁻ = 0, p⁺ = 2g -> {{p}} = g |
   *  +-------------------------+------------------------------+------------------------------+
   *
   */
  template<typename FEEvaluationPressure>
  inline void calculate_flux_boundary_face(VectorizedArray<value_type> &flux,
                                           unsigned int const          q,
                                           FEEvaluationPressure const  &fe_eval_pressure,
                                           OperatorType const          &operator_type,
                                           BoundaryTypeP const         &boundary_type,
                                           types::boundary_id const    boundary_id = types::boundary_id()) const
  {
    // element e⁻
    VectorizedArray<value_type> value_m = make_vectorized_array<value_type>(0.0);

    if(operator_type == OperatorType::full || operator_type == OperatorType::homogeneous)
    {
      value_m = fe_eval_pressure.get_value(q);
    }
    else if(operator_type == OperatorType::inhomogeneous)
    {
      // do nothing, value_m is already initialized with zeros
    }
    else
    {
      AssertThrow(false, ExcMessage("Specified OperatorType is not implemented!"));
    }

    // element e⁺
    VectorizedArray<value_type> value_p = make_vectorized_array<value_type>(0.0);

    if(operator_type == OperatorType::full)
    {
      if(boundary_type == BoundaryTypeP::Neumann)
      {
        value_p = value_m;
      }
      else if(boundary_type == BoundaryTypeP::Dirichlet)
      {
        VectorizedArray<value_type> g;
        typename std::map<types::boundary_id,std::shared_ptr<Function<dim> > >::iterator it;
        it = operator_data.bc->dirichlet_bc.find(boundary_id);
        Point<dim,VectorizedArray<value_type> > q_points = fe_eval_pressure.quadrature_point(q);
        evaluate_scalar_function(g,it->second,q_points,eval_time);

        value_p = -value_m + 2.0*inverse_scaling_factor_pressure*g;
      }
      else
      {
        AssertThrow(false,ExcMessage("Boundary type of face is invalid or not implemented."));
      }
    }
    else if(operator_type == OperatorType::homogeneous)
    {
      if(boundary_type == BoundaryTypeP::Neumann)
      {
        value_p = value_m;
      }
      else if(boundary_type == BoundaryTypeP::Dirichlet)
      {
        value_p = -value_m;
      }
      else
      {
        AssertThrow(false,ExcMessage("Boundary type of face is invalid or not implemented."));
      }
    }
    else if(operator_type == OperatorType::inhomogeneous)
    {
      if(boundary_type == BoundaryTypeP::Neumann)
      {
        // do nothing since value_p is already initialized with zeros
      }
      else if(boundary_type == BoundaryTypeP::Dirichlet)
      {
        VectorizedArray<value_type> g;
        typename std::map<types::boundary_id,std::shared_ptr<Function<dim> > >::iterator it;
        it = operator_data.bc->dirichlet_bc.find(boundary_id);
        Point<dim,VectorizedArray<value_type> > q_points = fe_eval_pressure.quadrature_point(q);
        evaluate_scalar_function(g,it->second,q_points,eval_time);

        value_p = 2.0*inverse_scaling_factor_pressure*g;
      }
      else
      {
        AssertThrow(false,ExcMessage("Boundary type of face is invalid or not implemented."));
      }
    }
    else
    {
      AssertThrow(false, ExcMessage("Specified OperatorType is not implemented!"));
    }

    calculate_flux(flux,value_m,value_p);
  }

  void cell_loop (const MatrixFree<dim,value_type>                 &data,
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

      do_cell_integral(fe_eval_pressure,fe_eval_velocity);

      fe_eval_velocity.distribute_local_to_global (dst);
    }
  }

  void face_loop (const MatrixFree<dim,value_type>                 &data,
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
          VectorizedArray<value_type> value_m = fe_eval_pressure.get_value(q);
          VectorizedArray<value_type> value_p = fe_eval_pressure_neighbor.get_value(q);

          VectorizedArray<value_type> flux;
          calculate_flux(flux,value_m,value_p);

          Tensor<1,dim,VectorizedArray<value_type> > flux_times_normal =
              flux*fe_eval_pressure.get_normal_vector(q);

          fe_eval_velocity.submit_value (flux_times_normal, q);
          fe_eval_velocity_neighbor.submit_value (-flux_times_normal, q); // minus sign since n⁺ = - n⁻
        }
        fe_eval_velocity.integrate (true,false);
        fe_eval_velocity_neighbor.integrate (true,false);

        fe_eval_velocity.distribute_local_to_global (dst);
        fe_eval_velocity_neighbor.distribute_local_to_global (dst);
      }
    }
  }

  void boundary_face_loop_hom_operator (const MatrixFree<dim,value_type>                &data,
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
        types::boundary_id boundary_id = data.get_boundary_id(face);
        BoundaryTypeP boundary_type = BoundaryTypeP::Undefined;

        if(operator_data.bc->dirichlet_bc.find(boundary_id) != operator_data.bc->dirichlet_bc.end())
          boundary_type = BoundaryTypeP::Dirichlet;
        else if(operator_data.bc->neumann_bc.find(boundary_id) != operator_data.bc->neumann_bc.end())
          boundary_type = BoundaryTypeP::Neumann;

        AssertThrow(boundary_type != BoundaryTypeP::Undefined,
            ExcMessage("Boundary type of face is invalid or not implemented."));

        fe_eval_velocity.reinit (face);
        fe_eval_pressure.reinit (face);
        fe_eval_pressure.read_dof_values(src);
        fe_eval_pressure.evaluate (true,false);

        for (unsigned int q=0; q<fe_eval_velocity.n_q_points; ++q)
        {
          VectorizedArray<value_type> flux;

          if(operator_data.use_boundary_data == true)
          {
            calculate_flux_boundary_face(flux,q,fe_eval_pressure,OperatorType::homogeneous,boundary_type);
          }
          else // use_boundary_data == false
          {
            VectorizedArray<value_type> value_m = fe_eval_pressure.get_value(q);
            // exterior value = interior value
            VectorizedArray<value_type> value_p = value_m;
            calculate_flux(flux,value_m,value_p);
          }

          Tensor<1,dim,VectorizedArray<value_type> > flux_times_normal =
              flux * fe_eval_pressure.get_normal_vector(q);

          fe_eval_velocity.submit_value (flux_times_normal, q);
        }
        fe_eval_velocity.integrate (true,false);
        fe_eval_velocity.distribute_local_to_global (dst);
      }
    }
  }

  void boundary_face_loop_full_operator (const MatrixFree<dim,value_type>                &data,
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
        types::boundary_id boundary_id = data.get_boundary_id(face);
        BoundaryTypeP boundary_type = BoundaryTypeP::Undefined;

        if(operator_data.bc->dirichlet_bc.find(boundary_id) != operator_data.bc->dirichlet_bc.end())
          boundary_type = BoundaryTypeP::Dirichlet;
        else if(operator_data.bc->neumann_bc.find(boundary_id) != operator_data.bc->neumann_bc.end())
          boundary_type = BoundaryTypeP::Neumann;

        AssertThrow(boundary_type != BoundaryTypeP::Undefined,
            ExcMessage("Boundary type of face is invalid or not implemented."));

        fe_eval_velocity.reinit (face);
        fe_eval_pressure.reinit (face);
        fe_eval_pressure.read_dof_values(src);
        fe_eval_pressure.evaluate (true,false);

        for (unsigned int q=0; q<fe_eval_velocity.n_q_points; ++q)
        {
          VectorizedArray<value_type> flux;

          if(operator_data.use_boundary_data == true)
          {
            calculate_flux_boundary_face(flux,q,fe_eval_pressure,OperatorType::full,boundary_type,boundary_id);
          }
          else // use_boundary_data == false
          {
            VectorizedArray<value_type> value_m = fe_eval_pressure.get_value(q);
            // exterior value = interior value
            VectorizedArray<value_type> value_p = value_m;
            calculate_flux(flux,value_m,value_p);
          }

          Tensor<1,dim,VectorizedArray<value_type> > flux_times_normal =
              flux * fe_eval_pressure.get_normal_vector(q);

          fe_eval_velocity.submit_value (flux_times_normal, q);
        }
        fe_eval_velocity.integrate (true,false);
        fe_eval_velocity.distribute_local_to_global (dst);
      }
    }
  }

  void cell_loop_inhom_operator (const MatrixFree<dim,value_type>                 &,
                                 parallel::distributed::Vector<value_type>        &,
                                 const parallel::distributed::Vector<value_type>  &,
                                 const std::pair<unsigned int,unsigned int>       &) const
  {

  }

  void face_loop_inhom_operator (const MatrixFree<dim,value_type>                 &,
                                 parallel::distributed::Vector<value_type>        &,
                                 const parallel::distributed::Vector<value_type>  &,
                                 const std::pair<unsigned int,unsigned int>       &) const
  {

  }

  void boundary_face_loop_inhom_operator (const MatrixFree<dim,value_type>                &data,
                                          parallel::distributed::Vector<value_type>       &dst,
                                          const parallel::distributed::Vector<value_type> &,
                                          const std::pair<unsigned int,unsigned int>      &face_range) const
  {
    if(operator_data.integration_by_parts_of_gradP == true &&
       operator_data.use_boundary_data == true)
    {
      FEFaceEval_Velocity_Velocity_linear fe_eval_velocity(data,this->fe_param,true,operator_data.dof_index_velocity);
      FEFaceEval_Pressure_Velocity_linear fe_eval_pressure(data,this->fe_param,true,operator_data.dof_index_pressure);

      for(unsigned int face=face_range.first; face<face_range.second; face++)
      {
        types::boundary_id boundary_id = data.get_boundary_id(face);
        BoundaryTypeP boundary_type = BoundaryTypeP::Undefined;

        if(operator_data.bc->dirichlet_bc.find(boundary_id) != operator_data.bc->dirichlet_bc.end())
          boundary_type = BoundaryTypeP::Dirichlet;
        else if(operator_data.bc->neumann_bc.find(boundary_id) != operator_data.bc->neumann_bc.end())
          boundary_type = BoundaryTypeP::Neumann;

        AssertThrow(boundary_type != BoundaryTypeP::Undefined,
            ExcMessage("Boundary type of face is invalid or not implemented."));

        fe_eval_velocity.reinit (face);
        fe_eval_pressure.reinit (face);

        for (unsigned int q=0; q<fe_eval_velocity.n_q_points; ++q)
        {
          VectorizedArray<value_type> flux;
          calculate_flux_boundary_face(flux,q,fe_eval_pressure,OperatorType::inhomogeneous,boundary_type,boundary_id);

          Tensor<1,dim,VectorizedArray<value_type> > flux_times_normal =
              flux * fe_eval_pressure.get_normal_vector(q);

          fe_eval_velocity.submit_value (-flux_times_normal, q); // minus sign since this term occurs on the rhs of the equation
        }
        fe_eval_velocity.integrate (true,false);
        fe_eval_velocity.distribute_local_to_global (dst);
      }
    }
  }

  MatrixFree<dim,value_type> const * data;
  GradientOperatorData<dim> operator_data;
  double mutable eval_time;

  // if the continuity equation of the incompressible Navier-Stokes
  // equations is scaled by a constant factor, the system of equations
  // is solved for a modified pressure p^* = 1/scaling_factor * p. Hence,
  // when applying the gradient operator to this modified pressure we have
  // to make sure that we also apply the correct boundary conditions for p^*,
  // i.e., g_p^* = 1/scaling_factor * g_p
  double inverse_scaling_factor_pressure;
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

  std::shared_ptr<BoundaryDescriptorNavierStokesU<dim> > bc;
};

template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename value_type>
class DivergenceOperator: public BaseOperator<dim>
{
public:
  enum class OperatorType {
    full,
    homogeneous,
    inhomogeneous
  };

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

  typedef DivergenceOperator<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type> This;

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
    data->loop (&This::cell_loop,&This::face_loop,
                &This::boundary_face_loop_hom_operator, this, dst, src);
  }

  void rhs (parallel::distributed::Vector<value_type> &dst,
            double const                              evaluation_time) const
  {
    dst = 0;
    rhs_add(dst,evaluation_time);
  }

  void rhs_add (parallel::distributed::Vector<value_type> &dst,
                double const                              evaluation_time) const
  {
    this->eval_time = evaluation_time;

    parallel::distributed::Vector<value_type> src;
    data->loop (&This::cell_loop_inhom_operator,&This::face_loop_inhom_operator,
                &This::boundary_face_loop_inhom_operator, this, dst, src);
  }

  void evaluate (parallel::distributed::Vector<value_type>       &dst,
                 const parallel::distributed::Vector<value_type> &src,
                 double const                                    evaluation_time) const
  {
    dst = 0;
    evaluate_add(dst,src,evaluation_time);
  }

  void evaluate_add (parallel::distributed::Vector<value_type>       &dst,
                     const parallel::distributed::Vector<value_type> &src,
                     double const                                    evaluation_time) const
  {
    this->eval_time = evaluation_time;

    data->loop (&This::cell_loop,&This::face_loop,
                &This::boundary_face_loop_full_operator, this, dst, src);
  }

private:
  template<typename FEEvaluationPressure, typename FEEvaluationVelocity>
  inline void do_cell_integral(FEEvaluationPressure &fe_eval_pressure,
                               FEEvaluationVelocity &fe_eval_velocity) const
  {
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
  }

  /*
   *  This function calculates the numerical flux for interior faces
   *  which is simply the average value (central flux).
   */
  inline void calculate_flux (Tensor<1,dim,VectorizedArray<value_type> >       &flux,
                              Tensor<1,dim,VectorizedArray<value_type> > const &value_m,
                              Tensor<1,dim,VectorizedArray<value_type> > const &value_p) const
  {
    flux = 0.5*(value_m + value_p);
  }

  /*
   *  This function calculates the numerical flux for boundary faces
   *  depending on the operator type, the type of the boundary face
   *  and the given boundary conditions.
   *
   *                            +------------------------------+------------------------------+---------------------------------------------+
   *                            | Dirichlet boundaries         | Neumann boundaries           | symmetry boundaries                         |
   *  +-------------------------+------------------------------+------------------------------+---------------------------------------------+
   *  | full operator           | u⁺ = -u⁻ + 2g   -> {{u}} = g | u⁺ = u⁻        -> {{u}} = u⁻ | u⁺ = u⁻ - 2 (u⁻*n)n -> {{u}} = u⁻ - (u⁻*n)n |
   *  +-------------------------+------------------------------+------------------------------+---------------------------------------------+
   *  | homogeneous operator    | u⁺ = -u⁻        -> {{u}} = 0 | u⁺ = u⁻        -> {{u}} = u⁻ | u⁺ = u⁻ - 2 (u⁻*n)n -> {{u}} = u⁻ - (u⁻*n)n |
   *  +-------------------------+------------------------------+------------------------------+---------------------------------------------+
   *  | inhomogeneous operator  | u⁻ = 0, u⁺ = 2g -> {{u}} = g | u⁻ = 0, u⁺ = 0 -> {{u}} = 0  | u⁻ = 0, u⁺ = 0      -> {{u}} = 0            |
   *  +-------------------------+------------------------------+------------------------------+---------------------------------------------+
   *
   */
  template<typename FEEvaluationVelocity>
  inline void calculate_flux_boundary_face(
      Tensor<1,dim,VectorizedArray<value_type> > &flux,
      unsigned int const                         q,
      FEEvaluationVelocity const                 &fe_eval_velocity,
      OperatorType const                         &operator_type,
      BoundaryTypeU const                        &boundary_type,
      types::boundary_id const                   boundary_id = types::boundary_id()) const
  {
    // element e⁻
    Tensor<1,dim,VectorizedArray<value_type> > value_m;

    if(operator_type == OperatorType::full || operator_type == OperatorType::homogeneous)
    {
      value_m = fe_eval_velocity.get_value(q);
    }
    else if(operator_type == OperatorType::inhomogeneous)
    {
      // do nothing, value_m is already initialized with zeros
    }
    else
    {
      AssertThrow(false, ExcMessage("Specified OperatorType is not implemented!"));
    }

    // element e⁺
    Tensor<1,dim,VectorizedArray<value_type> > value_p;

    if(operator_type == OperatorType::full)
    {
      if(boundary_type == BoundaryTypeU::Dirichlet)
      {
        Tensor<1,dim,VectorizedArray<value_type> > g;

        typename std::map<types::boundary_id,std::shared_ptr<Function<dim> > >::iterator it;
        it = operator_data.bc->dirichlet_bc.find(boundary_id);
        Point<dim,VectorizedArray<value_type> > q_points = fe_eval_velocity.quadrature_point(q);
        evaluate_vectorial_function(g,it->second,q_points,eval_time);

        value_p = -value_m + make_vectorized_array<value_type>(2.0)*g;
      }
      else if(boundary_type == BoundaryTypeU::Neumann)
      {
        value_p = value_m;
      }
      else if(boundary_type == BoundaryTypeU::Symmetry)
      {
        Tensor<1,dim,VectorizedArray<value_type> > normal_m = fe_eval_velocity.get_normal_vector(q);
        value_p = value_m - 2.0 * (value_m*normal_m) * normal_m;
      }
      else
      {
        AssertThrow(false,ExcMessage("Boundary type of face is invalid or not implemented."));
      }
    }
    else if(operator_type == OperatorType::homogeneous)
    {
      if(boundary_type == BoundaryTypeU::Dirichlet)
      {
        value_p = -value_m;
      }
      else if(boundary_type == BoundaryTypeU::Neumann)
      {
        value_p = value_m;
      }
      else if(boundary_type == BoundaryTypeU::Symmetry)
      {
        Tensor<1,dim,VectorizedArray<value_type> > normal_m = fe_eval_velocity.get_normal_vector(q);
        value_p = value_m - 2.0 * (value_m*normal_m) * normal_m;
      }
      else
      {
        AssertThrow(false,ExcMessage("Boundary type of face is invalid or not implemented."));
      }
    }
    else if(operator_type == OperatorType::inhomogeneous)
    {
      if(boundary_type == BoundaryTypeU::Dirichlet)
      {
        Tensor<1,dim,VectorizedArray<value_type> > g;

        typename std::map<types::boundary_id,std::shared_ptr<Function<dim> > >::iterator it;
        it = operator_data.bc->dirichlet_bc.find(boundary_id);
        Point<dim,VectorizedArray<value_type> > q_points = fe_eval_velocity.quadrature_point(q);
        evaluate_vectorial_function(g,it->second,q_points,eval_time);

        value_p = make_vectorized_array<value_type>(2.0)*g;
      }
      else if(boundary_type == BoundaryTypeU::Neumann)
      {
        // do nothing since value_p is already initialized with zeros
      }
      else if(boundary_type == BoundaryTypeU::Symmetry)
      {
        // do nothing since value_p is already initialized with zeros
      }
      else
      {
        AssertThrow(false,ExcMessage("Boundary type of face is invalid or not implemented."));
      }
    }
    else
    {
      AssertThrow(false, ExcMessage("Specified OperatorType is not implemented!"));
    }

    calculate_flux(flux,value_m,value_p);
  }

  void cell_loop (const MatrixFree<dim,value_type>                &data,
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

      do_cell_integral(fe_eval_pressure,fe_eval_velocity);

      fe_eval_pressure.distribute_local_to_global (dst);
    }
  }

  void face_loop (const MatrixFree<dim,value_type>                &data,
                  parallel::distributed::Vector<value_type>       &dst,
                  const parallel::distributed::Vector<value_type> &src,
                  const std::pair<unsigned int,unsigned int>      &face_range) const
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
          Tensor<1,dim,VectorizedArray<value_type> > value_m = fe_eval_velocity.get_value(q);
          Tensor<1,dim,VectorizedArray<value_type> > value_p = fe_eval_velocity_neighbor.get_value(q);
          Tensor<1,dim,VectorizedArray<value_type> > flux;
          calculate_flux(flux,value_m,value_p);

          VectorizedArray<value_type> flux_times_normal = flux * fe_eval_velocity.get_normal_vector(q);

          fe_eval_pressure.submit_value (flux_times_normal, q);
          fe_eval_pressure_neighbor.submit_value (-flux_times_normal, q); // minus sign since n⁺ = - n⁻
        }
        fe_eval_pressure.integrate (true,false);
        fe_eval_pressure_neighbor.integrate (true,false);

        fe_eval_pressure.distribute_local_to_global (dst);
        fe_eval_pressure_neighbor.distribute_local_to_global (dst);
      }
    }
  }

  void boundary_face_loop_hom_operator (const MatrixFree<dim,value_type>                &data,
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
        types::boundary_id boundary_id = data.get_boundary_id(face);
        BoundaryTypeU boundary_type = BoundaryTypeU::Undefined;

        if(operator_data.bc->dirichlet_bc.find(boundary_id) != operator_data.bc->dirichlet_bc.end())
          boundary_type = BoundaryTypeU::Dirichlet;
        else if(operator_data.bc->neumann_bc.find(boundary_id) != operator_data.bc->neumann_bc.end())
          boundary_type = BoundaryTypeU::Neumann;
        else if(operator_data.bc->symmetry_bc.find(boundary_id) != operator_data.bc->symmetry_bc.end())
          boundary_type = BoundaryTypeU::Symmetry;

        AssertThrow(boundary_type != BoundaryTypeU::Undefined,
            ExcMessage("Boundary type of face is invalid or not implemented."));

        fe_eval_pressure.reinit (face);

        fe_eval_velocity.reinit(face);
        fe_eval_velocity.read_dof_values(src);
        fe_eval_velocity.evaluate (true,false);

        for(unsigned int q=0;q<fe_eval_pressure.n_q_points;++q)
        {
          Tensor<1,dim,VectorizedArray<value_type> > flux;

          if(operator_data.use_boundary_data == true)
          {
            calculate_flux_boundary_face(flux,q,fe_eval_velocity,OperatorType::homogeneous,boundary_type);
          }
          else // use_boundary_data == false
          {
            Tensor<1,dim,VectorizedArray<value_type> > value_m = fe_eval_velocity.get_value(q);
            // exterior value = interior value
            Tensor<1,dim,VectorizedArray<value_type> > value_p = value_m;
            calculate_flux(flux,value_m,value_p);
          }

          VectorizedArray<value_type> flux_times_normal = flux * fe_eval_velocity.get_normal_vector(q);
          fe_eval_pressure.submit_value(flux_times_normal,q);
        }
        fe_eval_pressure.integrate(true,false);
        fe_eval_pressure.distribute_local_to_global(dst);
      }
    }
  }

  void boundary_face_loop_full_operator (const MatrixFree<dim,value_type>                &data,
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
        types::boundary_id boundary_id = data.get_boundary_id(face);
        BoundaryTypeU boundary_type = BoundaryTypeU::Undefined;

        if(operator_data.bc->dirichlet_bc.find(boundary_id) != operator_data.bc->dirichlet_bc.end())
          boundary_type = BoundaryTypeU::Dirichlet;
        else if(operator_data.bc->neumann_bc.find(boundary_id) != operator_data.bc->neumann_bc.end())
          boundary_type = BoundaryTypeU::Neumann;
        else if(operator_data.bc->symmetry_bc.find(boundary_id) != operator_data.bc->symmetry_bc.end())
          boundary_type = BoundaryTypeU::Symmetry;

        AssertThrow(boundary_type != BoundaryTypeU::Undefined,
            ExcMessage("Boundary type of face is invalid or not implemented."));

        fe_eval_pressure.reinit (face);

        fe_eval_velocity.reinit(face);
        fe_eval_velocity.read_dof_values(src);
        fe_eval_velocity.evaluate (true,false);

        for(unsigned int q=0;q<fe_eval_pressure.n_q_points;++q)
        {
          Tensor<1,dim,VectorizedArray<value_type> > flux;

          if(operator_data.use_boundary_data == true)
          {
            calculate_flux_boundary_face(flux,q,fe_eval_velocity,OperatorType::full,boundary_type,boundary_id);
          }
          else // use_boundary_data == false
          {
            Tensor<1,dim,VectorizedArray<value_type> > value_m = fe_eval_velocity.get_value(q);
            // exterior value = interior value
            Tensor<1,dim,VectorizedArray<value_type> > value_p = value_m;
            calculate_flux(flux,value_m,value_p);
          }

          VectorizedArray<value_type> flux_times_normal = flux * fe_eval_velocity.get_normal_vector(q);
          fe_eval_pressure.submit_value(flux_times_normal,q);
        }
        fe_eval_pressure.integrate(true,false);
        fe_eval_pressure.distribute_local_to_global(dst);
      }
    }
  }

  void cell_loop_inhom_operator (const MatrixFree<dim,value_type>                &,
                                 parallel::distributed::Vector<value_type>       &,
                                 const parallel::distributed::Vector<value_type> &,
                                 const std::pair<unsigned int,unsigned int>      &) const
  {

  }

  void face_loop_inhom_operator (const MatrixFree<dim,value_type>                 &,
                                 parallel::distributed::Vector<value_type>        &,
                                 const parallel::distributed::Vector<value_type>  &,
                                 const std::pair<unsigned int,unsigned int>       &) const
  {

  }

  void boundary_face_loop_inhom_operator (const MatrixFree<dim,value_type>                &data,
                                          parallel::distributed::Vector<value_type>       &dst,
                                          const parallel::distributed::Vector<value_type> &,
                                          const std::pair<unsigned int,unsigned int>      &face_range) const
  {
    if(operator_data.integration_by_parts_of_divU == true &&
       operator_data.use_boundary_data == true)
    {
      FEFaceEval_Velocity_Velocity_linear fe_eval_velocity(data,this->fe_param,true,operator_data.dof_index_velocity);
      FEFaceEval_Pressure_Velocity_linear fe_eval_pressure(data,this->fe_param,true,operator_data.dof_index_pressure);

      for(unsigned int face=face_range.first; face<face_range.second; face++)
      {
        types::boundary_id boundary_id = data.get_boundary_id(face);
        BoundaryTypeU boundary_type = BoundaryTypeU::Undefined;

        if(operator_data.bc->dirichlet_bc.find(boundary_id) != operator_data.bc->dirichlet_bc.end())
          boundary_type = BoundaryTypeU::Dirichlet;
        else if(operator_data.bc->neumann_bc.find(boundary_id) != operator_data.bc->neumann_bc.end())
          boundary_type = BoundaryTypeU::Neumann;
        else if(operator_data.bc->symmetry_bc.find(boundary_id) != operator_data.bc->symmetry_bc.end())
          boundary_type = BoundaryTypeU::Symmetry;

        AssertThrow(boundary_type != BoundaryTypeU::Undefined,
            ExcMessage("Boundary type of face is invalid or not implemented."));

        fe_eval_pressure.reinit (face);
        fe_eval_velocity.reinit(face);

        for(unsigned int q=0;q<fe_eval_pressure.n_q_points;++q)
        {
          Tensor<1,dim,VectorizedArray<value_type> > flux;
          calculate_flux_boundary_face(flux,q,fe_eval_velocity,OperatorType::inhomogeneous,boundary_type,boundary_id);

          VectorizedArray<value_type> flux_times_normal = flux * fe_eval_velocity.get_normal_vector(q);
          fe_eval_pressure.submit_value(-flux_times_normal,q); // minus sign since this term occurs on the rhs of the equation
        }
        fe_eval_pressure.integrate(true,false);
        fe_eval_pressure.distribute_local_to_global(dst);
      }
    }
  }

  MatrixFree<dim,value_type> const * data;
  DivergenceOperatorData<dim> operator_data;
  double mutable eval_time;
};

template<int dim>
struct ConvectiveOperatorData
{
  ConvectiveOperatorData ()
    :
    dof_index(0)
  {}

  unsigned int dof_index;

  std::shared_ptr<BoundaryDescriptorNavierStokesU<dim> > bc;
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

  typedef FEEvaluationWrapper<dim,fe_degree,fe_degree_xwall,n_actual_q_points_vel_nonlinear,dim,value_type,is_xwall>
    FEEval_Velocity_Velocity_nonlinear;
  typedef FEFaceEvaluationWrapper<dim,fe_degree,fe_degree_xwall,n_actual_q_points_vel_nonlinear,dim,value_type,is_xwall>
    FEFaceEval_Velocity_Velocity_nonlinear;

  typedef ConvectiveOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, value_type> This;

  void initialize(MatrixFree<dim,value_type> const  &mf_data,
                  ConvectiveOperatorData<dim> const &operator_data_in)
  {
    this->data = &mf_data;
    this->operator_data = operator_data_in;
  }

  void evaluate (parallel::distributed::Vector<value_type>       &dst,
                 parallel::distributed::Vector<value_type> const &src,
                 double const                                    evaluation_time) const
  {
    dst = 0;
    evaluate_add(dst,src,evaluation_time);
  }

  //TODO: OIF splitting approach
  void evaluate_oif (parallel::distributed::Vector<value_type>       &dst,
                     parallel::distributed::Vector<value_type> const &src,
                     double const                                    evaluation_time,
                     parallel::distributed::Vector<value_type> const &velocity) const
  {
    dst = 0;

    this->eval_time = evaluation_time;
    velocity_linearization = &velocity;

    data->loop(&This::cell_loop_OIF,&This::face_loop_OIF,
               &This::boundary_face_loop_OIF,this, dst, src);

    velocity_linearization = nullptr;
  }

  void evaluate_add (parallel::distributed::Vector<value_type>       &dst,
                     parallel::distributed::Vector<value_type> const &src,
                     double const                                    evaluation_time) const
  {
    this->eval_time = evaluation_time;

    data->loop(&This::cell_loop_nonlinear_operator,&This::face_loop_nonlinear_operator,
               &This::boundary_face_loop_nonlinear_operator,this, dst, src);
  }

  void apply_linearized (parallel::distributed::Vector<value_type>       &dst,
                         parallel::distributed::Vector<value_type> const &src,
                         parallel::distributed::Vector<value_type> const *vector_linearization,
                         double const                                    evaluation_time) const
  {
    dst = 0;

    apply_linearized_add(dst,src,vector_linearization,evaluation_time);
  }

  void apply_linearized_add (parallel::distributed::Vector<value_type>       &dst,
                             parallel::distributed::Vector<value_type> const &src,
                             parallel::distributed::Vector<value_type> const *vector_linearization,
                             double const                                    evaluation_time) const
  {
    this->eval_time = evaluation_time;
    velocity_linearization = vector_linearization;

    data->loop(&This::cell_loop_linearized_operator,&This::face_loop_linearized_operator,
               &This::boundary_face_loop_linearized_operator,this, dst, src);

    velocity_linearization = nullptr;
  }

  void apply_linearized_block_jacobi (parallel::distributed::Vector<value_type>       &dst,
                                      parallel::distributed::Vector<value_type> const &src,
                                      parallel::distributed::Vector<value_type> const *vector_linearization,
                                      double const                                    evaluation_time) const
  {
    dst = 0;
    apply_linearized_block_jacobi_add(dst,src,vector_linearization,evaluation_time);
  }

  void apply_linearized_block_jacobi_add (parallel::distributed::Vector<value_type>       &dst,
                                          parallel::distributed::Vector<value_type> const &src,
                                          parallel::distributed::Vector<value_type> const *vector_linearization,
                                          double const                                    evaluation_time) const
  {
    this->eval_time = evaluation_time;
    velocity_linearization = vector_linearization;

    data->loop(&This::cell_loop_linearized_operator,&This::face_loop_linearized_operator_block_jacobi,
               &This::boundary_face_loop_linearized_operator,this, dst, src);

    velocity_linearization = nullptr;
  }

  void calculate_diagonal(parallel::distributed::Vector<value_type>       &diagonal,
                          parallel::distributed::Vector<value_type> const *vector_linearization,
                          double const                                    evaluation_time) const
  {
    diagonal = 0;

    add_diagonal(diagonal,vector_linearization,evaluation_time);
  }

  void add_diagonal(parallel::distributed::Vector<value_type>       &diagonal,
                    parallel::distributed::Vector<value_type> const *vector_linearization,
                    double const                                    evaluation_time) const
  {
    this->eval_time = evaluation_time;
    velocity_linearization = vector_linearization;

    parallel::distributed::Vector<value_type>  src_dummy(diagonal);

    data->loop(&This::cell_loop_diagonal,&This::face_loop_diagonal,
               &This::boundary_face_loop_diagonal,this,diagonal,src_dummy);

    velocity_linearization = nullptr;
  }

  void add_block_jacobi_matrices(std::vector<LAPACKFullMatrix<value_type> >      &matrices,
                                 parallel::distributed::Vector<value_type> const *vector_linearization,
                                 double const                                    evaluation_time) const
  {
    this->eval_time = evaluation_time;
    velocity_linearization = vector_linearization;

    parallel::distributed::Vector<value_type>  src;

    data->loop(&This::cell_loop_calculate_block_jacobi_matrices,&This::face_loop_calculate_block_jacobi_matrices,
               &This::boundary_face_loop_calculate_block_jacobi_matrices, this, matrices, src);

    velocity_linearization = nullptr;
  }

  ConvectiveOperatorData<dim> const & get_operator_data() const
  {
    return operator_data;
  }


private:
  template<typename FEEvaluation>
  inline void do_cell_integral_nonlinear_operator(FEEvaluation &fe_eval) const
  {
    fe_eval.evaluate (true,false,false);
    for (unsigned int q=0; q<fe_eval.n_q_points; ++q)
    {
      // nonlinear convective flux F(u) = uu
      Tensor<1,dim,VectorizedArray<value_type> > u = fe_eval.get_value(q);
      Tensor<2,dim,VectorizedArray<value_type> > F = outer_product(u,u);
      fe_eval.submit_gradient (-F, q); // minus sign due to integration by parts
    }
    fe_eval.integrate (false,true);

    //TODO: energy preserving formulation of convective term
//    fe_eval.evaluate (true,true,false);
//    for (unsigned int q=0; q<fe_eval.n_q_points; ++q)
//    {
//      // nonlinear convective flux F(u) = uu
//      Tensor<1,dim,VectorizedArray<value_type> > u = fe_eval.get_value(q);
//      Tensor<2,dim,VectorizedArray<value_type> > F = outer_product(u,u);
//      VectorizedArray<value_type> divergence = fe_eval.get_divergence(q);
//      Tensor<1,dim,VectorizedArray<value_type> > div_term = -0.5*divergence*u;
//      fe_eval.submit_gradient (-F, q); // minus sign due to integration by parts
//      fe_eval.submit_value(div_term,q);
//    }
//    fe_eval.integrate (true,true);
    //TODO: energy preserving formulation of convective term
  }

  template<typename FEEvaluation>
  inline void do_cell_integral_linearized_operator(FEEvaluation &fe_eval,
                                                   FEEvaluation &fe_eval_linearization) const
  {
    fe_eval.evaluate (true,false,false);
    fe_eval_linearization.evaluate (true,false,false);

    for (unsigned int q=0; q<fe_eval.n_q_points; ++q)
    {
      Tensor<1,dim,VectorizedArray<value_type> > delta_u = fe_eval.get_value(q);
      Tensor<1,dim,VectorizedArray<value_type> > u = fe_eval_linearization.get_value(q);
      Tensor<2,dim,VectorizedArray<value_type> > F = outer_product(u,delta_u);
      fe_eval.submit_gradient (-(F+transpose(F)), q); // minus sign due to integration by parts
    }
    fe_eval.integrate (false,true);
  }

  /*
   *  Calculation of lambda according to Shahbazi et al.:
   *  lambda = max ( max |lambda(flux_jacobian_M)| , max |lambda(flux_jacobian_P)| )
   *         = max ( | 2*(uM)^T*normal | , | 2*(uM)^T*normal | )
   */
  inline void calculate_lambda(VectorizedArray<value_type>       &lambda,
                               VectorizedArray<value_type> const &uM_n,
                               VectorizedArray<value_type> const &uP_n) const
  {
    lambda = 2.0 * std::max(std::abs(uM_n), std::abs(uP_n));
  }

  /*
   *  Calculate Lax-Friedrichs flux for nonlinear operator on interior faces.
   */
  inline void calculate_flux_nonlinear_operator(Tensor<1,dim,VectorizedArray<value_type> >       &flux,
                                                Tensor<1,dim,VectorizedArray<value_type> > const &uM,
                                                Tensor<1,dim,VectorizedArray<value_type> > const &uP,
                                                Tensor<1,dim,VectorizedArray<value_type> > const &normalM) const
  {
    VectorizedArray<value_type> uM_n = uM*normalM;
    VectorizedArray<value_type> uP_n = uP*normalM;

    Tensor<1,dim,VectorizedArray<value_type> > average_normal_flux =
        make_vectorized_array<value_type>(0.5) * (uM*uM_n + uP*uP_n);

    Tensor<1,dim,VectorizedArray<value_type> > jump_value = uM - uP;

    VectorizedArray<value_type> lambda;
    calculate_lambda(lambda,uM_n,uP_n);

    flux = average_normal_flux + 0.5 * lambda * jump_value;
  }

  // TODO outflow BC
  inline void calculate_flux_nonlinear_operator_boundary(
      Tensor<1,dim,VectorizedArray<value_type> >       &flux,
      Tensor<1,dim,VectorizedArray<value_type> > const &uM,
      Tensor<1,dim,VectorizedArray<value_type> > const &uP,
      Tensor<1,dim,VectorizedArray<value_type> > const &normalM,
      VectorizedArray<value_type> const                &factor) const
  {
    VectorizedArray<value_type> uM_n = uM*normalM;
    VectorizedArray<value_type> uP_n = uP*normalM;

    Tensor<1,dim,VectorizedArray<value_type> > average_normal_flux =
        make_vectorized_array<value_type>(0.5) * (uM*uM_n + factor*uP*uP_n);

    Tensor<1,dim,VectorizedArray<value_type> > jump_value = uM - uP;

    VectorizedArray<value_type> lambda;
    calculate_lambda(lambda,uM_n,uP_n);

    flux = average_normal_flux + 0.5 * lambda * jump_value;
  }

  /*
   *  Calculate Lax-Friedrichs flux for linearized operator on interior faces.
   */
  inline void calculate_flux_linearized_operator(Tensor<1,dim,VectorizedArray<value_type> > &flux,
                                                 Tensor<1,dim,VectorizedArray<value_type> > &uM,
                                                 Tensor<1,dim,VectorizedArray<value_type> > &uP,
                                                 Tensor<1,dim,VectorizedArray<value_type> > &delta_uM,
                                                 Tensor<1,dim,VectorizedArray<value_type> > &delta_uP,
                                                 Tensor<1,dim,VectorizedArray<value_type> > &normalM) const
  {
    VectorizedArray<value_type> uM_n = uM*normalM;
    VectorizedArray<value_type> uP_n = uP*normalM;

    const VectorizedArray<value_type> delta_uM_n = delta_uM*normalM;
    const VectorizedArray<value_type> delta_uP_n = delta_uP*normalM;

    Tensor<1,dim,VectorizedArray<value_type> > average_normal_flux =
        make_vectorized_array<value_type>(0.5) *
        (uM*delta_uM_n + delta_uM*uM_n + uP*delta_uP_n + delta_uP*uP_n);

    Tensor<1,dim,VectorizedArray<value_type> > jump_value = delta_uM - delta_uP;

    VectorizedArray<value_type> lambda;
    calculate_lambda(lambda,uM_n,uP_n);

    flux = average_normal_flux + 0.5 * lambda * jump_value;
  }

  /*
   *  This function calculates the exterior velocity on boundary faces
   *  according to:
   *
   *  Dirichlet boundary: u⁺ = -u⁻ + 2g
   *  Neumann boundary:   u⁺ = u⁻
   *  symmetry boundary:  u⁺ = u⁻ -(u⁻*n)n - (u⁻*n)n = u⁻ - 2 (u⁻*n)n
  */
  template<typename FEEvaluation>
  inline void calculate_exterior_velocity_boundary_face(Tensor<1,dim,VectorizedArray<value_type> >       &uP,
                                                        Tensor<1,dim,VectorizedArray<value_type> > const &uM,
                                                        unsigned int const                               q,
                                                        FEEvaluation                                     &fe_eval,
                                                        BoundaryTypeU const                              &boundary_type,
                                                        types::boundary_id const                         boundary_id) const
  {
    if(boundary_type == BoundaryTypeU::Dirichlet)
    {
      Tensor<1,dim,VectorizedArray<value_type> > g;
      typename std::map<types::boundary_id,std::shared_ptr<Function<dim> > >::iterator it;
      it = operator_data.bc->dirichlet_bc.find(boundary_id);
      Point<dim,VectorizedArray<value_type> > q_points = fe_eval.quadrature_point(q);
      evaluate_vectorial_function(g,it->second,q_points,eval_time);

      uP = -uM + make_vectorized_array<value_type>(2.0)*g;
    }
    else if(boundary_type == BoundaryTypeU::Neumann)
    {
      uP = uM;
    }
    else if(boundary_type == BoundaryTypeU::Symmetry)
    {
      Tensor<1,dim,VectorizedArray<value_type> > normalM = fe_eval.get_normal_vector(q);
      uP = uM - 2. * (uM*normalM) * normalM;
    }
    else
    {
      AssertThrow(false,ExcMessage("Boundary type of face is invalid or not implemented."));
    }
  }

  /*
   *  Calculate Lax-Friedrichs flux for linearized operator on boundary faces.
   *
   *  Homogeneous linearized operator:
   *  Dirichlet boundary: delta_u⁺ = - delta_u⁻
   *  Neumann boundary:   delta_u⁺ = + delta_u⁻
   *  symmetry boundary:  delta_u⁺ = delta_u⁻ - 2 (delta_u⁻*n)n
   */
  template<typename FEEvaluation>
  inline void calculate_flux_linearized_operator_boundary_face(Tensor<1,dim,VectorizedArray<value_type> > &flux,
                                                               Tensor<1,dim,VectorizedArray<value_type> > &uM,
                                                               Tensor<1,dim,VectorizedArray<value_type> > &uP,
                                                               unsigned int const                         q,
                                                               FEEvaluation                               &fe_eval,
                                                               BoundaryTypeU const                        &boundary_type) const
  {
    // element e⁻
    Tensor<1,dim,VectorizedArray<value_type> > delta_uM = fe_eval.get_value(q);

    // element e⁺
    Tensor<1,dim,VectorizedArray<value_type> > delta_uP;

    if(boundary_type == BoundaryTypeU::Dirichlet)
    {
      delta_uP = - delta_uM;
    }
    else if(boundary_type == BoundaryTypeU::Neumann)
    {
      delta_uP = delta_uM;
    }
    else if(boundary_type == BoundaryTypeU::Symmetry)
    {
      Tensor<1,dim,VectorizedArray<value_type> > normalM = fe_eval.get_normal_vector(q);
      delta_uP = delta_uM - 2. * (delta_uM*normalM) * normalM;
    }
    else
    {
      AssertThrow(false,ExcMessage("Boundary type of face is invalid or not implemented."));
    }

    Tensor<1,dim,VectorizedArray<value_type> > normalM = fe_eval.get_normal_vector(q);
    calculate_flux_linearized_operator(flux,uM,uP,delta_uM,delta_uP,normalM);
  }


  /*
   *  Evaluation of nonlinear convective operator.
   */
  void cell_loop_nonlinear_operator (const MatrixFree<dim,value_type>                 &data,
                                     parallel::distributed::Vector<value_type>        &dst,
                                     const parallel::distributed::Vector<value_type>  &src,
                                     const std::pair<unsigned int,unsigned int>       &cell_range) const
  {
    FEEval_Velocity_Velocity_nonlinear fe_eval(data,this->fe_param,operator_data.dof_index);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);
      fe_eval.read_dof_values(src);

      do_cell_integral_nonlinear_operator(fe_eval);

      fe_eval.distribute_local_to_global(dst);
    }
  }

  void face_loop_nonlinear_operator (const MatrixFree<dim,value_type>                &data,
                                     parallel::distributed::Vector<value_type>       &dst,
                                     const parallel::distributed::Vector<value_type> &src,
                                     const std::pair<unsigned int,unsigned int>      &face_range) const
  {
    FEFaceEval_Velocity_Velocity_nonlinear fe_eval(data,this->fe_param,true,operator_data.dof_index);
    FEFaceEval_Velocity_Velocity_nonlinear fe_eval_neighbor(data,this->fe_param,false,operator_data.dof_index);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval.reinit(face);
      fe_eval.read_dof_values(src);
      fe_eval.evaluate(true,false);

      fe_eval_neighbor.reinit(face);
      fe_eval_neighbor.read_dof_values(src);
      fe_eval_neighbor.evaluate(true,false);

      for(unsigned int q=0;q<fe_eval.n_q_points;++q)
      {
        Tensor<1,dim,VectorizedArray<value_type> > uM = fe_eval.get_value(q);
        Tensor<1,dim,VectorizedArray<value_type> > uP = fe_eval_neighbor.get_value(q);
        Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval.get_normal_vector(q);

        Tensor<1,dim,VectorizedArray<value_type> > flux;
        calculate_flux_nonlinear_operator(flux,uM,uP,normal);

        fe_eval.submit_value(flux,q);
        fe_eval_neighbor.submit_value(-flux,q); // minus sign since n⁺ = - n⁻

        // TODO: energy preserving flux function
//        Tensor<1,dim,VectorizedArray<value_type> > uM = fe_eval.get_value(q);
//        Tensor<1,dim,VectorizedArray<value_type> > uP = fe_eval_neighbor.get_value(q);
//        Tensor<1,dim,VectorizedArray<value_type> > jump = uM - uP;
//        Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval.get_normal_vector(q);
//
//        Tensor<1,dim,VectorizedArray<value_type> > flux, flux_m, flux_p;
//        calculate_flux_nonlinear_operator(flux,uM,uP,normal);
//
//        flux_m = flux + 0.25 * jump*normal * uP;
//        flux_p = -flux + 0.25 * jump*normal * uM;
//
//        fe_eval.submit_value(flux_m,q);
//        fe_eval_neighbor.submit_value(flux_p,q);
        // TODO: energy preserving flux function
      }
      fe_eval.integrate(true,false);
      fe_eval_neighbor.integrate(true,false);

      fe_eval.distribute_local_to_global(dst);
      fe_eval_neighbor.distribute_local_to_global(dst);
    }
  }

  void boundary_face_loop_nonlinear_operator (const MatrixFree<dim,value_type>                &data,
                                              parallel::distributed::Vector<value_type>       &dst,
                                              const parallel::distributed::Vector<value_type> &src,
                                              const std::pair<unsigned int,unsigned int>      &face_range) const
  {
    FEFaceEval_Velocity_Velocity_nonlinear fe_eval(data,this->fe_param,true,operator_data.dof_index);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      types::boundary_id boundary_id = data.get_boundary_id(face);
      BoundaryTypeU boundary_type = BoundaryTypeU::Undefined;

      if(operator_data.bc->dirichlet_bc.find(boundary_id) != operator_data.bc->dirichlet_bc.end())
        boundary_type = BoundaryTypeU::Dirichlet;
      else if(operator_data.bc->neumann_bc.find(boundary_id) != operator_data.bc->neumann_bc.end())
        boundary_type = BoundaryTypeU::Neumann;
      else if(operator_data.bc->symmetry_bc.find(boundary_id) != operator_data.bc->symmetry_bc.end())
        boundary_type = BoundaryTypeU::Symmetry;

      AssertThrow(boundary_type != BoundaryTypeU::Undefined,
          ExcMessage("Boundary type of face is invalid or not implemented."));

      fe_eval.reinit(face);
      fe_eval.read_dof_values(src);
      fe_eval.evaluate(true,false);

      for(unsigned int q=0;q<fe_eval.n_q_points;++q)
      {
        // TODO standard formulation
        Tensor<1,dim,VectorizedArray<value_type> > uM = fe_eval.get_value(q);
        Tensor<1,dim,VectorizedArray<value_type> > uP;
        calculate_exterior_velocity_boundary_face(uP,uM,q,fe_eval,boundary_type,boundary_id);
        Tensor<1,dim,VectorizedArray<value_type> > normalM = fe_eval.get_normal_vector(q);

        Tensor<1,dim,VectorizedArray<value_type> > flux;
        calculate_flux_nonlinear_operator(flux,uM,uP,normalM);
        fe_eval.submit_value(flux,q);
        // TODO standard formulation

        // TODO outflow BC
//        Tensor<1,dim,VectorizedArray<value_type> > uM = fe_eval.get_value(q);
//        Tensor<1,dim,VectorizedArray<value_type> > uP;
//        calculate_exterior_velocity_boundary_face(uP,uM,q,fe_eval,boundary_type,boundary_id);
//        Tensor<1,dim,VectorizedArray<value_type> > normalM = fe_eval.get_normal_vector(q);
//
//        Tensor<1,dim,VectorizedArray<value_type> > flux;
//        bool use_outflow_boundary_condition = true;
//
//        // outflow: do nothing, factor = 1.0
//        // inflow: set convective flux to zero, value = -1.0
//        VectorizedArray<value_type> outflow_on_neumann_boundary
//          = make_vectorized_array<value_type>(1.0);
//
//        if(use_outflow_boundary_condition == true &&
//           boundary_type == BoundaryTypeU::Neumann)
//        {
//          VectorizedArray<value_type> uM_n = uM*normalM;
//
//          for(unsigned int v=0; v<VectorizedArray<value_type>::n_array_elements; ++v)
//            if(uM_n[v] < 0.0) //inflow
//              outflow_on_neumann_boundary[v] = -1.0;
//        }
//        calculate_flux_nonlinear_operator_boundary(flux,uM,uP,normalM,outflow_on_neumann_boundary);
//        fe_eval.submit_value(flux,q);
        // TODO outflow BC

        // TODO: energy preserving flux function
//        Tensor<1,dim,VectorizedArray<value_type> > uM = fe_eval.get_value(q);
//        Tensor<1,dim,VectorizedArray<value_type> > uP;
//        calculate_exterior_velocity_boundary_face(uP,uM,q,fe_eval,boundary_type,boundary_id);
//        Tensor<1,dim,VectorizedArray<value_type> > normalM = fe_eval.get_normal_vector(q);
//
//        Tensor<1,dim,VectorizedArray<value_type> > flux;
//        calculate_flux_nonlinear_operator(flux,uM,uP,normalM);
//        flux = flux + 0.25 * (uM-uP)*normalM * uP;
//        fe_eval.submit_value(flux,q);
        // TODO: energy preserving flux function
      }
      fe_eval.integrate(true,false);
      fe_eval.distribute_local_to_global(dst);
    }
  }

  /*
   *  Evaluate linearized convective operator (homogeneous part of operator).
   */
  void cell_loop_linearized_operator(const MatrixFree<dim,value_type>                &data,
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

      fe_eval_linearization.reinit(cell);
      fe_eval_linearization.read_dof_values(*velocity_linearization);

      do_cell_integral_linearized_operator(fe_eval,fe_eval_linearization);

      fe_eval.distribute_local_to_global (dst);
    }
  }

  void face_loop_linearized_operator (const MatrixFree<dim,value_type>                &data,
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
        Tensor<1,dim,VectorizedArray<value_type> > delta_uM = fe_eval.get_value(q);
        Tensor<1,dim,VectorizedArray<value_type> > delta_uP = fe_eval_neighbor.get_value(q);
        Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval.get_normal_vector(q);
        Tensor<1,dim,VectorizedArray<value_type> > flux;

        calculate_flux_linearized_operator(flux,uM,uP,delta_uM,delta_uP,normal);

        fe_eval.submit_value(flux,q);
        fe_eval_neighbor.submit_value(-flux,q); // minus sign since n⁺ = -n⁻
      }
      fe_eval.integrate(true,false);
      fe_eval_neighbor.integrate(true,false);

      fe_eval.distribute_local_to_global(dst);
      fe_eval_neighbor.distribute_local_to_global(dst);
    }
  }

  void boundary_face_loop_linearized_operator (const MatrixFree<dim,value_type>                &data,
                                               parallel::distributed::Vector<value_type>       &dst,
                                               const parallel::distributed::Vector<value_type> &src,
                                               const std::pair<unsigned int,unsigned int>      &face_range) const
  {
    FEFaceEval_Velocity_Velocity_nonlinear fe_eval(data,this->fe_param,true,operator_data.dof_index);
    FEFaceEval_Velocity_Velocity_nonlinear fe_eval_linearization(data,this->fe_param,true,operator_data.dof_index);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      types::boundary_id boundary_id = data.get_boundary_id(face);
      BoundaryTypeU boundary_type = BoundaryTypeU::Undefined;

      if(operator_data.bc->dirichlet_bc.find(boundary_id) != operator_data.bc->dirichlet_bc.end())
        boundary_type = BoundaryTypeU::Dirichlet;
      else if(operator_data.bc->neumann_bc.find(boundary_id) != operator_data.bc->neumann_bc.end())
        boundary_type = BoundaryTypeU::Neumann;
      else if(operator_data.bc->symmetry_bc.find(boundary_id) != operator_data.bc->symmetry_bc.end())
        boundary_type = BoundaryTypeU::Symmetry;

      AssertThrow(boundary_type != BoundaryTypeU::Undefined,
          ExcMessage("Boundary type of face is invalid or not implemented."));

      fe_eval.reinit (face);
      fe_eval.read_dof_values(src);
      fe_eval.evaluate(true,false);

      fe_eval_linearization.reinit (face);
      fe_eval_linearization.read_dof_values(*velocity_linearization);
      fe_eval_linearization.evaluate(true,false);

      for(unsigned int q=0;q<fe_eval.n_q_points;++q)
      {
        Tensor<1,dim,VectorizedArray<value_type> > uM = fe_eval_linearization.get_value(q);
        Tensor<1,dim,VectorizedArray<value_type> > uP;
        calculate_exterior_velocity_boundary_face(uP,uM,q,fe_eval_linearization,boundary_type,boundary_id);

        Tensor<1,dim,VectorizedArray<value_type> > flux;
        calculate_flux_linearized_operator_boundary_face(flux,uM,uP,q,fe_eval,boundary_type);

        fe_eval.submit_value(flux,q);
      }
      fe_eval.integrate(true,false);
      fe_eval.distribute_local_to_global(dst);
    }
  }

  /*
   *  OIF splitting approach
   */
  void cell_loop_OIF(const MatrixFree<dim,value_type>                &data,
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

      fe_eval_linearization.reinit(cell);
      fe_eval_linearization.read_dof_values(*velocity_linearization);

      fe_eval.evaluate (true,false,false);
      fe_eval_linearization.evaluate (true,false,false);

      for (unsigned int q=0; q<fe_eval.n_q_points; ++q)
      {
        Tensor<1,dim,VectorizedArray<value_type> > u = fe_eval.get_value(q);
        Tensor<1,dim,VectorizedArray<value_type> > u_ref = fe_eval_linearization.get_value(q);
        Tensor<2,dim,VectorizedArray<value_type> > F = outer_product(u,u_ref);
        fe_eval.submit_gradient (-F, q); // minus sign due to integration by parts
      }
      fe_eval.integrate (false,true);

      fe_eval.distribute_local_to_global (dst);
    }
  }

  void face_loop_OIF (const MatrixFree<dim,value_type>                &data,
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
        Tensor<1,dim,VectorizedArray<value_type> > uM_ref = fe_eval_linearization.get_value(q);
        Tensor<1,dim,VectorizedArray<value_type> > uP_ref = fe_eval_linearization_neighbor.get_value(q);
        Tensor<1,dim,VectorizedArray<value_type> > uM = fe_eval.get_value(q);
        Tensor<1,dim,VectorizedArray<value_type> > uP = fe_eval_neighbor.get_value(q);
        Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval.get_normal_vector(q);
        Tensor<1,dim,VectorizedArray<value_type> > flux;

        VectorizedArray<value_type> average_normal_velocity = 0.5 * ((uM_ref + uP_ref) * normal);
        flux = 0.5 * average_normal_velocity * (uM + uP) + 0.5 * std::abs(average_normal_velocity) * (uM-uP);

        fe_eval.submit_value(flux,q);
        fe_eval_neighbor.submit_value(-flux,q); // minus sign since n⁺ = -n⁻
      }
      fe_eval.integrate(true,false);
      fe_eval_neighbor.integrate(true,false);

      fe_eval.distribute_local_to_global(dst);
      fe_eval_neighbor.distribute_local_to_global(dst);
    }
  }

  void boundary_face_loop_OIF (const MatrixFree<dim,value_type>                &data,
                               parallel::distributed::Vector<value_type>       &dst,
                               const parallel::distributed::Vector<value_type> &src,
                               const std::pair<unsigned int,unsigned int>      &face_range) const
  {
    FEFaceEval_Velocity_Velocity_nonlinear fe_eval(data,this->fe_param,true,operator_data.dof_index);
    FEFaceEval_Velocity_Velocity_nonlinear fe_eval_linearization(data,this->fe_param,true,operator_data.dof_index);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      types::boundary_id boundary_id = data.get_boundary_id(face);
      BoundaryTypeU boundary_type = BoundaryTypeU::Undefined;

      if(operator_data.bc->dirichlet_bc.find(boundary_id) != operator_data.bc->dirichlet_bc.end())
        boundary_type = BoundaryTypeU::Dirichlet;
      else if(operator_data.bc->neumann_bc.find(boundary_id) != operator_data.bc->neumann_bc.end())
        boundary_type = BoundaryTypeU::Neumann;
      else if(operator_data.bc->symmetry_bc.find(boundary_id) != operator_data.bc->symmetry_bc.end())
        boundary_type = BoundaryTypeU::Symmetry;

      AssertThrow(boundary_type != BoundaryTypeU::Undefined,
          ExcMessage("Boundary type of face is invalid or not implemented."));

      fe_eval.reinit (face);
      fe_eval.read_dof_values(src);
      fe_eval.evaluate(true,false);

      fe_eval_linearization.reinit (face);
      fe_eval_linearization.read_dof_values(*velocity_linearization);
      fe_eval_linearization.evaluate(true,false);

      for(unsigned int q=0;q<fe_eval.n_q_points;++q)
      {
        Tensor<1,dim,VectorizedArray<value_type> > uM_ref = fe_eval_linearization.get_value(q);
        Tensor<1,dim,VectorizedArray<value_type> > uP_ref;
        calculate_exterior_velocity_boundary_face(uP_ref,uM_ref,q,fe_eval_linearization,boundary_type,boundary_id);

        Tensor<1,dim,VectorizedArray<value_type> > uM = fe_eval.get_value(q);
        Tensor<1,dim,VectorizedArray<value_type> > uP;
        calculate_exterior_velocity_boundary_face(uP,uM,q,fe_eval,boundary_type,boundary_id);

        Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval.get_normal_vector(q);

        Tensor<1,dim,VectorizedArray<value_type> > flux;
        VectorizedArray<value_type> average_normal_velocity = 0.5 * ((uM_ref + uP_ref) * normal);
        flux = 0.5 * average_normal_velocity * (uM + uP) + 0.5 * std::abs(average_normal_velocity) * (uM-uP);

        fe_eval.submit_value(flux,q);
      }
      fe_eval.integrate(true,false);
      fe_eval.distribute_local_to_global(dst);
    }
  }

  /*
   *  Block-jacobi operator: re-implement face_loop, cell_loop and boundary_face_loop are
   *  identical to linearized homogeneous operator.
   */
  void face_loop_linearized_operator_block_jacobi (const MatrixFree<dim,value_type>                &data,
                                                   parallel::distributed::Vector<value_type>       &dst,
                                                   const parallel::distributed::Vector<value_type> &src,
                                                   const std::pair<unsigned int,unsigned int>      &face_range) const
  {
    FEFaceEval_Velocity_Velocity_nonlinear fe_eval_linearization(data,this->fe_param,true,operator_data.dof_index);
    FEFaceEval_Velocity_Velocity_nonlinear fe_eval_linearization_neighbor(data,this->fe_param,false,operator_data.dof_index);

    // Perform face integral for element e⁻
    FEFaceEval_Velocity_Velocity_nonlinear fe_eval(data,this->fe_param,true,operator_data.dof_index);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval_linearization.reinit(face);
      fe_eval_linearization.read_dof_values(*velocity_linearization);
      fe_eval_linearization.evaluate(true, false);

      fe_eval_linearization_neighbor.reinit (face);
      fe_eval_linearization_neighbor.read_dof_values(*velocity_linearization);
      fe_eval_linearization_neighbor.evaluate(true, false);

      fe_eval.reinit(face);
      fe_eval.read_dof_values(src);
      fe_eval.evaluate(true, false);

      // integrate over face for element e⁻
      for(unsigned int q=0;q<fe_eval.n_q_points;++q)
      {
        Tensor<1,dim,VectorizedArray<value_type> > uM = fe_eval_linearization.get_value(q);
        Tensor<1,dim,VectorizedArray<value_type> > uP = fe_eval_linearization_neighbor.get_value(q);
        Tensor<1,dim,VectorizedArray<value_type> > delta_uM = fe_eval.get_value(q);
        Tensor<1,dim,VectorizedArray<value_type> > delta_uP; // set delta_uP to zero
        Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval.get_normal_vector(q);
        Tensor<1,dim,VectorizedArray<value_type> > flux;

        calculate_flux_linearized_operator(flux,uM,uP,delta_uM,delta_uP,normal);

        fe_eval.submit_value(flux,q);
      }
      fe_eval.integrate(true,false);
      fe_eval.distribute_local_to_global(dst);
    }

    // TODO: this has to be removed as soon as the new infrastructure is used that
    // allows to perform face integrals over all faces of the current element.
    // Perform face integral for element e⁺
    FEFaceEval_Velocity_Velocity_nonlinear fe_eval_neighbor(data,this->fe_param,false,operator_data.dof_index);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval_linearization.reinit(face);
      fe_eval_linearization.read_dof_values(*velocity_linearization);
      fe_eval_linearization.evaluate(true, false);

      fe_eval_linearization_neighbor.reinit (face);
      fe_eval_linearization_neighbor.read_dof_values(*velocity_linearization);
      fe_eval_linearization_neighbor.evaluate(true, false);

      fe_eval_neighbor.reinit (face);
      fe_eval_neighbor.read_dof_values(src);
      fe_eval_neighbor.evaluate(true, false);

      // integrate over face for element e⁺
      for(unsigned int q=0;q<fe_eval.n_q_points;++q)
      {
        Tensor<1,dim,VectorizedArray<value_type> > uM = fe_eval_linearization.get_value(q);
        Tensor<1,dim,VectorizedArray<value_type> > uP = fe_eval_linearization_neighbor.get_value(q);
        Tensor<1,dim,VectorizedArray<value_type> > delta_uM; // set delta_uM to zero
        Tensor<1,dim,VectorizedArray<value_type> > delta_uP = fe_eval_neighbor.get_value(q);
        // hack: minus sign since n⁺ = -n⁻ !!!
        Tensor<1,dim,VectorizedArray<value_type> > normal = - fe_eval_neighbor.get_normal_vector(q);
        Tensor<1,dim,VectorizedArray<value_type> > flux;
        // hack: note that uM and uP, delta_uM and delta_uP are interchanged !!!
        calculate_flux_linearized_operator(flux,uP,uM,delta_uP,delta_uM,normal);

        fe_eval_neighbor.submit_value(flux,q);
      }
      fe_eval_neighbor.integrate(true,false);
      fe_eval_neighbor.distribute_local_to_global(dst);
    }
  }


  /*
   *  Calculation of diagonal of linearized convective operator.
   */
  void cell_loop_diagonal (const MatrixFree<dim,value_type>                 &data,
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

      fe_eval.reinit(cell);

      VectorizedArray<value_type> local_diagonal_vector[fe_eval.tensor_dofs_per_cell*dim];

      for (unsigned int j=0; j<fe_eval.dofs_per_cell*dim; ++j)
      {
        for (unsigned int i=0; i<fe_eval.dofs_per_cell*dim; ++i)
          fe_eval.write_cellwise_dof_value(i,make_vectorized_array<value_type>(0.));
        fe_eval.write_cellwise_dof_value(j,make_vectorized_array<value_type>(1.));

        do_cell_integral_linearized_operator(fe_eval,fe_eval_linearization);

        local_diagonal_vector[j] = fe_eval.read_cellwise_dof_value(j);
      }

      for (unsigned int j=0; j<fe_eval.dofs_per_cell*dim; ++j)
        fe_eval.write_cellwise_dof_value(j,local_diagonal_vector[j]);

      fe_eval.distribute_local_to_global (dst);
    }
  }

  void face_loop_diagonal (const MatrixFree<dim,value_type>                &data,
                           parallel::distributed::Vector<value_type>       &dst,
                           const parallel::distributed::Vector<value_type> &,
                           const std::pair<unsigned int,unsigned int>      &face_range) const
  {
    FEFaceEval_Velocity_Velocity_nonlinear fe_eval(data,this->fe_param,true,operator_data.dof_index);
    FEFaceEval_Velocity_Velocity_nonlinear fe_eval_neighbor(data,this->fe_param,false,operator_data.dof_index);

    FEFaceEval_Velocity_Velocity_nonlinear fe_eval_linearization(data,this->fe_param,true,operator_data.dof_index);
    FEFaceEval_Velocity_Velocity_nonlinear fe_eval_linearization_neighbor(data,this->fe_param,false,operator_data.dof_index);

    // Perform face intergrals for element e⁻
    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval_linearization.reinit(face);
      fe_eval_linearization.read_dof_values(*velocity_linearization);
      fe_eval_linearization.evaluate(true, false);

      fe_eval_linearization_neighbor.reinit (face);
      fe_eval_linearization_neighbor.read_dof_values(*velocity_linearization);
      fe_eval_linearization_neighbor.evaluate(true, false);

      fe_eval.reinit(face);

      VectorizedArray<value_type> local_diagonal_vector[fe_eval.tensor_dofs_per_cell*dim];
      for (unsigned int j=0; j<fe_eval.dofs_per_cell*dim; ++j)
      {
        // set dof value j of element- to 1 and all other dof values of element- to zero
        for (unsigned int i=0; i<fe_eval.dofs_per_cell*dim; ++i)
          fe_eval.write_cellwise_dof_value(i,make_vectorized_array<value_type>(0.));
        fe_eval.write_cellwise_dof_value(j,make_vectorized_array<value_type>(1.));

        fe_eval.evaluate(true, false);

        for(unsigned int q=0;q<fe_eval.n_q_points;++q)
        {
          Tensor<1,dim,VectorizedArray<value_type> > uM = fe_eval_linearization.get_value(q);
          Tensor<1,dim,VectorizedArray<value_type> > uP = fe_eval_linearization_neighbor.get_value(q);
          Tensor<1,dim,VectorizedArray<value_type> > delta_uM = fe_eval.get_value(q);
          Tensor<1,dim,VectorizedArray<value_type> > delta_uP; // set delta_uP to zero
          Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval.get_normal_vector(q);
          Tensor<1,dim,VectorizedArray<value_type> > flux;

          calculate_flux_linearized_operator(flux,uM,uP,delta_uM,delta_uP,normal);

          fe_eval.submit_value(flux,q);
        }
        fe_eval.integrate(true,false);

        local_diagonal_vector[j] = fe_eval.read_cellwise_dof_value(j);
      }
      for (unsigned int j=0; j<fe_eval.dofs_per_cell*dim; ++j)
        fe_eval.write_cellwise_dof_value(j, local_diagonal_vector[j]);

      fe_eval.distribute_local_to_global(dst);
    }


    // TODO: this has to be removed as soon as the new infrastructure is used that
    // allows to perform face integrals over all faces of the current element.
    // Perform face intergrals for element e⁺
    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval_linearization.reinit(face);
      fe_eval_linearization.read_dof_values(*velocity_linearization);
      fe_eval_linearization.evaluate(true, false);

      fe_eval_linearization_neighbor.reinit (face);
      fe_eval_linearization_neighbor.read_dof_values(*velocity_linearization);
      fe_eval_linearization_neighbor.evaluate(true, false);

      fe_eval_neighbor.reinit (face);

      VectorizedArray<value_type> local_diagonal_vector_neighbor[fe_eval_neighbor.tensor_dofs_per_cell*dim];
      for (unsigned int j=0; j<fe_eval_neighbor.dofs_per_cell*dim; ++j)
      {
        // set dof value j of element+ to 1 and all other dof values of element+ to zero
        for (unsigned int i=0; i<fe_eval_neighbor.dofs_per_cell*dim; ++i)
          fe_eval_neighbor.write_cellwise_dof_value(i, make_vectorized_array<value_type>(0.));
        fe_eval_neighbor.write_cellwise_dof_value(j,make_vectorized_array<value_type>(1.));

        fe_eval_neighbor.evaluate(true, false);

        for(unsigned int q=0;q<fe_eval.n_q_points;++q)
        {
          Tensor<1,dim,VectorizedArray<value_type> > uM = fe_eval_linearization.get_value(q);
          Tensor<1,dim,VectorizedArray<value_type> > uP = fe_eval_linearization_neighbor.get_value(q);
          Tensor<1,dim,VectorizedArray<value_type> > delta_uP = fe_eval_neighbor.get_value(q);
          Tensor<1,dim,VectorizedArray<value_type> > delta_uM; // set delta_uM to zero
          // hack: minus sign since n⁺ = -n⁻ !!!
          Tensor<1,dim,VectorizedArray<value_type> > normal = - fe_eval_neighbor.get_normal_vector(q);
          Tensor<1,dim,VectorizedArray<value_type> > flux;
          // hack: note that uM and uP, delta_uM and delta_uP are interchanged !!!
          calculate_flux_linearized_operator(flux,uP,uM,delta_uP,delta_uM,normal);
          fe_eval_neighbor.submit_value(flux,q);
        }
        fe_eval_neighbor.integrate(true,false);

        local_diagonal_vector_neighbor[j] = fe_eval_neighbor.read_cellwise_dof_value(j);
      }
      for (unsigned int j=0; j<fe_eval_neighbor.dofs_per_cell*dim; ++j)
        fe_eval_neighbor.write_cellwise_dof_value(j, local_diagonal_vector_neighbor[j]);

      fe_eval_neighbor.distribute_local_to_global(dst);
    }
  }


  // TODO: this function has to be removed as soon as the new infrastructure is used that
  // allows to perform face integrals over all faces of the current element
  void boundary_face_loop_diagonal (const MatrixFree<dim,value_type>                 &data,
                                    parallel::distributed::Vector<value_type>        &dst,
                                    const parallel::distributed::Vector<value_type>  &,
                                    const std::pair<unsigned int,unsigned int>       &face_range) const
  {
    FEFaceEval_Velocity_Velocity_nonlinear fe_eval(data,this->fe_param,true,operator_data.dof_index);
    FEFaceEval_Velocity_Velocity_nonlinear fe_eval_linearization(data,this->fe_param,true,operator_data.dof_index);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      types::boundary_id boundary_id = data.get_boundary_id(face);
      BoundaryTypeU boundary_type = BoundaryTypeU::Undefined;

      if(operator_data.bc->dirichlet_bc.find(boundary_id) != operator_data.bc->dirichlet_bc.end())
        boundary_type = BoundaryTypeU::Dirichlet;
      else if(operator_data.bc->neumann_bc.find(boundary_id) != operator_data.bc->neumann_bc.end())
        boundary_type = BoundaryTypeU::Neumann;
      else if(operator_data.bc->symmetry_bc.find(boundary_id) != operator_data.bc->symmetry_bc.end())
        boundary_type = BoundaryTypeU::Symmetry;

      AssertThrow(boundary_type != BoundaryTypeU::Undefined,
          ExcMessage("Boundary type of face is invalid or not implemented."));

      fe_eval.reinit (face);

      fe_eval_linearization.reinit (face);
      fe_eval_linearization.read_dof_values(*velocity_linearization);
      fe_eval_linearization.evaluate(true,false);

      VectorizedArray<value_type> local_diagonal_vector[fe_eval.tensor_dofs_per_cell*dim];
      for (unsigned int j=0; j<fe_eval.dofs_per_cell*dim; ++j)
      {
        // set dof value j of element- to 1 and all other dof values of element- to zero
        for (unsigned int i=0; i<fe_eval.dofs_per_cell*dim; ++i)
          fe_eval.write_cellwise_dof_value(i,make_vectorized_array<value_type>(0.));
        fe_eval.write_cellwise_dof_value(j,make_vectorized_array<value_type>(1.));

        fe_eval.evaluate(true, false);

        for(unsigned int q=0;q<fe_eval.n_q_points;++q)
        {
          Tensor<1,dim,VectorizedArray<value_type> > uM = fe_eval_linearization.get_value(q);
          Tensor<1,dim,VectorizedArray<value_type> > uP;
          calculate_exterior_velocity_boundary_face(uP,uM,q,fe_eval_linearization,boundary_type,boundary_id);

          Tensor<1,dim,VectorizedArray<value_type> > flux;
          calculate_flux_linearized_operator_boundary_face(flux,uM,uP,q,fe_eval,boundary_type);

          fe_eval.submit_value(flux,q);
        }
        fe_eval.integrate(true,false);

        local_diagonal_vector[j] = fe_eval.read_cellwise_dof_value(j);
      }

      for (unsigned int j=0; j<fe_eval.dofs_per_cell*dim; ++j)
        fe_eval.write_cellwise_dof_value(j, local_diagonal_vector[j]);

      fe_eval.distribute_local_to_global(dst);
    }
  }

  void cell_loop_calculate_block_jacobi_matrices (const MatrixFree<dim,value_type>                 &data,
                                                  std::vector<LAPACKFullMatrix<value_type> >       &matrices,
                                                  const parallel::distributed::Vector<value_type>  &,
                                                  const std::pair<unsigned int,unsigned int>       &cell_range) const
  {
    FEEval_Velocity_Velocity_nonlinear fe_eval(data,this->fe_param,operator_data.dof_index);
    FEEval_Velocity_Velocity_nonlinear fe_eval_linearization(data,this->fe_param,operator_data.dof_index);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval_linearization.reinit(cell);
      fe_eval_linearization.read_dof_values(*velocity_linearization);

      fe_eval.reinit(cell);

      // Note that the velocity has dim components.
      unsigned int dofs_per_cell = fe_eval.dofs_per_cell*dim;

      for (unsigned int j=0; j<dofs_per_cell; ++j)
      {
        for (unsigned int i=0; i<dofs_per_cell; ++i)
          fe_eval.begin_dof_values()[i] = make_vectorized_array<value_type>(0.);
        fe_eval.begin_dof_values()[j] = make_vectorized_array<value_type>(1.);

        do_cell_integral_linearized_operator(fe_eval,fe_eval_linearization);

        for(unsigned int i=0; i<dofs_per_cell; ++i)
          for (unsigned int v=0; v<VectorizedArray<value_type>::n_array_elements; ++v)
            matrices[cell*VectorizedArray<value_type>::n_array_elements+v](i,j) += fe_eval.begin_dof_values()[i][v];
      }
    }
  }

  void face_loop_calculate_block_jacobi_matrices (const MatrixFree<dim,value_type>                &data,
                                                  std::vector<LAPACKFullMatrix<value_type> >      &matrices,
                                                  const parallel::distributed::Vector<value_type> &,
                                                  const std::pair<unsigned int,unsigned int>      &face_range) const
  {
    FEFaceEval_Velocity_Velocity_nonlinear fe_eval(data,this->fe_param,true,operator_data.dof_index);
    FEFaceEval_Velocity_Velocity_nonlinear fe_eval_neighbor(data,this->fe_param,false,operator_data.dof_index);

    FEFaceEval_Velocity_Velocity_nonlinear fe_eval_linearization(data,this->fe_param,true,operator_data.dof_index);
    FEFaceEval_Velocity_Velocity_nonlinear fe_eval_linearization_neighbor(data,this->fe_param,false,operator_data.dof_index);

    // Perform face intergrals for element e⁻.
    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval_linearization.reinit(face);
      fe_eval_linearization.read_dof_values(*velocity_linearization);
      fe_eval_linearization.evaluate(true, false);

      fe_eval_linearization_neighbor.reinit (face);
      fe_eval_linearization_neighbor.read_dof_values(*velocity_linearization);
      fe_eval_linearization_neighbor.evaluate(true, false);

      fe_eval.reinit(face);

      // Note that the velocity has dim components.
      unsigned int dofs_per_cell = fe_eval.dofs_per_cell*dim;

      for (unsigned int j=0; j<dofs_per_cell; ++j)
      {
        // set dof value j of element- to 1 and all other dof values of element- to zero
        for (unsigned int i=0; i<dofs_per_cell; ++i)
          fe_eval.begin_dof_values()[i] = make_vectorized_array<value_type>(0.);
        fe_eval.begin_dof_values()[j] = make_vectorized_array<value_type>(1.);

        fe_eval.evaluate(true, false);

        for(unsigned int q=0;q<fe_eval.n_q_points;++q)
        {
          Tensor<1,dim,VectorizedArray<value_type> > uM = fe_eval_linearization.get_value(q);
          Tensor<1,dim,VectorizedArray<value_type> > uP = fe_eval_linearization_neighbor.get_value(q);
          Tensor<1,dim,VectorizedArray<value_type> > delta_uM = fe_eval.get_value(q);
          Tensor<1,dim,VectorizedArray<value_type> > delta_uP; // set delta_uP to zero
          Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval.get_normal_vector(q);
          Tensor<1,dim,VectorizedArray<value_type> > flux;

          calculate_flux_linearized_operator(flux,uM,uP,delta_uM,delta_uP,normal);

          fe_eval.submit_value(flux,q);
        }
        fe_eval.integrate(true,false);

        for (unsigned int v=0; v<VectorizedArray<value_type>::n_array_elements; ++v)
        {
          const unsigned int cell_number = data.get_face_info(face).cells_minus[v];
          if (cell_number != numbers::invalid_unsigned_int)
            for(unsigned int i=0; i<dofs_per_cell; ++i)
              matrices[cell_number](i,j) += fe_eval.begin_dof_values()[i][v];
        }
      }
    }



    // TODO: This has to be removed as soon as the new infrastructure is used that
    // allows to perform face integrals over all faces of the current element.
    // Perform face intergrals for element e⁺.
    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval_linearization.reinit(face);
      fe_eval_linearization.read_dof_values(*velocity_linearization);
      fe_eval_linearization.evaluate(true, false);

      fe_eval_linearization_neighbor.reinit (face);
      fe_eval_linearization_neighbor.read_dof_values(*velocity_linearization);
      fe_eval_linearization_neighbor.evaluate(true, false);

      fe_eval_neighbor.reinit (face);

      // Note that the velocity has dim components.
      unsigned int dofs_per_cell = fe_eval_neighbor.dofs_per_cell*dim;

      for (unsigned int j=0; j<dofs_per_cell; ++j)
      {
        // set dof value j of element+ to 1 and all other dof values of element+ to zero
        for (unsigned int i=0; i<dofs_per_cell; ++i)
          fe_eval_neighbor.begin_dof_values()[i] = make_vectorized_array<value_type>(0.);
        fe_eval_neighbor.begin_dof_values()[j] = make_vectorized_array<value_type>(1.);

        fe_eval_neighbor.evaluate(true, false);

        for(unsigned int q=0;q<fe_eval.n_q_points;++q)
        {
          Tensor<1,dim,VectorizedArray<value_type> > uM = fe_eval_linearization.get_value(q);
          Tensor<1,dim,VectorizedArray<value_type> > uP = fe_eval_linearization_neighbor.get_value(q);
          Tensor<1,dim,VectorizedArray<value_type> > delta_uP = fe_eval_neighbor.get_value(q);
          Tensor<1,dim,VectorizedArray<value_type> > delta_uM; // set delta_uM to zero
          // hack: minus sign since n⁺ = -n⁻ !!!
          Tensor<1,dim,VectorizedArray<value_type> > normal = - fe_eval_neighbor.get_normal_vector(q);
          Tensor<1,dim,VectorizedArray<value_type> > flux;
          // hack: note that uM and uP, delta_uM and delta_uP are interchanged !!!
          calculate_flux_linearized_operator(flux,uP,uM,delta_uP,delta_uM,normal);
          fe_eval_neighbor.submit_value(flux,q);
        }
        fe_eval_neighbor.integrate(true,false);

        for (unsigned int v=0; v<VectorizedArray<value_type>::n_array_elements; ++v)
        {
          const unsigned int cell_number = data.get_face_info(face).cells_plus[v];
          if (cell_number != numbers::invalid_unsigned_int)
            for(unsigned int i=0; i<dofs_per_cell; ++i)
              matrices[cell_number](i,j) += fe_eval_neighbor.begin_dof_values()[i][v];
        }
      }
    }
  }

  // TODO: This function has to be removed as soon as the new infrastructure is used that
  // allows to perform face integrals over all faces of the current element.
  void boundary_face_loop_calculate_block_jacobi_matrices (const MatrixFree<dim,value_type>                &data,
                                                           std::vector<LAPACKFullMatrix<value_type> >      &matrices,
                                                           const parallel::distributed::Vector<value_type> &,
                                                           const std::pair<unsigned int,unsigned int>      &face_range) const
  {
    FEFaceEval_Velocity_Velocity_nonlinear fe_eval(data,this->fe_param,true,operator_data.dof_index);
    FEFaceEval_Velocity_Velocity_nonlinear fe_eval_linearization(data,this->fe_param,true,operator_data.dof_index);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      types::boundary_id boundary_id = data.get_boundary_id(face);
      BoundaryTypeU boundary_type = BoundaryTypeU::Undefined;

      if(operator_data.bc->dirichlet_bc.find(boundary_id) != operator_data.bc->dirichlet_bc.end())
        boundary_type = BoundaryTypeU::Dirichlet;
      else if(operator_data.bc->neumann_bc.find(boundary_id) != operator_data.bc->neumann_bc.end())
        boundary_type = BoundaryTypeU::Neumann;
      else if(operator_data.bc->symmetry_bc.find(boundary_id) != operator_data.bc->symmetry_bc.end())
        boundary_type = BoundaryTypeU::Symmetry;

      AssertThrow(boundary_type != BoundaryTypeU::Undefined,
          ExcMessage("Boundary type of face is invalid or not implemented."));

      fe_eval.reinit (face);

      fe_eval_linearization.reinit (face);
      fe_eval_linearization.read_dof_values(*velocity_linearization);
      fe_eval_linearization.evaluate(true,false);

      // Note that the velocity has dim components.
      unsigned int dofs_per_cell = fe_eval.dofs_per_cell*dim;

      for (unsigned int j=0; j<dofs_per_cell; ++j)
      {
        // set dof value j of element- to 1 and all other dof values of element- to zero
        for (unsigned int i=0; i<dofs_per_cell; ++i)
          fe_eval.begin_dof_values()[i] = make_vectorized_array<value_type>(0.);
        fe_eval.begin_dof_values()[j] = make_vectorized_array<value_type>(1.);

        fe_eval.evaluate(true, false);

        for(unsigned int q=0;q<fe_eval.n_q_points;++q)
        {
          Tensor<1,dim,VectorizedArray<value_type> > uM = fe_eval_linearization.get_value(q);
          Tensor<1,dim,VectorizedArray<value_type> > uP;
          calculate_exterior_velocity_boundary_face(uP,uM,q,fe_eval_linearization,boundary_type,boundary_id);

          Tensor<1,dim,VectorizedArray<value_type> > flux;
          calculate_flux_linearized_operator_boundary_face(flux,uM,uP,q,fe_eval,boundary_type);

          fe_eval.submit_value(flux,q);
        }
        fe_eval.integrate(true,false);

        for (unsigned int v=0; v<VectorizedArray<value_type>::n_array_elements; ++v)
        {
          const unsigned int cell_number = data.get_face_info(face).cells_minus[v];
          if (cell_number != numbers::invalid_unsigned_int)
            for(unsigned int i=0; i<dofs_per_cell; ++i)
              matrices[cell_number](i,j) += fe_eval.begin_dof_values()[i][v];
        }
      }
    }
  }

  MatrixFree<dim,value_type> const * data;
  ConvectiveOperatorData<dim> operator_data;
  mutable double eval_time;
  mutable parallel::distributed::Vector<value_type> const * velocity_linearization;
};

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_NAVIER_STOKES_OPERATORS_H_ */
