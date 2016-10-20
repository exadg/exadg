/*
 * DGSpalartAllmarasModel.h
 *
 *  Created on: Aug 1, 2016
 *      Author: krank
 */

#ifndef INCLUDE_DGSPALARTALLMARASMODEL_H_
#define INCLUDE_DGSPALARTALLMARASMODEL_H_

#include "DGNavierStokesDualSplittingXWall.h"
#include "FE_Parameters.h"

using namespace dealii;
const double vt_initial_value = 0.0005;

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
class DGSpalartAllmarasModel
{
public:
  typedef double value_type;
  static const bool is_xwall = true;
  static const unsigned int n_actual_q_points_vel_linear = xwall_quad_rule;

  /*
   * nomenclature typdedef FEEvaluationWrapper:
   * FEEval_name1_name2 : name1 specifies the dof handler, name2 the quadrature formula
   * example: FEEval_Pressure_Velocity_linear: dof handler for pressure (scalar quantity),
   * quadrature formula with fe_degree_velocity+1 quadrature points
   */

  typedef FEEvaluationWrapper<dim,fe_degree,fe_degree_xwall,n_actual_q_points_vel_linear,dim,value_type,is_xwall> FEEval_Velocity_Velocity_nonlinear;
  typedef FEFaceEvaluationWrapper<dim,fe_degree,fe_degree_xwall,n_actual_q_points_vel_linear,dim,value_type,is_xwall> FEFaceEval_Velocity_Velocity_nonlinear;
  //situation is equal to pressure: do not want to have enrichment but need same quadrature rule as with enrichment
  typedef FEEvaluationWrapperPressure<dim,fe_degree,fe_degree_xwall,n_actual_q_points_vel_linear,1,value_type,is_xwall> FEEval_Vt_Velocity_nonlinear;
  typedef FEFaceEvaluationWrapperPressure<dim,fe_degree,fe_degree_xwall,n_actual_q_points_vel_linear,1,value_type,is_xwall> FEFaceEval_Vt_Velocity_nonlinear;

  // constructor
  DGSpalartAllmarasModel(FEParameters<dim> const &parameter)
    :
    viscosity(parameter.viscosity),
    fe_param(parameter),
    dof_index_velocity(0),
    dof_index_vt(3),
    inverse_mass_matrix_operator(),
    vel(nullptr),
    kappa(0.41),
    cb1(0.1355),
    cb2(0.622),
    cb3(2./3.),
    cv1(7.1),
    cw1(cb1/(kappa*kappa)+(1.+cb2)/cb3),
    cw2(0.3),
    cw3(2.),
    ct3(1.2),
    ct4(0.5)
  {}

  // destructor
  virtual ~DGSpalartAllmarasModel(){}

  virtual void setup (MatrixFree<dim,value_type> &data,
                      Mapping<dim> &mapping,
                      std::set<types::boundary_id>    &dirichlet_bc_indicator,
                      std::set<types::boundary_id>    &neumann_bc_indicator);

  double get_viscosity() const
  {
    return viscosity;
  }

  void evaluate (MatrixFree<dim,value_type> const &data,
                 parallel::distributed::Vector<value_type> *src_vel,
                 const parallel::distributed::Vector<value_type> &src_vt,
                 parallel::distributed::Vector<value_type>       &dst
                 );

protected:

  const double viscosity;

  FEParameters<dim> const &fe_param;

  unsigned int dof_index_velocity;
  unsigned int dof_index_vt;

  InverseMassMatrixOperator<dim,fe_degree,value_type,1> inverse_mass_matrix_operator;

  std::set<types::boundary_id> dirichlet_boundary;
  std::set<types::boundary_id> neumann_boundary;

  AlignedVector<VectorizedArray<value_type> > array_penalty_parameter;

  parallel::distributed::Vector<value_type>* vel;

  // Returns the current factor by which array_penalty_parameter() is
  // multiplied in the definition of the interior penalty parameter through
  // get_array_penalty_parameter()[cell] * get_penalty_factor().
  value_type get_penalty_factor() const
  {
    return (fe_degree + 1.0) * (fe_degree + 1.0);
  }

private:
  // Computes the array penalty parameter for later use of the symmetric
  // interior penalty method. Called in reinit().
  void compute_array_penalty_parameter(MatrixFree<dim,value_type> &data, const Mapping<dim> &mapping);

  // compute cell
  void local_evaluate_spalart_allmaras (const MatrixFree<dim,value_type>                 &data,
                                        parallel::distributed::Vector<value_type>        &dst,
                                        const parallel::distributed::Vector<value_type>  &src,
                                        const std::pair<unsigned int,unsigned int>       &cell_range) const;

  // compute face
  void local_evaluate_spalart_allmaras_face (const MatrixFree<dim,value_type>                 &data,
                                             parallel::distributed::Vector<value_type>        &dst,
                                             const parallel::distributed::Vector<value_type>  &src,
                                             const std::pair<unsigned int,unsigned int>       &face_range) const;

  // compute boundary face
  void local_evaluate_spalart_allmaras_boundary_face (const MatrixFree<dim,value_type>                 &data,
                                                      parallel::distributed::Vector<value_type>        &dst,
                                                      const parallel::distributed::Vector<value_type>  &src,
                                                      const std::pair<unsigned int,unsigned int>       &cell_range) const;

  const value_type kappa;
  const value_type cb1;
  const value_type cb2;
  const value_type cb3;
  const value_type cv1;
  const value_type cw1;
  const value_type cw2;
  const value_type cw3;
  const value_type ct3;
  const value_type ct4;
};

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void DGSpalartAllmarasModel<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
setup (MatrixFree<dim,value_type> &data,
       Mapping<dim> &mapping,
       std::set<types::boundary_id>   &dirichlet_bc_indicator,
       std::set<types::boundary_id>   &neumann_bc_indicator)
{
  dirichlet_boundary = dirichlet_bc_indicator;
  neumann_boundary = neumann_bc_indicator;
  // inverse mass matrix operator
  inverse_mass_matrix_operator.initialize(data,dof_index_vt,0);

  compute_array_penalty_parameter(data,mapping);

}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void DGSpalartAllmarasModel<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
compute_array_penalty_parameter(MatrixFree<dim,value_type> &data, const Mapping<dim> &mapping)
{
  // Compute penalty parameter for each cell
  array_penalty_parameter.resize(data.n_macro_cells()+data.n_macro_ghost_cells());
  QGauss<dim> quadrature(fe_degree+1);
  FEValues<dim> fe_values(mapping,
                          data.get_dof_handler(dof_index_vt).get_fe(),
                          quadrature, update_JxW_values);
  QGauss<dim-1> face_quadrature(fe_degree+1);
  FEFaceValues<dim> fe_face_values(mapping, data.get_dof_handler(dof_index_vt).get_fe(), face_quadrature, update_JxW_values);

  for (unsigned int i=0; i<data.n_macro_cells()+data.n_macro_ghost_cells(); ++i)
    for (unsigned int v=0; v<data.n_components_filled(i); ++v)
      {
        typename DoFHandler<dim>::cell_iterator cell = data.get_cell_iterator(i,v,dof_index_vt);
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

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void DGSpalartAllmarasModel<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
evaluate (MatrixFree<dim,value_type> const &data,
          parallel::distributed::Vector<value_type> *src_vel,
          const parallel::distributed::Vector<value_type> &src_vt,
          parallel::distributed::Vector<value_type>       &dst)
{
  src_vel->update_ghost_values();
  vel = src_vel;

  dst = 0;

  data.loop (&DGSpalartAllmarasModel<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::local_evaluate_spalart_allmaras,
             &DGSpalartAllmarasModel<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::local_evaluate_spalart_allmaras_face,
             &DGSpalartAllmarasModel<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::local_evaluate_spalart_allmaras_boundary_face,
             this, dst, src_vt);

  inverse_mass_matrix_operator.apply_inverse_mass_matrix(dst,dst);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void DGSpalartAllmarasModel<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
local_evaluate_spalart_allmaras(const MatrixFree<dim,value_type>                 &data,
                                parallel::distributed::Vector<value_type>        &dst,
                                const parallel::distributed::Vector<value_type>  &src,
                                const std::pair<unsigned int,unsigned int>       &cell_range) const
{
  FEEval_Velocity_Velocity_nonlinear fe_eval_velocity(data,&fe_param,dof_index_velocity);
  FEEval_Vt_Velocity_nonlinear fe_eval_vt(data,&fe_param,dof_index_vt);

  AlignedVector<VectorizedArray<value_type> > wdist;
  AlignedVector<VectorizedArray<value_type> > tauw;
  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
  {
    fe_eval_velocity.reinit(cell);
    fe_eval_velocity.read_dof_values(*vel);
    fe_eval_velocity.evaluate (true,true,false);
    fe_eval_vt.reinit(cell);
    fe_eval_vt.read_dof_values(src);
    fe_eval_vt.evaluate (true,true,false);

    fe_eval_vt.fill_wdist_and_tauw(cell,wdist,tauw);
    for (unsigned int q=0; q<fe_eval_velocity.n_q_points; ++q)
    {
      Tensor<1,dim,VectorizedArray<value_type> > submit_gradient = Tensor<1,dim,VectorizedArray<value_type> >();
      // nonlinear convective flux F(u) = vtu
      Tensor<1,dim,VectorizedArray<value_type> > u = fe_eval_velocity.get_value(q);
      VectorizedArray<value_type> vt               = fe_eval_vt.get_value(q);
      Tensor<1,dim,VectorizedArray<value_type> > F = u * vt;
      submit_gradient += -F;
//      fe_eval_vt.submit_gradient (-F, q); // minus sign due to integration by parts

      // viscous flux
      submit_gradient += (viscosity+std::max(vt,make_vectorized_array<value_type>(0.)))/cb3*fe_eval_vt.get_gradient(q);

      // source terms
      {
        VectorizedArray<value_type> submit_value = VectorizedArray<value_type>();

        VectorizedArray<value_type> vt = std::max(fe_eval_vt.get_value(q),make_vectorized_array(0.));
        Tensor<1,dim,VectorizedArray<value_type> > grad_vt = fe_eval_vt.get_gradient(q);

        Tensor<2,dim,VectorizedArray<value_type> > grad_vel = fe_eval_velocity.get_gradient(q);
        Tensor<2,dim,VectorizedArray<value_type> > Om;
        VectorizedArray<value_type> OmddOm = VectorizedArray<value_type>();
        for (unsigned int i = 0; i<dim; i++)
          for (unsigned int j = 0; j<dim; j++)
            Om[i][j] = 0.5 * (grad_vel[i][j] - grad_vel[j][i]);
        for (unsigned int i = 0; i<dim; i++)
          for (unsigned int j = 0; j<dim; j++)
            OmddOm += Om[i][j] * Om[i][j];
        VectorizedArray<value_type> S = std::sqrt(2.*OmddOm);
        VectorizedArray<value_type> chi = vt/viscosity;
        VectorizedArray<value_type> fv1 = chi * chi * chi /(chi * chi * chi  + cv1 * cv1 * cv1);
        VectorizedArray<value_type> fv2 = 1. - chi / (1. + chi * fv1);
        VectorizedArray<value_type> Sbar = vt/(kappa * kappa * wdist[q]*wdist[q])*fv2;
        VectorizedArray<value_type> Stilde = S + Sbar;
//        for (unsigned int v = 0; v< VectorizedArray<value_type>::n_array_elements;v++)
//          if(Sbar[v] < -0.7 *S[v])
//            Stilde[v] = S[v] * (1. + (0.7 * 0.7 * S[v] + 0.9 * Sbar[v])/((0.9-2.*0.7)* S[v]-Sbar[v]));
        VectorizedArray<value_type> r   = vt / (Stilde * kappa * kappa * wdist[q] * wdist[q]);
        VectorizedArray<value_type> g   = r + cw2 * (r * r * r * r * r * r - r);
        VectorizedArray<value_type> fw  = g * std::pow(((1.+ cw3 * cw3 * cw3 * cw3 * cw3 * cw3)/(g * g * g * g * g * g + cw3 * cw3 * cw3 * cw3 * cw3 * cw3)),1./6.);

        VectorizedArray<value_type> ft2 = VectorizedArray<value_type>();//ct3 * std::exp(-ct4 * chi * chi);

//        Tensor<1,dim,VectorizedArray<value_type> > submit_gradient_add = vt * grad_vt *cb2/cb3;
        submit_value += -grad_vt * grad_vt *cb2/cb3;

        submit_value += -cb1 * (1. - ft2) * Stilde * vt;

        submit_value += -(cb1/(kappa * kappa)* ft2 - cw1 * fw)* (vt / wdist[q])*(vt / wdist[q]);

        vt = fe_eval_vt.get_value(q);
        for (unsigned int v = 0; v < VectorizedArray<value_type>::n_array_elements; v++)
          if(vt[v] < 0.)
          {
            submit_value[v] = 0.;
//            for(unsigned int i=0;i<dim;i++)
//              submit_gradient_add[i][v] = 0.;
          }
//        submit_gradient += submit_gradient_add;
        fe_eval_vt.submit_value(submit_value,q);
      }

      // submit gradient
      fe_eval_vt.submit_gradient(submit_gradient,q);
    }
    fe_eval_vt.integrate (true,true);
    fe_eval_vt.distribute_local_to_global(dst);
  }
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void DGSpalartAllmarasModel<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
local_evaluate_spalart_allmaras_face(const MatrixFree<dim,value_type>                 &data,
                                     parallel::distributed::Vector<value_type>        &dst,
                                     const parallel::distributed::Vector<value_type>  &src,
                                     const std::pair<unsigned int,unsigned int>       &face_range) const
{
  FEFaceEval_Velocity_Velocity_nonlinear fe_eval_velocity(data,&fe_param,true,dof_index_velocity);
  FEFaceEval_Velocity_Velocity_nonlinear fe_eval_velocity_neighbor(data,&fe_param,false,dof_index_velocity);
  FEFaceEval_Vt_Velocity_nonlinear fe_eval_vt(data,&fe_param,true,dof_index_vt);
  FEFaceEval_Vt_Velocity_nonlinear fe_eval_vt_neighbor(data,&fe_param,false,dof_index_vt);

  for(unsigned int face=face_range.first; face<face_range.second; face++)
  {
    fe_eval_velocity.reinit(face);
    fe_eval_velocity.read_dof_values(*vel);
    fe_eval_velocity.evaluate(true, false);

    fe_eval_velocity_neighbor.reinit (face);
    fe_eval_velocity_neighbor.read_dof_values(*vel);
    fe_eval_velocity_neighbor.evaluate(true,false);

    fe_eval_vt.reinit(face);
    fe_eval_vt.read_dof_values(src);
    fe_eval_vt.evaluate(true, true);

    fe_eval_vt_neighbor.reinit (face);
    fe_eval_vt_neighbor.read_dof_values(src);
    fe_eval_vt_neighbor.evaluate(true,true);

    VectorizedArray<value_type> sigmaF =
      std::max(fe_eval_vt.read_cell_data(array_penalty_parameter),
          fe_eval_vt_neighbor.read_cell_data(array_penalty_parameter)) *
      get_penalty_factor();

    for(unsigned int q=0;q<fe_eval_velocity.n_q_points;++q)
    {
      // values to submit
      VectorizedArray<value_type> submit_value = VectorizedArray<value_type>();
      Tensor<1,dim,VectorizedArray<value_type> > submit_gradient = Tensor<1,dim,VectorizedArray<value_type> >();

      //initialize some common variables
      Tensor<1,dim,VectorizedArray<value_type> > uM = fe_eval_velocity.get_value(q);
      Tensor<1,dim,VectorizedArray<value_type> > uP = fe_eval_velocity_neighbor.get_value(q);
      Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval_vt.get_normal_vector(q);

      const VectorizedArray<value_type> vtM = fe_eval_vt.get_value(q);
      const VectorizedArray<value_type> vtP = fe_eval_vt_neighbor.get_value(q);
      const VectorizedArray<value_type> jump_value = vtM - vtP;

      // convective flux
      {
        const VectorizedArray<value_type> uM_n = uM*normal;
        const VectorizedArray<value_type> uP_n = uP*normal;

        // calculation of lambda according to Shahbazi et al., i.e.
        // lambda = max ( max |lambda(flux_jacobian_M)| , max |lambda(flux_jacobian_P)| )
        // where the maximum eigenvalue of the flux Jacobian is the
        // maximum eigenvalue of (u^T * normal) * I + u * normal^T, which is
        // abs(2*u^T*normal) (this can be verified by rank-1 matrix algebra)
        const VectorizedArray<value_type> lambda = std::max(std::abs(uM_n), std::abs(uP_n));

        const VectorizedArray<value_type> average_normal_flux = ( vtM*uM_n + vtP*uP_n) * make_vectorized_array<value_type>(0.5);
        const VectorizedArray<value_type> lf_flux = average_normal_flux + 0.5 * lambda * jump_value;
        submit_value += lf_flux;
      }

      // viscous flux
      {
        VectorizedArray<value_type> facM = (viscosity + std::max(vtM,make_vectorized_array<value_type>(0.)))/cb3;
        VectorizedArray<value_type> facP = (viscosity + std::max(vtP,make_vectorized_array<value_type>(0.)))/cb3;

        //harmonic weighting of variable viscosity, see Schott and Rasthofer et al. (2015)
        VectorizedArray<value_type> fac_harmonic_weight_penalty = facM * facP / (facM + facP);
        VectorizedArray<value_type> average_gradient =
            (fe_eval_vt.get_gradient(q) + fe_eval_vt_neighbor.get_gradient(q))
            * fac_harmonic_weight_penalty * normal;

        average_gradient = average_gradient - jump_value * sigmaF * 2. * fac_harmonic_weight_penalty;

        submit_gradient += -jump_value * normal * fac_harmonic_weight_penalty;
//        fe_eval.submit_normal_gradient(-0.5*jump_value,q);
//        fe_eval_neighbor.submit_normal_gradient(-0.5*jump_value,q);
        submit_value += -average_gradient;
//        fe_eval.submit_value(-average_gradient,q);
//        fe_eval_neighbor.submit_value(average_gradient,q);
      }

//      submit_value += -(std::max(vtM,make_vectorized_array<value_type>(0.)) * fe_eval_vt.get_gradient(q)
//                      + std::max(vtP,make_vectorized_array<value_type>(0.)) * fe_eval_vt_neighbor.get_gradient(q))* 0.5 * normal *cb2/cb3;
      //submit values
      fe_eval_vt.submit_value(submit_value,q);
      fe_eval_vt_neighbor.submit_value(-submit_value,q);
      //submit gradients
      fe_eval_vt.submit_gradient(submit_gradient,q);
      fe_eval_vt_neighbor.submit_gradient(submit_gradient,q);
    }
    fe_eval_vt.integrate(true,true);
    fe_eval_vt.distribute_local_to_global(dst);
    fe_eval_vt_neighbor.integrate(true,true);
    fe_eval_vt_neighbor.distribute_local_to_global(dst);
  }
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void DGSpalartAllmarasModel<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
local_evaluate_spalart_allmaras_boundary_face(const MatrixFree<dim,value_type>                 &data,
                                              parallel::distributed::Vector<value_type>        &dst,
                                              const parallel::distributed::Vector<value_type>  &src,
                                              const std::pair<unsigned int,unsigned int>       &face_range) const
{
  FEFaceEval_Velocity_Velocity_nonlinear fe_eval_velocity(data,&fe_param,true,dof_index_velocity);
  FEFaceEval_Vt_Velocity_nonlinear fe_eval_vt(data,&fe_param,true,dof_index_vt);

  for(unsigned int face=face_range.first; face<face_range.second; face++)
  {
    fe_eval_velocity.reinit(face);
    fe_eval_velocity.read_dof_values(*vel);
    fe_eval_velocity.evaluate(true, false);

    fe_eval_vt.reinit(face);
    fe_eval_vt.read_dof_values(src);
    fe_eval_vt.evaluate(true, true);

    VectorizedArray<value_type> sigmaF =
      fe_eval_vt.read_cell_data(array_penalty_parameter) *
      get_penalty_factor();

    if (dirichlet_boundary.find(data.get_boundary_indicator(face))
        != dirichlet_boundary.end())
    {
      for(unsigned int q=0;q<fe_eval_velocity.n_q_points;++q)
      {
        // values to submit
        VectorizedArray<value_type> submit_value = VectorizedArray<value_type>();
        Tensor<1,dim,VectorizedArray<value_type> > submit_gradient = Tensor<1,dim,VectorizedArray<value_type> >();
        // some definitions
        Tensor<1,dim,VectorizedArray<value_type> > uM = fe_eval_velocity.get_value(q);

        Tensor<1,dim,VectorizedArray<value_type> > g_n = Tensor<1,dim,VectorizedArray<value_type> >();//walls only
        Tensor<1,dim,VectorizedArray<value_type> > uP = -uM + make_vectorized_array<value_type>(2.0)*g_n;
        Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval_vt.get_normal_vector(q);

        const VectorizedArray<value_type> g_vt_n(make_vectorized_array<value_type>(0.)); //only walls considered
        const VectorizedArray<value_type> vtM = fe_eval_vt.get_value(q);
        const VectorizedArray<value_type> vtP = -vtM + make_vectorized_array<value_type>(2.0)*g_vt_n;
        const VectorizedArray<value_type> jump_value = vtM - vtP;

        //convective flux
        {
          const VectorizedArray<value_type> uM_n = uM*normal;
          const VectorizedArray<value_type> uP_n = uP*normal;

          // calculation of lambda according to Shahbazi et al., i.e.
          // lambda = max ( max |lambda(flux_jacobian_M)| , max |lambda(flux_jacobian_P)| )
          // where the maximum eigenvalue of the flux Jacobian is the
          // maximum eigenvalue of (u^T * normal) * I + u * normal^T, which is
          // abs(2*u^T*normal) (this can be verified by rank-1 matrix algebra)
          const VectorizedArray<value_type> lambda = std::max(std::abs(uM_n), std::abs(uP_n));

          const VectorizedArray<value_type> average_normal_flux = ( vtM*uM_n + vtP*uP_n) * make_vectorized_array<value_type>(0.5);
          const VectorizedArray<value_type> lf_flux = average_normal_flux + 0.5 * lambda * jump_value;

//          fe_eval_vt.submit_value(lf_flux,q);
          submit_value += lf_flux;
        }

        // viscous flux
        {
          //use that the eddy viscosity is zero at the wall
          VectorizedArray<value_type> facM = (viscosity + std::max(vtM,make_vectorized_array<value_type>(0.)))/cb3;

          //harmonic weighting of variable viscosity, see Schott and Rasthofer et al. (2015)
          //this term is difficult to specify including all the measures for robustness and fulfilling the boundary conditions
          //so we apply a minor simplification here and just state the following
          VectorizedArray<value_type> average_gradient =
              fe_eval_vt.get_gradient(q)*facM * normal;

          average_gradient = average_gradient - jump_value * sigmaF * facM;

          submit_gradient += -jump_value * normal * facM * 0.5;
          submit_value += -average_gradient;
        }

        fe_eval_vt.submit_value(submit_value,q);
        fe_eval_vt.submit_gradient(submit_gradient,q);
      }
    }
    else if (neumann_boundary.find(data.get_boundary_indicator(face))
              != neumann_boundary.end())
    {
      for(unsigned int q=0;q<fe_eval_velocity.n_q_points;++q)
      {
        // values to submit
        VectorizedArray<value_type> submit_value = VectorizedArray<value_type>();
        Tensor<1,dim,VectorizedArray<value_type> > submit_gradient = Tensor<1,dim,VectorizedArray<value_type> >();
        Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval_velocity.get_normal_vector(q);
        // convective flux
        {
          // on GammaN: vt⁺ = vt⁻, u⁺ = u⁻
          Tensor<1,dim,VectorizedArray<value_type> > uM = fe_eval_velocity.get_value(q);
          Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval_velocity.get_normal_vector(q);
          const VectorizedArray<value_type> uM_n = uM*normal;

          const VectorizedArray<value_type> vtM = fe_eval_vt.get_value(q);
          const VectorizedArray<value_type> average_normal_flux = vtM*uM_n;
          const VectorizedArray<value_type> lf_flux = average_normal_flux;

          submit_value += lf_flux;
        }
//        fe_eval_vt.submit_value(lf_flux,q);

        // viscous flux
        {
          //harmonic weighting of variable viscosity, see Schott and Rasthofer et al. (2015)
          VectorizedArray<value_type> fac_harmonic_weight_penalty = make_vectorized_array<value_type>(0.);
          VectorizedArray<value_type> average_gradient =
              fe_eval_vt.get_normal_gradient(q,true)*0.;

          average_gradient = average_gradient - 0. * sigmaF * 2. * fac_harmonic_weight_penalty;

          submit_gradient += -0. * normal * fac_harmonic_weight_penalty;
  //        fe_eval.submit_normal_gradient(-0.5*jump_value,q);
  //        fe_eval_neighbor.submit_normal_gradient(-0.5*jump_value,q);
          submit_value += -average_gradient;
  //        fe_eval.submit_value(-average_gradient,q);
  //        fe_eval_neighbor.submit_value(average_gradient,q);
        }

        fe_eval_vt.submit_value(submit_value,q);
        fe_eval_vt.submit_gradient(submit_gradient,q);
      }
    }

    fe_eval_vt.integrate(true,true);
    fe_eval_vt.distribute_local_to_global(dst);
  }
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
class DGNavierStokesDualSplittingXWallSpalartAllmaras : public DGNavierStokesDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>
{
public:
  typedef typename DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::value_type value_type;
  typedef typename DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::FEFaceEval_Velocity_Velocity_linear FEFaceEval_Velocity_Velocity_linear;
  typedef typename DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::FEEval_Velocity_Velocity_linear FEEval_Velocity_Velocity_linear;

  enum class DofHandlerSelector{
    velocity = static_cast<int>(DGNavierStokesDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::DofHandlerSelector::velocity),
    pressure = static_cast<int>(DGNavierStokesDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::DofHandlerSelector::pressure),
    wdist_tauw = static_cast<int>(DGNavierStokesDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::DofHandlerSelector::wdist_tauw),
    vt = static_cast<int>(DGNavierStokesDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::DofHandlerSelector::n_variants),
    n_variants = static_cast<typename std::underlying_type<DofHandlerSelector>::type >(vt)+1
  };

  //same quadrature rules as in DGNavierStokesDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>
  enum class QuadratureSelector{
    velocity = static_cast<int>(DGNavierStokesDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::QuadratureSelector::velocity),
    pressure = static_cast<int>(DGNavierStokesDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::QuadratureSelector::pressure),
    velocity_nonlinear = static_cast<int>(DGNavierStokesDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::QuadratureSelector::velocity_nonlinear),
    enriched = static_cast<int>(DGNavierStokesDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::QuadratureSelector::enriched),
    n_variants = static_cast<typename std::underlying_type<QuadratureSelector>::type >(enriched)+1
  };

  DGNavierStokesDualSplittingXWallSpalartAllmaras(parallel::distributed::Triangulation<dim> const &triangulation,
                                                  InputParametersNavierStokes<dim> const          &parameter)
    :
      DGNavierStokesDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>(triangulation,parameter),
      fe_vt(QGaussLobatto<1>(fe_degree+1)),
      dof_handler_vt(triangulation),
      spalart_allmaras(this->fe_param)
  {
  }

  virtual ~DGNavierStokesDualSplittingXWallSpalartAllmaras(){}

  unsigned int get_dof_index_vt() const
  {
    return static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::vt);
  }

  FE_DGQArbitraryNodes<dim> const & get_fe_vt() const
  {
    return *fe_vt;
  }

  DoFHandler<dim> const & get_dof_handler_vt() const
  {
    return dof_handler_vt;
  }

  void setup (const std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> > periodic_face_pairs,
              std_cxx11::shared_ptr<BoundaryDescriptorNavierStokes<dim> > boundary_descriptor_velocity,
              std_cxx11::shared_ptr<BoundaryDescriptorNavierStokes<dim> > boundary_descriptor_pressure,
              std_cxx11::shared_ptr<FieldFunctionsNavierStokes<dim> > field_functions);

  void prescribe_initial_condition_vt(parallel::distributed::Vector<value_type> &vt) const
  {
    vt = vt_initial_value;
  };

  // initialization of vectors
  void initialize_vector_vt(parallel::distributed::Vector<value_type> &src) const
  {
    this->data.initialize_dof_vector(src,
        static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::vt));
  }

  void evaluate_spalart_allmaras(parallel::distributed::Vector<value_type>       *src_vel,
                                 parallel::distributed::Vector<value_type> const &src_vt,
                                 parallel::distributed::Vector<value_type> &dst)
  {
    spalart_allmaras.evaluate(this->data,src_vel,src_vt,dst);
  }

  void set_eddy_viscosity(const parallel::distributed::Vector<value_type>  &src);

  void local_set_eddyviscosity (const MatrixFree<dim,value_type>                 &data,
                                parallel::distributed::Vector<value_type>        &,
                                const parallel::distributed::Vector<value_type>  &src,
                                const std::pair<unsigned int,unsigned int>   &cell_range);

  void local_set_eddyviscosity_face (const MatrixFree<dim,value_type>                 &data,
                                     parallel::distributed::Vector<value_type>        &,
                                     const parallel::distributed::Vector<value_type>  &src,
                                     const std::pair<unsigned int,unsigned int>   &face_range);
  void local_set_eddyviscosity_boundary_face (const MatrixFree<dim,value_type>                 & data,
                                              parallel::distributed::Vector<value_type>        &,
                                              const parallel::distributed::Vector<value_type>  &src,
                                              const std::pair<unsigned int,unsigned int>   &face_range);

  void rhs_viscous (parallel::distributed::Vector<value_type>       &dst,
                              const parallel::distributed::Vector<value_type> &src,
                              const parallel::distributed::Vector<value_type> &vt);

private:
  virtual void create_dofs();

  virtual void data_reinit(typename MatrixFree<dim,value_type>::AdditionalData & additional_data);

  FE_DGQArbitraryNodes<dim> fe_vt;

  DoFHandler<dim>  dof_handler_vt;

  DGSpalartAllmarasModel<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule> spalart_allmaras;
};

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void DGNavierStokesDualSplittingXWallSpalartAllmaras<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
setup (const std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> > periodic_face_pairs,
        std_cxx11::shared_ptr<BoundaryDescriptorNavierStokes<dim> > boundary_descriptor_velocity,
        std_cxx11::shared_ptr<BoundaryDescriptorNavierStokes<dim> > boundary_descriptor_pressure,
        std_cxx11::shared_ptr<FieldFunctionsNavierStokes<dim> > field_functions)
{
  DGNavierStokesDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
  setup(periodic_face_pairs,
  boundary_descriptor_velocity,
  boundary_descriptor_pressure,
  field_functions);

  spalart_allmaras.setup(this->data,this->mapping,this->dirichlet_boundary,this->neumann_boundary);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void DGNavierStokesDualSplittingXWallSpalartAllmaras<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
create_dofs()
{

  // enumerate degrees of freedom
  // multigrid solvers for enrichment not supported
  this->dof_handler_u.distribute_dofs(*this->fe_u);
  this->dof_handler_p.distribute_dofs(this->fe_p);
  this->dof_handler_p.distribute_mg_dofs(this->fe_p);
  this->dof_handler_wdist.distribute_dofs(this->fe_wdist);
  this->dof_handler_vt.distribute_dofs(this->fe_vt);

  unsigned int ndofs_per_cell_velocity    = Utilities::fixed_int_power<fe_degree+1,dim>::value*dim;
  unsigned int ndofs_per_cell_xwall    = Utilities::fixed_int_power<fe_degree_xwall+1,dim>::value*dim;
  unsigned int ndofs_per_cell_pressure    = Utilities::fixed_int_power<fe_degree_p+1,dim>::value;
  unsigned int ndofs_per_cell_vt    = Utilities::fixed_int_power<fe_degree+1,dim>::value;

  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  pcout << std::endl << "Discontinuous finite element discretization:" << std::endl << std::endl
    << "Velocity:" << std::endl
    << "  degree of 1D polynomials:\t"  << std::fixed << std::setw(10) << std::right << fe_degree
    << " (polynomial) and " << std::setw(10) << std::right << fe_degree_xwall << " (enrichment) " << std::endl
    << "  number of dofs per cell:\t"   << std::fixed << std::setw(10) << std::right << ndofs_per_cell_velocity
    << " (polynomial) and " << std::setw(10) << std::right << ndofs_per_cell_xwall << " (enrichment) " << std::endl
    << "  number of dofs (velocity):\t" << std::fixed << std::setw(10) << std::right << this->dof_handler_u.n_dofs() << std::endl
    << "Pressure:" << std::endl
    << "  degree of 1D polynomials:\t"  << std::fixed << std::setw(10) << std::right << fe_degree_p << std::endl
    << "  number of dofs per cell:\t"   << std::fixed << std::setw(10) << std::right << ndofs_per_cell_pressure << std::endl
    << "  number of dofs (pressure):\t" << std::fixed << std::setw(10) << std::right << this->dof_handler_p.n_dofs() << std::endl
    << "Eddy viscosity:" << std::endl
    << "  degree of 1D polynomials:\t"  << std::fixed << std::setw(10) << std::right << fe_degree << std::endl
    << "  number of dofs per cell:\t"   << std::fixed << std::setw(10) << std::right << ndofs_per_cell_vt << std::endl
    << "  number of dofs (eddy visc):\t" << std::fixed << std::setw(10) << std::right << this->dof_handler_vt.n_dofs() << std::endl;
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void DGNavierStokesDualSplittingXWallSpalartAllmaras<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
data_reinit(typename MatrixFree<dim,value_type>::AdditionalData & additional_data)
{
  std::vector<const DoFHandler<dim> * >  dof_handler_vec;

  dof_handler_vec.resize(static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::n_variants));
  dof_handler_vec[static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::velocity)] = &this->dof_handler_u;
  dof_handler_vec[static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::pressure)] = &this->dof_handler_p;
  dof_handler_vec[static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::wdist_tauw)] = &this->dof_handler_wdist;
  dof_handler_vec[static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::vt)] = &dof_handler_vt;

  std::vector<const ConstraintMatrix *> constraint_matrix_vec;
  constraint_matrix_vec.resize(static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::n_variants));
  ConstraintMatrix constraint_u, constraint_p;
  constraint_u.close();
  constraint_p.close();
  this->initialize_constraints(additional_data.periodic_face_pairs_level_0);
  constraint_matrix_vec[static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::velocity)] = &constraint_u;
  constraint_matrix_vec[static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::pressure)] = &constraint_p;
  constraint_matrix_vec[static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::wdist_tauw)] = &this->constraint_periodic;
  constraint_matrix_vec[static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::vt)] = &constraint_p;

  std::vector<Quadrature<1> > quadratures;

  // resize quadratures
  quadratures.resize(static_cast<typename std::underlying_type<QuadratureSelector>::type >(QuadratureSelector::n_variants));
  // velocity
  quadratures[static_cast<typename std::underlying_type<QuadratureSelector>::type >(QuadratureSelector::velocity)]
              = QGauss<1>(fe_degree+1);
  // pressure
  quadratures[static_cast<typename std::underlying_type<QuadratureSelector>::type >(QuadratureSelector::pressure)]
              = QGauss<1>(fe_degree_p+1);
  // exact integration of nonlinear convective term
  quadratures[static_cast<typename std::underlying_type<QuadratureSelector>::type >(QuadratureSelector::velocity_nonlinear)]
              = QGauss<1>(fe_degree + (fe_degree+2)/2);
  // high-order integration of enrichment
  quadratures[static_cast<typename std::underlying_type<QuadratureSelector>::type >(QuadratureSelector::enriched)]
              = QGauss<1>(xwall_quad_rule);

  this->data.reinit (this->mapping, dof_handler_vec, constraint_matrix_vec, quadratures, additional_data);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void DGNavierStokesDualSplittingXWallSpalartAllmaras<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
set_eddy_viscosity(const parallel::distributed::Vector<value_type>  &src)
{
  parallel::distributed::Vector<value_type> dummy;
  this->data.loop(&DGNavierStokesDualSplittingXWallSpalartAllmaras<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::local_set_eddyviscosity,
                   &DGNavierStokesDualSplittingXWallSpalartAllmaras<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::local_set_eddyviscosity_face,
                   &DGNavierStokesDualSplittingXWallSpalartAllmaras<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::local_set_eddyviscosity_boundary_face,
                   this, dummy, src);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void DGNavierStokesDualSplittingXWallSpalartAllmaras<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
local_set_eddyviscosity (const MatrixFree<dim,value_type>                 &data,
                     parallel::distributed::Vector<value_type>        &,
                     const parallel::distributed::Vector<value_type>  &src,
                     const std::pair<unsigned int,unsigned int>   &cell_range)
{
  FEEvaluationWrapperPressure<dim,fe_degree,fe_degree_xwall,xwall_quad_rule,1,value_type,true> fe_eval_vt(data,&this->fe_param,3);

  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
  {
    fe_eval_vt.reinit(cell);
    fe_eval_vt.read_dof_values(src);
    fe_eval_vt.evaluate(true,false);
//      fe_eval_vt.fill_wdist_and_tauw(cell,wdist,tauw);
    for(unsigned int q=0; q< fe_eval_vt.n_q_points; q++)
    {
      VectorizedArray<value_type>  vt = std::max(fe_eval_vt.get_value(q),make_vectorized_array<value_type>(0.));
      VectorizedArray<value_type>  chi = vt/this->fe_param.viscosity;
      VectorizedArray<value_type>  fv1 = chi * chi * chi / (chi * chi * chi + 7.1 * 7.1 * 7.1);
      this->viscous_operator.set_viscous_coefficient_cell(cell,q,vt * fv1 + this->fe_param.viscosity);
    }
  }
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void DGNavierStokesDualSplittingXWallSpalartAllmaras<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
local_set_eddyviscosity_face (const MatrixFree<dim,value_type>                 &data,
                          parallel::distributed::Vector<value_type>        &,
                          const parallel::distributed::Vector<value_type>  &src,
                          const std::pair<unsigned int,unsigned int>   &face_range)
{
  FEFaceEvaluationWrapperPressure<dim,fe_degree,fe_degree_xwall,xwall_quad_rule,1,value_type,true> fe_eval_vt(data,&this->fe_param,true,3);
  FEFaceEvaluationWrapperPressure<dim,fe_degree,fe_degree_xwall,xwall_quad_rule,1,value_type,true> fe_eval_vt_neighbor(data,&this->fe_param,false,3);

//    AlignedVector<VectorizedArray<Number> > wdist;
//    AlignedVector<VectorizedArray<Number> > tauw;
  for(unsigned int face=face_range.first; face<face_range.second; face++)
  {
    fe_eval_vt.reinit(face);
    fe_eval_vt_neighbor.reinit(face);
    fe_eval_vt.read_dof_values(src);
    fe_eval_vt_neighbor.read_dof_values(src);
    fe_eval_vt.evaluate(true,false);
    fe_eval_vt_neighbor.evaluate(true,false);
//      fe_eval_vt.fill_wdist_and_tauw(face,wdist,tauw);
    for(unsigned int q=0; q< fe_eval_vt.n_q_points; q++)
    {
      VectorizedArray<value_type>  vt = std::max(fe_eval_vt.get_value(q),make_vectorized_array<value_type>(0.));
      VectorizedArray<value_type>  chi = vt/this->fe_param.viscosity;
      VectorizedArray<value_type>  fv1 = chi * chi * chi / (chi * chi * chi + 7.1 * 7.1 * 7.1);
      this->viscous_operator.set_viscous_coefficient_face(face,q,vt * fv1 + this->fe_param.viscosity);
    }
//      fe_eval_vt_neighbor.fill_wdist_and_tauw(face,wdist,tauw);
    for(unsigned int q=0; q< fe_eval_vt_neighbor.n_q_points; q++)
    {
      VectorizedArray<value_type>  vt = std::max(fe_eval_vt_neighbor.get_value(q),make_vectorized_array<value_type>(0.));
      VectorizedArray<value_type>  chi = vt/this->fe_param.viscosity;
      VectorizedArray<value_type>  fv1 = chi * chi * chi / (chi * chi * chi + 7.1 * 7.1 * 7.1);
      this->viscous_operator.set_viscous_coefficient_face_neighbor(face,q,vt * fv1 + this->fe_param.viscosity);
    }
  }
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void DGNavierStokesDualSplittingXWallSpalartAllmaras<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
local_set_eddyviscosity_boundary_face (const MatrixFree<dim,value_type>                 & data,
                                   parallel::distributed::Vector<value_type>        &,
                                   const parallel::distributed::Vector<value_type>  &src,
                                   const std::pair<unsigned int,unsigned int>   &face_range)
{
  FEFaceEvaluationWrapperPressure<dim,fe_degree,fe_degree_xwall,xwall_quad_rule,1,value_type,true> fe_eval_vt(data,&this->fe_param,true,3);

  for(unsigned int face=face_range.first; face<face_range.second; face++)
  {
    fe_eval_vt.reinit(face);
    fe_eval_vt.read_dof_values(src);
    fe_eval_vt.evaluate(true,false);
    for(unsigned int q=0; q< fe_eval_vt.n_q_points; q++)
    {
      VectorizedArray<value_type>  vt = std::max(fe_eval_vt.get_value(q),make_vectorized_array<value_type>(0.));
      VectorizedArray<value_type>  chi = vt/this->fe_param.viscosity;
      VectorizedArray<value_type>  fv1 = chi * chi * chi / (chi * chi * chi + 7.1 * 7.1 * 7.1);
      this->viscous_operator.set_viscous_coefficient_face(face,q,vt * fv1 + this->fe_param.viscosity);
    }
  }
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
void DGNavierStokesDualSplittingXWallSpalartAllmaras<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::
rhs_viscous (parallel::distributed::Vector<value_type>       &dst,
               const parallel::distributed::Vector<value_type> &src,
               const parallel::distributed::Vector<value_type> &vt)
{
  set_eddy_viscosity(vt);
  DGNavierStokesDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>::rhs_viscous(dst,src);
}

#endif /* INCLUDE_DGSPALARTALLMARASMODEL_H_ */
