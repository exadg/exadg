/*
 * DGConvDiffOperation.h
 *
 *  Created on: Aug 2, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_DGCONVDIFFOPERATION_H_
#define INCLUDE_DGCONVDIFFOPERATION_H_

using namespace dealii;

#include "../include/InverseMassMatrix.h"
#include "../include/ScalarConvectionDiffusionOperators.h"

#include "../include/BoundaryDescriptorConvDiff.h"
#include "../include/FieldFunctionsConvDiff.h"

#include "InputParametersConvDiff.h"

template<int dim, int fe_degree, typename value_type>
class DGConvDiffOperation
{
public:

  DGConvDiffOperation(parallel::distributed::Triangulation<dim> const &triangulation,
                      InputParametersConvDiff const                   &param_in)
    :
    fe(QGaussLobatto<1>(fe_degree+1)),
    mapping(fe_degree),
    dof_handler(triangulation),
    param(param_in)
  {}

  void setup(std_cxx11::shared_ptr<BoundaryDescriptorConvDiff<dim> > boundary_descriptor_in,
             std_cxx11::shared_ptr<FieldFunctionsConvDiff<dim> >     field_functions_in)
  {
    ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
    pcout << std::endl << "Setup convection-diffusion operation ..." << std::endl;

    boundary_descriptor = boundary_descriptor_in;
    field_functions = field_functions_in;

    create_dofs();

    initialize_matrix_free();

    setup_operators();

    pcout << std::endl << "... done!" << std::endl;
  }

  void initialize_solution_vector(parallel::distributed::Vector<value_type> &src) const
  {
    data.initialize_dof_vector(src);
  }

  void prescribe_initial_conditions(parallel::distributed::Vector<value_type> &src,
                                    const double                              evaluation_time) const
  {
    field_functions->analytical_solution->set_time(evaluation_time);
    VectorTools::interpolate(dof_handler, *(field_functions->analytical_solution), src);
  }

  // getters
  MatrixFree<dim,value_type> const & get_data() const
  {
    return data;
  }

  void evaluate(parallel::distributed::Vector<value_type>       &dst,
                parallel::distributed::Vector<value_type> const &src,
                const value_type                                evaluation_time) const
  {
    if(param.runtime_optimization == false) //apply volume and surface integrals for each operator separately
    {
      if(param.equation_type == EquationTypeConvDiff::Diffusion)
      {
        diffusive_operator.evaluate(dst,src,evaluation_time);
      }
      else if(param.equation_type == EquationTypeConvDiff::Convection)
      {
        convective_operator.evaluate(dst,src,evaluation_time);
      }
      else if(param.equation_type == EquationTypeConvDiff::ConvectionDiffusion)
      {
        diffusive_operator.evaluate(dst,src,evaluation_time);
        convective_operator.evaluate_add(dst,src,evaluation_time);
      }
      else
      {
        AssertThrow(param.equation_type == EquationTypeConvDiff::Diffusion ||
                    param.equation_type == EquationTypeConvDiff::Convection ||
                    param.equation_type == EquationTypeConvDiff::ConvectionDiffusion,
                    ExcMessage("Specified equation type for convection-diffusion problem not implemented."));
      }

      // shift diffusive and convective term to the rhs of the equation
      dst *= -1.0;

      if(param.right_hand_side == true)
      {
        rhs_operator.evaluate_add(dst,evaluation_time);
      }
    }
    else // param.runtime_optimization == true
    {
      convection_diffusion_operator.evaluate(dst,src,evaluation_time);
    }

    // apply inverse mass matrix
    inverse_mass_matrix_operator.apply_inverse_mass_matrix(dst,dst);
  }

  void vmult(parallel::distributed::Vector<value_type>       &dst,
             parallel::distributed::Vector<value_type> const &src)
  {
    apply(dst,src);
  }

private:
  void create_dofs()
  {
    // enumerate degrees of freedom
    dof_handler.distribute_dofs(fe);

    unsigned int ndofs_per_cell = Utilities::fixed_int_power<fe_degree+1,dim>::value;

    ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

    pcout << std::endl
          << "Discontinuous Galerkin finite element discretization:" << std::endl << std::endl;

    print_parameter(pcout,"degree of 1D polynomials",fe_degree);
    print_parameter(pcout,"number of dofs per cell",ndofs_per_cell);
    print_parameter(pcout,"number of dofs (total)",dof_handler.n_dofs());
  }

  void initialize_matrix_free()
  {
    // quadrature formula used to perform integrals
    QGauss<1> quadrature (fe_degree+1);

    // initialize matrix_free_data
    typename MatrixFree<dim,value_type>::AdditionalData additional_data;
    additional_data.mpi_communicator = MPI_COMM_WORLD;
    additional_data.tasks_parallel_scheme =
      MatrixFree<dim,value_type>::AdditionalData::partition_partition;
    additional_data.build_face_info = true;
    additional_data.mapping_update_flags = (update_gradients | update_JxW_values |
                        update_quadrature_points | update_normal_vectors |
                        update_values);

    ConstraintMatrix dummy;
    dummy.close();
    data.reinit (mapping, dof_handler, dummy, quadrature, additional_data);
  }

  void setup_operators()
  {
    // mass matrix operator
    ScalarConvDiffOperators::MassMatrixOperatorData mass_matrix_operator_data;
    mass_matrix_operator_data.dof_index = 0;
    mass_matrix_operator_data.quad_index = 0;
    mass_matrix_operator.initialize(data,mass_matrix_operator_data);

    // inverse mass matrix operator
    // dof_index = 0, quad_index = 0
    inverse_mass_matrix_operator.initialize(data,0,0);

    // convective operator
    ScalarConvDiffOperators::ConvectiveOperatorData<dim> convective_operator_data;
    convective_operator_data.dof_index = 0;
    convective_operator_data.quad_index = 0;
    convective_operator_data.numerical_flux_formulation = param.numerical_flux_convective_operator;
    convective_operator_data.bc = boundary_descriptor;
    convective_operator_data.velocity = field_functions->velocity;
    convective_operator.initialize(data,convective_operator_data);

    // diffusive operator
    ScalarConvDiffOperators::DiffusiveOperatorData<dim> diffusive_operator_data;
    diffusive_operator_data.dof_index = 0;
    diffusive_operator_data.quad_index = 0;
    diffusive_operator_data.IP_factor = param.IP_factor;
    diffusive_operator_data.diffusivity = param.diffusivity;
    diffusive_operator_data.bc = boundary_descriptor;
    diffusive_operator.initialize(mapping,data,diffusive_operator_data);

    // rhs operator
    ScalarConvDiffOperators::RHSOperatorData<dim> rhs_operator_data;
    rhs_operator_data.dof_index = 0;
    rhs_operator_data.quad_index = 0;
    rhs_operator_data.rhs = field_functions->right_hand_side;
    rhs_operator.initialize(data,rhs_operator_data);

    // convection-diffusion operator (also includes rhs operator)
    ScalarConvDiffOperators::ConvectionDiffusionOperatorData<dim> conv_diff_operator_data;
    conv_diff_operator_data.conv_data = convective_operator_data;
    conv_diff_operator_data.diff_data = diffusive_operator_data;
    conv_diff_operator_data.rhs_data = rhs_operator_data;
    convection_diffusion_operator.initialize(mapping, data, conv_diff_operator_data);
  }

  void apply(parallel::distributed::Vector<value_type>       &dst,
             parallel::distributed::Vector<value_type> const &src) const
  {
//    // mass matrix operator
//    if(param.problem_type == ProblemType::Steady)
//    {
//      dst = 0;
//    }
//    else if(param.problem_type == ProblemType::Unsteady)
//    {
//      mass_matrix_operator.apply(dst,src);
//      dst *= scaling_factor_time_derivative_term;
//    }
//    else
//    {
//      AssertThrow(param.problem_type == ProblemType::Steady ||
//                  param.problem == ProblemType::Unsteady,
//                  ExcMessage("Specified problem type for convection-diffusion equation not implemented."));
//    }
//
//    // diffusive and convective operator
//    if(param.equation_type == EquationTypeConvDiff::Diffusion)
//    {
//      diffusive_operator.apply_add(dst,src);
//    }
//    else if(param.equation_type == EquationTypeConvDiff::Convection)
//    {
//      // TODO: ensure that evaluation_time is set correctly
//      convective_operator.apply_add(dst,src,evaluation_time);
//    }
//    else if(param.equation_type == EquationTypeConvDiff::ConvectionDiffusion)
//    {
//      diffusive_operator.apply_add(dst,src);
//      // TODO: ensure that evaluation_time is set correctly
//      convective_operator.apply_add(dst,src,evaluation_time);
//    }
//    else
//    {
//      AssertThrow(param.equation_type == EquationTypeConvDiff::Diffusion ||
//                  param.equation_type == EquationTypeConvDiff::Convection ||
//                  param.equation_type == EquationTypeConvDiff::ConvectionDiffusion,
//                  ExcMessage("Specified equation type for convection-diffusion problem not implemented."));
//    }
  }

  FE_DGQArbitraryNodes<dim> fe;
  MappingQGeneric<dim> mapping;
  DoFHandler<dim> dof_handler;

  MatrixFree<dim,value_type> data;

  InputParametersConvDiff const &param;

  std_cxx11::shared_ptr<BoundaryDescriptorConvDiff<dim> > boundary_descriptor;
  std_cxx11::shared_ptr<FieldFunctionsConvDiff<dim> > field_functions;

  ScalarConvDiffOperators::MassMatrixOperator<dim, fe_degree, value_type> mass_matrix_operator;
  InverseMassMatrixOperator<dim,fe_degree,value_type> inverse_mass_matrix_operator;
  ScalarConvDiffOperators::ConvectiveOperator<dim, fe_degree, value_type> convective_operator;
  ScalarConvDiffOperators::DiffusiveOperator<dim, fe_degree, value_type> diffusive_operator;
  ScalarConvDiffOperators::RHSOperator<dim, fe_degree, value_type> rhs_operator;

  // convection-diffusion operator for runtime optimization (also includes rhs operator)
  ScalarConvDiffOperators::ConvectionDiffusionOperator<dim, fe_degree, value_type> convection_diffusion_operator;
};


#endif /* INCLUDE_DGCONVDIFFOPERATION_H_ */
