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

  void setup(std_cxx11::shared_ptr<BoundaryDescriptor<dim> > boundary_descriptor_in,
             std_cxx11::shared_ptr<FieldFunctions<dim> >     field_functions_in)
  {
    boundary_descriptor = boundary_descriptor_in;
    field_functions = field_functions_in;

    create_dofs();

    initialize_matrix_free();

    setup_operators();
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

private:
  void create_dofs()
  {
    // enumerate degrees of freedom
    dof_handler.distribute_dofs(fe);

    unsigned int ndofs_per_cell = Utilities::fixed_int_power<fe_degree+1,dim>::value;

    ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
    pcout << std::endl << "Discontinuous finite element discretization:" << std::endl << std::endl
      << "  degree of 1D polynomials:\t"  << std::fixed << std::setw(10) << std::right << fe_degree << std::endl
      << "  number of dofs per cell:\t"   << std::fixed << std::setw(10) << std::right << ndofs_per_cell << std::endl
      << "  number of dofs (total):\t" << std::fixed << std::setw(10) << std::right << dof_handler.n_dofs() << std::endl;
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
    // inverse mass matrix operator
    // dof_index = 0, quad_index = 0
    inverse_mass_matrix_operator.initialize(data,0,0);

    // convective operator
    ScalarConvDiffOperators::ConvectiveOperatorData<dim> convective_operator_data;
    convective_operator_data.dof_index = 0;
    convective_operator_data.quad_index = 0;
    convective_operator_data.bc = boundary_descriptor;
    convective_operator_data.velocity = field_functions->velocity;
    convective_operator.initialize(data,convective_operator_data);

    // diffusive operator
    ScalarConvDiffOperators::DiffusiveOperatorData<dim> diffusive_operator_data;
    diffusive_operator_data.dof_index = 0;
    diffusive_operator_data.quad_index = 0;
    diffusive_operator_data.IP_formulation = InteriorPenaltyFormulation::SIPG;
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

  FE_DGQArbitraryNodes<dim> fe;
  MappingQ<dim> mapping;
  DoFHandler<dim> dof_handler;

  MatrixFree<dim,value_type> data;

  InputParametersConvDiff const &param;

  std_cxx11::shared_ptr<BoundaryDescriptor<dim> > boundary_descriptor;
  std_cxx11::shared_ptr<FieldFunctions<dim> > field_functions;

  InverseMassMatrixOperator<dim,fe_degree,value_type> inverse_mass_matrix_operator;
//  ScalarConvDiffOperators::MassMatrixOperator<dim, fe_degree, value_type> mass_matrix_operator;
  ScalarConvDiffOperators::ConvectiveOperator<dim, fe_degree, value_type> convective_operator;
  ScalarConvDiffOperators::DiffusiveOperator<dim, fe_degree, value_type> diffusive_operator;
  ScalarConvDiffOperators::RHSOperator<dim, fe_degree, value_type> rhs_operator;

  // convection-diffusion operator (also includes rhs operator) for runtime optimization
  ScalarConvDiffOperators::ConvectionDiffusionOperator<dim, fe_degree, value_type> convection_diffusion_operator;
};


#endif /* INCLUDE_DGCONVDIFFOPERATION_H_ */
