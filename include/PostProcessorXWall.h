/*
 * PostProcessorXWall.h
 *
 *  Created on: Aug 8, 2016
 *      Author: krank
 */

#ifndef INCLUDE_POSTPROCESSORXWALL_H_
#define INCLUDE_POSTPROCESSORXWALL_H_

#include "SpaldingsLaw.h"
#include "PostProcessor.h"

  template <int dim>
  class Postprocessor : public DataPostprocessor<dim>
  {
    static const unsigned int number_vorticity_components = (dim==2) ? 1 : dim;
  public:
    Postprocessor (const unsigned int partition,const double viscosity)
      :
      partition (partition),
      viscosity(viscosity)
    {}
    virtual ~Postprocessor(){};

    virtual
    std::vector<std::string>
    get_names() const
    {
      // must be kept in sync with get_data_component_interpretation and
      // compute_derived_quantities_vector
      std::vector<std::string> solution_names (dim, "velocity");
      solution_names.push_back ("wdist");
      solution_names.push_back ("tauw");
      for (unsigned int d=0; d<dim; ++d)
        solution_names.push_back ("velocity_xwall");
      for (unsigned int d=0; d<number_vorticity_components; ++d)
        solution_names.push_back ("vorticity");
      solution_names.push_back ("owner");

      return solution_names;
    }

    virtual
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    get_data_component_interpretation() const
    {
      std::vector<DataComponentInterpretation::DataComponentInterpretation>
        interpretation(2*dim+number_vorticity_components+3, DataComponentInterpretation::component_is_part_of_vector);
      // wdist
      interpretation[dim] = DataComponentInterpretation::component_is_scalar;
      // tauw
      interpretation[dim+1] = DataComponentInterpretation::component_is_scalar;
      // owner
      interpretation.back() = DataComponentInterpretation::component_is_scalar;
      //vorticity
      if(dim == 2)
        interpretation[2*dim+number_vorticity_components+1] = DataComponentInterpretation::component_is_scalar;

      return interpretation;
    }

    virtual UpdateFlags get_needed_update_flags () const
    {
      return update_values | update_quadrature_points;
    }

    virtual void
    compute_derived_quantities_vector (const std::vector<Vector<double> >              &uh,
                                       const std::vector<std::vector<Tensor<1,dim> > > &/*duh*/,
                                       const std::vector<std::vector<Tensor<2,dim> > > &/*dduh*/,
                                       const std::vector<Point<dim> >                  &/*normals*/,
                                       const std::vector<Point<dim> >                  &,
                                       std::vector<Vector<double> >                    &computed_quantities) const
    {
      const unsigned int n_quadrature_points = uh.size();

      Assert (computed_quantities.size() == n_quadrature_points,            ExcInternalError());
      //the vorticity also comes in a length of dim in the 2D case
      Assert (uh[0].size() == 2*dim+2*dim+2,        ExcInternalError());

      AlignedVector<double> wdist;
      wdist.resize(n_quadrature_points,1.);
      AlignedVector<double> tauw;
      tauw.resize(n_quadrature_points,1.);
      for (unsigned int q=0; q<n_quadrature_points; ++q)
      {
        wdist[q] = uh[q](2*dim);
        tauw[q] = uh[q](2*dim+1);
      }
      SpaldingsLawEvaluation<dim, double, double > spalding(viscosity);
      spalding.reinit(wdist,tauw,n_quadrature_points);
      for (unsigned int q=0; q<n_quadrature_points; ++q)
        {
          const double enrichment_func = spalding.enrichment(q);
          for (unsigned int d=0; d<dim; ++d)
            computed_quantities[q](d)
              = (uh[q](d) + uh[q](dim+d) * enrichment_func);

          // wdist
          computed_quantities[q](dim) = wdist[q];

          // tauw
          computed_quantities[q](dim+1) = tauw[q];

          // velocity_xwall
          for (unsigned int d=0; d<dim; ++d)
            computed_quantities[q](dim+2+d) = uh[q](dim+d);

          // vorticity
          for (unsigned int d=0; d<number_vorticity_components; ++d)
            computed_quantities[q](2*dim+2+d) = uh[q](2*dim+2+d)+uh[q](2*dim+2+d + dim)*enrichment_func;

          // owner
          computed_quantities[q](2*dim+number_vorticity_components+2) = partition;
        }

    }

  private:
    const unsigned int partition;
    const double viscosity;
  };

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  class PostProcessorXWall: public PostProcessor<dim,fe_degree,fe_degree_p,fe_degree_xwall,n_q_points_1d_xwall>
  {
  public:
    PostProcessorXWall(std_cxx11::shared_ptr<DGNavierStokesBase<dim,fe_degree,fe_degree_p,fe_degree_xwall,n_q_points_1d_xwall> >  ns_operation,
                       InputParametersNavierStokes const &param_in):
      PostProcessor<dim,fe_degree,fe_degree_p,fe_degree_xwall,n_q_points_1d_xwall>(ns_operation,param_in),
      ns_operation_xw_(std::dynamic_pointer_cast<DGNavierStokesDualSplittingXWall<dim,fe_degree,fe_degree_p,fe_degree_xwall,n_q_points_1d_xwall> > (ns_operation))
    {
    }

    void do_postprocessing(parallel::distributed::Vector<double> const &velocity,
                           parallel::distributed::Vector<double> const &pressure,
                           parallel::distributed::Vector<double> const &vorticity,
                           parallel::distributed::Vector<double> const &divergence,
                           double const time,
                           unsigned int const time_step_number)
    {
      this->time_ = time;
      this->time_step_number_ = time_step_number;

      const double EPSILON = 1.0e-10; // small number which is much smaller than the time step size
      if( time > (this->param.output_start_time + this->output_counter_* this->param.output_interval_time - EPSILON))
      {
        write_output(velocity,pressure,vorticity,divergence);
        ++(this->output_counter_);
      }
    };
protected:
    std_cxx11::shared_ptr< const DGNavierStokesDualSplittingXWall<dim,fe_degree,fe_degree_p,fe_degree_xwall,n_q_points_1d_xwall> > ns_operation_xw_;
private:
    void write_output(parallel::distributed::Vector<double> const &velocity,
                      parallel::distributed::Vector<double> const &pressure,
                      parallel::distributed::Vector<double> const &vorticity,
                      parallel::distributed::Vector<double> const &divergence);

  };

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void PostProcessorXWall<dim,fe_degree,fe_degree_p,fe_degree_xwall,n_q_points_1d_xwall>::
  write_output(parallel::distributed::Vector<double> const &velocity,
               parallel::distributed::Vector<double> const &pressure,
               parallel::distributed::Vector<double> const &vorticity,
               parallel::distributed::Vector<double> const &divergence)
 {
    ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
    pcout << std::endl << "OUTPUT << Write data at time t = " << std::scientific << std::setprecision(4) << this->time_ << std::endl;

    // velocity + xwall dofs
    const FESystem<dim> joint_fe (this->ns_operation_->get_fe_u(), 1, //velocity
                                  ns_operation_xw_->get_fe_wdist(), 1, //wdist
                                  ns_operation_xw_->get_fe_wdist(), 1, //tauw
                                  this->ns_operation_->get_fe_u(), 1 //vorticity
                                  );
    DoFHandler<dim> joint_dof_handler (this->ns_operation_->get_dof_handler_u().get_triangulation());
    joint_dof_handler.distribute_dofs (joint_fe);
    IndexSet joint_relevant_set;
    DoFTools::extract_locally_relevant_dofs(joint_dof_handler, joint_relevant_set);
    parallel::distributed::Vector<double>
      joint_solution (joint_dof_handler.locally_owned_dofs(), joint_relevant_set, MPI_COMM_WORLD);
    std::vector<types::global_dof_index> loc_joint_dof_indices (joint_fe.dofs_per_cell),
      loc_vel_dof_indices (this->ns_operation_->get_fe_u().dofs_per_cell),
      loc_wdist_dof_indices(ns_operation_xw_->get_fe_wdist().dofs_per_cell);
    typename DoFHandler<dim>::active_cell_iterator
      joint_cell = joint_dof_handler.begin_active(),
      joint_endc = joint_dof_handler.end(),
      vel_cell = this->ns_operation_->get_dof_handler_u().begin_active(),
      wdist_cell = ns_operation_xw_->get_dof_handler_wdist().begin_active();

    for (; joint_cell != joint_endc; ++joint_cell, ++vel_cell
    , ++ wdist_cell
    )
      if (joint_cell->is_locally_owned())
      {
        joint_cell->get_dof_indices (loc_joint_dof_indices);
        vel_cell->get_dof_indices (loc_vel_dof_indices);
        wdist_cell->get_dof_indices (loc_wdist_dof_indices);
        for (unsigned int i=0; i<joint_fe.dofs_per_cell; ++i)
          switch (joint_fe.system_to_base_index(i).first.first)
            {
            case 0: //velocity
              Assert (joint_fe.system_to_base_index(i).first.second == 0,
                      ExcInternalError());
              joint_solution (loc_joint_dof_indices[i]) =
                velocity(loc_vel_dof_indices[ joint_fe.system_to_base_index(i).second ]);
              break;
            case 1: //wdist, necessary to reconstruct velocity
              Assert (joint_fe.system_to_base_index(i).first.second == 0,
                      ExcInternalError());
              joint_solution (loc_joint_dof_indices[i]) =
                  (*(this->ns_operation_->get_fe_parameters().wdist))
                (loc_wdist_dof_indices[ joint_fe.system_to_base_index(i).second ]);
              break;
            case 2: //tauw, necessary to reconstruct velocity
              Assert (joint_fe.system_to_base_index(i).first.second == 0,
                      ExcInternalError());
              joint_solution (loc_joint_dof_indices[i]) =
                  (*(this->ns_operation_->get_fe_parameters().tauw))
                (loc_wdist_dof_indices[ joint_fe.system_to_base_index(i).second ]);
              break;
            case 3: //vorticity
              Assert (joint_fe.system_to_base_index(i).first.second == 0,
                      ExcInternalError());
              joint_solution (loc_joint_dof_indices[i]) =
                  vorticity(loc_vel_dof_indices[ joint_fe.system_to_base_index(i).second ]);
              break;
            default:
              AssertThrow (false, ExcInternalError());
              break;
            }
      }

  joint_solution.update_ghost_values();

  Postprocessor<dim> postprocessor (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD),this->ns_operation_->get_viscosity());

  DataOut<dim> data_out;
  data_out.attach_dof_handler(joint_dof_handler);
  data_out.add_data_vector(joint_solution, postprocessor);

  pressure.update_ghost_values();
  data_out.add_data_vector (this->ns_operation_->get_dof_handler_p(),pressure, "p");
  {
    std_cxx11::shared_ptr< const DGNavierStokesDualSplittingXWallSpalartAllmaras<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> > nsa;
    nsa = std::dynamic_pointer_cast<const DGNavierStokesDualSplittingXWallSpalartAllmaras<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> > (ns_operation_xw_);
    if(nsa != nullptr)
    {
      data_out.add_data_vector (nsa->get_dof_handler_vt(),divergence, "vt");
    }
  }

  if(this->param.compute_divergence == true)
  {
    AssertThrow(false,ExcMessage("currently not supported"));
//    std::vector<std::string> divergence_names (dim, "divergence");
//    std::vector<DataComponentInterpretation::DataComponentInterpretation>
//      divergence_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
//    data_out.add_data_vector (this->ns_operation_->get_dof_handler_u(),divergence, divergence_names, divergence_component_interpretation);
  }

  std::ostringstream filename;
  filename << "output/"
           << this->param.output_prefix
           << "_Proc"
           << Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)
           << "_"
           << this->output_counter_
           << ".vtu";

  data_out.build_patches (this->ns_operation_->get_mapping(),5, DataOut<dim>::curved_inner_cells);

  std::ofstream output (filename.str().c_str());
  data_out.write_vtu (output);

  if ( Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    std::vector<std::string> filenames;
    for (unsigned int i=0;i<Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);++i)
    {
      std::ostringstream filename;
      filename << this->param.output_prefix
               << "_Proc"
               << i
               << "_"
               << this->output_counter_
               << ".vtu";

        filenames.push_back(filename.str().c_str());
    }
    std::string master_name = "output/" + this->param.output_prefix + "_" + Utilities::int_to_string(this->output_counter_) + ".pvtu";
    std::ofstream master_output (master_name.c_str());
    data_out.write_pvtu_record (master_output, filenames);
  }
}

#endif
