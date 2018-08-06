#ifndef SOLVER_CG_WRAPPER
#define SOLVER_CG_WRAPPER

template<int dim, typename VectorType = Vector<double>>
class SolverCGWrapper : public SolverCG<VectorType>
{
public:
  SolverCGWrapper(SolverControl & cn) : SolverCG<VectorType>(cn)
  {
  }

  void
  print_vectors(const unsigned int,
                const VectorType & /*solution*/,
                const VectorType &,
                const VectorType &) const
  {
    double error = 0.0;

    //        auto & triangulation = dof_handler.get_triangulation();
    //        auto & fe = dof_handler.get_fe();
    //
    //        Vector<double> norm_per_cell(
    //                triangulation.n_active_cells());
    //
    //        VectorTools::integrate_difference(
    //                dof_handler,
    //                solution,
    //                exact_solution,
    //                norm_per_cell,
    //                QGauss<dim>(fe.degree + 1),
    //                VectorTools::L2_norm);
    //
    //        error = VectorTools::compute_global_error(
    //                triangulation,
    //                norm_per_cell,
    //                VectorTools::L2_norm);
    //        {
    //            std::cout << error << std::endl;
    ////            DataOut<dim> data_out;
    ////
    ////            data_out.attach_dof_handler(dof_handler);
    ////            data_out.add_data_vector(solution, "solution");
    ////            data_out.build_patches(30);
    ////
    ////            const std::string filename = "solution";
    ////            std::ofstream output_pressure("output/" + filename +
    ///".vtu");
    ////            data_out.write_vtu(output_pressure);
    ////            exit(0);
    //        }

    this->history.push_back(error);
  }

  std::vector<double> &
  get_history_data() const
  {
    return history;
  }

private:
  mutable std::vector<double> history;
};
#endif