#pragma once

#if USE_PAGMO
#include <pagmo/types.hpp>
#include <pagmo/threading.hpp>
#include <pagmo/population.hpp>
#endif

#include "differentiation.hpp"
#include "parameter.hpp"

#if USE_OPTIM
#include "../../third_party/optim/optim.hpp"
#include "Eigen/Core"
#endif

// DEBUG
#include <iostream>

namespace tds 
{

class optim_gd
{

public:  // pagmo interface
  pagmo::population evolve(pagmo::population pop) const
  {
    auto& prob = pop.get_problem();
    auto dim = prob.get_nx();
    const auto bounds = prob.get_bounds();
    const auto& lb = bounds.first;
    const auto& ub = bounds.second;
    auto NP = pop.size();

    // Sanity Check
    if (!prob.has_gradient()) {
      pagmo_throw(std::invalid_argument, "Problem doesn't have a gradient provided: " + get_name() + " cannot apply gradient descent");
    }

    if (prob.is_stochastic()) {
      pagmo_throw(std::invalid_argument, "Problem is stochastic: " + get_name() + " cannot apply gradient descent");
    }

    if (prob.get_nf() != 1u) {
      pagmo_throw(std::invalid_argument, "Problem has multiple objectives: " + get_name() + " cannot apply gradient descent");
    }

    // auto fit = pop.get_f();
    auto result = pop.get_x(); 

    // Optim should be initialized to use EIGEN
    auto problem_func = [](const optim::Vec_t& vals_inp, optim::Vec_t* grad_out, void* opt_data) -> double{
      pagmo::problem* actual_prob = reinterpret_cast<pagmo::problem*>(opt_data);

      // Have to convert Vec_t back to vector for the problem
      pagmo::vector_double std_x(vals_inp.data(), vals_inp.data() + vals_inp.size());

      const double obj_val = actual_prob->fitness(std_x)[0];

      if (grad_out) {
        // *grad_out = optim::Vec_t(actual_prob->gradient(std_x).data());
        auto std_grad = actual_prob->gradient(std_x);
        *grad_out = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(std_grad.data(), std_grad.size()); 
      }

      return obj_val;
    };

    optim::algo_settings_t settings;

    settings.gd_settings.method = 6;
    settings.gd_settings.par_step_size = 0.1;
    settings.iter_max = 100; // maxium iteration bounds
    settings.vals_bound = true;

    auto std_lower_bounds = prob.get_bounds().first;
    auto std_upper_bounds = prob.get_bounds().second;
    settings.lower_bounds = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(std_lower_bounds.data(), std_lower_bounds.size());
    settings.upper_bounds = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(std_upper_bounds.data(), std_upper_bounds.size());

    for (size_t i = 0; i < result.size(); i++) {
      optim::Vec_t input_x = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(result[i].data(), result[i].size());
      auto input_x_copy = input_x; // DEBUG
      bool success = optim::gd(input_x, problem_func, reinterpret_cast<void*>(&prob), settings);
      // success will be true if the algorithm converges
      // if (!success) {
      //   pagmo_throw(std::runtime_error, "GD failed at individual " + std::to_string(i));
      // }
      pagmo::vector_double output_x(input_x.data(), input_x.data() + input_x.size());
      pop.set_x(i, output_x);
      
      // DEBUG
      std::cout << "Population " << i << " on the island finished with convergence: " << success << std::endl;
      std::cout << "Input: " << std::endl;
      std::cout << input_x_copy << std::endl;
      std::cout << "Output: " << std::endl;
      std::cout << input_x << std::endl;
      std::cout << std::endl;
    }

    return pop;

  }
  
  std::string get_name() const { return std::string("Optim Gradient Descent"); }
  
  pagmo::thread_safety get_thread_safety() const { return pagmo::thread_safety::basic; }


};

}







