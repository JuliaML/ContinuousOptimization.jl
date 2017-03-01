module ContinuousOptimization

using LearningStrategies
const LS = LearningStrategies

#=

First, we need LearningStrategy types to do individual components of Optim.optimize:

function optimize{T, M<:Optimizer}(d, initial_x::Array{T}, method::M, options::OptimizationOptions)

-- LS.TimeLimit
    t0 = time() # Initial time stamp used to control early stopping by options.time_limit

-- ignore?
    if length(initial_x) == 1 && typeof(method) <: NelderMead
        error("Use optimize(f, scalar, scalar) for 1D problems")
    end

-- TODO
    state = initial_state(method, options, d, initial_x)

-- LS.ShowStatus, LS.Tracer, LS.IterFunction
    tr = OptimizationTrace{typeof(method)}()
    tracing = options.store_trace || options.show_trace || options.extended_trace || options.callback != nothing
    stopped, stopped_by_callback, stopped_by_time_limit = false, false, false

- LS.GradConverged
    x_converged, f_converged = false, false
    g_converged = if typeof(method) <: NelderMead
        nmobjective(state.f_simplex, state.m, state.n) < options.g_tol
    elseif  typeof(method) <: ParticleSwarm || typeof(method) <: SimulatedAnnealing
        g_converged = false
    else
        vecnorm(state.g, Inf) < options.g_tol
    end

    converged = g_converged
    iteration = 0

-- part of LS.ShowStatus's pre_hook
    options.show_trace && print_header(method)
    trace!(tr, state, iteration, method, options)

-- ignore... part of MetaLearner
    while !converged && !stopped && iteration < options.iterations

-- LS.MaxIter
        iteration += 1

-- LS.GradientLearner
        update_state!(d, state, method) && break # it returns true if it's forced by something in update! to stop (eg dx_dg == 0.0 in BFGS)
        update_g!(d, state, method)

-- LS.FunctionConverged, LS.GradConverged, LS.ParamsConverged
        x_converged, f_converged,
        g_converged, converged = assess_convergence(state, options)
        # We don't use the Hessian for anything if we have declared convergence,
        # so we might as well not make the (expensive) update if converged == true
        !converged && update_h!(d, state, method)

-- LS.ShowStatus, LS.Tracer
        # If tracing, update trace with trace!. If a callback is provided, it
        # should have boolean return value that controls the variable stopped_by_callback.
        # This allows for early stopping controlled by the callback.
        if tracing
            stopped_by_callback = trace!(tr, state, iteration, method, options)
        end

-- LS.TimeLimit
        # Check time_limit; if none is provided it is NaN and the comparison
        # will always return false.
        stopped_by_time_limit = time()-t0 > options.time_limit ? true : false

-- ignore... part of MetaLearner
        # Combine the two, so see if the stopped flag should be changed to true
        # and stop the while loop
        stopped = stopped_by_callback || stopped_by_time_limit ? true : false
    end # while

    after_while!(d, state, method, options)

-- TODO: how do we nicely summarize the sub-learners??
    return MultivariateOptimizationResults(state.method_string,
                                            initial_x,
                                            state.x,
                                            Float64(state.f_x),
                                            iteration,
                                            iteration == options.iterations,
                                            x_converged,
                                            options.x_tol,
                                            f_converged,
                                            options.f_tol,
                                            g_converged,
                                            options.g_tol,
                                            tr,
                                            state.f_calls,
                                            state.g_calls,
                                            state.h_calls)
end


Here's a rough outline of how we'd do this with MetaLearner:
-----------------------------------------------------------

# keep vector of: ‖θ_true - θ‖
tracer = Tracer(Float64, (model,i) -> norm(θ - params(model)))

# build the MetaLearner
learner = make_learner(
    GradientLearner(...),
    TimeLimit(60), # stop after 60 seconds
    MaxIter(1000), # stop after 1000 iterations
    ShowStatus(100), # show a status update before iterating and every 100 iterations
    Converged(params, tol=1e-6), # similar to x_converged for the function-case
    Converged(output, tol=1e-6), # similar to f_converged for the function-case
    Converged(grad, tol=1e-6, every=10), # similar to g_converged for the function-case
                                         # note: we can also only check every ith iteration
)

# learn is like optimize.
learn!(model, learner)

# note: for the function minimization case, it "iterates" over obs == nothing, since the
#   `x` in f(x) is treated as learnable parameters in Transformations.Differentiable, NOT input

------------------------------------------------------------
=#

end # module
