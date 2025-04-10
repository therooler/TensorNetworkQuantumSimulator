using TensorNetworkQuantumSimulator
const TN = TensorNetworkQuantumSimulator

using ITensorNetworks

using NamedGraphs.NamedGraphGenerators: named_grid

function main()
    nx = 5
    ny = 5
    nq = nx * ny

    # the graph is your main friend in working with the TNs
    g = named_grid((nx, ny))

    dt = 0.05

    hx = 1.0
    hz = 0.8
    J = 0.5

    # pauli rotations are tuples like `(pauli_string, [site_labels], parameter)`
    layer = []
    append!(layer, ("X", [v], 2*hx*dt) for v in vertices(g))
    append!(layer, ("Z", [v], 2*hz*dt) for v in vertices(g))
    append!(layer, ("ZZ", pair, 2*J*dt) for pair in edges(g));

    # observables are tuples like `(pauli_string, [site_labels], optional:coefficient)`
    # it's important that the `site_labels` match the names of the vertices of the graph `g`
    obs = ("Z", [(3, 3)])  # right in the middle

    # the number of circuit layers
    nl = 50

    # the initial state
    ψ = zerostate(g)

    # an array to keep track of expectations
    expectations = Float64[real(expect(ψ, obs))]

    # evolve! The first evaluation will take significantly longer because of compulation.
    for l in 1:nl
        #printing
        println("Layer $l")

        # apply layer
        t = @timed ψ = apply(layer, ψ);

        # push expectation to list
        push!(expectations, real(expect(ψ, obs)))

        # printing
        println("    Took time: $(t.time) [s]. Max bond dimension: $(maxlinkdim(ψ))")
    end
end

main()