function expect(tn1::ITensorNetwork, tn2::ITensorNetwork; imag_tol=1e-14)
    # this is pure state fidelity

    val = inner(tn1, tn2; alg="bp")

    return val
end


function expect(tn::ITensorNetwork, obs::Tuple; max_loop_size=0, imag_tol=1e-14)
    # unpack
    op = obs[1]
    qinds = obs[2]
    if length(obs) == 2
        coeff = 1.0
    else
        coeff = obs[3]
    end

    # TODO: can we pass BP caches somewhere?
    # this extra exception may not be needed, but it should be a bit faster
    if max_loop_size == 0
        val = (ITensorNetworks.expect(tn, op, qinds)*coeff)[] # [] to get the scalar value
    else
        val = expect_loopcorrect(tn, [string(o) for o in op], qinds, max_loop_size) * coeff
    end


    return val
end

# boundary MPS
function expect(ψIψ::BoundaryMPSBeliefPropagationCache, obs::Tuple; kwargs...)

    op_string = obs[1]
    qinds = obs[2]
    if length(obs) == 2
        coeff = 1.0
    else
        coeff = obs[3]
    end

    # length of qinds should be the same as the number of indices in op_string
    @assert length(qinds) == length(op_string) "Pauli string $(op_string) does not match the number of indices $(qinds)."

    if length(qinds) == 1
        v = qinds[]
        rdm = one_site_rdm(ψIψ, v; kwargs...)
        rdm /= tr(rdm)
        s = only(filter(i -> plev(i) == 0, inds(rdm)))
        val = (rdm*ITensors.op(op_string, s))[]
    elseif length(qinds) == 2
        v1, v2 = qinds
        if !(v1[1] == v2[1])
            # TODO:
            throw(ArgumentError("Operators needs to be in the same column, got columns $(v1[1]) and $(v2[1])."))
        end
        # TODO: extend this to arbitrary bodyness

        rdm = two_site_rdm(ψIψ, v1, v2; kwargs...)
        s1, s2 = first(filter(i -> plev(i) == 0, inds(rdm))),
        last(filter(i -> plev(i) == 0, inds(rdm)))
        val = ((rdm*ITensors.op(string(op_string[1]), s1))*ITensors.op(string(op_string[2]), s2))[] /
              ((rdm*ITensors.op("I", s1))*ITensors.op("I", s2))[]
    else
        throw(ArgumentError("Only 1 or 2 qubit operators are supported."))
    end

    return val * coeff
end


function expect_loopcorrect(
    ψIψ::BeliefPropagationCache, op_strings::Vector, verts::Vector, max_circuit_size::Int64; max_genus::Int64=2, bp_update_kwargs=default_cache_update_kwargs(ψIψ)
)

    ψIψ_tn = tensornetwork(ψIψ)
    ψIψ_vs = [ψIψ_tn[operator_vertex(ψIψ_tn, v)] for v in verts]
    sinds = [commonind(ψIψ_tn[ket_vertex(ψIψ_tn, v)], ψIψ_vs[i]) for (i, v) in enumerate(verts)]
    operators = [ITensors.op(op_strings[i], sinds[i]) for i in 1:length(op_strings)]

    ψOψ = update_factors(ψIψ, Dictionary([(v, "operator") for v in verts], operators))

    s = scalar(ψIψ)
    ψIψ = normalize(ψIψ)
    denom = s * corrected_free_energy(ψIψ, max_circuit_size; max_genus)

    ψOψ = update(ψOψ; bp_update_kwargs...)
    s = scalar(ψOψ)
    ψOψ = normalize(ψOψ)
    numer = s * corrected_free_energy(ψOψ, max_circuit_size; max_genus)

    return numer / denom
end

function expect_loopcorrect(ψIψ::BeliefPropagationCache, op_string, vert, max_circuit_size::Int64; kwargs...)
    return expect_loopcorrect(ψIψ, [op_string], [vert], max_circuit_size; kwargs...)
end

function expect_loopcorrect(ψ::ITensorNetwork, args...; bp_update_kwargs=(; maxiter=25, tol=1e-8), kwargs...)
    ψIψ = BeliefPropagationCache(QuadraticFormNetwork(ψ))
    ψIψ = update(ψIψ; bp_update_kwargs...)
    return expect_loopcorrect(ψIψ, args...; bp_update_kwargs, kwargs...)
end


