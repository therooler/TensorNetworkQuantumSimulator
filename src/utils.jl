getnqubits(g::NamedGraph) = length(g.vertices)
getnqubits(tninds::IndsNetwork) = length(tninds.data_graph.vertex_data)


## Truncate a tensor network down to a maximum bond dimension
"""
    truncate(ψ::ITensorNetwork, maxdim; cutoff=nothing, bp_update_kwargs=get_global_bp_update_kwargs())

Truncate the ITensorNetwork `ψ` to a maximum bond dimension `maxdim` using the specified singular value cutoff.
"""
function truncate(ψ::ITensorNetwork, maxdim; cutoff=nothing, bp_update_kwargs=get_global_bp_update_kwargs())
    ψψ = build_bp_cache(ψ; bp_update_kwargs...)
    apply_kwargs = (; maxdim, cutoff, normalize=false)
    for e in edges(ψ)
        s1, s2 = only(siteinds(ψ, src(e))), only(siteinds(ψ, dst(e)))
        id = ITensors.op("I", s1) * ITensors.op("I", s2)
        ψ, ψψ = apply(id, ψ, ψψ; reset_all_messages=false, apply_kwargs)
    end

    ψψ = updatecache(ψψ; bp_update_kwargs...)

    return ψ, ψψ
end



function trace(Q::ITensorNetwork)
    d = getphysicaldim(siteinds(Q))
    if d == 2
        vec = [1.0, 1.0]
    elseif d == 4
        vec = [1.0, 0.0, 0.0, 0.0]
    else
        throwdimensionerror()
    end

    val = ITensorNetworks.inner(ITensorNetwork(v -> vec, siteinds(Q)), Q; alg="bp")
    return val
end


function get_global_cache_update_kwargs(alg::Algorithm)
    alg == Algorithm("bp") && return get_global_bp_update_kwargs()
    alg == Algorithm("boundarymps") && return get_global_boundarymps_update_kwargs()
    error("No update parameters known for that algorithm")
end


stringtosymbols(str) = [Symbol(s) for s in str]

## 
getphysicaldim(indices::IndsNetwork) = dim(first(indices.data_graph.vertex_data.values))
getphysicaldim(indices::AbstractVector{<:Index}) = first(indices)
getphysicaldim(index::Index) = dim(index)


function getmode(indices::IndsNetwork)
    d = getphysicaldim(indices)
    if d == 2
        return "Schrödinger"
    elseif d == 4
        return "Heisenberg"
    else
        throwdimensionerror()
    end
end


function throwdimensionerror()
    throw(ArgumentError("Only physical dimensions 2 and 4 are supported."))
end



# 