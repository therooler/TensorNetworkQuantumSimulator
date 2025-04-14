getnqubits(g::NamedGraph) = length(g.vertices)
getnqubits(tninds::IndsNetwork) = length(tninds.data_graph.vertex_data)


getphysicaldim(indices::IndsNetwork) = dim(first(indices.data_graph.vertex_data.values))
getphysicaldim(indices::AbstractVector{<:Index}) = first(indices)
getphysicaldim(index::Index) = dim(index)


## 
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

function trace(Q::ITensorNetwork)
    d = getphysicaldim(siteinds(Q))
    if d == 2
        vec = [1.0, 1.0]
    elseif d == 4
        vec = [1.0, 0.0, 0.0, 0.0]
    else
        throwdimensionerror()
    end

    val = ITensorNetworks.inner(ITensorNetwork(v -> vec, siteinds(Q)), Q; alg = "bp")
    return val
end



stringtosymbols(str) = [Symbol(s) for s in str]

function throwdimensionerror()
    throw(ArgumentError("Only physical dimensions 2 and 4 are supported."))
end

function get_global_cache_update_kwargs(alg::Algorithm)
    alg == Algorithm("bp") && return get_global_bp_update_kwargs()
    alg == Algorithm("boundarymps") && return get_global_boundarymps_update_kwargs()
    error("No update parameters known for that algorithm")
end

function ITensorNetworks.truncate(
    ψ::ITensorNetwork;
    cache_update_kwargs = get_global_bp_update_kwargs(),
    kwargs...,
)
    ψ_vidal = VidalITensorNetwork(ψ; cache_update_kwargs, kwargs...)
    return ITensorNetwork(ψ_vidal)
end
# 
