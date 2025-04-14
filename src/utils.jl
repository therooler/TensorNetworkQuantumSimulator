getnqubits(g::NamedGraph) = length(g.vertices)
getnqubits(tninds::IndsNetwork) = length(tninds.data_graph.vertex_data)

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


function get_global_cache_update_kwargs(alg::Algorithm)
    alg == Algorithm("bp") && return get_global_bp_update_kwargs()
    alg == Algorithm("boundarymps") && return get_global_boundarymps_update_kwargs()
    error("No update parameters known for that algorithm")
end

## Truncate a tensor network down to a maximum bond dimension
"""
    truncate(ψ::ITensorNetwork, maxdim; cutoff=nothing, bp_update_kwargs=get_global_bp_update_kwargs())

Truncate the ITensorNetwork `ψ` to a maximum bond dimension `maxdim` using the specified singular value cutoff.
"""
function ITensorNetworks.truncate(
    ψ::ITensorNetwork;
    cache_update_kwargs = get_global_bp_update_kwargs(),
    kwargs...,
)
    ψ_vidal = VidalITensorNetwork(ψ; cache_update_kwargs, kwargs...)
    return ITensorNetwork(ψ_vidal)
end
# 
