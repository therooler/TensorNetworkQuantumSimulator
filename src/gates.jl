# conversion of a tuple circuit to an ITensor circuit
function toitensor(circuit, sinds::IndsNetwork)
    return [toitensor(gate, sinds) for gate in circuit]
end

# conversion of the tuple gate to an ITensor
function toitensor(gate::Tuple, sinds::IndsNetwork)
    gate_symbol = gate[1]
    gate_inds = gate[2]
    θ = gate[3]

    # if it is a NamedEdge, we need to convert it to a tuple
    gate_inds = _ensuretuple(gate_inds)

    return paulirotation(gate_symbol, gate_inds, θ, sinds)
end

# conversion retruns the gate itself if it is already
function toitensor(gate::ITensor, sinds::IndsNetwork)
    return gate
end


function paulirotation(generator, qinds, θ, tninds::IndsNetwork)
    # ("XX", 0.3, (1, 2))) could be the gate part

    # get the right indices from the IndsNetwork
    indices = [only(tninds[ind]) for ind in qinds]

    return paulirotation(generator, θ, indices)
end

"""
    paulirotation(generator::String, θ, indices::Vector{Index})

Returns an ITensor representing a Pauli rotation gate acting on the qubits specified by `indices`. 
The generator is a string of Pauli matrices, e.g. "X" or "Y", "Z", or "XY". 
The angle of rotation is specified by `θ`.
"""
function paulirotation(generator, θ, indices)
    d = getphysicaldim(first(indices))
    heisenberg = d == 4

    nqubits = length(generator)
    @assert length(indices) == nqubits "The number of indices must match the length of the gate generator."
    U = paulirotationmatrix(generator, θ)

    if heisenberg
        # transform into PTMs
        U = PP.calculateptm(U, heisenberg=true)
    end

    # check for physical dimension matching
    # TODO


    # define legs of the tensor
    legs = (indices..., [ind' for ind in indices]...)

    # create the ITensor
    return itensor(transpose(U), legs)

end


"""
    paulirotationmatrix(generator, θ)
"""
function paulirotationmatrix(generator, θ)
    symbols = stringtosymbols(generator)
    pauli_rot = PP.PauliRotation(symbols, 1:length(symbols))
    return PP.tomatrix(pauli_rot, θ)
end


# conversion of the gate indices to a tuple
function _ensuretuple(gate_inds::Union{Tuple,AbstractArray})
    return gate_inds
end

function _ensuretuple(gate_inds::NamedEdge)
    return (gate_inds.src, gate_inds.dst)
end