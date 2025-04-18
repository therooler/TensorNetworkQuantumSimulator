using TensorNetworkQuantumSimulator
const TN = TensorNetworkQuantumSimulator

using ITensorNetworks: siteinds, ITensorNetwork

using NamedGraphs.NamedGraphGenerators: named_grid
using Statistics

using ITensors

function strip_qn(tn::ITensorNetwork)
  tn = copy(tn)
  for v in vertices(tn)
    tn[v] = dense(tn[v])
  end
  return tn
end

function main()

  ITensors.disable_warn_order()

  nx, ny = 3,3
  #g = TN.heavy_hexagonal_lattice(nx, ny)
  g = named_grid((nx,ny))
  nqubits = length(vertices(g))

  println("Quantum numbers disabled for simulating the dynamics of the Heisenberg model on a 3x3 grid \n")
  s = TN.siteinds("S=1/2", g; conserve_qns =false)
  ψ = ITensorNetwork(v -> isodd(first(v) + last(v)) ? "↑" : "↓", s)

  maxdim, cutoff = 2, 1e-14
  apply_kwargs = (; maxdim, cutoff, normalize=true)
  #Parameters for BP, as the graph is not a tree (it has loops), we need to specify these
  set_global_bp_update_kwargs!(maxiter = 25, tol = 1e-6)
  J, Δ = 1.0, 0.5
  no_trotter_steps = 3
  δt = 0.25

  ec = edge_color(g, 4)


  # pauli rotations are tuples like `(pauli_string, [site_labels], parameter)`
  layer = []
  append!(layer, ("Rxxyy", pair, 2*J*δt) for pair in reduce(vcat, ec))
  append!(layer, ("Rzz", pair, 2*Δ*δt) for pair in reduce(vcat, ec))
  #append!(layer, ("Rz", [v], 2*Δ*δt) for v in vertices(g))

  ψψ = build_bp_cache(ψ)
  z_obs = [("Z", [v]) for v in vertices(g)]
  init_z = sum(TN.expect(ψψ, z_obs))
  println("Initial total magnetisation is $init_z")

  # evolve! The first evaluation will take significantly longer because of compulation.
  for l = 1:no_trotter_steps
    println("Applying Layer $l")

    t1 = @timed ψ, ψψ, errors =
        apply(layer, ψ, ψψ; apply_kwargs, update_every = 1, verbose = false);

    println("Maximum gate error for this layer is $(maximum(errors))") 

  end

  ψ, ψψ = TN.symmetric_gauge(ψ; cache_update_kwargs = get_global_bp_update_kwargs())

  z_bp = TN.expect(ψψ, z_obs)
  final_z = sum(z_bp)
  println("BP Measured Final total magnetisation is $final_z")

  z_bmps = TN.expect(ψ, z_obs; alg = "boundarymps", cache_construction_kwargs = (; message_rank = 2))
  final_z_bmps = sum(z_bmps)
  println("Boundary MPS Measured Final total magnetisation is $final_z_bmps")

  z_exact = TN.expect(ψ, z_obs; alg = "exact")

  @show sum(abs.(z_bmps - z_exact))
  @show sum(abs.(z_bp - z_exact))
end

main()
