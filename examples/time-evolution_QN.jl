using TensorNetworkQuantumSimulator
const TN = TensorNetworkQuantumSimulator

using ITensorNetworks

using NamedGraphs.NamedGraphGenerators: named_grid
using Statistics

using ITensorMPS

using ITensors

function main()

  ITensors.disable_warn_order()

  nx, ny = 2,2
  g = TN.heavy_hexagonal_lattice(nx, ny)
  nqubits = length(vertices(g))
  s = siteinds("S=1/2", g; conserve_qns = true)
  ψ = ITensorNetwork(v -> isodd(first(v)) ? "↑" : "↓", s)

  maxdim, cutoff = 2, 1e-14
  apply_kwargs = (; maxdim, cutoff, normalize=true)
  #Parameters for BP, as the graph is not a tree (it has loops), we need to specify these
  set_global_bp_update_kwargs!(maxiter = 25, tol = 1e-6)
  J, Δ = 1.0, 0.5
  no_trotter_steps = 5
  δt = 0.25

  ec = edge_color(g, 3)


  # pauli rotations are tuples like `(pauli_string, [site_labels], parameter)`
  layer = []
  append!(layer, ("Rxxyy", pair, 2*J*δt) for pair in reduce(vcat, ec))
  append!(layer, ("Rzz", pair, 2*Δ*δt) for pair in reduce(vcat, ec))

  ψψ = build_bp_cache(ψ)
  z_obs = [("Z", [v]) for v in vertices(g)]
  init_z = sum(expect(ψψ, z_obs))
  println("Initial total magnetisation is $init_z")

  # evolve! The first evaluation will take significantly longer because of compulation.
  for l = 1:no_trotter_steps
    println("Applying Layer $l")

    t1 = @timed ψ, ψψ, errors =
        apply(layer, ψ, ψψ; apply_kwargs, update_every = 1, verbose = false);

  end

  final_z = sum(expect(ψψ, z_obs))
  println("Bp Measured Final total magnetisation is $final_z")

  final_z_exact = sum(expect(ψ, z_obs; alg = "exact"))
  println("Exact Final total magnetisation is $final_z_exact")

end

main()
