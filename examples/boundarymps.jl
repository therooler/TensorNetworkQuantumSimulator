include("../src/ITensorNetworksExamples.jl")

using Random
Random.seed!(1634)

function main()
  nx, ny = 5, 5
  χ = 2
  ITensors.disable_warn_order()
  bp_update_kwargs = (; maxiter=40, tol=1e-12, message_update_kwargs = (; message_update_function = ms -> make_eigs_real.(default_message_update(ms))))
  boundary_mps_kwargs = (; message_update_kwargs = (; niters = 30, tolerance = 1e-10))
  gs = [
    (named_grid((nx, 1)), "line"),
    (named_hexagonal_lattice_graph(nx - 2, ny -2), "hexagonal"),
    (named_grid((nx, ny)), "square"),
    (lieb_lattice_grid(nx, ny), "lieb"),
  ]
  for (g, g_str) in gs
    println("Testing for $g_str lattice with $(nv(g)) vertices")
    s = siteinds("S=1/2", g)
    ψ = random_tensornetwork(ComplexF64, s; link_space=χ)
    s = siteinds(ψ)
    v_centre = first(center(g))

    println("Computing single site expectation value via various means")
    boundary_mps_ranks = [1, 2, 4, 8, 16, 32]
    for r in boundary_mps_ranks
      sz_boundarymps = expect(ψ, ("Z", [v_centre]); alg = "boundarymps", cache_construction_kwargs = (; message_rank = r))
      println("Boundary MPS Value for Z at Rank $r is $sz_boundarymps")
    end

    sz_exact = expect(ψ, ("Z", [v_centre]); alg = "exact")
    println("Exact value for Z is $sz_exact")

    if !is_tree(g)
      vn = first(filter(v -> first(v) == first(v_centre), neighbors(g, v_centre)))
      println("Computing two site, neighboring, expectation value via various means")
      boundary_mps_ranks = [1, 2, 4, 8, 16, 32]
      for r in boundary_mps_ranks
        sz_boundarymps = expect(ψ, ("ZZ", [v_centre, vn]); alg = "boundarymps", cache_construction_kwargs = (; message_rank = r))
        println("Boundary MPS Value for ZZ at Rank $r is $sz_boundarymps")
      end

      #sz_exact = exact_expect(ψ, ["Z", "Z"], [v_centre, vn])
      #println("Exact value for ZZ is $sz_exact")
    end
  end
end

main()
