using Pkg #to be removed
Pkg.activate("/Users/daan/Desktop/TimedtdvpTest/TimedTDVP") #to be removed
using MPSKit, MPSKitModels, TensorKit, Plots

function my_transverse_field_ising(gs)
    L = length(gs)
    lattice = InfiniteChain(L)
    ZZ = rmul!(σᶻᶻ(), -1)
    X = rmul!(σˣ(), -1)
    a = @mpoham sum(ZZ{i,j} for (i, j) in nearest_neighbours(lattice))
    b = @mpoham sum(gs[i] * X{i} for i in vertices(lattice))
    return a + b
end

function my_timedependent_ising(gl,gs,gr,f)
    L = length(gs)
    lattice  = InfiniteChain(1)
    latticeL = InfiniteChain(L)
    ZZ = rmul!(σᶻᶻ(), -1)
    X = rmul!(σˣ(), -1)

    ZZl = @mpoham sum(ZZ{i,j} for (i, j) in nearest_neighbours(lattice))
    ZZm = @mpoham sum(ZZ{i,j} for (i, j) in nearest_neighbours(latticeL))
    ZZr = @mpoham sum(ZZ{i,j} for (i, j) in nearest_neighbours(lattice))

    Xl = @mpoham sum(gl * X{i} for i in vertices(lattice))
    Xm = @mpoham sum(gs[i] * X{i} for i in vertices(latticeL))
    Xr = @mpoham sum(gr * X{i} for i in vertices(lattice))

    H1 = Window(ZZl,ZZm,ZZr)
    H2 = Window(Xl,Xm,Xr)
    return LazySum([H1,MultipliedOperator(H2,f)])
end

function my_expectation_value(Ψwindow::WindowMPS,O::Window{A,A,A}) where {A<:TrivialTensorMap{ComplexSpace, 1, 1, Matrix{ComplexF64}}}
    left   = expectation_value(Ψwindow.left, O.left)
    middle = expectation_value(Ψwindow, O.middle)
    right  = expectation_value(Ψwindow.right, O.right)
    return vcat(left,middle,right)
end

function my_finalize(t, Ψ, H, envs, si, tosave)
    push!(tosave, my_expectation_value(Ψ, si))
    return Ψ, envs
end

# WindowMPS as bath
#-------------------

#define the hamiltonian
H = transverse_field_ising(; g=0.3)
sx, sy, sz = σˣ(), σʸ(), σᶻ()

#initilizing a random mps
Ψ = InfiniteMPS([ℂ^2], [ℂ^12])

#Finding the groundstate
(Ψ, envs, _) = find_groundstate(Ψ, H, VUMPS(; maxiter=400))

len = 20
middle = round(Int, len / 2)

# make a WindowMPS by promoting len sites to the window part
# by setting fixleft=fixright=true we indicate that the infinite parts will not change
Ψwindow = WindowMPS(Ψ, len; fixleft=true, fixright=true)
#apply a single spinflip at the middle site
@tensor Ψwindow.AC[middle][-1 -2; -3] := Ψwindow.AC[middle][-1, 1, -3] * sx[-2, 1];
normalize!(Ψwindow);

# create the environment
# note: this method is only defined for fixleft=true=fixright=true WindowMPS and assumes the same H for left and right infinite environments
# if it errors for these reasons use H  = Window(Hleft,Hmiddle,Hright) instead
envs = environments(Ψwindow, H);
szdat = [expectation_value(Ψwindow, sz)]

#setup for time_evolve
alg = TDVP2(; trscheme=truncdim(20),
            finalize=(t, Ψ, H, envs) -> my_finalize(t, Ψ, H, envs, sz,szdat));
t_span = 0:0.05:3.0
Ψwindow, envs = time_evolve!(Ψwindow, H, t_span, alg, envs);

display(heatmap(real.(reduce((a, b) -> [a b], szdat))))

# WindowMPS as interpolation
#----------------------------

# The Hamiltonian wil be -(∑_{<i,j>} Z_i Z_j + f(t) * ∑_{<i>} g_i X_i)
gl = 3.0
gr = 4.0
L = 10
gs = range(gl, gr; length=L); #interpolating values for g

Hl = my_transverse_field_ising([gl]);
Hr = my_transverse_field_ising([gr]);

Hfin = my_transverse_field_ising(gs);
# Note: it is important that the lattice for the finite part of the WindowMPS is infinite.
#       For a finite lattice @mpoham is smart and does not construct the terms in the MPOHamiltonian
#       that will not be used due to the boundary. For a WindowMPS there is no boundary and we thus
#       we need the MPO for the infinite lattice.

Hwindow = Window(Hl, Hfin, Hr);
sx, sy, sz = σˣ(), σʸ(), σᶻ()

#initilizing a random mps
D = 12
Ψl = InfiniteMPS([ℂ^2], [ℂ^D])
Ψr = InfiniteMPS([ℂ^2], [ℂ^D]) #we do not want Ψr === ψl
#Ts = map(i->TensorMap(rand,ComplexF64,ℂ^D*ℂ^2,ℂ^D),eachindex(gs))
Ts = fill(TensorMap(rand, ComplexF64, ℂ^D * ℂ^2, ℂ^D), L);
Ψwindow = WindowMPS(Ψl, Ts, Ψr);

# finding the groundstate
(Ψwindow, envs, _) = find_groundstate(Ψwindow, Hwindow);
# the alg for find_groundstate has to be a Window(alg_left,alg_middle,alg_right).
# If no alg is specified the default is Window(VUMP(),DMRG(),VUMPS())
es = real.(expectation_value(Ψwindow, Hwindow, envs));
# note that es[1] is the expectation_value of Ψwindow.left and es[end] that of Ψwindow.right
scatter(0:(L + 1), es; label="")

# WindowMPS for non-uniform quench
#---------------------------------

#define the hamiltonian
g_uni = 3.0
gl = 3.0
gr = 4.0
L = 40

Hl = my_transverse_field_ising([g_uni]);
Hr = my_transverse_field_ising([g_uni]);
Hfin = my_transverse_field_ising([g_uni]);
Hgs = Window(Hl, Hfin, Hr);

D = 12
Ψl = InfiniteMPS([ℂ^2], [ℂ^D])
Ψr = InfiniteMPS([ℂ^2], [ℂ^D]) #we do not want Ψr === ψl
#Ts = map(i->TensorMap(rand,ComplexF64,ℂ^D*ℂ^2,ℂ^D),eachindex(gs))
Ts = fill(TensorMap(rand, ComplexF64, ℂ^D * ℂ^2, ℂ^D), L);
Ψwindow = WindowMPS(Ψl, Ts, Ψr);

# finding the groundstate
(Ψwindow, envs_gs, _) = find_groundstate(Ψwindow, Hgs);

#define the quench Hamiltonian
f(t) = 0.1*t #we take a linear ramp
gs = range(gl, gr; length=L); #interpolating values for g
Hquench = my_timedependent_ising(gl,gs,gr,f);
# Hquench is a time-dependent Hamiltonian i.e. we can do H(t) to get the instantanious Hamiltonian.
# Note: To get an expectation_value of a time-dependent Hamiltonian one needs to give H(t) tot he function.

envs = environments(Ψwindow,Hquench);

sdat = [my_expectation_value(Ψwindow, Window(sx,sx,sx))]

#setup for time_evolve
left_alg = rightalg = TDVP();
middle_alg =  TDVP2(; trscheme=truncdim(20));
alg = WindowTDVP(;left=left_alg,middle=middle_alg,right=rightalg,
            finalize=(t, Ψ, H, envs) -> my_finalize(t, Ψ, H, envs, Window(sx,sx,sx), sdat));
t_span = 0:0.005:1.0

Ψwindow, envs = time_evolve!(Ψwindow, Hquench, t_span, alg, envs; verbose=true);

display(heatmap(t_span,0:L+1,real.(reduce((a, b) -> [a b], sdat)),xlabel="t",ylabel="i"))
