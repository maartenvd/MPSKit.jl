using TensorKit
using MPSKit
using MPSKit: UniformGauging, uniform_gauge
using BenchmarkTools

A = [MPSTensor(2, 1, 3), MPSTensor(2, 3, 3), MPSTensor(2, 3, 1)];

alg = UniformGauging(; verbosity=0)
AL, AR, CR = uniform_gauge(A; alg);

space.(AL)
space.(AR)
space.(CR)

AL2, AR2, CR2 = uniform_gauge(AR; alg)

@benchmark uniform_gauge($A; alg) seconds = 10
