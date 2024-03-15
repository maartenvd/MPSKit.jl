"""
    struct SvdCut <: Algorithm end

An algorithm that uses truncated SVD to change the bond dimension of a ψ.

# Fields
- `trscheme::TruncationScheme = notrunc()` : The truncation scheme to use.
"""
@kwdef struct SvdCut <: Algorithm
    trscheme::TruncationScheme = notrunc()
end

changebonds(ψ::AbstractFiniteMPS, alg::SvdCut) = changebonds!(copy(ψ), alg)
function changebonds!(ψ::AbstractFiniteMPS, alg::SvdCut)
    for i in (length(ψ) - 1):-1:1
        U, S, V, = tsvd(ψ.CR[i]; trunc=alg.trscheme, alg=TensorKit.SVD())
        AL′ = ψ.AL[i] * U
        ψ.AC[i] = (AL′, complex(S))
        AR′ = _transpose_front(V * _transpose_tail(ψ.AR[i + 1]))
        ψ.AC[i + 1] = (complex(S), AR′)
    end
    return normalize!(ψ)
end

function changebonds(ψ::DenseMPO, alg::SvdCut)
    return convert(DenseMPO, changebonds(convert(InfiniteMPS, ψ), alg))
end
function changebonds(ψ::MPOMultiline, alg::SvdCut)
    return convert(MPOMultiline, changebonds(convert(MPSMultiline, ψ), alg))
end
function changebonds(ψ::MPSMultiline, alg::SvdCut)
    return Multiline(map(x -> changebonds(x, alg), ψ.data))
end
function changebonds(ψ::InfiniteMPS, alg::SvdCut)
    copied = complex.(ψ.AL)
    ncr = ψ.CR[1]

    for i in 1:length(ψ)
        U, ncr, = tsvd(ψ.CR[i]; trunc=alg.trscheme, alg=TensorKit.SVD())
        copied[i] = copied[i] * U
        copied[i + 1] = _transpose_front(U' * _transpose_tail(copied[i + 1]))
    end

    return InfiniteMPS(uniform_gauge(copied, complex(ncr))...)
end

function changebonds(ψ, H, alg::SvdCut, envs=environments(ψ, H))
    return (changebonds(ψ, alg), envs)
end
