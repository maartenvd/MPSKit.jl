function leading_boundary(ψ::InfiniteMPS, H::DenseMPO, alg::GradientGrassmann,
                          envs=environments(ψ, H))
    multi, envs, err = leading_boundary(convert(MPSMultiline, ψ),
                                        convert(MPOMultiline, H), alg, envs)
    ψ = convert(InfiniteMPS, multi)
    return ψ, envs, err
end

function leading_boundary(ψ::MPSMultiline, H, alg::GradientGrassmann,
                          envs=environments(ψ, H))
    res = optimize(GrassmannMPS.fg,
                   GrassmannMPS.ManifoldPoint(ψ, envs),
                   alg.method;
                   (transport!)=GrassmannMPS.transport!,
                   retract=GrassmannMPS.retract,
                   inner=GrassmannMPS.inner,
                   (scale!)=GrassmannMPS.scale!,
                   (add!)=GrassmannMPS.add!,
                   (finalize!)=alg.finalize!,
                   isometrictransport=true)
    x, fx, gx, numfg, normgradhistory = res
    return x.state, x.envs, normgradhistory[end]
end
