import pypuf.simulation, pypuf.io, pypuf.attack, pypuf.metrics



puf = pypuf.simulation.ArbiterPUF(n=64, noisiness=.25, seed=3)
pypuf.metrics.reliability(puf, seed=3).mean()
challenges = pypuf.io.random_inputs(n=puf.challenge_length, N=500, seed=2)
responses_mean = puf.r_eval(5, challenges).mean(axis=-1)
crps = pypuf.io.ChallengeResponseSet(challenges, responses_mean)
for crp in crps:
    print(crp)
#attack = pypuf.attack.LeastSquaresRegression(crps, feature_map=lambda cs: pypuf.simulation.ArbiterPUF.transform_atf(cs, k=1)[:, 0, :])
#model = attack.fit()