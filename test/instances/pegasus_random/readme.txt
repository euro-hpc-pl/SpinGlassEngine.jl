These are random instances generated for DWave Advantage_system6.1
We used "nice" coordinate system, so some parts of machine are unused
------------------------------------------------------------------------
------------------------------------------------------------------------
Size description:
P4 - 3x3x3 with 216 spins and 1324 couplings
P8 - 7x7x3 with 1176 spins and 8107 couplings (1 missing edge compared to "perfect" P8 graph)
P12 - 11x11x3 with 2900 spins and 20 592 couplings (4 missnig nodes and 60 missing edges compared to "perfect" P12 graph)
P16 - 15x15x3 with 5 376 spins and 38 615 couplings (24 missnig nodes and 341 missing edges compared to "perfect" P16 graph)
------------------------------------------------------------------------
Category description:
(Here category describes how we generated biases and couplings values)
RAU - RAndom Uniform: biases (h) are in range [-0.1, 0.1], coupling are in range [-1, 1]
RCO - Random Couplings Only: biases are set to 0, coupling are in range [-1, 1]
AC3 - Anti-Cluser: biases are in range [-1/9, 1/9], couplings within unit cells are in range [-1/3, 1/3]
		   couplings between unit cells are in range [-1,1] (so 3 times stronger than within unit cell)
-------------------------------------------------------------------------
Final Remarks:
1. Here We define pegasus unit cell similarly to DWaves documentation, so 1 unit cell consist of 24 spins grouped in 3 "chimera-like" K4 graphs.
   This definition is consistent with SpinGlassPEPS convention.
2. files with _sg.txt affix are in pegasus_lattice format
3. files with _dv.pkl affix are pickled python dictionaries (h, J), prepared for DWave ocean software (sampler.sample_ising())