These are random instances generated for DWave Advantage_prototype1.1
------------------------------------------------------------------------
------------------------------------------------------------------------
Size description:
Z2 - zephyr lattice with 159 nodes and 1205 edges (1 missing node and 19 missing edges compared to "perfect" Z2 graph)
Z3 - zephyr lattice with 332 nodes and 2735 edges (4 missing nodes and 73 missing edges compared to "perfect" Z3 graph)
Z4 - zephyr lattice with 563 nodes and 4790 edges (13 missing nodes and 242 missing edges compared to "perfect" Z4 graph)

------------------------------------------------------------------------
Category description:
(Here category describes how we generated biases and couplings values)
RAU - RAndom Uniform: biases (h) are in range [-0.1, 0.1], coupling are in range [-1, 1]
RCO - Random Couplings Only: biases are set to 0, coupling are in range [-1, 1]
AC3 - Anti-Cluser: biases are in range [-1/9, 1/9], couplings within unit cells are in range [-1/3, 1/3]
		   couplings between unit cells are in range [-1,1] (so 3 times stronger than within unit cell) - Work in progress
-------------------------------------------------------------------------
Final Remarks:
1. Here We define zephyr unit cell in rather unique way. See picture.
   This definition is consistent with SpinGlassPEPS convention.
2. files with _sg.txt affix are in zephyr_lattice format
3. files with _dv.pkl affix are pickled python dictionaries (h, J), prepared for DWave ocean software (sampler.sample_ising())