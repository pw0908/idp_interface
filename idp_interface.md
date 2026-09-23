# Sequence disorder produce interfacial potentials in biocondensates
* Author: Pierre J. Walker
* Journals: Macromolecules, Biomacromolecules, PRX

* I have provided all the relevant literature under `literature`
## Motivation
* Interfacial properties of in the form of coacervates, in the form of polyelectrolytes and intrinsically disordered proteins, are becoming particularly important in understanding how systems such as biocondensates, membraneless organelles, and lipid nanoparticles form and interact with their environment. See work by Yifan Dai, Pavan Inguva and others.
* In particular, the formation of electrostatic potential at interfaces is of growing interest for not only biological understanding, but applications such as the use of biocondensate interfaces as catalysts.
* The formation of electrostatic potentials at interfaces is relatively well understood for electrolyte mixture as well as less-trivial cases such as neutral polar solvents and water (see work on the Stockmayer fluid, experimental papers regarding droplets as catalyst by Wei Min, etc.). However, the case of biocondensates is substantially more complex due to the inherent complexity of the macromolecules, with complex sequence dependencies, resulting in complex interfaces and dependencies on external factors such as pH dependency, temperature and salt concentration.
* It is because of these complexities that it is often preferrable, to develop fundamental understanding, to develop simplified representations of these chemical complexities. A typical representation of biocondensates / intrinsically disordered proteins is to treat them a polyelectrolyte chains of identically-sized, positive and negative chains. See work by Pengfei Zhang, Pierre Walker, Fredrickson, Artem Rumynastev, etc. In these cases, the identity of the component can be reduce to the sequence disorder and net charge. Whether it be through analytical theory or coarse-grained molecular dynamics simulations, such approaches have proved invaluable in understanding the single-chain and phase behavior.
* While molecular dynamics may provide the most complete description of our test system, computational limitations make it challenging to fully resolve the interface, let alone any interfacial potential, particularly for large macromolecules.
* It is for this reason we opt to use analytical theory instead. While self-consistent field theory could be applicable, it's limitations in reproducing the phase behavior of such coarse grained system make it inadequate for our purposes.
* As such, we choose to use liquid state theory instead. Specifically, we use the variant developed by Zhang et al. 2016, based on the MSA theory developed by Blum et al., which has had much success in reproducing the phase- and interfacial-behavior of such macromolecules. See works by Zhang, Walker, etc.

## Hypothesis
My expectation is that for highly ordered sequences (e.g. perfect diblocks), the perfect symmetry will result in no electrostatic potential at the interface. However, once sequence disorder is introduced, the break in symmetry will force a non-zero electrostatic potential at the interface, even if the sequence is net-neutral. However, at the fully disordered limit (alternating blocks), symmetry is returned and there should no longer be an electrostatic potential. Coupled with changes in density of the coacervate with sequence disorder, there will be a non-trivial relationship between $\kappa$ (measure of sequence disorder) and the electrostatic potential at the interface.

## Method
For the purposes of this project, we will be making us of the Clapeyron.jl and ClassicalDFT.jl packages. Any code produced in this project should be kept outside of these two packages as external implementations.

For computational efficiency, only implement the simplified forms of the equations which assume all beads are of the the same size ($\sigma=3\times 10^{-10}$m). For simplicity later on, I suggest focusing on short chains (10-20mers).

For all steps, make sure to save results in compressed files for future analysis. When plotting figures, I have provided my default rcParams file under `resources/rcParams.txt`. Use PyPlot.jl.

Once benchmarking is complete, submit jobs through slurm on the cluster. I have provided the generic sbatch header under `resources/slurm_header.txt`.

### Step 1: Bulk validation
* As a first step, I would like you to implement LS theory into Clapeyron.jl. I have provided you a validated implementation under `resources/Helmholtz_LS.jl` which does make use of Clapeyron tools but the main function, `a`, is not compatible with Clapeyron as it expresses the reduced free energy as a normalized free energy density. 
* You should implement this as an `a_res` function, much like all other EOS in Clapeyron.
* As this will be an electrolyte EoS, two eos will need to be produced: a neutral term (f_hs, part of f_chain and f_chi) and the charge terms (f_el and part of f_chain). We will assume a constant dielectric constant model is used for the permittivity.
* Once implemented, it should be easy enough to benchmark our implementation against `resources/Helmholtz_LS.jl`. 
* Once this is done, I'd like you to obtain the phase diagrams for a net-neutral IDP with equal number of cation and anion beads. I'd suggest picking 4-5 sequences with varying degrees of sequence disorder.
* Deliverable: phase diagrams of 4-5 IDP sequences of varying sequence disorder.

### Step 2: Extension of ClassicalDFT.jl to LS
* As of now, ClassicalDFT.jl should have most of the tools needed to study this problem: electrostatic support and chain propagators. I just haven't tested it for the combination of the two.
* To ensure that these two implementations work, I would like you to implement the free energy functionals for LS, making use of existing FMT and PC-SAFT implementations to obtain the neutral terms, and following the existing implementation of DH for the charged terms. Once these are implemented, run uniform profiles and compare the chemical potentials obtained in Clapeyron compared to ClassicalDFT. If these match, we're good to go. 
* Deliverable: a benchmark between Clapeyron and ClassicalDFT implementations of LS theory.

### Step 3: Studying the IDP interface
* For the 4-5 IDP sequences selected in step 1, run interfacial calculations along the two-phase boundary to examine how the density profiles change at the interface. Make sure to record the interfacial tension and electrostatic potential profiles. These should only ever be 1D calculations.
* If you need a larger number of points, you can use threading to allocate across multiple CPUs. Note that enzyme, the tools used to obtain derivatives in ClassicalDFT, only begins threading across multiple CPUs if there are 1000 points per thread.
* Deliverable: Figures examining the interfacial tension along the two-phase boundary for 4-5 IDP sequences, along with a few electrostatic potential profile figures.

## Results

All three steps are complete and cross-validated (Clapeyron.jl `a_res` vs. `resources/Helmholtz_LS.jl` to machine precision; the ClassicalDFT.jl functional vs. Clapeyron's own bulk chemical potential, also to machine precision, 151/151 unit tests). A consolidated write-up with figures is published at
[claude.ai/artifact/C1MfvzYGx7VATp2iMkUCMz](https://claude.ai/artifact/C1MfvzYGx7VATp2iMkUCMz); the summary below is the short version.

**The hypothesis holds, with a sharper, exact statement of *why*.** Both symmetric extremes (the perfect diblock and strict alternation) give an electrostatic potential of *exactly* zero, and intermediate disorder gives a real, nonzero potential — confirmed for a 7-sequence family traced along their full Step 1 binodals (~80 points each), and for a 1,340-sequence survey at fixed $\ell_B/\sigma=1$. The exact mechanism: reversing bead order ($i \leftrightarrow N+1-i$) is always a symmetry of the neutral hard-sphere/chain physics for a linear chain. For an **anti-palindrome** ($Z_i = -Z_{N+1-i}$ for every bead, which both the diblock and the alternating sequence are), that same relabeling is *also* exactly a global charge flip — forcing the net local charge density to equal its own negative everywhere, i.e. to vanish identically, independent of how segregated the sequence is. A **true palindrome** ($Z_i=+Z_{N+1-i}$) gets the same relabeling symmetry but *not* the charge flip, so nothing cancels — and empirically it can carry the *largest* potential found in this study.

**$\kappa_{\mathrm{seq}}$ (the Sawle–Ghosh/Das–Pappu clustering parameter used for sequence selection in Step 1) is not, by itself, the right predictor of the potential.** A second, windowed order parameter was developed —
$$\kappa_{\mathrm{sym}} = \frac{1}{\kappa_{\mathrm{sym}}^{\max}(N)}\sum_{j=1}^{N-5}\Big(\sum_{i=j}^{j+5}\big(Z_i+Z_{N+1-i}\big)\Big)^2,$$
normalized by the true achievable maximum for a net-neutral chain of length $N$ (not a loose analytic ceiling) — that measures local deviation from the anti-palindrome condition specifically, rather than clustering in general. $\kappa_{\mathrm{sym}}=0$ exactly whenever a sequence is anti-palindromic (in whole or in part), and organizes the achievable electrostatic potential far more cleanly than $\kappa_{\mathrm{seq}}$ across the whole survey.

**For a 20-mer, the entire achievable $(\kappa_{\mathrm{seq}},\kappa_{\mathrm{sym}})$ region can be mapped exactly, not just sampled.** A net-neutral 20-bead chain has exactly $\binom{20}{10}=184{,}756$ distinct arrangements — few enough to enumerate directly. Doing so shows $\kappa_{\mathrm{seq}}$ itself takes only 419 distinct exact values at this length (a genuinely discrete quantity, not a sampling artifact), and that the achievable region is triangular: rising from the origin to $\kappa_{\mathrm{sym}}=1.0$ exactly at $\kappa_{\mathrm{seq}}=0.702$ (the sequence $[+5,-10,+5]$), then falling back to the diblock at $(1,0)$. Some regions inside that triangle are confirmed combinatorially empty rather than merely under-sampled — e.g. no sequence of this length reaches $\kappa_{\mathrm{seq}}\approx0.8$ with $\kappa_{\mathrm{sym}}\approx0.2$.
