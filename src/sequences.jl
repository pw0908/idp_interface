"""
    sigma_blob(charges, g)

Mean local charge-asymmetry `<σ>` over all windows of length `g` in
`charges` (the blob-based statistic underlying the Das-Pappu/Sawle-Ghosh
sequence charge-patterning order parameter `κ`). For each window,
`σ = (f+ - f-)^2 / (f+ + f-)` where `f±` are the fractions of +/- beads
*within that window* (0 when the window has no charged beads at all).
"""
function sigma_blob(charges::AbstractVector{<:Integer}, g::Int)
    N = length(charges)
    g <= N || throw(ArgumentError("blob size $g exceeds sequence length $N"))
    nwindows = N - g + 1
    total = 0.0
    for start in 1:nwindows
        np = 0
        nm = 0
        for i in start:start+g-1
            c = charges[i]
            c > 0 && (np += 1)
            c < 0 && (nm += 1)
        end
        fp = np / g
        fm = nm / g
        denom = fp + fm
        total += denom == 0 ? 0.0 : (fp - fm)^2 / denom
    end
    return total / nwindows
end

"""
    kappa_seq(charges; blobsizes=(5,6))

The Das-Pappu/Sawle-Ghosh `κ` charge-patterning order parameter: `κ = 0`
for a well-mixed (locally charge-balanced) sequence, `κ = 1` for the most
segregated arrangement achievable with the same composition (same counts
of +, -, and neutral beads). Averages `sigma_blob` over the given blob
sizes, then normalizes by the same quantity evaluated on the maximally
segregated permutation of the same composition (all + beads first, then
all neutrals, then all - beads).
"""
function kappa_seq(charges::AbstractVector{<:Integer}; blobsizes=(5, 6))
    N = length(charges)
    sizes = Tuple(g for g in blobsizes if g <= N)
    isempty(sizes) && throw(ArgumentError("sequence of length $N is shorter than every blob size in $blobsizes"))

    sigma_bar = sum(sigma_blob(charges, g) for g in sizes) / length(sizes)

    npos = count(>(0), charges)
    nneg = count(<(0), charges)
    nneu = N - npos - nneg
    segregated = vcat(fill(1, npos), fill(0, nneu), fill(-1, nneg))
    sigma_max = sum(sigma_blob(segregated, g) for g in sizes) / length(sizes)

    sigma_max == 0 && return 0.0
    return sigma_bar / sigma_max
end

"""
    block_sequence(N, blocksize)

A net-neutral, equal +/- sequence of length `N` (`N` even) built from
repeating `[+1 × blocksize, -1 × blocksize]` blocks. `blocksize == N÷2`
gives the perfect diblock (`κ=1`); `blocksize == 1` gives the fully
alternating sequence (`κ≈0`). `N÷2` must be divisible by `blocksize`.
"""
function block_sequence(N::Int, blocksize::Int)
    iseven(N) || throw(ArgumentError("N must be even for an equal +/- sequence"))
    half = N ÷ 2
    half % blocksize == 0 || throw(ArgumentError("blocksize $blocksize must divide N/2 = $half"))
    charges = Int[]
    sign = 1
    while length(charges) < N
        append!(charges, fill(sign, blocksize))
        sign = -sign
    end
    return charges
end

"""
    default_sequence_family(N=20)

The 4-5 net-neutral, equal +/- sequences (10-20mers, per the project spec)
used throughout Steps 1-3, spanning `kappa_seq` from the fully alternating
limit to the perfect-diblock limit via `block_sequence`.
"""
function default_sequence_family(N::Int=20)
    half = N ÷ 2
    blocksizes = filter(b -> half % b == 0, 1:half)
    # pick up to 5 blocksizes roughly log-spaced across the divisors of N/2,
    # always including the two limits (1 and half).
    chosen = if length(blocksizes) <= 5
        blocksizes
    else
        idx = round.(Int, range(1, length(blocksizes), length=5))
        unique(blocksizes[idx])
    end
    return [(b, block_sequence(N, b)) for b in chosen]
end

"""
    mirror_symmetry(charges)

`mean(charges[i]*charges[end+1-i])` -- the *signed* autocorrelation of a
charge sequence with its own mirror image, `∈[-1,1]`. This sign is not a
cosmetic choice -- it distinguishes two physically distinct kinds of
reflective structure:

- `+1`: a perfect **palindrome** (`charges[i]==charges[end+1-i]` for all
  `i`). Relabeling the chain's bead indices `i↔N+1-i` is always a symmetry
  of the neutral hard-sphere/chain physics (a linear chain has no intrinsic
  direction) -- for a palindrome this relabeling leaves every charge
  unchanged too, so it implies *no* constraint on the resulting charge
  density profile. A palindrome can still have strong local clustering
  (arbitrary `kappa_seq`) and a genuinely nonzero interfacial electrostatic
  potential -- confirmed directly (`results/neutral_sample`, sequences like
  `[1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,1]`, a true palindrome
  with `kappa_seq=0.147` and a real potential jump `≈0.0099`).
- `-1`: a perfect **anti-palindrome** (`charges[i]==-charges[end+1-i]` for
  all `i`, e.g. any clean diblock -- reversing a diblock exactly negates
  it -- or strict alternation for even `N`). Here the same bead-relabeling
  symmetry *also* flips every charge, which forces the net local charge
  density `Σᵢ Zᵢρᵢ(x)` to equal its own negative at every point in space --
  i.e. exactly zero everywhere, and hence exactly zero electrostatic
  potential, *regardless of `kappa_seq`* (this is the rigorous reason
  `block1`, κ≈0.026, and `block10`, κ=1, both give identically zero
  potential despite being at opposite ends of the `kappa_seq` range).
- `0`: a sequence whose mirror is, on average, uncorrelated with itself.

An earlier version of this function returned `|mean(...)|`, deliberately
conflating palindromes and anti-palindromes as equally "symmetric" -- wrong,
since only anti-palindromes carry the symmetry-enforced zero-potential
guarantee; a true palindrome is not similarly protected. Keep the sign.
"""
function mirror_symmetry(charges::AbstractVector{<:Integer})
    N = length(charges)
    return sum(charges[i] * charges[N + 1 - i] for i in 1:N) / N
end

"""
    sequence_asymmetry(charges)

`1 - abs(mirror_symmetry(charges))`: `0` for a perfectly (anti-)palindromic
sequence (either kind of reflective regularity), `1` for one whose mirror is
on average uncorrelated with itself. Use this (rather than
`mirror_symmetry` directly) only when palindrome/anti-palindrome really
should be pooled together -- e.g. as a rough "how regular is this sequence"
summary for something insensitive to the palindrome/anti-palindrome
distinction. For anything related to the electrostatic potential, use the
signed `mirror_symmetry` instead -- the sign is exactly what determines
whether zero potential is guaranteed.
"""
sequence_asymmetry(charges::AbstractVector{<:Integer}) = 1 - abs(mirror_symmetry(charges))

function _kappa_sym_raw(charges::AbstractVector{<:Integer}, window::Int)
    N = length(charges)
    nwindows = N - window + 1
    nwindows >= 1 || throw(ArgumentError("window $window exceeds sequence length $N"))
    raw = 0
    for j in 1:nwindows
        S = 0
        for i in j:(j + window - 1)
            S += charges[i] + charges[N + 1 - i]
        end
        raw += S^2
    end
    return raw
end

"""
    kappa_sym_max(N; window=6)

The true achievable maximum of `kappa_sym`'s raw (unnormalized) sum for a
net-neutral, equal-`+`/`-` sequence of length `N` -- i.e. the composition-
conditioned normalizer `kappa_sym` actually divides by, in the same spirit
as `kappa_seq`'s own worst-case-permutation normalization (rather than a
simple analytical ceiling that a real composition can't reach; an earlier
version of `kappa_sym` used exactly such a ceiling, `(2*window)^2*(N-window+1)`,
which is why its values used to top out around `~0.48` instead of reaching
`1`).

The maximizing sequence is the single-transition palindrome
`[+1×(N/4), -1×(N/2), +1×(N/4)]` (requires `N` divisible by `4`) --
justified two ways: (1) analytically, for any exact palindrome
`Zᵢ+Z_{N+1-i}=2Zᵢ` identically, so `kappa_sym`'s raw sum reduces to `4×`
the sum of squared plain windowed charge sums, maximized by the most-blocky
(single-transition) balanced arrangement; (2) confirmed directly by brute
force (200,000 random shuffles of ten `+1`/ten `-1` beads for `N=20`: none
exceeded this reference's raw value of `1040`).
"""
function kappa_sym_max(N::Int; window::Int=6)
    N % 4 == 0 || throw(ArgumentError("kappa_sym_max's reference construction needs N divisible by 4"))
    ref = vcat(fill(1, N ÷ 4), fill(-1, N ÷ 2), fill(1, N ÷ 4))
    return _kappa_sym_raw(ref, window)
end

"""
    kappa_sym(charges; window=6)

A *windowed* mirror-asymmetry statistic, complementary to `mirror_symmetry`'s
single whole-sequence correlation the same way `kappa_seq`'s local blob
averaging complements a single global composition statistic:
```
kappa_sym = (1/kappa_sym_max(N)) * Σ_{j=1}^{N-window+1} ( Σ_{i=j}^{j+window-1} (Zᵢ + Z_{N+1-i}) )²
```
Each inner term `Zᵢ + Z_{N+1-i}` is exactly `0` whenever position `i` and its
mirror partner are anti-symmetric (`Zᵢ==-Z_{N+1-i}`) -- the specific,
rigorously-protected condition that forces zero interfacial electrostatic
potential (see `mirror_symmetry`'s docstring) -- and nonzero (`±2`) whenever
they are not (whether because they're palindromic, `Zᵢ==Z_{N+1-i}`, or just
uncorrelated). Squaring and summing over a sliding window therefore measures
*local deviation from the anti-palindrome condition specifically*, rather
than `mirror_symmetry`'s signed correlation, which spends a sign
distinguishing palindrome from anti-palindrome but can't (by construction,
being a single whole-sequence average) say anything about *where* along the
chain the reflective structure lives. `kappa_sym=0` exactly for `block1`
and `block10` (both global anti-palindromes, confirmed to give exactly zero
electrostatic potential) -- and, unlike `mirror_symmetry`, would also read
`0` for a sequence that is anti-palindromic in some *regions* but not
others, distinguishing that case from a globally-uncorrelated one that
`mirror_symmetry` alone cannot.

Normalized by `kappa_sym_max(N; window)`, the *true* achievable maximum for
a net-neutral sequence of this length (not a simple analytical ceiling --
see `kappa_sym_max`'s docstring), so `kappa_sym∈[0,1]` with `1` genuinely
reachable (by the single-transition palindrome `kappa_sym_max` itself is
built from).
"""
function kappa_sym(charges::AbstractVector{<:Integer}; window::Int=6)
    N = length(charges)
    raw = _kappa_sym_raw(charges, window)
    return raw / kappa_sym_max(N; window=window)
end

"""
    random_blocky_sequence(N, rng; max_block=4)

A net-neutral, equal +/- sequence of length `N` (`N` even) with no imposed
symmetry, built by concatenating random runs of length `1:max_block`
(alternating sign, run lengths clipped so the +/- counts never overshoot
`N÷2` each) -- i.e. genuinely disordered but still "blocky" rather than
single-bead-alternating, unlike `block_sequence`'s perfectly periodic
patterns. Intended for Step 3's asymmetric test sequences (`kappa_seq`
between the alternating and diblock limits, but without `block_sequence`'s
palindromic/periodic structure).
"""
function random_blocky_sequence(N::Int, rng::Random.AbstractRNG; max_block::Int=4)
    iseven(N) || throw(ArgumentError("N must be even for an equal +/- sequence"))
    half = N ÷ 2
    charges = Int[]
    remaining = [half, half]  # [remaining +, remaining -]
    sign = rand(rng, (1, -1))
    while length(charges) < N
        idx = sign > 0 ? 1 : 2
        run = min(rand(rng, 1:max_block), remaining[idx])
        if run == 0
            sign = -sign
            continue
        end
        append!(charges, fill(sign, run))
        remaining[idx] -= run
        sign = -sign
    end
    return charges
end

"""
    high_kappa_asymmetric_sequence(N, rng; n_anti=div(N,4), nswaps=0)

Targeted constructive generator for the region `random_blocky_sequence`
essentially never reaches by chance: high `kappa_seq` (strongly blocky)
*combined with* a deliberately-controlled `kappa_sym` spread, rather than
whatever combination falls out of pure random block generation.

For each mirror pair `(i, N+1-i)`, `i in 1:N÷2`, either **anti**-mirrors it
(`charges[N+1-i] = -charges[i]`, the `kappa_sym`-lowering rule) or
**palindromically** mirrors it (`charges[N+1-i] = +charges[i]`, generally
raising `kappa_sym`); `n_anti` of the `N÷2` pairs get the anti rule.
Net-neutrality of the *whole* sequence constrains how many of each first-half
sign can be anti- vs. palindrome-classified -- worked out fully generally
(not assuming a balanced first half) by writing the total `+` count as
`p_fh + s_plus + a_minus` (`p_fh` = first half's own `+` count, `s_plus`/
`a_minus` = palindrome/anti pairs with the relevant sign) and solving for
net-neutrality (`=N/2`): this forces `s_plus = (N/2-n_anti)/2` *regardless of
`p_fh`*, and therefore (for the *most blocky* -- i.e. maximally imbalanced --
first half `select_binodal_indices`-style construction below) `p_fh` itself
must be chosen as `(N/2+n_anti)/2`, letting `n_anti` come entirely from the
`+` side. This is the piece the first (now-superseded) version of this
function got wrong: it fixed the first half to always be perfectly balanced
(`block_sequence(N/2, N/4)`), which forces a periodic (not diblock) pattern
at `n_anti=N/2` (confirmed empirically: `kappa_seq≈0.4`, matching
`block_sequence(N,N/4)`, not the true diblock `kappa_seq=1`). Adapting the
first half's own composition to `n_anti` via the relation above instead lets
`n_anti=0` reach the exact `kappa_sym`-maximizing palindrome
(`[+N/4,-N/2,+N/4]`, confirmed analytically: for any exact palindrome
`S_j` reduces to `2×(plain windowed sum)`, so maximizing `kappa_sym` among
palindromes is exactly maximizing blockiness, achieved by the single-
transition first half) and `n_anti=N/2` reach the *exact* diblock
(`kappa_seq=1`, `kappa_sym=0`, matching `block_sequence(N,N/2)` exactly).

`nswaps` (random `+`/`-` position transpositions within the first half,
applied before pair classification) reintroduces controlled diversity across
random draws for a fixed `n_anti`, at the cost of some blockiness. Still
only *targets* a region of `(kappa_seq, kappa_sym)` space -- the actual
values need computing after construction.

`p_fh` (the first half's own `+` count) defaults to `(N/2+n_anti)/2`, the
most-imbalanced (hence most blocky) choice net-neutrality allows for a given
`n_anti` -- but that is only the *top* of a valid range,
`[(N/2-n_anti)/2, (N/2+n_anti)/2]` (derived from the same net-neutrality
argument above, without assuming the maximal-imbalance choice); any `p_fh`
in that range keeps the construction net-neutral. **This matters**: at the
default (fixed) `p_fh`, both `kappa_seq` and `kappa_sym` end up almost
entirely determined by `n_anti` alone (confirmed empirically: at `nswaps=0`,
`n_anti=6` always gives `kappa_seq≈0.77` with `kappa_sym` confined to
`[0.10,0.38]` -- it cannot reach, say, `0.5` or `0.8` at that `kappa_seq` no
matter how many times you resample). Varying `p_fh` across its valid range
(not just its default endpoint) is the actual extra degree of freedom needed
to decouple `kappa_seq` and `kappa_sym` and cover the plane between the
`n_anti`-anchored bands -- not a fundamental limitation of `N=20` itself
(`kappa_sym`'s raw values are integers over a denominator of `1040`, i.e.
resolution `~0.001`, far finer than the gaps a fixed-`p_fh` sweep leaves).
"""
function high_kappa_asymmetric_sequence(N::Int, rng::Random.AbstractRNG; n_anti::Int=div(N, 4),
                                         nswaps::Int=0, p_fh::Union{Int,Nothing}=nothing)
    iseven(N) || throw(ArgumentError("N must be even for an equal +/- sequence"))
    half = N ÷ 2
    iseven(n_anti) || throw(ArgumentError("n_anti must be even to keep the sequence net-neutral"))
    0 <= n_anti <= half || throw(ArgumentError("n_anti must be in [0, $half]"))

    n_same = half - n_anti
    p_fh_lo, p_fh_hi = (half - n_anti) ÷ 2, (half + n_anti) ÷ 2
    p_fh = something(p_fh, p_fh_hi)
    p_fh_lo <= p_fh <= p_fh_hi || throw(ArgumentError("p_fh=$p_fh outside the net-neutral-feasible range [$p_fh_lo, $p_fh_hi] for n_anti=$n_anti"))
    first_half = vcat(fill(1, p_fh), fill(-1, half - p_fh))  # single transition: maximally blocky given p_fh
    for _ in 1:nswaps
        pp, mp = findall(==(1), first_half), findall(==(-1), first_half)
        (isempty(pp) || isempty(mp)) && break  # single-sign first half (e.g. n_anti=half): nothing to swap
        i, j = rand(rng, pp), rand(rng, mp)
        first_half[i], first_half[j] = first_half[j], first_half[i]
    end

    # General net-neutrality split (not assuming the default max-imbalance
    # p_fh): exactly n_same/2 "+"-valued AND n_same/2 "-"-valued pairs must
    # be palindrome-classified regardless of p_fh; the remaining n_anti
    # anti pairs split as a_plus from the "+" side and a_minus=n_anti-a_plus
    # from the "-" side, with a_plus=p_fh-n_same/2 (reduces to "all from +"
    # exactly when p_fh is at its default upper bound, recovering the
    # original special case).
    n_same_half = n_same ÷ 2
    a_plus = p_fh - n_same_half
    a_minus = n_anti - a_plus
    plus_pos, minus_pos = findall(==(1), first_half), findall(==(-1), first_half)
    anti_set = Set(vcat(Random.shuffle(rng, plus_pos)[1:a_plus],
                         Random.shuffle(rng, minus_pos)[1:a_minus]))

    charges = Vector{Int}(undef, N)
    charges[1:half] = first_half
    for i in 1:half
        charges[N + 1 - i] = i in anti_set ? -first_half[i] : first_half[i]
    end
    return charges
end

"""
    perturbed_diblock_sequence(N, rng; nswaps=1)

Start from the exact diblock (`block_sequence(N, N÷2)`, `kappa_seq=1`) and
apply `nswaps` random `+`/`-` position transpositions anywhere in the
sequence, net-neutral by construction (a swap just exchanges which position
holds which sign). The most direct way to sample sequences that are only
*slightly* disordered relative to the diblock, rather than
`high_kappa_asymmetric_sequence`'s from-scratch construction. Empirically
(500-trial sweeps), `nswaps=1` keeps `kappa_seq>0.75` about a quarter of the
time (reaching `kappa_sym` up to ~0.06); `nswaps=2` about 8% of the time
(reaching `kappa_sym` up to ~0.16); higher `nswaps` rapidly become rare.
Restricting the swapped positions to a small window around the block
boundary (tried first) is *worse*, not better, for keeping `kappa_seq` high
at fixed `nswaps`: concentrating multiple defects inside a region comparable
to `kappa_seq`'s own blob window (5-6 beads) does more damage per swap than
spreading them across the whole chain.
"""
function perturbed_diblock_sequence(N::Int, rng::Random.AbstractRNG; nswaps::Int=1)
    iseven(N) || throw(ArgumentError("N must be even for an equal +/- sequence"))
    seq = block_sequence(N, N ÷ 2)
    for _ in 1:nswaps
        i = rand(rng, findall(==(1), seq))
        j = rand(rng, findall(==(-1), seq))
        seq[i], seq[j] = seq[j], seq[i]
    end
    return seq
end
