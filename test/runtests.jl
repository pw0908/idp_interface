using Test

# test_LS_validation.jl include()s resources/Helmholtz_LS.jl, which defines a
# bare `LS` function in this file's scope -- run it last so it doesn't shadow
# `IDPInterface.LS` for any other test file (which must otherwise qualify
# every call as `IDPInterface.LS`, as test_LS_validation.jl itself does).
include("test_sequences.jl")
include("test_DFT_bulk.jl")
include("test_LS_validation.jl")
