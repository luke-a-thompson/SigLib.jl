module SigLib

include("library.jl")
include("errors.jl")
include("buffers.jl")
include("c_api.jl")
include("lengths.jl")
include("bases.jl")
include("cache.jl")
include("signatures.jl")
include("branched_signatures.jl")
include("log_signatures.jl")
include("log_sig_workspace.jl")
include("branched_log_signatures.jl")

const signature = sig
const signature_length = sig_length
const branched_signature = branched_sig
const branched_signature_length = branched_sig_length

export sig, sig_length, signature, signature_length
export clear_log_sig_cache, set_cache_dir, log_sig, log_sig_combine, log_sig_join
export log_sig_length, logsig_to_sig, prepare_log_sig, sig_to_log_sig
export LogSigWorkspace, log_sig!
export branched_sig, branched_sig_combine, branched_sig_length, branched_sig_to_log_sig
export branched_log_sig, branched_log_sig_length, prepare_branched_sig, prepare_branched_log_sig
export branched_signature, branched_signature_length
export SigLibError, CPSigError
export words, words_of_length, lyndon_words, lyndon_words_of_length, is_lyndon
export word_to_idx, idx_to_word, lyndon_bracket, lyndon_brackets, log_sig_basis
export trees, trees_of_order, rooted_trees, tree_to_idx, idx_to_tree, branched_log_sig_basis

end
