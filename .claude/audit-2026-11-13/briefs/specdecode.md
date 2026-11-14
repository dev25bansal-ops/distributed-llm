# Fix brief: specdecode

Audit findings for the files in your exclusive scope. For EACH finding:
reproduce-or-refute first (run the check command or read the cited code
carefully), then either FIX it or record an explicit refutation. Never
leave a finding unaddressed. After fixing, extend or add a test in the
named test file and run it. Do NOT commit; leave changes in the tree.

## [crit] src/distllm/core/distributed_speculative.py:1297
**Summary**: DistributedSpeculativeDecoder._verify_tokens sampled path calls the shared accept_token kernel with logits row prefix_len+i-1 (one row early) while its own inline p computation uses prefix_len+i, so the accept/reject decision is made from the wrong position's distribution.
**Failure scenario**: With a remote draft (draft_logprobs available, temperature>0 - the default mode of this decoder), the verifier consults row L-2+i instead of L-1+i. A stub with p(draft token)=4.5e-5 at its own row is ACCEPTED (accepted_count=1), and a token the target endorses with p=0.9993 at its own row is REJECTED (accepted_count=0). Every sampled acceptance decision is therefore drawn from the wrong distribution, silently corrupting generated text whenever distributed speculative decoding runs. The inline comment claims accept_token 'recomputes the identical decision' - it does not (regression from the M-02 refactor).
**Check command**: `python -c "import torch, math; from distllm.core.distributed_speculative import DistributedSpeculativeDecoder as D; d = D(target_forward=lambda i, **k: None, draft_model=object(), temperature=1.0, device='cpu'); t = torch.zeros(1, 4, 16); t[0, 1, 7] = 10.0; t[0, 2, 9] = 10.0; n = d._verify_tokens(torch.tensor([[10, 11, 12]]), torch.tensor([[7]]), t, draft_logprobs=[math.log(0.999)]); print('accepted', n)"  # prints 'accepted 1'; expected 0 - token 7 has p=4.5e-5 at its own row (row 2)`
**Reproduced by auditor**: True

## [high] src/distllm/core/speculative_decoder.py:257
**Summary**: _draft_forward records the draft probability q from the UNMASKED softmax while the token itself is sampled from the top-k masked (renormalized) distribution, so verification runs min(1, p/q) against the wrong q and over-accepts.
**Failure scenario**: Default constructor has top_k=20, temperature=1.0. With top_k=1 the draft always proposes the top token (true proposal probability 1.0) but q is recorded as the unmasked softmax value (0.3), so acceptance min(1, p/q) becomes 1.0 instead of min(1, p) despite p=0.70: reproduced 300/300 acceptances where exact speculative sampling gives ~70%. The output distribution no longer matches the target model's - the core correctness guarantee of speculative decoding is broken (same pattern in SelfSpeculativeDecoder._draft_forward, lines 811-816).
**Check command**: `python .tmp-audit-specverify.py  # section C2: 'accepted 300/300 trials' with p_target=0.70, expected ~70%`
**Reproduced by auditor**: True

## [med] src/distllm/core/speculative_decoder.py:513
**Summary**: verify_batch's in-loop EOS check `break`s before appending a per-sequence result, so the returned results list is shorter than the batch - the caller's per-sequence token mapping desyncs and torch.tensor([r[2] for r in results]) produces too few next tokens.
**Failure scenario**: Two batched sequences whose draft tokens hit the tokenizer EOS mid-chain: the for-loop breaks out of the EOS-matching sequence's verification without appending, so results == [] (reproduced; len 0 instead of 2). scheduler.step then indexes next_tokens[i] for every sequence in the batch - IndexError whenever any sequence's draft touches EOS, or silently misaligned tokens when some other sequence did produce a result.
**Check command**: `python .tmp-audit-specstats.py  # section A prints: results: [] -> len: 0 (expected 2)`
**Reproduced by auditor**: True

## [med] src/distllm/core/speculative_decoder.py:492
**Summary**: verify_batch indexes logits2d[i] (row i) to verify draft token i, but request_pipeline passes FULL per-sequence logits (prefill rows = prompt positions), where draft token 0 must be verified against row L-1; interior prompt rows are consulted instead.
**Failure scenario**: A prefill sequence of length 4 whose draft token 7 is endorsed exactly by row 3 (the row that predicts it): verify_batch returns (0, [9], 9) - the endorsed draft is rejected AND the emitted 'next token' 9 comes from row 0, the first prompt position's distribution, so the very first generated token is corrupted. Both callers are affected: request_pipeline.py:614 ([all_logits[i:i+1]]) and :736 ([last_logits[i:i+1]] for prefill sequences in mixed batches).
**Check command**: `python -c "import torch; from distllm.core.speculative_decoder import SpeculativeDecoder; L=torch.zeros(1,4,16); L[0,:3,9]=10.0; L[0,3,7]=10.0; print(SpeculativeDecoder(device='cpu').verify_batch([[7]], [L]))"  # prints [(0, [9], 9)]; expected to accept draft 7`
**Reproduced by auditor**: True

## [med] src/distllm/core/speculative_decoder.py:442
**Summary**: verify_batch ignores its temperature parameter and draft_logits_list entirely - it always does greedy argmax-match acceptance and emits the target argmax as the next token, so the batch speculative paths silently ignore the request's sampling temperature.
**Failure scenario**: A batch request with temperature=1.0 (or any sampled config) goes through _generate_local_batch/_run_async_pipeline_batch (request_pipeline.py:614, :736) which never pass temperature, and verify_batch has no sampled branch anyway: every decision is argmax-vs-draft and next_token is always an argmax - sampling degenerates to greedy, unlike the non-spec path which calls self._sample with the request temperature. The draft_logits_list argument that was carefully generated for rejection sampling is dead weight. Also, even with temperature explicitly passed, identical decisions result: temperature=0.0 and temperature=1.0 with q~0 return the same (1, [5], 5) although a rejection sampler must reject q<=0.
**Check command**: `python -c "import torch; from distllm.core.speculative_decoder import SpeculativeDecoder; L=torch.zeros(1,1,16); L[0,0,5]=10.0; d=SpeculativeDecoder(device='cpu'); print(d.verify_batch([[5]], [L], temperature=0.0), d.verify_batch([[5]], [L], temperature=1.0, draft_logits_list=[torch.zeros(1,16)]))"  # both print (1, [5], 5)`
**Reproduced by auditor**: True

## [med] src/distllm/core/speculative_decoder.py:1016
**Summary**: MultiDraftSpeculativeDecoder computes accepted = generated.shape[1] - prompt_len (every emitted token, including target-fallback correction tokens from rounds with zero consensus and full-rejection rounds) against total_proposed = sum(consensus_lengths), so acceptance-rate statistics count the wrong path and can be inflated to 1.0 (or be absent) with zero real draft acceptances.
**Failure scenario**: A run where the consensus draft is proposed but rejected by the target, and the emitted token is the target's correction: reported stats show accepted=1, total_proposed=1, acceptance_rate=1.0, while true draft acceptances = 0. Rounds with consensus_len==0 (target-fallback tokens) also inflate 'accepted'. Consumers of these stats (adaptive speculation length, auto-disable thresholds, dashboards) are misled into thinking the draft is perfect.
**Check command**: `python .tmp-audit-specstats.py  # section B prints acceptance_rate 1.0, true draft acceptances 0`
**Reproduced by auditor**: True
