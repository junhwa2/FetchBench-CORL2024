"""VLM+VORM closed-loop (ThinkGrasp-style) integration for FetchBench.

Groups the pieces that were previously scattered across the fetch package:
  sim_rpc.py       - file-RPC transport (server side; the vorm master mirrors it)
  reveal_client.py - Gemini-ER reveal-reasoning VLM client (grasp vs remove)
  server_task.py   - FetchMeshCuroboGORunVLMServer (reset/observe/decide/execute)
  server_entry.py  - hydra entry that builds the env once and serves the loop

See VLM_SIM_RPC_SPEC.md for the protocol and how to launch it.
"""
