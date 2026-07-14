# Lessons

- Do not call a migration complete because new contracts wrap an old path. Verify that default
  owned, injected, pooled, and explicit execution actually traverse the new contract without
  fabricated bindings or legacy planners.
- An async protocol is not async-native when its maintained implementation delegates blocking
  work through `asyncio.to_thread()`. Use native async backend operations; isolate thread bridges
  to explicitly caller-supplied legacy synchronous integrations.
- Architecture reviews must inspect the runtime call graph, not only contract definitions and
  passing tests.
- Keep async compatibility bridges at explicit external boundaries. Maintained backends and pool
  lifecycles must implement the async contract natively rather than inheriting a sync core.
- Adapter SPIs should provide normalized, named field information directly. Do not make each
  adapter parse `Optional`, `Annotated`, and `list` independently or recover field identity from
  mutable run-context state.
- Do not increment a version for an unshipped contract. Version changes represent compatibility
  with a released or otherwise consumed interface, not iterations within an uncommitted refactor.
