"""SkVM tasks — ports of SkVM microbenchmark generators as prompt_profiler tasks.

Currently wired:
  - structured_gen: gen.text.structured L3 (deeply nested JSON generation)

Each primitive lives in its own task module (class_name -> BaseTask subclass
with a distinct `name` used as the cube `dataset` and the feature `task`).
"""
