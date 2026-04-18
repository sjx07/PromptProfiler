# scripts/list_enable_code_responses.py
from core.store import CubeStore
from analyze import ExecutionQuery

store = CubeStore("runs/wtq_mvp.db", read_only=True)
# print(store.list_configs())  # warm up the store and print available configs
rows = (ExecutionQuery(store)
        .has_feature("enable_cot")
        .columns(["execution_id", "query_id", "config_id", "raw_response"])
        .rows()[:10])

for r in rows:
    print(f"[{r['execution_id']}] cfg={r['config_id']} q={r['query_id']}")
    print(r["raw_response"])
    print("-" * 60)