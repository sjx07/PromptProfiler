import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union


NodeType = Literal["section", "rule", "group"]
logger = logging.getLogger(__name__)


@dataclass
class RuleItem:
    """
    Atomic rule unit with metadata for attribution / traceability.
    """
    text: str
    rule_id: Optional[str] = None
    rule_kind: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    node_id: Optional[str] = None
    parent_id: Optional[str] = None


@dataclass
class RuleGroup:
    """
    A named group of rules inside a section (e.g., "Brand Consistency").
    """
    title: str
    level: int
    group_id: Optional[str] = None
    children: List[Union["RuleItem", "RuleSection"]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_id: Optional[str] = None


@dataclass
class RuleSection:
    title: str
    level: int = 1
    content: str = ""

    children: List[Union["RuleItem", "RuleGroup", "RuleSection"]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    node_id: Optional[str] = None
    parent_id: Optional[str] = None

    def iter_rule_children(self) -> List[Union["RuleItem", "RuleGroup"]]:
        """Return non-section children (RuleItem and RuleGroup only).

        This replaces iter_rule_nodes_in_order() -- ordering is now implicit
        in the children list, so no metadata['rule_nodes_order'] needed.
        """
        return [c for c in self.children if not isinstance(c, RuleSection)]


def _get_node_id(node: Union[RuleSection, RuleGroup, RuleItem]) -> Optional[str]:
    """Extract the identifier from any node type."""
    if isinstance(node, RuleSection):
        return node.node_id
    elif isinstance(node, RuleGroup):
        return node.group_id
    elif isinstance(node, RuleItem):
        return node.node_id or node.rule_id
    return None


@dataclass
class RuleMask:
    """Tracks enabled/disabled state for rule nodes, separate from the tree structure."""
    _mask: Dict[str, bool] = field(default_factory=dict)

    @classmethod
    def all_enabled(cls, tree: "RuleTree") -> "RuleMask":
        """Build a mask with all rule node_ids set to True."""
        mask = cls()
        for node_id, node in tree._index.items():
            if isinstance(node, RuleItem):
                mask._mask[node_id] = True
        return mask

    def toggle_subtree(self, tree: "RuleTree", node_id: str, enabled: bool) -> None:
        """Enable/disable all atomic rules under a subtree."""
        for nid in tree.iter_subtree_ids(node_id):
            node = tree._index.get(nid)
            if isinstance(node, RuleItem):
                self._mask[nid] = enabled

    def apply(self, updates: Dict[str, bool]) -> None:
        """Bulk update mask entries."""
        for rid, en in updates.items():
            self._mask[rid] = bool(en)

    def is_rule_enabled(self, rule_id: str) -> bool:
        """Check if a specific rule is enabled. Missing key defaults to True."""
        return self._mask.get(rule_id, True)

    def is_node_enabled(self, tree: "RuleTree", node_id: str) -> bool:
        """Check if a node is enabled.

        For RuleItem: direct mask lookup.
        For section/group: True if any descendant rule is enabled.
        TODO: discuss case for section/group
        """
        node = tree._index.get(node_id)
        if node is None:
            return False
        if isinstance(node, RuleItem):
            return self.is_rule_enabled(node_id)
        # sections/groups are considered enabled if they have any enabled rules
        for nid in tree.iter_subtree_ids(node_id):
            n = tree._index.get(nid)
            if isinstance(n, RuleItem) and self.is_rule_enabled(nid):
                return True
        return False

    def enabled_rule_ids(self, tree: "RuleTree") -> List[str]:
        """Return sorted list of enabled rule IDs."""
        out = []
        for nid, node in tree._index.items():
            if isinstance(node, RuleItem) and self.is_rule_enabled(nid):
                out.append(nid)
        out.sort()
        return out

    def snapshot(self) -> Dict[str, bool]:
        """Return a copy of the current mask state."""
        return dict(self._mask)

    def restore(self, mask: Dict[str, bool]) -> None:
        """Replace mask state with a snapshot."""
        self._mask = dict(mask)


@dataclass
class RuleTree:
    """Single tree structure replacing TreeIndex + NodeRef.

    Owns the recursive tree (roots) and provides O(1) lookup via a derived index.
    """
    roots: List[RuleSection] = field(default_factory=list)
    mask: RuleMask = field(default_factory=RuleMask)
    _index: Dict[str, Union[RuleSection, RuleGroup, RuleItem]] = field(default_factory=dict)

    def _rebuild_index(self) -> None:
        """DFS walk roots -> children, populate _index from node_id -> obj."""
        self._index.clear()

        def walk(node: Union[RuleSection, RuleGroup, RuleItem]) -> None:
            nid = _get_node_id(node)
            if nid:
                self._index[nid] = node
            if isinstance(node, (RuleSection, RuleGroup)):
                for child in node.children:
                    walk(child)

        for root in self.roots:
            walk(root)

    def get(self, node_id: str) -> Optional[Union[RuleSection, RuleGroup, RuleItem]]:
        """O(1) lookup by node ID."""
        return self._index.get(node_id)

    def iter_subtree_ids(self, node_id: str) -> List[str]:
        """Return node_id plus all descendant IDs (DFS)."""
        if node_id not in self._index:
            return []
        out: List[str] = []
        stack = [node_id]
        while stack:
            cur = stack.pop()
            out.append(cur)
            node = self._index.get(cur)
            if node is None:
                continue
            # Collect children IDs from actual children lists
            if isinstance(node, (RuleSection, RuleGroup)):
                children_ids = []
                for child in node.children:
                    cid = _get_node_id(child)
                    if cid:
                        children_ids.append(cid)
                # reverse for DFS order consistency
                for cid in reversed(children_ids):
                    stack.append(cid)
        return out

    def is_enabled(self, node_id: str) -> bool:
        """Delegates to self.mask.is_node_enabled(self, node_id)."""
        return self.mask.is_node_enabled(self, node_id)

    @property
    def section_ids(self) -> List[str]:
        """Return IDs of all top-level sections (roots), preserving order."""
        return [s.node_id for s in self.roots if s.node_id]


def _norm_text(s: str) -> str:
    return " ".join((s or "").strip().split())


def parse_sections(
    data: Union[Dict[str, Any], List[Dict[str, Any]]],
    *,
    drop_duplicate_text_rules: bool = True,
) -> "RuleTree":
    """
    Accepts either:
      - a Dict containing {"sections": [...]}
      - a List[section_dict]

    Returns:
      RuleTree with roots, derived index, and all-enabled mask.
    """

    # -------- normalize input --------
    if isinstance(data, dict):
        sections_data = data.get("sections", []) or []
        if not isinstance(sections_data, list):
            raise TypeError("data['sections'] must be a list")
    elif isinstance(data, list):
        sections_data = data
    else:
        raise TypeError(f"Unsupported input type: {type(data)}")

    def node_id_of(d: Dict[str, Any]) -> Optional[str]:
        return d.get("id") or d.get("path")

    def parse_section(sec_d: Dict[str, Any], parent_id: Optional[str]) -> RuleSection:
        sid = node_id_of(sec_d)
        if not sid:
            raise ValueError(f"Section missing 'id' or 'path': {sec_d}")
        title = sec_d.get("title", "") or ""
        level = int(sec_d.get("level", 1) or 1)
        content = sec_d.get("content", "") or ""

        # metadata: keep everything except core fields
        metadata = {k: v for k, v in sec_d.items() if k not in ["title", "level", "content", "children"]}
        children_data = sec_d.get("children", []) or []

        content_norm = _norm_text(content)

        sec_obj = RuleSection(
            title=title,
            level=level,
            content=content,
            children=[],
            metadata=metadata,
            node_id=sid,
            parent_id=parent_id,
        )

        # Build ordered children: (order_key, node)
        children_ordered: List[Tuple[int, Union[RuleItem, RuleGroup, RuleSection]]] = []
        fallback_counter = 1

        def child_order(ch: Dict[str, Any], fallback: int) -> int:
            o = ch.get("order", None)
            return o if isinstance(o, int) else fallback

        for ch in children_data:
            ctype = ch.get("node_type")
            cid = node_id_of(ch)
            if not ctype or not cid:
                raise ValueError(f"Child node missing 'node_type' or 'id/path': {ch}")

            # ---- atomic rule ----
            if ctype == "rule":
                txt = ch.get("content", "") or ""
                kind = ch.get("rule_kind", None)
                txt_norm = _norm_text(txt)

                # drop duplicate "text" rule repeating section content
                if drop_duplicate_text_rules and kind == "text" and txt_norm and txt_norm == content_norm:
                    fallback_counter += 1
                    continue

                ri = RuleItem(
                    text=txt,
                    rule_id=cid,
                    rule_kind=kind,
                    metadata={k: v for k, v in ch.items() if k not in ["content"]},
                    node_id=cid,
                    parent_id=sid,
                )

                o = child_order(ch, fallback_counter)
                children_ordered.append((o, ri))
                fallback_counter += 1
                continue

            # ---- group ----
            if ctype == "rule_group":
                gtitle = ch.get("title", "") or ""
                glevel = int(ch.get("level", level + 1) or (level + 1))
                gmeta = {k: v for k, v in ch.items() if k not in ["title", "level", "content", "children"]}
                group_children: List[Union[RuleItem, RuleSection]] = []

                for gc in (ch.get("children") or []):
                    if gc.get("node_type") == "rule":
                        rid = node_id_of(gc)
                        if not rid:
                            raise ValueError(f"Group child rule missing 'id' or 'path': {gc}")
                        rtxt = gc.get("content", "") or ""
                        rkind = gc.get("rule_kind", None)

                        gri = RuleItem(
                            text=rtxt,
                            rule_id=rid,
                            rule_kind=rkind,
                            metadata={k: v for k, v in gc.items() if k not in ["content"]},
                            node_id=rid,
                            parent_id=cid,
                        )
                        group_children.append(gri)
                    elif gc.get("children"):
                        # rare: nested section-like inside group
                        sub = parse_section(gc, parent_id=cid)
                        # preserve current behavior: add to parent section's children
                        children_ordered.append((child_order(gc, fallback_counter), sub))

                gobj = RuleGroup(
                    title=gtitle,
                    level=glevel,
                    group_id=cid,
                    children=group_children,
                    metadata=gmeta,
                    parent_id=sid,
                )

                o = child_order(ch, fallback_counter)
                children_ordered.append((o, gobj))
                fallback_counter += 1
                continue

            # ---- subsection ----
            if ch.get("children"):
                sub = parse_section(ch, parent_id=sid)
                children_ordered.append((child_order(ch, fallback_counter), sub))
                fallback_counter += 1
                continue

            fallback_counter += 1

        if not children_data:
            leaf_rule_id = f"{sid}/text" if sid else "text"
            ri = RuleItem(
                text=content,
                rule_id=leaf_rule_id,
                rule_kind="text",
                metadata={},
                node_id=leaf_rule_id,
                parent_id=sid,
            )
            children_ordered.append((1, ri))

        children_ordered.sort(key=lambda x: x[0])
        sec_obj.children = [x[1] for x in children_ordered]

        return sec_obj

    # parse top-level sections
    roots: List[RuleSection] = []
    for s in sections_data or []:
        sec = parse_section(s, parent_id=None)
        roots.append(sec)

    # Build tree with derived index and all-enabled mask
    tree = RuleTree(roots=roots)
    tree._rebuild_index()
    tree.mask = RuleMask.all_enabled(tree)

    return tree


def parse_sections_with_index(
    data: Union[Dict[str, Any], List[Dict[str, Any]]],
    *,
    drop_duplicate_text_rules: bool = True,
) -> Tuple[List[RuleSection], "RuleTree"]:
    """
    Backward-compat wrapper.

    Accepts either:
      - a Dict containing {"sections": [...]}
      - a List[section_dict]

    Returns:
      (sections, tree) -- tree is a RuleTree (replaces old TreeIndex).
    """
    tree = parse_sections(data, drop_duplicate_text_rules=drop_duplicate_text_rules)
    return tree.roots, tree


def iter_enabled_group_rules(g: RuleGroup, mask: "RuleMask") -> List[RuleItem]:
    return [
        ri for ri in g.children
        if isinstance(ri, RuleItem) and mask.is_rule_enabled(_get_node_id(ri) or '') and (ri.text or "").strip()
    ]


def has_any_enabled_rules_in_section(sec: RuleSection, mask: "RuleMask") -> bool:
    for node in sec.iter_rule_children():
        if isinstance(node, RuleItem) and mask.is_rule_enabled(_get_node_id(node) or '') and (node.text or "").strip():
            return True
        if isinstance(node, RuleGroup) and iter_enabled_group_rules(node, mask):
            return True
    for child in sec.children:
        if isinstance(child, RuleSection) and has_any_enabled_rules_in_section(child, mask):
            return True
    return False


def _segment_to_section(
    segment: Dict[str, Any],
    *,
    prompt_id: str,
    segment_index: int,
) -> Dict[str, Any]:
    """Convert a collector segment block to section format."""
    section_id = f"{prompt_id}/seg_{segment_index}"
    children: List[Dict[str, Any]] = []
    atomic_rules = segment.get("atomic_rules") or []

    for i, rule in enumerate(atomic_rules):
        children.append(
            {
                "node_type": "rule",
                "id": f"{section_id}/{rule.get('rule_id') or f'rule_{i}'}",
                "content": rule.get("text", ""),
                "rule_kind": rule.get("rule_kind") or "rule",
                "order": i + 1,
            }
        )

    text = segment.get("text", "") or segment.get("content", "")
    if not children and text.strip():
        children.append(
            {
                "node_type": "rule",
                "id": f"{section_id}/text",
                "content": text.strip(),
                "rule_kind": "text",
                "order": 1,
            }
        )

    return {
        "id": section_id,
        "path": section_id,
        "title": segment.get("functional_component", "") or segment.get("title", "segment"),
        "level": int(segment.get("level", 1) or 1),
        "content": text,
        "children": children,
        "functional_component": segment.get("functional_component"),
        "is_system": segment.get("is_system", True),
        "start_line": segment.get("start_line", 0),
        "end_line": segment.get("end_line", 0),
    }


def _extract_prompt_sections(prompt: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Get sections from collector prompt item (supports both sections and segments)."""
    sections = prompt.get("sections")
    if isinstance(sections, list):
        return list(sections)

    segments = prompt.get("segments")
    if isinstance(segments, list):
        prompt_id = str(prompt.get("id") or "prompt")
        return [
            _segment_to_section(seg, prompt_id=prompt_id, segment_index=i)
            for i, seg in enumerate(segments)
        ]
    return []


def _normalize_component_filter(
    functional_components: Optional[Union[str, Sequence[str]]],
) -> Optional[set[str]]:
    if functional_components is None:
        return None

    if isinstance(functional_components, str):
        parts = [p.strip() for p in functional_components.split(",")]
    else:
        parts = [str(p).strip() for p in functional_components]

    normalized = {p for p in parts if p}
    return normalized or None


def _score_prompt_for_selection(prompt: Dict[str, Any]) -> Tuple[float, float, int, str]:
    """Deterministic prompt ranking when multiple candidates match."""
    status_score = 1.0 if prompt.get("analysis_status") == "success" else 0.0
    try:
        confidence = float(prompt.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    section_count = len(_extract_prompt_sections(prompt))
    prompt_id = str(prompt.get("id", ""))
    return (status_score, confidence, section_count, prompt_id)


def _select_collector_prompts(
    prompts: List[Dict[str, Any]],
    *,
    prompt_id: Optional[str],
    prompt_name: Optional[str],
    prompt_index: Optional[int],
    semantic_stage: Optional[str],
    semantic_subtype: Optional[str],
    repo_name: Optional[str],
    analysis_status: Optional[str],
) -> List[Dict[str, Any]]:
    selected = list(prompts)

    if analysis_status:
        selected = [p for p in selected if p.get("analysis_status") == analysis_status]
    if semantic_stage:
        selected = [p for p in selected if p.get("semantic_stage") == semantic_stage]
    if semantic_subtype:
        selected = [p for p in selected if p.get("semantic_subtype") == semantic_subtype]
    if repo_name:
        selected = [p for p in selected if p.get("repo_name") == repo_name]
    if prompt_id:
        selected = [p for p in selected if str(p.get("id")) == str(prompt_id)]
    if prompt_name:
        selected = [p for p in selected if str(p.get("name")) == str(prompt_name)]

    if prompt_index is not None:
        if prompt_index < 0 or prompt_index >= len(selected):
            raise ValueError(
                f"prompt_index={prompt_index} is out of range for {len(selected)} matched prompts"
            )
        selected = [selected[prompt_index]]

    return selected


def _load_sections_from_collector_prompts(
    data: Dict[str, Any],
    *,
    prompt_id: Optional[str],
    prompt_name: Optional[str],
    prompt_index: Optional[int],
    semantic_stage: Optional[str],
    semantic_subtype: Optional[str],
    repo_name: Optional[str],
    functional_components: Optional[Union[str, Sequence[str]]],
    merge_prompts: bool,
    analysis_status: Optional[str],
) -> List[Dict[str, Any]]:
    prompts = data.get("prompts")
    if not isinstance(prompts, list):
        raise ValueError("Expected a 'prompts' list in collector analyzed JSON")

    selected = _select_collector_prompts(
        prompts,
        prompt_id=prompt_id,
        prompt_name=prompt_name,
        prompt_index=prompt_index,
        semantic_stage=semantic_stage,
        semantic_subtype=semantic_subtype,
        repo_name=repo_name,
        analysis_status=analysis_status,
    )
    if not selected:
        raise ValueError("No prompts matched the collector selection filters")

    if not merge_prompts and len(selected) > 1:
        ranked = sorted(selected, key=_score_prompt_for_selection, reverse=True)
        chosen = ranked[0]
        logger.warning(
            "Multiple prompts matched (%d). Using highest-confidence prompt: id=%s name=%s",
            len(selected),
            chosen.get("id"),
            chosen.get("name"),
        )
        selected = [chosen]

    component_filter = _normalize_component_filter(functional_components)
    sections_data: List[Dict[str, Any]] = []
    for prompt in selected:
        for section in _extract_prompt_sections(prompt):
            if component_filter:
                comp = (
                    section.get("functional_component")
                    or section.get("title")
                    or ""
                )
                if comp not in component_filter:
                    continue
            sections_data.append(section)

    if not sections_data:
        raise ValueError("No sections remained after collector prompt filtering")

    return sections_data


def load_sections_from_classification_results(
    filepath: str,
    *,
    prompt_id: Optional[str] = None,
    prompt_name: Optional[str] = None,
    prompt_index: Optional[int] = None,
    semantic_stage: Optional[str] = None,
    semantic_subtype: Optional[str] = None,
    repo_name: Optional[str] = None,
    functional_components: Optional[Union[str, Sequence[str]]] = None,
    merge_prompts: bool = False,
    analysis_status: Optional[str] = "success",
) -> "RuleTree":
    """Load rule sections from classification results JSON.

    Supported input formats:
    - {"sections": [...]} rule tree
    - [{"sections": [...], "score"/"Score": ...}, ...] scored candidates
    - DomainPromptCollector analyzed JSON: {"prompts": [...]}

    For collector files, you can select prompts via filters and optionally
    merge multiple prompts into one section set.

    Note: duplicate text rules are preserved for attribution traceability.
    """
    import json
    from pathlib import Path

    path = Path(filepath)
    with open(path, "r") as f:
        data = json.load(f)

    sections_data: List[Dict[str, Any]]

    # 1) Handle scored list format: pick highest score if present
    if isinstance(data, list):
        if data and "sections" in data[0]:
            score_max = 0.0
            sections_data = data[0]["sections"]
            for entry in data:
                if "Score" in entry or "score" in entry:
                    score = entry.get("Score", entry.get("score", 0.0))
                    if score > score_max:
                        score_max = score
                        sections_data = entry["sections"]
        else:
            raise ValueError(f"Expected 'sections' key in first list element")

    # 2) Plain {"sections": ...} format
    elif isinstance(data, dict) and "sections" in data:
        sections_data = data["sections"]

    # 3) DomainPromptCollector analyzed format
    elif isinstance(data, dict) and "prompts" in data:
        sections_data = _load_sections_from_collector_prompts(
            data,
            prompt_id=prompt_id,
            prompt_name=prompt_name,
            prompt_index=prompt_index,
            semantic_stage=semantic_stage,
            semantic_subtype=semantic_subtype,
            repo_name=repo_name,
            functional_components=functional_components,
            merge_prompts=merge_prompts,
            analysis_status=analysis_status,
        )

    else:
        raise ValueError(f"Unexpected format in {filepath}")

    if not isinstance(sections_data, list):
        raise TypeError("Resolved sections data must be a list")

    return parse_sections(sections_data, drop_duplicate_text_rules=False)

# Backward-compat alias for attribution modules that import TreeIndex
TreeIndex = RuleTree
