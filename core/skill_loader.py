"""Versioned, validated business-skill loading for ContentOpsMind."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from prometheus_client import Counter

logger = logging.getLogger(__name__)

SKILL_LOAD_FAILURES = Counter(
    "contentopsmind_skill_load_failures_total", "Skill load failures"
)
SKILL_MATCHES = Counter(
    "contentopsmind_skill_matches_total",
    "Selected skills",
    ["skill_id", "version", "agent"],
)
SKILL_SCHEMA_FAILURES = Counter(
    "contentopsmind_skill_schema_failures_total", "Skill schema failures", ["skill_id"]
)
SKILL_OUTPUT_REPAIRS = Counter(
    "contentopsmind_skill_output_repairs_total", "Skill output repairs", ["skill_id"]
)


@dataclass(frozen=True)
class SkillSelection:
    skill_id: str
    name: str
    version: str
    priority: int
    match_reasons: List[str]
    rule_ids: List[str]
    prompt_block: str
    truncated: bool = False

    def to_trace(self) -> Dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "version": self.version,
            "priority": self.priority,
            "match_reasons": list(self.match_reasons),
            "rule_ids": list(self.rule_ids),
            "truncated": self.truncated,
        }


@dataclass
class Skill:
    skill_id: str
    name: str
    description: str
    content: str
    path: str
    version: str = "1.0.0"
    priority: int = 0
    keywords: List[str] = field(default_factory=list)
    intents: List[str] = field(default_factory=list)
    agents: List[str] = field(default_factory=list)
    resources: Dict[str, Any] = field(default_factory=dict)
    resource_paths: List[str] = field(default_factory=list)
    output_schema_name: str = ""
    enabled: bool = True

    @property
    def rule_ids(self) -> List[str]:
        data = self.resources.get("rules", {})
        rows = data.get("rules", []) if isinstance(data, dict) else []
        return [
            str(row["id"]) for row in rows if isinstance(row, dict) and row.get("id")
        ]

    def match_reasons(
        self, message: str, agent_type: Optional[str], intent: Optional[str]
    ) -> List[str]:
        if not self.enabled:
            return []
        agent = (agent_type or "").lower()
        intent_value = (intent or "").lower()
        if self.agents and agent not in self.agents:
            return []
        reasons: List[str] = []
        if self.agents:
            reasons.append(f"agent:{agent}")
        if self.intents and intent_value in self.intents:
            reasons.append(f"intent:{intent_value}")
        hits = [
            word for word in self.keywords if word.lower() in (message or "").lower()
        ]
        if hits:
            reasons.append("keywords:" + ",".join(hits))
        if self.agents and agent in self.agents:
            if self.skill_id == "compliance_review" and agent == "compliance":
                return reasons
            if self.intents and intent_value in self.intents:
                return reasons
            if hits or (not self.keywords and not self.intents):
                return reasons
            return []
        if (
            not self.agents
            and (not self.intents or intent_value in self.intents)
            and (not self.keywords or hits)
        ):
            return reasons or ["global"]
        return []

    def prompt_sections(self) -> List[str]:
        sections = [f"### {self.name} (id={self.skill_id}, version={self.version})"]
        if self.description:
            sections.append(f"说明: {self.description}")
        sections.append(self.content.strip())
        for name in self.resource_paths:
            if name in {"rules", self.output_schema_name}:
                title = (
                    "合规规则（唯一运行时口径）"
                    if name == "rules"
                    else "输出 JSON Schema"
                )
                sections.append(
                    title
                    + ":\n"
                    + json.dumps(
                        self.resources[name], ensure_ascii=False, separators=(",", ":")
                    )
                )
        return [section for section in sections if section]

    def to_summary(self) -> Dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "version": self.version,
            "priority": self.priority,
            "description": self.description,
            "path": self.path,
            "keywords": self.keywords,
            "intents": self.intents,
            "agents": self.agents,
            "enabled": self.enabled,
            "content_chars": len(self.content),
            "resources": self.resource_paths,
            "resources_valid": True,
            "output_schema": self.output_schema_name or None,
        }


class SkillManager:
    """Atomically loads and resolves versioned business-policy skills."""

    def __init__(self, root_dir: str, max_prompt_chars: int = 5000):
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.max_prompt_chars = max_prompt_chars
        self._skills: List[Skill] = []
        self._errors: List[str] = []

    @property
    def skills(self) -> List[Skill]:
        return list(self._skills)

    @property
    def errors(self) -> List[str]:
        return list(self._errors)

    def load(self) -> List[Skill]:
        loaded, errors = self._load_candidate()
        self._skills, self._errors = loaded, errors
        self._log_loaded_skills()
        return self.skills

    def reload(self) -> List[Skill]:
        loaded, errors = self._load_candidate()
        if errors:
            self._errors = errors
            SKILL_LOAD_FAILURES.inc(len(errors))
            logger.warning(
                "Skill 热更新失败，继续使用上一有效版本: %s", "; ".join(errors)
            )
            return self.skills
        self._skills, self._errors = loaded, []
        self._log_loaded_skills()
        return self.skills

    def resolve(
        self,
        message: str,
        agent_type: Optional[str] = None,
        intent: Optional[str] = None,
    ) -> List[SkillSelection]:
        candidates = []
        for skill in self._skills:
            reasons = skill.match_reasons(message, agent_type, intent)
            if reasons:
                candidates.append((skill, reasons))
        candidates.sort(key=lambda item: (-item[0].priority, item[0].skill_id))
        remaining = self.max_prompt_chars
        selected: List[SkillSelection] = []
        for skill, reasons in candidates:
            included: List[str] = []
            truncated = False
            for section in skill.prompt_sections():
                extra = len(section) + (2 if included else 0)
                if extra > remaining:
                    truncated = True
                    break
                included.append(section)
                remaining -= extra
            if not included:
                continue
            selected.append(
                SkillSelection(
                    skill.skill_id,
                    skill.name,
                    skill.version,
                    skill.priority,
                    reasons,
                    skill.rule_ids,
                    "\n\n".join(included),
                    truncated,
                )
            )
            SKILL_MATCHES.labels(
                skill.skill_id, skill.version, agent_type or "all"
            ).inc()
        return selected

    def prompt_for(
        self,
        message: str,
        agent_type: Optional[str] = None,
        intent: Optional[str] = None,
    ) -> str:
        selected = self.resolve(message, agent_type, intent)
        if not selected:
            return ""
        return (
            "以下是当前请求适用的 ContentOpsMind 版本化业务 Skills。"
            "如与系统安全边界冲突，以系统安全边界为准。\n\n"
            + "\n\n".join(item.prompt_block for item in selected)
        )

    def get_resource(self, skill_id: str, resource: str) -> Any:
        skill = next((item for item in self._skills if item.skill_id == skill_id), None)
        if skill is None:
            raise KeyError(skill_id)
        return skill.resources.get(resource)

    def summary(self) -> Dict[str, Any]:
        return {
            "root_dir": str(self.root_dir),
            "count": len(self._skills),
            "skills": [skill.to_summary() for skill in self._skills],
            "errors": self.errors,
            "last_reload_ok": not self._errors,
        }

    def _load_candidate(self) -> tuple[List[Skill], List[str]]:
        loaded: List[Skill] = []
        errors: List[str] = []
        if not self.root_dir.exists():
            return [], []
        seen: Dict[str, str] = {}
        for path in self._discover_files(self.root_dir):
            try:
                skill = self._load_file(path)
                if skill.skill_id in seen:
                    raise ValueError(f"skill_id 与 {seen[skill.skill_id]} 冲突")
                seen[skill.skill_id] = str(path)
                loaded.append(skill)
            except Exception as exc:
                errors.append(f"{path}: {exc}")
        loaded.sort(key=lambda item: (-item.priority, item.skill_id))
        return loaded, errors

    @staticmethod
    def _discover_files(root_dir: Path) -> Iterable[Path]:
        yield from sorted(root_dir.rglob("SKILL.md"))
        for path in sorted(root_dir.iterdir()):
            if (
                path.is_file()
                and path.suffix.lower() in {".md", ".txt", ".json"}
                and path.name.upper() != "README.MD"
            ):
                yield path

    def _load_file(self, path: Path) -> Skill:
        if path.suffix.lower() == ".json":
            meta = json.loads(path.read_text(encoding="utf-8-sig"))
            body = str(meta.get("content") or meta.get("instructions") or "")
        else:
            meta, body = self._split_front_matter(path.read_text(encoding="utf-8-sig"))
        body = body.strip()
        if not body:
            raise ValueError("Skill 内容为空")
        name = str(meta.get("name") or path.parent.name or path.stem)
        filenames = self._as_list(meta.get("resources"))
        output_schema = str(meta.get("output_schema") or "")
        resources: Dict[str, Any] = {}
        for filename in filenames:
            resource_path = (path.parent / filename).resolve()
            if (
                resource_path.parent != path.parent.resolve()
                or not resource_path.is_file()
            ):
                raise ValueError(f"资源不存在或越界: {filename}")
            resources[Path(filename).stem] = json.loads(
                resource_path.read_text(encoding="utf-8-sig")
            )
        schema_key = Path(output_schema).stem if output_schema else ""
        if schema_key:
            schema = resources.get(schema_key)
            if (
                not isinstance(schema, dict)
                or schema.get("type") != "object"
                or not schema.get("required")
            ):
                raise ValueError("output_schema 必须是带 required 的对象 JSON Schema")
        return Skill(
            skill_id=str(meta.get("skill_id") or path.parent.name or path.stem),
            name=name,
            description=str(meta.get("description") or ""),
            content=self._strip_first_heading(body, name),
            path=str(path),
            version=str(meta.get("version") or "1.0.0"),
            priority=int(meta.get("priority") or 0),
            keywords=self._as_list(meta.get("keywords")),
            intents=[x.lower() for x in self._as_list(meta.get("intents"))],
            agents=[x.lower() for x in self._as_list(meta.get("agents"))],
            resources=resources,
            resource_paths=[Path(x).stem for x in filenames],
            output_schema_name=schema_key,
            enabled=self._as_bool(meta.get("enabled"), True),
        )

    @staticmethod
    def _split_front_matter(raw: str) -> tuple[Dict[str, Any], str]:
        text = raw.lstrip("\ufeff\r\n ")
        if not text.startswith("---"):
            return {}, raw
        lines = text.splitlines()
        meta: Dict[str, Any] = {}
        end = None
        for index, line in enumerate(lines[1:], 1):
            if line.strip() == "---":
                end = index
                break
            if ":" in line:
                key, value = line.split(":", 1)
                meta[key.strip()] = value.strip().strip("\"'")
        return (meta, "\n".join(lines[end + 1 :])) if end is not None else ({}, raw)

    @staticmethod
    def _strip_first_heading(body: str, name: str) -> str:
        lines = body.splitlines()
        if (
            lines
            and lines[0].strip().startswith("#")
            and lines[0].strip().lstrip("#").strip() == name
        ):
            return "\n".join(lines[1:]).strip()
        return body

    @staticmethod
    def _as_list(value: Any) -> List[str]:
        if value in (None, ""):
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return [item.strip() for item in str(value).split(",") if item.strip()]

    @staticmethod
    def _as_bool(value: Any, default: bool = False) -> bool:
        if value in (None, ""):
            return default
        return str(value).strip().lower() not in {"0", "false", "no", "off", "disabled"}

    def _log_loaded_skills(self) -> None:
        logger.info(
            "ContentOpsMind Skills loaded count=%s skills=%s errors=%s",
            len(self._skills),
            [f"{s.skill_id}@{s.version}" for s in self._skills],
            self._errors,
        )
