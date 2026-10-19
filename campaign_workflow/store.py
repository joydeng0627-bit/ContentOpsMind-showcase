from __future__ import annotations

import json
import time
import uuid
from typing import Any, Dict, List, Optional

import redis

from .models import (
    ApprovalRecord,
    CampaignWorkflow,
    ChannelGenerationResult,
    ContentVersion,
    GenerationAttempt,
    GenerationAttemptState,
    ReviewContext,
    ReviewResult,
    WorkflowExecutionEvent,
)


class WorkflowNotFoundError(KeyError):
    pass


class WorkflowConflictError(RuntimeError):
    pass


class WorkflowLockLostError(WorkflowConflictError):
    """The caller no longer owns the workflow execution lease."""

    code = "workflow_lock_lost"


class CampaignWorkflowStore:
    """Redis-backed durable store isolated from conversation memory keys."""

    GLOBAL_INDEX = "campaign:workflows"
    _SNAPSHOT_COMMIT_SCRIPT = """
-- DELIVERY_SNAPSHOT_COMMIT_V1
if ARGV[1] == '' or redis.call('GET', KEYS[1]) ~= ARGV[1] then return -3 end
local checks = cjson.decode(ARGV[2])
local indexes = cjson.decode(ARGV[3])
local writes = cjson.decode(ARGV[4])
for key, expected in pairs(checks) do
  local kind = redis.call('TYPE', key)['ok']
  if kind ~= 'none' and kind ~= 'string' then return -5 end
  local actual = redis.call('GET', key)
  if expected == false then
    if actual then return -1 end
  elseif actual ~= expected then return -1 end
end
for key, expected in pairs(indexes) do
  local kind = redis.call('TYPE', key)['ok']
  if kind ~= 'none' and kind ~= 'zset' then return -5 end
  local actual = redis.call('ZRANGE', key, 0, -1)
  table.sort(actual)
  table.sort(expected)
  if #actual ~= #expected then return -1 end
  for i = 1, #actual do if actual[i] ~= expected[i] then return -1 end end
end
-- Validate all destination types and arguments BEFORE any write. Lua does not roll back errors.
for _, op in ipairs(writes) do
  local kind = redis.call('TYPE', op['key'])['ok']
  if op['op'] == 'set' then
    if (kind ~= 'none' and kind ~= 'string') or type(op['value']) ~= 'string' then return -5 end
  elseif op['op'] == 'zadd' then
    if (kind ~= 'none' and kind ~= 'zset') or not tonumber(op['score']) or type(op['member']) ~= 'string' then return -5 end
  else return -5 end
end
for _, op in ipairs(writes) do
  if op['op'] == 'set' then redis.call('SET', op['key'], op['value'])
  else redis.call('ZADD', op['key'], op['score'], op['member']) end
end
return 1
"""
    _VERSION_SET_SCRIPT = """
-- IMMUTABLE_CONTENT_VERSION_V1
local owner = redis.call('GET', KEYS[1])
if (ARGV[1] ~= '' and owner ~= ARGV[1]) or (ARGV[1] == '' and owner) then return -3 end
local function same(a, b)
  if type(a) ~= type(b) then return false end
  if type(a) ~= 'table' then return a == b end
  for k, v in pairs(a) do if not same(v, b[k]) then return false end end
  for k, _ in pairs(b) do if a[k] == nil then return false end end
  return true
end
local raw = redis.call('GET', KEYS[2])
if raw then
  local before = cjson.decode(raw)
  local after = cjson.decode(ARGV[2])
  before['status'] = nil
  after['status'] = nil
  -- Missing new legacy fields remain missing; do not retrofit historical hashes.
  for k, v in pairs(after) do
    if before[k] == nil and (v == '' or type(v) == 'table' and next(v) == nil) then after[k] = nil end
  end
  if not same(before, after) then return -4 end
end
local kind = redis.call('TYPE', KEYS[3])['ok']
if kind ~= 'none' and kind ~= 'zset' then return -5 end
redis.call('SET', KEYS[2], ARGV[2])
redis.call('ZADD', KEYS[3], ARGV[3], ARGV[4])
return 1
"""
    _CAS_SCRIPT = """
if ARGV[5] ~= '' and redis.call('GET', KEYS[4]) ~= ARGV[5] then return -3 end
if ARGV[5] == '' and redis.call('GET', KEYS[4]) then return -3 end
local current = redis.call('GET', KEYS[1])
if not current then return -2 end
local parsed = cjson.decode(current)
if tonumber(parsed['revision']) ~= tonumber(ARGV[1]) then return -1 end
redis.call('SET', KEYS[1], ARGV[2])
redis.call('ZADD', KEYS[2], ARGV[3], ARGV[4])
redis.call('ZADD', KEYS[3], ARGV[3], ARGV[4])
return 1
"""
    _GUARDED_SET_INDEX_SCRIPT = """
if ARGV[1] ~= '' and redis.call('GET', KEYS[1]) ~= ARGV[1] then return -3 end
if ARGV[1] == '' and redis.call('GET', KEYS[1]) then return -3 end
redis.call('SET', KEYS[2], ARGV[2])
redis.call('ZADD', KEYS[3], ARGV[3], ARGV[4])
return 1
"""
    _SAVE_REVIEW_SCRIPT = """
if ARGV[1] ~= '' and redis.call('GET', KEYS[1]) ~= ARGV[1] then return -3 end
if ARGV[1] == '' and redis.call('GET', KEYS[1]) then return -3 end
if redis.call('EXISTS', KEYS[2]) == 1 then return -4 end
redis.call('SET', KEYS[2], ARGV[2])
redis.call('ZADD', KEYS[3], ARGV[3], ARGV[4])
redis.call('SET', KEYS[4], ARGV[4])
return 1
"""
    _SAVE_REVIEW_CONTEXT_SCRIPT = """
if ARGV[1] ~= '' and redis.call('GET', KEYS[1]) ~= ARGV[1] then return -3 end
if ARGV[1] == '' and redis.call('GET', KEYS[1]) then return -3 end
local existing_context = redis.call('GET', KEYS[2])
if existing_context and existing_context ~= ARGV[2] then return -4 end
local existing_pointer = redis.call('GET', KEYS[3])
if existing_pointer and existing_pointer ~= ARGV[3] then return -4 end
if not existing_context then redis.call('SET', KEYS[2], ARGV[2]) end
redis.call('SET', KEYS[3], ARGV[3])
return 1
"""
    _SAVE_GENERATION_RESULT_SCRIPT = """
if ARGV[1] ~= '' and redis.call('GET', KEYS[1]) ~= ARGV[1] then return -3 end
if ARGV[1] == '' and redis.call('GET', KEYS[1]) then return -3 end
local current = redis.call('GET', KEYS[2])
if current then
  local parsed = cjson.decode(current)
  if tonumber(parsed['result_revision']) ~= tonumber(ARGV[2]) then return -1 end
elseif tonumber(ARGV[2]) ~= 0 then
  return -1
end
redis.call('SET', KEYS[2], ARGV[3])
redis.call('ZADD', KEYS[3], ARGV[4], ARGV[5])
return 1
"""
    _COMMIT_GENERATION_SCRIPT = """
-- GENERATION_BATCH_COMMIT_V1
if ARGV[5] ~= '' and redis.call('GET', KEYS[4]) ~= ARGV[5] then return -3 end
if ARGV[5] == '' and redis.call('GET', KEYS[4]) then return -3 end
if redis.call('EXISTS', KEYS[6]) == 1 then return 2 end
local current = redis.call('GET', KEYS[1])
if not current then return -2 end
local workflow = cjson.decode(current)
if tonumber(workflow['revision']) ~= tonumber(ARGV[1]) then return -1 end
if workflow['active_generation_attempt_id'] ~= ARGV[6] then return -5 end
local attempt_raw = redis.call('GET', KEYS[5])
if not attempt_raw then return -6 end
local attempt = cjson.decode(attempt_raw)
if attempt['workflow_id'] ~= ARGV[4]
  or attempt['brief_hash'] ~= ARGV[7]
  or attempt['knowledge_snapshot_hash'] ~= ARGV[8]
  or attempt['generation_context_hash'] ~= ARGV[9] then return -5 end
local channels = cjson.decode(ARGV[10])
if #attempt['channels'] ~= #channels then return -5 end
for i = 1, #channels do
  if attempt['channels'][i] ~= channels[i] then return -5 end
end
local count = tonumber(ARGV[13])
for i = 1, count do
  local result_key = KEYS[8 + (3 * count) + i]
  local result_raw = redis.call('GET', result_key)
  if not result_raw then return -6 end
  local staged = cjson.decode(result_raw)
  if staged['status'] ~= 'ready_to_commit'
    or staged['workflow_id'] ~= ARGV[4]
    or staged['generation_attempt_id'] ~= ARGV[6]
    or staged['brief_hash'] ~= ARGV[7]
    or staged['knowledge_snapshot_hash'] ~= ARGV[8]
    or staged['generation_context_hash'] ~= ARGV[9]
    or staged['channel'] ~= channels[i] then return -5 end
end
for i = 1, count do
  local offset = 13 + ((i - 1) * 6)
  local version_id = ARGV[offset + 1]
  local version_payload = ARGV[offset + 2]
  local version_score = ARGV[offset + 3]
  local review_id = ARGV[offset + 4]
  local review_payload = ARGV[offset + 5]
  local review_score = ARGV[offset + 6]
  local version_key = KEYS[8 + i]
  local review_key = KEYS[8 + count + i]
  local review_slot_key = KEYS[8 + (2 * count) + i]
  if redis.call('EXISTS', version_key) == 1 or redis.call('EXISTS', review_key) == 1 then return -4 end
  redis.call('SET', version_key, version_payload)
  redis.call('ZADD', KEYS[7], version_score, version_id)
  redis.call('SET', review_key, review_payload)
  redis.call('ZADD', KEYS[8], review_score, review_id)
  redis.call('SET', review_slot_key, review_id)
end
redis.call('SET', KEYS[1], ARGV[2])
redis.call('ZADD', KEYS[2], ARGV[3], ARGV[4])
redis.call('ZADD', KEYS[3], ARGV[3], ARGV[4])
redis.call('SET', KEYS[5], ARGV[11])
redis.call('SET', KEYS[6], ARGV[12])
return 1
"""
    _SAVE_AGGREGATION_SCRIPT = """
-- IMMUTABLE_REVIEW_AGGREGATION_V1
if ARGV[5] ~= '' and redis.call('GET', KEYS[4]) ~= ARGV[5] then return -3 end
if ARGV[5] == '' and redis.call('GET', KEYS[4]) then return -3 end
local current = redis.call('GET', KEYS[1])
if not current then return -2 end
local parsed = cjson.decode(current)
if tonumber(parsed['revision']) ~= tonumber(ARGV[1]) then return -1 end
local function same(a, b)
  if type(a) ~= type(b) then return false end
  if type(a) ~= 'table' then return a == b end
  for k, v in pairs(a) do if not same(v, b[k]) then return false end end
  for k, _ in pairs(b) do if a[k] == nil then return false end end
  return true
end
for index = 5, #KEYS do
  local raw = redis.call('GET', KEYS[index])
  if not raw then return -4 end
  local before = cjson.decode(raw)
  local after = cjson.decode(ARGV[index + 1])
  before['status'] = nil
  after['status'] = nil
  for k, v in pairs(after) do
    if before[k] == nil and (v == '' or type(v) == 'table' and next(v) == nil) then after[k] = nil end
  end
  if not same(before, after) then return -4 end
end
for index = 2, 3 do
  local kind = redis.call('TYPE', KEYS[index])['ok']
  if kind ~= 'none' and kind ~= 'zset' then return -5 end
end
redis.call('SET', KEYS[1], ARGV[2])
redis.call('ZADD', KEYS[2], ARGV[3], ARGV[4])
redis.call('ZADD', KEYS[3], ARGV[3], ARGV[4])
for index = 5, #KEYS do
  redis.call('SET', KEYS[index], ARGV[index + 1])
end
return 1
"""
    _UNLOCK_SCRIPT = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('DEL', KEYS[1])
end
return 0
"""
    _RENEW_LOCK_SCRIPT = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('EXPIRE', KEYS[1], ARGV[2])
end
return 0
"""
    _OWNS_LOCK_SCRIPT = """
if redis.call('GET', KEYS[1]) == ARGV[1] then return 1 end
return 0
"""

    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url, decode_responses=True)
        self.redis.ping()

    @staticmethod
    def workflow_key(workflow_id: str) -> str:
        return f"campaign:workflow:{workflow_id}"

    @staticmethod
    def user_index_key(user_id: str) -> str:
        return f"campaign:user:{user_id}:workflows"

    @staticmethod
    def lock_key(workflow_id: str) -> str:
        return f"campaign:lock:{workflow_id}"

    @staticmethod
    def version_key(version_id: str) -> str:
        return f"campaign:version:{version_id}"

    @staticmethod
    def version_index_key(workflow_id: str) -> str:
        return f"campaign:workflow:{workflow_id}:versions"

    @staticmethod
    def review_key(review_id: str) -> str:
        return f"campaign:review:{review_id}"

    @staticmethod
    def review_index_key(workflow_id: str) -> str:
        return f"campaign:workflow:{workflow_id}:reviews"

    @staticmethod
    def review_slot_key(
        workflow_id: str, version_id: str, review_type: str, review_context_hash: str
    ) -> str:
        context = review_context_hash or "legacy"
        return f"campaign:workflow:{workflow_id}:review-slot:{version_id}:{review_type}:{context}"

    @staticmethod
    def review_context_key(review_context_hash: str) -> str:
        return f"campaign:review-context:{review_context_hash}"

    @staticmethod
    def review_context_pointer_key(workflow_id: str, version_id: str) -> str:
        return f"campaign:workflow:{workflow_id}:review-context:{version_id}"

    @staticmethod
    def event_key(event_id: str) -> str:
        return f"campaign:event:{event_id}"

    @staticmethod
    def event_index_key(workflow_id: str) -> str:
        return f"campaign:workflow:{workflow_id}:events"

    @staticmethod
    def approval_key(approval_id: str) -> str:
        return f"campaign:approval:{approval_id}"

    @staticmethod
    def approval_index_key(workflow_id: str) -> str:
        return f"campaign:workflow:{workflow_id}:approvals"

    @staticmethod
    def lock_audit_key(execution_id: str) -> str:
        return f"campaign:lock-audit:{execution_id}"

    @staticmethod
    def lock_audit_index_key(workflow_id: str) -> str:
        return f"campaign:workflow:{workflow_id}:lock-audits"

    @staticmethod
    def generation_attempt_key(generation_attempt_id: str) -> str:
        return f"campaign:generation-attempt:{generation_attempt_id}"

    @staticmethod
    def generation_attempt_index_key(workflow_id: str) -> str:
        return f"campaign:workflow:{workflow_id}:generation-attempts"

    @staticmethod
    def generation_result_key(generation_attempt_id: str, channel: str) -> str:
        digest = uuid.uuid5(uuid.NAMESPACE_URL, channel).hex[:16]
        return f"campaign:generation-result:{generation_attempt_id}:{digest}"

    @staticmethod
    def generation_result_index_key(generation_attempt_id: str) -> str:
        return f"campaign:generation-attempt:{generation_attempt_id}:results"

    @staticmethod
    def generation_commit_key(generation_attempt_id: str) -> str:
        return f"campaign:generation-attempt:{generation_attempt_id}:commit"

    @staticmethod
    def _serialize(workflow: CampaignWorkflow) -> str:
        return workflow.model_dump_json()

    def create(self, workflow: CampaignWorkflow) -> CampaignWorkflow:
        key = self.workflow_key(workflow.workflow_id)
        payload = self._serialize(workflow)
        if not self.redis.set(key, payload, nx=True):
            raise WorkflowConflictError(f"Workflow 已存在: {workflow.workflow_id}")
        score = workflow.created_at.timestamp()
        pipe = self.redis.pipeline(transaction=True)
        pipe.zadd(self.GLOBAL_INDEX, {workflow.workflow_id: score})
        pipe.zadd(self.user_index_key(workflow.user_id), {workflow.workflow_id: score})
        pipe.execute()
        return workflow

    def get(self, workflow_id: str) -> CampaignWorkflow:
        raw = self.redis.get(self.workflow_key(workflow_id))
        if not raw:
            raise WorkflowNotFoundError(workflow_id)
        return CampaignWorkflow.model_validate_json(raw)

    def list(
        self, user_id: Optional[str] = None, limit: int = 50
    ) -> List[CampaignWorkflow]:
        index = self.user_index_key(user_id) if user_id else self.GLOBAL_INDEX
        ids = self.redis.zrevrange(index, 0, max(0, min(limit, 200)) - 1)
        workflows: List[CampaignWorkflow] = []
        for workflow_id in ids:
            try:
                workflows.append(self.get(workflow_id))
            except WorkflowNotFoundError:
                self.redis.zrem(index, workflow_id)
        return workflows

    def save(
        self,
        workflow: CampaignWorkflow,
        expected_revision: int,
        lock_token: Optional[str] = None,
    ) -> CampaignWorkflow:
        if workflow.revision != expected_revision + 1:
            raise WorkflowConflictError("保存时 revision 必须递增 1")
        score = workflow.updated_at.timestamp()
        result = self.redis.eval(
            self._CAS_SCRIPT,
            4,
            self.workflow_key(workflow.workflow_id),
            self.GLOBAL_INDEX,
            self.user_index_key(workflow.user_id),
            self.lock_key(workflow.workflow_id),
            expected_revision,
            self._serialize(workflow),
            score,
            workflow.workflow_id,
            lock_token or "",
        )
        if result == -2:
            raise WorkflowNotFoundError(workflow.workflow_id)
        if result == -3:
            raise WorkflowLockLostError("执行租约已失效，拒绝保存 Workflow")
        if result != 1:
            raise WorkflowConflictError("Workflow 已被其他请求更新，请刷新后重试")
        return workflow

    def save_review_aggregation(
        self,
        workflow: CampaignWorkflow,
        versions: List[ContentVersion],
        *,
        expected_revision: int,
        lock_token: Optional[str] = None,
    ) -> CampaignWorkflow:
        if workflow.revision != expected_revision + 1:
            raise WorkflowConflictError("聚合保存时 revision 必须递增 1")
        score = workflow.updated_at.timestamp()
        keys = [
            self.workflow_key(workflow.workflow_id),
            self.GLOBAL_INDEX,
            self.user_index_key(workflow.user_id),
            self.lock_key(workflow.workflow_id),
            *(self.version_key(version.version_id) for version in versions),
        ]
        args = [
            expected_revision,
            self._serialize(workflow),
            score,
            workflow.workflow_id,
            lock_token or "",
            *(version.model_dump_json() for version in versions),
        ]
        result = self.redis.eval(self._SAVE_AGGREGATION_SCRIPT, len(keys), *keys, *args)
        if result == -2:
            raise WorkflowNotFoundError(workflow.workflow_id)
        if result == -3:
            raise WorkflowLockLostError("执行租约已失效，拒绝保存审核聚合")
        if result != 1:
            raise WorkflowConflictError("审核聚合使用了陈旧的 Workflow revision")
        return workflow

    def _guarded_set_index(
        self,
        workflow_id: str,
        data_key: str,
        index_key: str,
        member: str,
        score: float,
        payload: str,
        lock_token: Optional[str],
    ) -> None:
        result = self.redis.eval(
            self._GUARDED_SET_INDEX_SCRIPT,
            3,
            self.lock_key(workflow_id),
            data_key,
            index_key,
            lock_token or "",
            payload,
            score,
            member,
        )
        if result == -3:
            raise WorkflowLockLostError("执行租约已失效，拒绝写入工作流子资源")

    def save_content_version(
        self, version: ContentVersion, lock_token: Optional[str] = None
    ) -> ContentVersion:
        self.get(version.workflow_id)
        result = self.redis.eval(
            self._VERSION_SET_SCRIPT,
            3,
            self.lock_key(version.workflow_id),
            self.version_key(version.version_id),
            self.version_index_key(version.workflow_id),
            lock_token or "",
            version.model_dump_json(),
            version.created_at.timestamp(),
            version.version_id,
        )
        if result == -3:
            raise WorkflowLockLostError("执行租约已失效，拒绝写入文案版本")
        if result != 1:
            raise WorkflowConflictError(
                "文案版本的正文及审核输入不可原地修改，请创建新版本"
            )
        return version

    def delivery_snapshot(self, workflow_id: str) -> Dict[str, Any]:
        """Capture every delivery dependency; atomic commit rechecks values AND membership."""
        checks = {
            self.workflow_key(workflow_id): self.redis.get(
                self.workflow_key(workflow_id)
            )
        }
        indexes = {}
        for index, key_fn in (
            (self.version_index_key(workflow_id), self.version_key),
            (self.review_index_key(workflow_id), self.review_key),
            (self.approval_index_key(workflow_id), self.approval_key),
        ):
            ids = self.redis.zrange(index, 0, -1)
            indexes[index] = sorted(ids)
            for member, raw in zip(
                ids, self.redis.mget([key_fn(member) for member in ids])
            ):
                checks[key_fn(member)] = raw
        for version_id in indexes[self.version_index_key(workflow_id)]:
            pointer = self.review_context_pointer_key(workflow_id, version_id)
            context_hash = self.redis.get(pointer)
            checks[pointer] = context_hash
            if context_hash:
                key = self.review_context_key(context_hash)
                checks[key] = self.redis.get(key)
        return {"checks": checks, "indexes": indexes}

    def commit_delivery_snapshot(
        self,
        workflow_id: str,
        snapshot: Dict[str, Any],
        writes: List[Dict[str, Any]],
        lock_token: str,
    ) -> None:
        checks = {
            key: value if value is not None else False
            for key, value in snapshot["checks"].items()
        }
        keys = sorted(
            set(checks) | set(snapshot["indexes"]) | {op["key"] for op in writes}
        )
        result = self.redis.eval(
            self._SNAPSHOT_COMMIT_SCRIPT,
            1 + len(keys),
            self.lock_key(workflow_id),
            *keys,
            lock_token,
            json.dumps(checks),
            json.dumps(snapshot["indexes"]),
            json.dumps(writes),
        )
        if result == -3:
            raise WorkflowLockLostError("执行租约已失效，拒绝提交审批或交付快照")
        if result != 1:
            raise WorkflowConflictError(
                "审批或交付依赖已变化，或存储结构无效，请刷新后重试"
            )

    def commit_approval(
        self,
        workflow: CampaignWorkflow,
        versions: List[ContentVersion],
        approval: ApprovalRecord,
        snapshot: Dict[str, Any],
        lock_token: str,
    ) -> None:
        before = json.loads(snapshot["checks"][self.workflow_key(workflow.workflow_id)])
        if (
            workflow.revision != before["revision"] + 1
            or approval.committed_revision != workflow.revision
        ):
            raise WorkflowConflictError("审批提交 revision 必须递增 1")
        approval_key = self.approval_key(approval.approval_id)
        snapshot["checks"][approval_key] = None
        writes = [
            {
                "op": "set",
                "key": self.workflow_key(workflow.workflow_id),
                "value": workflow.model_dump_json(),
            },
            {"op": "set", "key": approval_key, "value": approval.model_dump_json()},
        ]
        for version in versions:
            old = ContentVersion.model_validate_json(
                snapshot["checks"][self.version_key(version.version_id)]
            )
            if old.model_dump(exclude={"status"}) != version.model_dump(
                exclude={"status"}
            ):
                raise WorkflowConflictError("审批不得修改文案或审核输入")
            writes.append(
                {
                    "op": "set",
                    "key": self.version_key(version.version_id),
                    "value": version.model_dump_json(),
                }
            )
        for index, member, score in (
            (self.GLOBAL_INDEX, workflow.workflow_id, workflow.updated_at.timestamp()),
            (
                self.user_index_key(workflow.user_id),
                workflow.workflow_id,
                workflow.updated_at.timestamp(),
            ),
            (
                self.approval_index_key(workflow.workflow_id),
                approval.approval_id,
                approval.created_at.timestamp(),
            ),
        ):
            writes.append(
                {"op": "zadd", "key": index, "member": member, "score": score}
            )
        self.commit_delivery_snapshot(
            workflow.workflow_id, snapshot, writes, lock_token
        )

    def save_generation_attempt(
        self, attempt: GenerationAttempt, lock_token: Optional[str] = None
    ) -> GenerationAttempt:
        self.get(attempt.workflow_id)
        self._guarded_set_index(
            attempt.workflow_id,
            self.generation_attempt_key(attempt.generation_attempt_id),
            self.generation_attempt_index_key(attempt.workflow_id),
            attempt.generation_attempt_id,
            attempt.created_at.timestamp(),
            attempt.model_dump_json(),
            lock_token,
        )
        return attempt

    def get_generation_attempt(
        self, generation_attempt_id: str
    ) -> Optional[GenerationAttempt]:
        raw = self.redis.get(self.generation_attempt_key(generation_attempt_id))
        return GenerationAttempt.model_validate_json(raw) if raw else None

    def save_generation_result(
        self,
        result: ChannelGenerationResult,
        *,
        expected_revision: int,
        lock_token: Optional[str] = None,
    ) -> ChannelGenerationResult:
        stored = self.redis.eval(
            self._SAVE_GENERATION_RESULT_SCRIPT,
            3,
            self.lock_key(result.workflow_id),
            self.generation_result_key(result.generation_attempt_id, result.channel),
            self.generation_result_index_key(result.generation_attempt_id),
            lock_token or "",
            expected_revision,
            result.model_dump_json(),
            result.updated_at.timestamp(),
            result.channel,
        )
        if stored == -3:
            raise WorkflowLockLostError("执行租约已失效，拒绝保存生成阶段结果")
        if stored != 1:
            raise WorkflowConflictError("生成阶段结果已被其他执行更新")
        return result

    def get_generation_result(
        self, generation_attempt_id: str, channel: str
    ) -> Optional[ChannelGenerationResult]:
        raw = self.redis.get(self.generation_result_key(generation_attempt_id, channel))
        return ChannelGenerationResult.model_validate_json(raw) if raw else None

    def list_generation_results(
        self, generation_attempt_id: str
    ) -> List[ChannelGenerationResult]:
        channels = self.redis.zrange(
            self.generation_result_index_key(generation_attempt_id), 0, -1
        )
        values = (
            self.redis.mget(
                [
                    self.generation_result_key(generation_attempt_id, channel)
                    for channel in channels
                ]
            )
            if channels
            else []
        )
        return [
            ChannelGenerationResult.model_validate_json(value)
            for value in values
            if value
        ]

    def generation_commit(self, generation_attempt_id: str) -> Optional[Dict[str, Any]]:
        raw = self.redis.get(self.generation_commit_key(generation_attempt_id))
        return json.loads(raw) if raw else None

    def commit_generation_batch(
        self,
        workflow: CampaignWorkflow,
        attempt: GenerationAttempt,
        results: List[ChannelGenerationResult],
        versions: List[ContentVersion],
        validations: List[ReviewResult],
        *,
        expected_revision: int,
        lock_token: Optional[str] = None,
    ) -> bool:
        if workflow.revision != expected_revision + 1:
            raise WorkflowConflictError("生成批量提交时 revision 必须递增 1")
        channels = sorted(attempt.channels)
        by_result = {item.channel: item for item in results}
        by_version = {item.channel: item for item in versions}
        by_validation = {item.version_id: item for item in validations}
        ordered_results = [by_result[channel] for channel in channels]
        ordered_versions = [by_version[channel] for channel in channels]
        ordered_validations = [
            by_validation[version.version_id] for version in ordered_versions
        ]
        committed_attempt = attempt.model_copy(
            update={
                "state": GenerationAttemptState.COMMITTED,
                "updated_at": workflow.updated_at,
            }
        )
        commit_payload = json.dumps(
            {
                "generation_attempt_id": attempt.generation_attempt_id,
                "version_ids": [item.version_id for item in ordered_versions],
                "workflow_revision": workflow.revision,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        keys = [
            self.workflow_key(workflow.workflow_id),
            self.GLOBAL_INDEX,
            self.user_index_key(workflow.user_id),
            self.lock_key(workflow.workflow_id),
            self.generation_attempt_key(attempt.generation_attempt_id),
            self.generation_commit_key(attempt.generation_attempt_id),
            self.version_index_key(workflow.workflow_id),
            self.review_index_key(workflow.workflow_id),
            *(self.version_key(item.version_id) for item in ordered_versions),
            *(self.review_key(item.review_id) for item in ordered_validations),
            *(
                self.review_slot_key(
                    workflow.workflow_id,
                    item.version_id,
                    item.review_type,
                    item.review_context_hash,
                )
                for item in ordered_validations
            ),
            *(
                self.generation_result_key(attempt.generation_attempt_id, item.channel)
                for item in ordered_results
            ),
        ]
        args: List[Any] = [
            expected_revision,
            self._serialize(workflow),
            workflow.updated_at.timestamp(),
            workflow.workflow_id,
            lock_token or "",
            attempt.generation_attempt_id,
            attempt.brief_hash,
            attempt.knowledge_snapshot_hash,
            attempt.generation_context_hash,
            json.dumps(channels, ensure_ascii=False, separators=(",", ":")),
            committed_attempt.model_dump_json(),
            commit_payload,
            len(channels),
        ]
        for version, validation in zip(ordered_versions, ordered_validations):
            args.extend(
                [
                    version.version_id,
                    version.model_dump_json(),
                    version.created_at.timestamp(),
                    validation.review_id,
                    validation.model_dump_json(),
                    validation.created_at.timestamp(),
                ]
            )
        stored = self.redis.eval(
            self._COMMIT_GENERATION_SCRIPT, len(keys), *keys, *args
        )
        if stored == -2:
            raise WorkflowNotFoundError(workflow.workflow_id)
        if stored == -3:
            raise WorkflowLockLostError("执行租约已失效，拒绝提交生成版本")
        if stored == -4:
            raise WorkflowConflictError("生成版本或渠道校验记录已存在")
        if stored == -5:
            raise WorkflowConflictError("GenerationAttempt 上下文不匹配，禁止提交")
        if stored == -6:
            raise WorkflowConflictError("GenerationAttempt 结果不完整，禁止提交")
        if stored == -1:
            raise WorkflowConflictError("生成批量提交使用了陈旧的 Workflow revision")
        return stored == 1

    def list_content_versions(self, workflow_id: str) -> List[ContentVersion]:
        self.get(workflow_id)
        ids = self.redis.zrange(self.version_index_key(workflow_id), 0, -1)
        values = (
            self.redis.mget([self.version_key(version_id) for version_id in ids])
            if ids
            else []
        )
        return [ContentVersion.model_validate_json(value) for value in values if value]

    def save_review(
        self, workflow_id: str, review: ReviewResult, lock_token: Optional[str] = None
    ) -> ReviewResult:
        self.get(workflow_id)
        result = self.redis.eval(
            self._SAVE_REVIEW_SCRIPT,
            4,
            self.lock_key(workflow_id),
            self.review_key(review.review_id),
            self.review_index_key(workflow_id),
            self.review_slot_key(
                workflow_id,
                review.version_id,
                review.review_type,
                review.review_context_hash,
            ),
            lock_token or "",
            review.model_dump_json(),
            review.created_at.timestamp(),
            review.review_id,
        )
        if result == -3:
            raise WorkflowLockLostError("执行租约已失效，拒绝写入审核结果")
        if result == -4:
            raise WorkflowConflictError("ReviewResult 为不可变记录，禁止覆盖")
        return review

    def save_review_context(
        self, workflow_id: str, context: ReviewContext, lock_token: Optional[str] = None
    ) -> ReviewContext:
        self.get(workflow_id)
        result = self.redis.eval(
            self._SAVE_REVIEW_CONTEXT_SCRIPT,
            3,
            self.lock_key(workflow_id),
            self.review_context_key(context.review_context_hash),
            self.review_context_pointer_key(workflow_id, context.version_id),
            lock_token or "",
            context.model_dump_json(),
            context.review_context_hash,
        )
        if result == -3:
            raise WorkflowLockLostError("执行租约已失效，拒绝冻结 ReviewContext")
        if result == -4:
            raise WorkflowConflictError("ContentVersion 已绑定不同的 ReviewContext")
        return context

    def get_review_context(
        self, workflow_id: str, version_id: str
    ) -> Optional[ReviewContext]:
        self.get(workflow_id)
        context_hash = self.redis.get(
            self.review_context_pointer_key(workflow_id, version_id)
        )
        if not context_hash:
            return None
        raw = self.redis.get(self.review_context_key(context_hash))
        return ReviewContext.model_validate_json(raw) if raw else None

    def latest_review(
        self,
        workflow_id: str,
        version_id: str,
        review_type: str,
        review_context_hash: str,
    ) -> Optional[ReviewResult]:
        self.get(workflow_id)
        review_id = self.redis.get(
            self.review_slot_key(
                workflow_id, version_id, review_type, review_context_hash
            )
        )
        if not review_id:
            return None
        raw = self.redis.get(self.review_key(review_id))
        return ReviewResult.model_validate_json(raw) if raw else None

    def list_reviews(self, workflow_id: str) -> List[ReviewResult]:
        self.get(workflow_id)
        ids = self.redis.zrange(self.review_index_key(workflow_id), 0, -1)
        values = (
            self.redis.mget([self.review_key(review_id) for review_id in ids])
            if ids
            else []
        )
        return [ReviewResult.model_validate_json(value) for value in values if value]

    def save_event(
        self, event: WorkflowExecutionEvent, lock_token: Optional[str] = None
    ) -> WorkflowExecutionEvent:
        self.get(event.workflow_id)
        score = (
            (event.started_at or event.finished_at).timestamp()
            if (event.started_at or event.finished_at)
            else time.time()
        )
        self._guarded_set_index(
            event.workflow_id,
            self.event_key(event.event_id),
            self.event_index_key(event.workflow_id),
            event.event_id,
            score,
            event.model_dump_json(),
            lock_token,
        )
        return event

    def list_events(self, workflow_id: str) -> List[WorkflowExecutionEvent]:
        self.get(workflow_id)
        ids = self.redis.zrange(self.event_index_key(workflow_id), 0, -1)
        values = (
            self.redis.mget([self.event_key(event_id) for event_id in ids])
            if ids
            else []
        )
        return [
            WorkflowExecutionEvent.model_validate_json(value)
            for value in values
            if value
        ]

    def save_approval(
        self, approval: ApprovalRecord, lock_token: Optional[str] = None
    ) -> ApprovalRecord:
        self.get(approval.workflow_id)
        self._guarded_set_index(
            approval.workflow_id,
            self.approval_key(approval.approval_id),
            self.approval_index_key(approval.workflow_id),
            approval.approval_id,
            approval.created_at.timestamp(),
            approval.model_dump_json(),
            lock_token,
        )
        return approval

    def list_approvals(self, workflow_id: str) -> List[ApprovalRecord]:
        self.get(workflow_id)
        ids = self.redis.zrange(self.approval_index_key(workflow_id), 0, -1)
        values = (
            self.redis.mget([self.approval_key(approval_id) for approval_id in ids])
            if ids
            else []
        )
        return [ApprovalRecord.model_validate_json(value) for value in values if value]

    def save_lock_audit(
        self, workflow_id: str, execution_id: str, audit: Dict[str, Any]
    ) -> None:
        payload = json.dumps(audit, ensure_ascii=False, separators=(",", ":"))
        score = float(audit.get("acquired_at") or time.time())
        pipe = self.redis.pipeline(transaction=True)
        pipe.set(self.lock_audit_key(execution_id), payload)
        pipe.zadd(self.lock_audit_index_key(workflow_id), {execution_id: score})
        pipe.execute()

    def list_lock_audits(self, workflow_id: str) -> List[Dict[str, Any]]:
        ids = self.redis.zrange(self.lock_audit_index_key(workflow_id), 0, -1)
        values = (
            self.redis.mget([self.lock_audit_key(execution_id) for execution_id in ids])
            if ids
            else []
        )
        return [json.loads(value) for value in values if value]

    def detail(self, workflow_id: str) -> dict:
        workflow = self.get(workflow_id)
        generation_attempt = (
            self.get_generation_attempt(workflow.active_generation_attempt_id)
            if workflow.active_generation_attempt_id
            else None
        )
        generation_results = (
            self.list_generation_results(generation_attempt.generation_attempt_id)
            if generation_attempt
            else []
        )
        events = self.list_events(workflow_id)
        versions = self.list_content_versions(workflow_id)
        reviews = self.list_reviews(workflow_id)
        latest_reviews: Dict[tuple, ReviewResult] = {}
        for review in reviews:
            key = (review.version_id, review.review_type, review.review_context_hash)
            current = latest_reviews.get(key)
            if current is None or (review.attempt, review.created_at) > (
                current.attempt,
                current.created_at,
            ):
                latest_reviews[key] = review
        review_progress: Dict[str, Dict[str, dict]] = {}
        for review in latest_reviews.values():
            if review.review_type not in {
                "fact_check",
                "compliance",
                "content_quality",
            }:
                continue
            review_progress.setdefault(review.version_id, {})[review.review_type] = {
                "decision": review.decision.value,
                "attempt": review.attempt,
                "request_id": review.request_id,
                "error_code": review.error_code,
                "review_context_hash": review.review_context_hash,
            }
        total_cost = sum(item.estimated_cost_usd for item in events)
        total_tokens = sum(item.token_usage for item in events)
        total_input_tokens = sum(item.input_tokens for item in events)
        total_output_tokens = sum(item.output_tokens for item in events)
        total_model_calls = sum(item.model_calls for item in events)
        usage_available = any(item.model_calls > 0 for item in events)
        priced_events = [item for item in events if item.model_calls > 0]
        cost_available = bool(priced_events) and all(
            item.cost_available for item in priced_events
        )
        failed_events = [item for item in events if item.status.value == "failed"]
        lock_audits = self.list_lock_audits(workflow_id)
        detail = {
            "workflow": workflow.model_dump(mode="json"),
            "versions": [item.model_dump(mode="json") for item in versions],
            "reviews": [item.model_dump(mode="json") for item in reviews],
            "approvals": [
                item.model_dump(mode="json")
                for item in self.list_approvals(workflow_id)
            ],
            "review_contexts": [
                context.model_dump(mode="json")
                for version in versions
                if (context := self.get_review_context(workflow_id, version.version_id))
                is not None
            ],
            "events": [item.model_dump(mode="json") for item in events],
            "execution_summary": {
                "usage_scope": "workflow_cumulative",
                "usage_accounting_version": (
                    2
                    if events and all(e.usage_accounting_version == 2 for e in events)
                    else 1
                ),
                "usage_complete": bool(events)
                and all(
                    e.usage_accounting_version == 2 and e.usage_complete for e in events
                ),
                "revision_count": sum(v.version > 1 for v in versions),
                "model_calls": total_model_calls,
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "total_tokens": total_tokens,
                "estimated_cost_usd": round(total_cost, 6),
                "usage_available": usage_available,
                "cost_available": cost_available,
                "retry_count": sum(max(0, item.attempt - 1) for item in events),
                "failed_stage_count": len(failed_events),
                "lock_renewals": sum(
                    int(item.get("renew_count", 0)) for item in lock_audits
                ),
                "lock_lost_count": sum(
                    1 for item in lock_audits if item.get("outcome") == "lost"
                ),
                "last_lock_outcome": (
                    lock_audits[-1].get("outcome") if lock_audits else None
                ),
            },
            "recoverability": {
                "recoverable": workflow.status.value
                in {"generating", "reviewing", "review_incomplete"}
                or (
                    workflow.status.value == "failed"
                    and workflow.current_stage.value
                    in {"knowledge_retrieval", "content_generation"}
                ),
                "resume_from": (
                    workflow.current_stage.value
                    if workflow.status.value
                    in {"failed", "generating", "reviewing", "review_incomplete"}
                    else None
                ),
            },
            "generation_resolution": (
                {
                    "generation_attempt_id": generation_attempt.generation_attempt_id,
                    "state": generation_attempt.state.value,
                    "brief_hash": generation_attempt.brief_hash,
                    "knowledge_snapshot_hash": generation_attempt.knowledge_snapshot_hash,
                    "generation_context_hash": generation_attempt.generation_context_hash,
                    "channels": generation_attempt.channels,
                    "results": [
                        {
                            "channel": item.channel,
                            "status": item.status.value,
                            "rejected_constraints": item.rejected_constraints,
                            "conflicts": [
                                conflict.model_dump(mode="json")
                                for conflict in item.conflicts
                            ],
                            "unresolved_fields": item.unresolved_fields,
                            "phase_attempts": item.phase_attempts,
                            "request_ids": {
                                "copywriting": item.copywriting_request_id,
                                "style": item.style_request_id,
                                "validation_revision": item.validation_revision_request_id,
                            },
                            "error_code": item.error_code,
                        }
                        for item in generation_results
                    ],
                }
                if generation_attempt
                else None
            ),
            "review_progress": review_progress,
            "review_aggregation": {
                "status": workflow.status.value,
                "approval_status": workflow.approval_status,
                "remediation": (
                    "retry_review"
                    if workflow.status.value == "review_incomplete"
                    else (
                        "human_required"
                        if workflow.approval_status == "human_required"
                        else (
                            "revision_allowed"
                            if workflow.status.value == "changes_requested"
                            else "none"
                        )
                    )
                ),
            },
            "citations": [
                {
                    "title": item.get("title", ""),
                    "source_url": item.get("source_url", ""),
                    "document_type": item.get("document_type", ""),
                    "score": item.get("score"),
                }
                for item in workflow.knowledge_snapshot.documents
            ],
        }
        from .eligibility import delivery_eligibility

        eligibility = delivery_eligibility(detail)
        execution_status = (
            "failed"
            if workflow.status.value == "failed"
            else (
                "partial_success"
                if workflow.status.value == "review_incomplete"
                else (
                    "pending"
                    if workflow.status.value in {"draft", "validated", "planned"}
                    else (
                        "running"
                        if workflow.status.value
                        in {
                            "retrieving",
                            "generating",
                            "reviewing",
                            "fact_checking",
                            "compliance_reviewing",
                        }
                        else (
                            "cancelled"
                            if workflow.status.value == "cancelled"
                            else "success"
                        )
                    )
                )
            )
        )
        detail.update(
            {
                key: eligibility[key]
                for key in (
                    "review_status",
                    "review_complete",
                    "approval_status",
                    "publish_allowed",
                )
            }
        )
        detail["execution_status"] = execution_status
        detail["delivery_eligibility"] = eligibility
        detail["review_aggregation"].update(
            {
                key: eligibility[key]
                for key in ("review_status", "review_complete", "publish_allowed")
            }
        )
        return detail

    def acquire_lock(self, workflow_id: str, ttl_seconds: int = 300) -> Optional[str]:
        token = uuid.uuid4().hex
        acquired = self.redis.set(
            self.lock_key(workflow_id), token, nx=True, ex=ttl_seconds
        )
        return token if acquired else None

    def release_lock(self, workflow_id: str, token: str) -> bool:
        return bool(
            self.redis.eval(self._UNLOCK_SCRIPT, 1, self.lock_key(workflow_id), token)
        )

    def renew_lock(self, workflow_id: str, token: str, ttl_seconds: int) -> bool:
        return bool(
            self.redis.eval(
                self._RENEW_LOCK_SCRIPT,
                1,
                self.lock_key(workflow_id),
                token,
                ttl_seconds,
            )
        )

    def owns_lock(self, workflow_id: str, token: str) -> bool:
        return bool(
            self.redis.eval(
                self._OWNS_LOCK_SCRIPT, 1, self.lock_key(workflow_id), token
            )
        )

    def health(self) -> bool:
        return bool(self.redis.ping())
