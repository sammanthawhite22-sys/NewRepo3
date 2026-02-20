# Astra AI - Copilot Instructions

## Project Overview

**Astra AI** is a safety-native AI system combining Go concurrency (worker pools) with safety validation (JSA Shield), semantic memory (vector embeddings + RAG), and LLM orchestration (multi-step planning + tool routing).

### Core Design Principle

Safety gates at every layer: inputs → orchestration planning → concurrency execution → semantic grounding → JSA validation → output. No AI output returns to user without passing JSA Shield analysis.

---

## Architecture (4 Layers)

### 1. **JSA Shield** (`jsa-shield/jsa.go`)
Hazard detection on all inputs/outputs.

**Key types:**
- `HazardLevel`: `Critical`, `High`, `Medium`, `Low` (config-driven patterns in `config.yaml`)
- `VerificationGate`: Function-based checks (mark `Critical: true` to fail-safe)
- `JSAAnalysis`: Result with `IsSafe`, `Explanation`, `PassedGates`, `FailedGates`

**Usage pattern:**
```go
shield := jsashield.NewJSAShield("MyShield")
shield.AddHazardPattern("malicious", jsashield.Critical)
shield.AddVerificationGate(jsashield.VerificationGate{
    Name: "CustomCheck",
    Check: func() bool { return userTrusted },
    Critical: true,
})
analysis := shield.Analyze(input)
if !analysis.IsSafe {
    logAudit(analysis) // Fallback with explanation
    return ErrorResponse
}
```
All user-facing outputs must pass `shield.Analyze()` before return.

---
`concurrency-engine/worker.go`)
Worker pool for parallel task processing.

**Key types:**
- `WorkerPool(numWorkers, queueSize)`: Manages N workers draining `TaskQueue`
- `Task`: `{ID, Input, Output, Error, Duration, Retries}`
- `PoolMetrics`: `{TotalTasks, CompletedTasks, FailedTasks, StartTime, EndTime}`

**Usage pattern:**
```go
pool := concurrency.NewWorkerPool(4, 100)
pool.Start()
tasks := []*concurrency.Task{{ID: "t1", Input: data1}, ...}
results := pool.Process(tasks)  // Waits for all to complete
pool.Close()  // Critical: signals workers to drain & exit
metrics := pool.GetMetrics()  // Check completion & failure rates
```
Results maintain order. Set `MaxRetries` and `TimeoutSecs` in config for resilience
**Critical detail:** Workers drain `TaskQueue` in a for-range loop; ensure `Close()` is called to signal completion.

---

### 3. **Semantic Memory + RAG** (Python, `semantic-memory/`)
Vector embeddings, caching, and document grounding with JSA audit trails.

**Key modules:**
- `memory.py`: `SemanticMemory(embedding_dim=768)` with `.remember(text, metadata)` → generates embeddings + cache; `.retrieve(query, top_k=5)` → ranked semantic results with scores
- `vector_db.py`: Pluggable backend (in-memory default; Pinecone via `PINECONE_API_KEY` env var); implements VectorRecord search
- `rag.py`: `RAGPipeline` retrieves docs, grounds LLM outputs, manages document sources (vector_store, knowledge_base, conversation)
- `SemanticCache`: LRU cache with TTL (default 3600s), configurable `cache_size: 1000`
- Config: `vector_dimension: 768`, `embedding_model: "sentence-transformers/all-MiniLM-L6-v2"`, `retrieval_threshold: 0.3`

**Usage pattern:**
```python
memory = SemanticMemory(embedding_dim=768)
memory.remember("Important context", metadata={"source": "docs"})
results = memory.retrieve("search query", top_k=5)  # [RetrievalResult{text, score, metadata}, ...]
memory.add_context("user", "What is safety?")  # Conversation history
context = memory.get_context(limit=10)  # Ordered conversation thread
rag = RAGPipeline(memory)
grounded = rag.ground_response(llm_output, original_query)
```

**Key insight:** All retrieval results include JSA audit IDs (`jsa_id` on Embedding) for audit tracing.

---

### 4. **Orchestration & Tools** (Python, `orchestration/`)
LLM routing, tool selection, and multi-step planning with sandboxed execution.

**Key types:**
- `ToolType` enum: `SEARCH`, `RETRIEVE`, `ANALYZE`, `GENERATE`, `CALCULATE`, `TRANSFORM`
- `Tool(name, description, tool_type, handler)`: Registered tools execute via `tool.execute(input_data)`
- `ExecutionPlan(plan_id, steps[])`: Multi-step decomposition with tool pre-assignment per step
- `PlanStep`: `{step_num, description, tool, input_data, dependencies, result}`
- `ToolDefinition` (in `tools.py`): Full tool definition with sandbox level, audit requirements, timeout controls
- `SafeToolExecutor`: Sandboxed execution with JSA verification before/after
- `Orchestrator.plan(query)` → `ExecutionPlan`; `.execute_plan(plan)` → results in order

**Usage pattern:**
```python
orchestrator = Orchestrator("MainOrch")
orchestrator.register_tool(Tool("search", "Search documents", ToolType.SEARCH, handler_func))
plan = orchestrator.plan("Find and analyze documents")
executed = orchestrator.execute_plan(plan)  # Runs sequentially, respects dependencies
```

**Tool Security:** Tools support sandbox levels (RESTRICTED, LIMITED, STANDARD, PRIVILEGED). Define execution time limits and output constraints in `ToolDefinition`. All invocations logged via `ToolInvocation` with JSA verification before return.

---

### 5. **Enterprise Features** (Python, `orchestration/`)
Compliance tracking, monitoring, and audit trails for SOC2/HIPAA.

**Compliance layer:**
- `ComplianceManager`: Central compliance event logging
- `AuditEvent`: Tracks event_type, actor, resource, action, status, data_classification
- `DataClassification`: PUBLIC, INTERNAL, CONFIDENTIAL, RESTRICTED
- `CompliancePolicy`: Track SOC2, HIPAA, GDPR, ISO27001 requirements
- Every operation logs audit events with event_id, timestamp, status

**Monitoring layer:**
- `OperationalMetrics`: Tracks TotalTasks, CompletedTasks, FailedTasks, AvgLatency
- `HealthStatus`: System health checks (all layers operational)
- Metrics exposed via `/stats` API endpoint

**Usage pattern:**
```python
compliance_mgr = ComplianceManager()
event = AuditEvent(
    event_type="QUERY_EXECUTED",
    actor="user_123",
    resource="llm_output",
    action="generate",
    data_classification=DataClassification.CONFIDENTIAL
)
event_id = compliance_mgr.log_event(event)  # Returns unique event_id
```
---

## Configuration

Central config in `config/config.yaml`. All modules load this at startup. Key sections:

| Section | Critical Settings | Default |
|---------|-------------------|---------|
| `concurrency_engine` | `workers`, `queue_size`, `timeout_seconds`, `max_retries` | 4 workers, 100 queue, 30s timeout, 3 retries |
| `jsa_shield` | `enabled: true`, `critical_check: true`, `hazard_patterns` | All patterns in config; add new ones there |
| `semantic_memory` | `vector_dimension`, `embedding_model`, `cache_size`, `retrieval_threshold` | 768-dim, sentence-transformers, 1000 cache, 0.3 threshold |
| `orchestration` | `llm_model`, `enable_planning: true`, `registered_tools` | gpt-4, multi-step planning enabled |
| `api` | `port`, `debug` | 5000, debug=true in dev |
| `safety` | `strict_mode: true`, `audit_all_outputs: true` | Always true in production |

Do not hardcode layer parameters; always read from config.

---

## Developer Workflows

### Run Full Integration Test

```powershell
cd c:\astra-ai
go run main.go  # Tests all 4 layers: JSA → Concurrency → Memory → Orchestration
```

### Run Python Integration Test

```powershell
# Requires semantic-memory, orchestration modules available
python test_integration.py  # Full 4-layer test: Orchestration → Memory → RAG
```

### Run Python API Server

```powershell
# Install dependencies from requirements.txt
pip install flask requests pyyaml python-dotenv pydantic
python api.py  # Starts Flask on :5000; exposes /query, /plan, /remember, /retrieve, /stats
```

### Testing a Single Component

**JSA Shield:** Add patterns to `config.yaml` under `jsa_shield.hazard_patterns`, then test in `main.go`'s JSA block.

**Concurrency Engine:** Modify task creation in `main.go`'s Layer 2 block; adjust worker count via config. Test with `pool.Process(tasks)` and inspect `pool.GetMetrics()`.

**Semantic Memory:** Call `.remember()` to store embeddings, `.retrieve()` for search; test in API `/remember` and `/retrieve` routes or directly in Python with `test_integration.py`.

**Orchestration & Tools:** Use `.plan()` to decompose query, `.execute_plan()` to run steps; inspect step tool assignments and results. Verify tool descriptions guide LLM routing correctly.

### Adding a New Tool

1. Define `Tool(name, description, ToolType.*, handler_func)` in `orchestration/tools.py`
2. Register with `Orchestrator().register_tool(tool)` in initialization
3. LLM will auto-route steps to registered tools based on description matching
4. Tool `handler` receives `input_data: Dict` and returns JSON-serializable output
5. Wrap with `SafeToolExecutor` for sandbox enforcement and JSA verification

---

## Data Flow & Safety Integration

```
User Query (REST API /query)
    ↓
[Orchestration] plan(query) → ExecutionPlan with pre-assigned tools
    ↓
[Concurrency] Submit plan steps as parallel tasks to WorkerPool
    ↓
[Semantic Memory] retrieve(query, top_k=5) → grounding docs
    ↓
[Tool Execution] Each tool's handler() runs in worker goroutine
    ↓
[RAG] Ground tool results in retrieved documents
    ↓
[JSA Shield] analyze(result) → IsSafe check on final output
    ↓
If safe: Return result to user
If not safe: Return fallback explanation from analysis
```

**Critical**: Every user-facing output passes JSA Shield. Tool handlers can fail/error without blocking pool; retries via `max_retries` config.

---

## Common Implementation Patterns

### Query Processing Pipeline (from `api.py`)

```python
# 1. Receive query
user_query = request.json["query"]

# 2. Plan multi-step execution
plan = self.orchestrator.plan(user_query)  # Tools pre-assigned

# 3. Retrieve grounding docs
docs = self.memory.retrieve(user_query, top_k=5)

# 4. Execute plan (concurrency engine runs tool handlers)
result = self.orchestrator.execute_plan(plan)

# 5. Return with metrics
return {
    "query": user_query,
    "plan_id": result.plan_id,
    "steps": result.steps,
    "retrieved_docs": len(docs)
}
```

### Safety-First Response

```go
// Process result through ALL layers
llmOutput := orchestrator.execute_plan(plan)
analysis := shield.Analyze(llmOutput)

if analysis.IsSafe {
    return llmOutput
} else {
    // Log audit event (includes analysis.Explanation)
    compliance.log_event(AuditEvent{
        event_type: "SAFETY_GATE_FAILED",
        resource: "llm_output",
        explanation: analysis.Explanation
    })
    return "I cannot provide that response: " + analysis.Explanation
}
```

### Parallel Task Submission

```go
// Decompose query into independent subtasks
subtasks := []*concurrency.Task{
    {ID: "parse", Input: query},
    {ID: "intent", Input: query},
    {ID: "context", Input: query},
}

pool := concurrency.NewWorkerPool(3, 10)
pool.Start()
results := pool.Process(subtasks)
pool.Close()

// results[0], results[1], results[2] in order
```

---

## Key Integration Points

| Module | Entry Point | Integrates With | Purpose |
|--------|-------------|-----------------|---------|
| `api.py` | Flask routes `/query`, `/plan`, `/remember` | Orchestrator, Memory, LLMClient | User-facing REST API |
| `orchestrator.py` | `.plan()`, `.execute_plan()` | Tool handlers, LLMClient | Multi-step reasoning |
| `semantic_memory/memory.py` | `.remember()`, `.retrieve()` | Vector DB, RAG pipeline | Context grounding |
| `orchestration/tools.py` | `ToolRegistry`, `SafeToolExecutor` | All layers | Sandboxed tool execution & routing |
| `orchestration/compliance.py` | `log_event(AuditEvent)` | All layers | Audit trail & SOC2/HIPAA compliance |
| `orchestration/monitoring.py` | `OperationalMetrics`, `HealthStatus` | All layers | Observability & health checks |
| `jsa-shield/jsa.go` | `shield.Analyze(input)` | All outputs | Safety validation gate |
| `concurrency-engine/worker.go` | `pool.Process(tasks)` | Tool handlers | Parallel execution |

**New features must route through**: Orchestration (+ compliance) → Tools → Concurrency → Semantic Memory → JSA Shield.

---

## Code Style & Conventions

- **Go**: CamelCase exports (`NewWorkerPool`, `Analyze`); snake_case unexported (`taskQueue`); handle channels with for-range and Close()
- **Python**: PEP 8; use `@dataclass` for type-safe data containers (see `orchestrator.py` for patterns)
- **Config**: YAML; nest under section key; use snake_case for keys
- **Naming**: Full words (`HazardLevel`, not `Hz`); descriptive type names (`ExecutionPlan`, not `Plan`)
- **Comments**: Explain why (e.g., "Critical gate blocks output if malicious patterns detected") not just what
- **Errors**: Return `JSAAnalysis` with explanation or `ToolResult` with error field; avoid silent failures

## Quick Reference

| File | Purpose | Key Types |
|------|---------|-----------|
| `jsa-shield/jsa.go` | Hazard detection & safety gates | `HazardLevel`, `VerificationGate`, `JSAAnalysis` |
| `concurrency-engine/worker.go` | Parallel task processing | `WorkerPool`, `Task`, `PoolMetrics` |
| `concurrency-engine/channels.go` | Channel coordination & fan-out | `Channel`, `FanOut` |
| `orchestration/orchestrator.py` | Multi-step planning & tool routing | `ExecutionPlan`, `PlanStep`, `Tool`, `ToolType` |
| `orchestration/tools.py` | Tool sandboxing & execution | `ToolDefinition`, `ToolSandboxLevel`, `SafeToolExecutor`, `ToolInvocation` |
| `semantic-memory/memory.py` | Vector search & RAG grounding | `SemanticMemory`, `Embedding`, `RetrievalResult`, `RAGPipeline` |
| `semantic-memory/vector_db.py` | Vector database backend | `VectorRecord`, pluggable DB (in-memory/Pinecone) |
| `api.py` | REST API with all 4 layers | `AstraAPI`, routes: `/query`, `/plan`, `/remember`, `/retrieve`, `/stats` |
| `config/config.yaml` | Central configuration | All layer parameters (workers, patterns, models, cache, etc.) |
| `main.go` | Go integration test | Demonstrates all 4 layers in sequence |
| `test_integration.py` | Python integration test | Tests Orchestration → Memory → RAG pipeline |
| `orchestration/compliance.py` | Audit trail & enterprise compliance | `ComplianceManager`, `AuditEvent`, `DataClassification`, `ComplianceStandard` |
| `orchestration/monitoring.py` | Observability & health checks | `OperationalMetrics`, `HealthStatus` |
