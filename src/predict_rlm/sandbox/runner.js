// Concurrent tool call runner for RLMs
// Based on DSPy's runner.js but with support for parallel async tool calls
// and skill package installation.
//
// Key difference: Multiple tool calls can be in-flight simultaneously.
// The host processes them in parallel and sends responses with matching IDs.
//
// Protocol: JSON-RPC 2.0 (matching dspy 3.1.3+)

import pyodideModule from "npm:pyodide/pyodide.js";
import { readLines } from "https://deno.land/std@0.186.0/io/mod.ts";

// =============================================================================
// JSON-RPC 2.0 Helpers
// =============================================================================

const JSONRPC_PROTOCOL_ERRORS = {
  ParseError: -32700,
  InvalidRequest: -32600,
  MethodNotFound: -32601,
};

const JSONRPC_APP_ERRORS = {
  SyntaxError: -32000,
  NameError: -32001,
  TypeError: -32002,
  ValueError: -32003,
  AttributeError: -32004,
  IndexError: -32005,
  KeyError: -32006,
  RuntimeError: -32007,
  CodeInterpreterError: -32008,
  Unknown: -32099,
};

const jsonrpcResult = (result, id) =>
  JSON.stringify({ jsonrpc: "2.0", result, id });

const jsonrpcError = (code, message, id, data = null) => {
  const err = { code, message };
  if (data) err.data = data;
  return JSON.stringify({ jsonrpc: "2.0", error: err, id });
};

const jsonrpcRequest = (method, params, id) =>
  JSON.stringify({ jsonrpc: "2.0", method, params, id });

/** Replace contiguous runs of binary characters (surrogates, control chars)
 *  with a human-readable placeholder so JSON.stringify produces valid UTF-8. */
const filterBinary = (s) => {
  if (typeof s !== 'string') return s;
  return s.replace(
    /[\uD800-\uDFFF\x00-\x08\x0B\x0C\x0E-\x1F]+/g,
    (m) => `<${m.length} binary bytes>`,
  );
};

// =============================================================================
// Python Code Templates
// =============================================================================

const PYTHON_SETUP_CODE = `
import sys, io, json
old_stdout, old_stderr = sys.stdout, sys.stderr
buf_stdout, buf_stderr = io.StringIO(), io.StringIO()
sys.stdout, sys.stderr = buf_stdout, buf_stderr

def last_exception_args():
    return json.dumps(sys.last_exc.args) if sys.last_exc else None

class FinalOutput(BaseException):
    # Control-flow exception to signal completion (like StopIteration)
    pass

# Restore tool globals from persistent module (survives async state corruption)
if '_repl_tools' in sys.modules:
    _repl_tools = sys.modules['_repl_tools']
    for _name in dir(_repl_tools):
        if not _name.startswith('_'):
            globals()[_name] = getattr(_repl_tools, _name)

# Default SUBMIT for single-output signatures (e.g., Program of Thought).
# Only define if not already registered with typed signatures.
if 'SUBMIT' not in dir():
    def SUBMIT(output):
        raise FinalOutput({"output": output})
`;

// Generate async tool wrapper - all tools are async for concurrent execution
// Store tools in a dedicated module namespace to survive async state corruption.
// We also create global aliases that reference the module for convenience.
const makeToolWrapper = (toolName, _parameters = []) => {
  // Special handling for predict tool - extract Pydantic schemas from signature
  if (toolName === "predict") {
    return makePredictWrapper();
  }

  return `
import sys, json, types
from pyodide.ffi import JsProxy

# Create persistent module for tools if it doesn't exist
if '_repl_tools' not in sys.modules:
    sys.modules['_repl_tools'] = types.ModuleType('_repl_tools')
_repl_tools = sys.modules['_repl_tools']

# Helper to serialize values (handles Pydantic models, dataclasses, etc.)
def _serialize_for_json(obj):
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_serialize_for_json(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _serialize_for_json(v) for k, v in obj.items()}
    # Pydantic v2
    if hasattr(obj, 'model_dump'):
        return obj.model_dump()
    # Pydantic v1
    if hasattr(obj, 'dict') and hasattr(obj, '__fields__'):
        return obj.dict()
    # Dataclass
    if hasattr(obj, '__dataclass_fields__'):
        import dataclasses
        return dataclasses.asdict(obj)
    # Fallback: try to convert to dict or str
    try:
        return dict(obj)
    except (TypeError, ValueError):
        return str(obj)

# Define tool function and store in module
async def _tool_${toolName}(*args, **kwargs):
    try:
        # Serialize args/kwargs safely (handles Pydantic models)
        safe_args = _serialize_for_json(list(args))
        safe_kwargs = _serialize_for_json(kwargs)
        payload = json.dumps({"args": safe_args, "kwargs": safe_kwargs})
    except Exception as e:
        raise TypeError(f"Failed to serialize arguments for ${toolName}: {e}")

    try:
        result = await _js_tool_call("${toolName}", payload)
        if isinstance(result, str):
            try:
                return json.loads(result)
            except (json.JSONDecodeError, ValueError):
                return result
        return result.to_py() if isinstance(result, JsProxy) else result
    except Exception as e:
        # Re-raise with context but don't corrupt state
        raise RuntimeError(f"Tool ${toolName} call failed: {e}") from e

setattr(_repl_tools, '${toolName}', _tool_${toolName})

# Also store the serialization helper in the module for reuse
setattr(_repl_tools, '_serialize_for_json', _serialize_for_json)

# Create global alias that delegates to module (survives even if global is cleared)
${toolName} = _repl_tools.${toolName}
`;
};

// Special wrapper for predict tool that extracts Pydantic schemas from signatures
const makePredictWrapper = () => {
  return `
import sys, json, types, re
from pyodide.ffi import JsProxy

# Create persistent module for tools if it doesn't exist
if '_repl_tools' not in sys.modules:
    sys.modules['_repl_tools'] = types.ModuleType('_repl_tools')
_repl_tools = sys.modules['_repl_tools']

# Helper to serialize values (handles Pydantic models, dataclasses, etc.)
def _serialize_for_json(obj):
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_serialize_for_json(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _serialize_for_json(v) for k, v in obj.items()}
    # Pydantic v2
    if hasattr(obj, 'model_dump'):
        return obj.model_dump()
    # Pydantic v1
    if hasattr(obj, 'dict') and hasattr(obj, '__fields__'):
        return obj.dict()
    # Dataclass
    if hasattr(obj, '__dataclass_fields__'):
        import dataclasses
        return dataclasses.asdict(obj)
    # Fallback: try to convert to dict or str
    try:
        return dict(obj)
    except (TypeError, ValueError):
        return str(obj)

# Helper to safely convert objects to JSON-serializable form
def _make_json_safe(obj):
    """Recursively ensure an object is JSON-serializable."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    elif isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_safe(item) for item in obj]
    else:
        # Convert any non-serializable object to string
        return str(obj)

# Helper to extract Pydantic schemas from signature for custom types
def _get_pydantic_schemas(sig):
    """Extract JSON schemas for custom Pydantic types referenced in a signature.

    Looks for capitalized type names (e.g., PageTask, Address) in the signature
    that aren't built-in DSPy types and have model_json_schema() method.

    Returns dict mapping type name to JSON schema.

    NOTE: When running inside asyncio.gather(), the call stack changes and frame
    introspection may not find user-defined models. We handle this by also checking
    the __main__ module's namespace directly, which persists across async contexts.
    """
    schemas = {}
    # Find ALL capitalized identifiers in the signature. This handles any
    # nesting depth (e.g. Optional[list[TaskItem]], list[Optional[TaskItem]]).
    # Non-Pydantic names are filtered by the model_json_schema() check below.
    pattern = r'(?<![.\\w])([A-Z][A-Za-z0-9_]*)'
    for match in re.finditer(pattern, sig):
        name = match.group(1)
        # Skip built-in DSPy types and Python typing
        if name in ('Image', 'List', 'Optional', 'Dict', 'Any', 'Union', 'Literal', 'Tuple', 'Set'):
            continue
        # Check if type exists and is a Pydantic model
        # First, check __main__ module directly - this works even in async contexts
        # where frame introspection fails (e.g., inside asyncio.gather)
        import sys
        cls = None

        # Priority 1: Check __main__ module globals (REPL-defined classes)
        main_module = sys.modules.get('__main__')
        if main_module and hasattr(main_module, '__dict__'):
            if name in main_module.__dict__:
                cls = main_module.__dict__[name]

        # Priority 2: Check _repl_tools module (persistent storage)
        if cls is None and '_repl_tools' in sys.modules:
            _repl_tools = sys.modules['_repl_tools']
            if hasattr(_repl_tools, name):
                cls = getattr(_repl_tools, name)

        # Priority 3: Traverse the call stack (works for sync calls)
        if cls is None:
            import inspect
            frame = inspect.currentframe()
            while frame:
                if name in frame.f_globals:
                    cls = frame.f_globals[name]
                    break
                if name in frame.f_locals:
                    cls = frame.f_locals[name]
                    break
                frame = frame.f_back

        if cls and hasattr(cls, 'model_json_schema'):
            try:
                schema = cls.model_json_schema()
                # Make the schema JSON-safe to prevent serialization errors
                # This handles any non-serializable objects that might be in the schema
                safe_schema = _make_json_safe(schema)
                # Verify it can be serialized
                json.dumps(safe_schema)
                schemas[name] = safe_schema
            except Exception as e:
                # Log but continue - schema extraction failure shouldn't break the call
                print(f"Warning: Failed to extract schema for {name}: {e}")
    return schemas

# DSPy Prediction-like wrapper: the top-level predict() return supports
# both attribute access (result.answer) and subscript access (result["answer"]).
# Uses composition (_store) + __getattribute__ so data fields ALWAYS win over
# methods. result.items returns the stored list, not a method — eliminating the
# dict-method collision that wastes iterations on "builtin_function_or_method
# is not iterable" errors.
class _PredictResult:
    __slots__ = ('_store',)

    def __init__(self, d):
        object.__setattr__(self, '_store', dict(d) if d else {})

    def __getattribute__(self, key):
        # Data fields take priority over class methods. This is the key
        # difference from __getattr__ (which only fires AFTER MRO lookup).
        if not key.startswith('_'):
            store = object.__getattribute__(self, '_store')
            if key in store:
                return store[key]
        return object.__getattribute__(self, key)

    def __getitem__(self, key):
        return object.__getattribute__(self, '_store')[key]

    def __setitem__(self, key, value):
        object.__getattribute__(self, '_store')[key] = value

    def __contains__(self, key):
        return key in object.__getattribute__(self, '_store')

    def __iter__(self):
        return iter(object.__getattribute__(self, '_store'))

    def __len__(self):
        return len(object.__getattribute__(self, '_store'))

    def __repr__(self):
        return f"PredictResult({object.__getattribute__(self, '_store')!r})"

    # Fallback methods — only reached when no data field has this name
    # (because __getattribute__ checks _store first).
    def keys(self):
        return list(object.__getattribute__(self, '_store').keys())

    def values(self):
        return list(object.__getattribute__(self, '_store').values())

    def items(self):
        return list(object.__getattribute__(self, '_store').items())

    def get(self, key, default=None):
        return object.__getattribute__(self, '_store').get(key, default)

# Reconstruct Pydantic model instances from predict result dicts.
# When the LM defines e.g. TaskItem and uses "-> tasks: list[TaskItem]",
# the host returns plain dicts (JSON-RPC).  This wraps them back into
# model instances so that attribute access (task.title) works naturally.
# The final result is wrapped in _PredictResult so the top level also
# supports attribute access (result.tasks) and matches DSPy conventions.
def _reconstruct_output_types(sig, result):
    if not isinstance(result, dict):
        return result
    import re as _re, sys as _sys
    outputs_part = sig.split("->")[1] if "->" in sig else ""
    _builtins = {'Image', 'List', 'Dict', 'Any', 'Optional', 'Union',
                 'Literal', 'Tuple', 'Set', 'BaseModel'}
    # Match field: Type, field: list[Type], field: Optional[Type],
    # field: Optional[list[Type]], etc. The wrapper group explicitly matches
    # any sequence of Optional[, list[, List[ prefixes before the type name.
    for match in _re.finditer(r'(\\w+)\\s*:\\s*((?:Optional\\[|list\\[|List\\[)*)\\s*([A-Z][A-Za-z0-9_]*)', outputs_part):
        field_name = match.group(1)
        wrapper = match.group(2) or ''
        type_name = match.group(3)
        is_list = 'list[' in wrapper.lower()
        if type_name in _builtins:
            continue
        # Look up model class in sandbox scope
        cls = None
        _main = _sys.modules.get('__main__')
        if _main and hasattr(_main, type_name):
            cls = getattr(_main, type_name)
        if cls is None and '_repl_tools' in _sys.modules:
            cls = getattr(_sys.modules['_repl_tools'], type_name, None)
        if cls is None or not hasattr(cls, 'model_validate') or not hasattr(cls, 'model_fields'):
            continue
        # Subclass with extra='allow' so the LM can add fields after prediction
        from pydantic import ConfigDict as _ConfigDict
        cls = type(cls.__name__, (cls,), {
            'model_config': _ConfigDict(extra='allow'),
        })
        # Reconstruct dicts into model instances. With extra='allow' on the
        # subclass, extra fields survive — no data is lost.
        value = result.get(field_name)
        if value is None:
            continue
        try:
            if is_list and isinstance(value, list):
                if all(isinstance(item, dict) for item in value):
                    result[field_name] = [cls.model_validate(item) for item in value]
            elif isinstance(value, dict):
                result[field_name] = cls.model_validate(value)
        except Exception:
            pass  # Fall back to plain dicts if reconstruction fails
    return _PredictResult(result)

# Define predict tool with schema extraction
async def _tool_predict(*args, **kwargs):
    try:
        # Extract signature from args or kwargs
        sig = args[0] if args else kwargs.get('signature', '')

        # Ensure sig is a string (defensive programming)
        if not isinstance(sig, str):
            sig = str(sig) if sig else ''

        # Get Pydantic schemas for custom types in the signature
        pydantic_schemas = _get_pydantic_schemas(sig)

        # Serialize args/kwargs safely (handles Pydantic models)
        safe_args = _serialize_for_json(list(args))
        safe_kwargs = _serialize_for_json(kwargs)

        # Build payload with optional schemas
        payload_dict = {"args": safe_args, "kwargs": safe_kwargs}
        if pydantic_schemas:
            payload_dict["pydantic_schemas"] = pydantic_schemas
        payload = json.dumps(payload_dict)
    except Exception as e:
        raise TypeError(f"Failed to serialize arguments for predict: {e}")

    try:
        result = await _js_tool_call("predict", payload)
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except (json.JSONDecodeError, ValueError):
                pass
        else:
            result = result.to_py() if isinstance(result, JsProxy) else result
        # Reconstruct Pydantic model instances for output fields so that
        # both task.title (attribute) and task["title"] (subscript) work.
        if isinstance(result, dict):
            result = _reconstruct_output_types(sig, result)
        return result
    except Exception as e:
        # Re-raise with context but don't corrupt state
        raise RuntimeError(f"Tool predict call failed: {e}") from e

setattr(_repl_tools, 'predict', _tool_predict)

# Also store helpers in the module for reuse
setattr(_repl_tools, '_serialize_for_json', _serialize_for_json)
setattr(_repl_tools, '_make_json_safe', _make_json_safe)
setattr(_repl_tools, '_get_pydantic_schemas', _get_pydantic_schemas)

# Create global alias that delegates to module (survives even if global is cleared)
predict = _repl_tools.predict
`;
};

const makeSubmitWrapper = (outputs) => {
  // Inlined Pydantic-aware serializer. SUBMIT accepts Pydantic instances
  // (top-level, nested in dicts, inside lists) and normalizes them to plain
  // JSON-safe types before raising FinalOutput. Without this, FinalOutput's
  // dict contains Pydantic instances that fail JSON serialization at the
  // transport boundary, silently producing a confusing 'FINAL returned
  // NoneType' error. Falls back to str() for anything else unserializable
  // so FinalOutput is never unparseable at the transport boundary.
  const serializerDef = `
import json as _submit_json
def _submit_to_jsonsafe(_v):
    if _v is None or isinstance(_v, (str, int, float, bool)):
        return _v
    if hasattr(_v, 'model_dump'):
        return _submit_to_jsonsafe(_v.model_dump(mode='python'))
    if hasattr(_v, 'dict') and hasattr(_v, '__fields__'):
        return _submit_to_jsonsafe(_v.dict())
    if isinstance(_v, dict):
        return {_k: _submit_to_jsonsafe(_x) for _k, _x in _v.items()}
    if isinstance(_v, (list, tuple, set)):
        return [_submit_to_jsonsafe(_x) for _x in _v]
    try:
        _submit_json.dumps(_v)
        return _v
    except (TypeError, ValueError):
        return str(_v)
`;

  if (!outputs || outputs.length === 0) {
    return `${serializerDef}
def SUBMIT(output):
    raise FinalOutput(_submit_to_jsonsafe({"output": output}))
`;
  }

  const sigParts = outputs.map(o => {
    let part = o.name;
    if (o.type) part += `: ${o.type}`;
    return part;
  });
  const dictParts = outputs.map(o => `"${o.name}": ${o.name}`);

  return `${serializerDef}
def SUBMIT(${sigParts.join(', ')}):
    raise FinalOutput(_submit_to_jsonsafe({${dictParts.join(', ')}}))
`;
};

// =============================================================================
// Concurrent Tool Call Support
// =============================================================================

globalThis.addEventListener("unhandledrejection", (event) => {
  event.preventDefault();
  console.log(jsonrpcError(
    JSONRPC_APP_ERRORS.Unknown,
    `Unhandled async error: ${event.reason?.message || event.reason}`,
    null
  ));
});

// Suppress console.log during initialization to avoid corrupting JSON protocol
const originalLog = console.log;
console.log = () => {};

const pyodide = await pyodideModule.loadPyodide({
  stdout: () => {},
  stderr: (msg) => console.error(msg),
});

// Pre-install packages only if PYODIDE_PREINSTALL=1 (default for real usage, skip for fast tests)
const preinstall = Deno.env.get("PYODIDE_PREINSTALL") !== "0";
if (preinstall) {
  await pyodide.loadPackage("micropip");

  // Collect all packages to install: defaults + skill packages
  const defaultPackages = ['pandas', 'pydantic'];
  const skillPackagesEnv = Deno.env.get("SKILL_PACKAGES") || "";
  const skillPackages = skillPackagesEnv
    .split(",")
    .map(p => p.trim())
    .filter(Boolean);
  const allPackages = [...defaultPackages, ...skillPackages];

  await pyodide.runPythonAsync(`
import micropip

# Install packages silently
await micropip.install(${JSON.stringify(allPackages)}, verbose=False)

# Import default packages now so they're ready and cached
import pandas
import pydantic
`);
}

// Restore stdout/console.log for normal operation (JSON output)
console.log = originalLog;
pyodide.setStdout({ batched: (msg) => console.log(msg) });

const stdinReader = readLines(Deno.stdin);
let requestIdCounter = 0;

// Store registered tools so we can re-inject them before each execution
// This protects against state corruption during long async executions
let registeredTools = [];  // [{name, params}, ...]
let registeredOutputs = null;  // output field definitions

// Pending tool calls waiting for responses
const pendingToolCalls = new Map(); // requestId -> { resolve, reject }

// Flag indicating code execution is in progress (tool calls may arrive)
let codeExecutionInProgress = false;
let responseReaderPromise = null;

// Cancellation flag for the response reader - allows fast exit on error
let responseReaderCancelled = false;

// Tool call bridge - sends JSON-RPC request and returns promise
// Multiple calls can be in-flight simultaneously
async function toolCallBridge(name, argsJson) {
  const requestId = `tc_${Date.now()}_${++requestIdCounter}`;

  // Create promise that will be resolved when response arrives
  const resultPromise = new Promise((resolve, reject) => {
    pendingToolCalls.set(requestId, { resolve, reject });
  });

  // Parse args to extract positional and keyword args
  const parsedArgs = JSON.parse(argsJson);

  // Send tool call request to host as JSON-RPC request
  console.log(jsonrpcRequest("tool_call", {
    name: name,
    args: parsedArgs.args || [],
    kwargs: parsedArgs.kwargs || {},
    ...(parsedArgs.pydantic_schemas ? { pydantic_schemas: parsedArgs.pydantic_schemas } : {}),
  }, requestId));

  // Wait for our specific response (dispatcher will route it)
  return resultPromise;
}

// Handle a JSON-RPC tool response by routing it to the pending request
function handleToolResponse(response) {
  const pending = pendingToolCalls.get(response.id);
  if (!pending) {
    console.error(JSON.stringify({
      warning: `Received response for unknown request: ${response.id}`
    }));
    return false;
  }

  pendingToolCalls.delete(response.id);

  if (response.error) {
    pending.reject(new Error(response.error.message || "Tool call failed"));
  } else {
    const result = response.result;
    pending.resolve(result.value);
  }
  return true;
}

// Helper to create a timeout promise
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// Background response reader - runs during code execution
// Reads tool responses and routes them to pending requests
// Uses non-blocking reads with timeout to allow fast cancellation
async function responseReader() {
  // Track if we have an in-flight read that needs to be resolved
  let pendingRead = null;

  while (!responseReaderCancelled && (codeExecutionInProgress || pendingToolCalls.size > 0)) {
    // Check if we have pending requests to wait for
    if (pendingToolCalls.size === 0 && codeExecutionInProgress) {
      // No pending requests but code is still running, yield and check again
      await sleep(10);
      continue;
    }

    // Start a read if we don't have one pending
    if (!pendingRead) {
      pendingRead = stdinReader.next();
    }

    // Race between the read and a timeout
    // This allows us to check the cancellation flag periodically
    const result = await Promise.race([
      pendingRead.then(r => ({ ...r, timeout: false })),
      sleep(100).then(() => ({ timeout: true }))
    ]);

    if (result.timeout) {
      // Timeout - check cancellation flag and loop
      continue;
    }

    // Got a result, clear the pending read
    pendingRead = null;

    if (result.done) {
      // stdin closed, reject all pending
      for (const [id, pending] of pendingToolCalls) {
        pending.reject(new Error("stdin closed while waiting for tool response"));
      }
      pendingToolCalls.clear();
      break;
    }

    try {
      const response = JSON.parse(result.value);
      // JSON-RPC response: has id and either result or error
      if (response.id && (response.result !== undefined || response.error !== undefined)) {
        handleToolResponse(response);
      } else {
        console.error(JSON.stringify({
          warning: `Unexpected message during execution: ${result.value}`
        }));
      }
    } catch (e) {
      console.error(JSON.stringify({
        warning: `Failed to parse response: ${e.message}`
      }));
    }
  }
}

// Expose the bridge to Python
pyodide.globals.set("_js_tool_call", toolCallBridge);

// Environment variables setup
try {
  const env_vars = (Deno.args[0] ?? "").split(",").filter(Boolean);
  for (const key of env_vars) {
    const val = Deno.env.get(key);
    if (val !== undefined) {
      pyodide.runPython(`
import os
os.environ[${JSON.stringify(key)}] = ${JSON.stringify(val)}
      `);
    }
  }
} catch (e) {
  console.error("Error setting environment variables in Pyodide:", e);
}

// =============================================================================
// Main Loop — JSON-RPC 2.0 dispatch
// =============================================================================

while (true) {
  const { value: line, done } = await stdinReader.next();
  if (done) break;

  let input;
  try {
    input = JSON.parse(line);
  } catch (error) {
    console.log(jsonrpcError(
      JSONRPC_PROTOCOL_ERRORS.ParseError,
      "Invalid JSON input: " + error.message,
      null
    ));
    continue;
  }

  if (typeof input !== 'object' || input === null) {
    console.log(jsonrpcError(
      JSONRPC_PROTOCOL_ERRORS.InvalidRequest,
      "Input is not a JSON object",
      null
    ));
    continue;
  }

  // --- JSON-RPC request (has method field) ---
  if (input.method) {
    const method = input.method;
    const params = input.params || {};
    const requestId = input.id; // undefined for notifications

    // sync_file — notification (no response)
    if (method === "sync_file") {
      try {
        await Deno.writeFile(
          params.host_path || params.virtual_path,
          pyodide.FS.readFile(params.virtual_path)
        );
      } catch (e) {
        console.error(`sync_file failed for ${params.virtual_path}: ${e.message || e}`);
      }
      continue;
    }

    // shutdown
    if (method === "shutdown") break;

    // mount_file
    if (method === "mount_file") {
      const hostPath = params.host_path;
      const virtualPath = params.virtual_path || hostPath;
      try {
        const contents = await Deno.readFile(hostPath);
        const dirs = virtualPath.split('/').slice(1, -1);
        let cur = '';
        for (const d of dirs) {
          cur += '/' + d;
          try {
            pyodide.FS.mkdir(cur);
          } catch (e) {
            const isExists = e.errno === 20
              || e.code === 'EEXIST'
              || (e.message && e.message.includes('File exists'));
            if (!isExists) {
              throw e;
            }
          }
        }
        pyodide.FS.writeFile(virtualPath, contents);
        console.log(jsonrpcResult({ mounted: virtualPath }, requestId));
      } catch (e) {
        console.log(jsonrpcError(
          JSONRPC_APP_ERRORS.RuntimeError,
          "Failed to mount file: " + e.message,
          requestId
        ));
      }
      continue;
    }

    // mkdir_p — create directory and all parents in MEMFS
    if (method === "mkdir_p") {
      const dirPath = params.path;
      try {
        const parts = dirPath.split('/').filter(Boolean);
        let cur = '';
        for (const d of parts) {
          cur += '/' + d;
          try {
            pyodide.FS.mkdir(cur);
          } catch (e) {
            // Pyodide FS throws ErrnoError with errno=20 (EEXIST) — ignore it
            const isExists = e.errno === 20
              || e.code === 'EEXIST'
              || (e.message && e.message.includes('File exists'));
            if (!isExists) {
              throw e;
            }
          }
        }
        console.log(jsonrpcResult({ created: dirPath }, requestId));
      } catch (e) {
        console.log(jsonrpcError(
          JSONRPC_APP_ERRORS.RuntimeError,
          "Failed to create directory: " + (e.message || String(e)),
          requestId
        ));
      }
      continue;
    }

    // list_dir — recursively list all files under a virtual directory
    if (method === "list_dir") {
      const dirPath = params.path;
      try {
        const files = [];
        function walk(dir) {
          let entries;
          try {
            entries = pyodide.FS.readdir(dir);
          } catch (e) {
            return; // directory doesn't exist
          }
          for (const entry of entries) {
            if (entry === '.' || entry === '..') continue;
            const fullPath = dir + '/' + entry;
            const stat = pyodide.FS.stat(fullPath);
            if (pyodide.FS.isFile(stat.mode)) {
              files.push(fullPath);
            } else if (pyodide.FS.isDir(stat.mode)) {
              walk(fullPath);
            }
          }
        }
        walk(dirPath);
        console.log(jsonrpcResult({ files: files }, requestId));
      } catch (e) {
        console.log(jsonrpcError(
          JSONRPC_APP_ERRORS.RuntimeError,
          "Failed to list directory: " + e.message,
          requestId
        ));
      }
      continue;
    }

    // inject_var — write large variable to virtual FS
    if (method === "inject_var") {
      const { name, value } = params;
      try {
        try { pyodide.FS.mkdir('/tmp'); } catch (e) { /* exists */ }
        try { pyodide.FS.mkdir('/tmp/dspy_vars'); } catch (e) { /* exists */ }
        pyodide.FS.writeFile(
          `/tmp/dspy_vars/${name}.json`,
          new TextEncoder().encode(value)
        );
        console.log(jsonrpcResult({ injected: name }, requestId));
      } catch (e) {
        console.log(jsonrpcError(
          JSONRPC_APP_ERRORS.RuntimeError,
          `Failed to inject var: ${e.message}`,
          requestId
        ));
      }
      continue;
    }

    // register — tools and/or output fields
    if (method === "register") {
      const toolNames = [];

      if (params.tools) {
        registeredTools = [];
        for (const tool of params.tools) {
          const name = typeof tool === 'string' ? tool : tool.name;
          const toolParams = typeof tool === 'string' ? [] : (tool.parameters || []);

          registeredTools.push({ name, params: toolParams });
          pyodide.runPython(makeToolWrapper(name, toolParams));
          toolNames.push(name);
        }
      }

      if (params.outputs) {
        registeredOutputs = params.outputs;
        pyodide.runPython(makeSubmitWrapper(params.outputs));
      }

      console.log(jsonrpcResult({
        tools: toolNames,
        outputs: params.outputs ? params.outputs.map(o => o.name) : []
      }, requestId));
      continue;
    }

    // execute — run Python code
    if (method === "execute") {
      const code = params.code || "";

      try {
        // Suppress ALL stdout during package loading to avoid corrupting JSON protocol
        const originalLog = console.log;
        console.log = () => {};
        pyodide.setStdout({ batched: () => {} });
        try {
          await pyodide.loadPackagesFromImports(code, {
            messageCallback: () => {},
            errorCallback: (msg) => console.error(msg),
          });
        } finally {
          console.log = originalLog;
          pyodide.setStdout({ batched: (msg) => console.log(msg) });
        }

        pyodide.runPython(PYTHON_SETUP_CODE);

        // Re-inject tool wrappers before each execution to protect against state corruption.
        for (const { name, params: toolParams } of registeredTools) {
          pyodide.runPython(makeToolWrapper(name, toolParams));
        }
        if (registeredOutputs) {
          pyodide.runPython(makeSubmitWrapper(registeredOutputs));
        }

        // Start response reader for concurrent tool calls
        codeExecutionInProgress = true;
        responseReaderPromise = responseReader();

        // Run the user's code
        const result = await pyodide.runPythonAsync(code);

        // Signal code execution complete
        codeExecutionInProgress = false;

        // Always wait for responseReader to fully stop before reading next command.
        await responseReaderPromise;

        const capturedStdout = pyodide.runPython("buf_stdout.getvalue()");
        pyodide.runPython("sys.stdout, sys.stderr = old_stdout, old_stderr");

        let output = (result === null || result === undefined) ? capturedStdout : (result.toJs?.() ?? result);
        if (typeof output === 'string') output = filterBinary(output);
        console.log(jsonrpcResult({ output }, requestId));
      } catch (error) {
        codeExecutionInProgress = false;

        // Signal the response reader to stop immediately
        responseReaderCancelled = true;

        // Wait for responseReader to exit (should be within 100ms due to timeout loop)
        if (responseReaderPromise) {
          try {
            await responseReaderPromise;
          } catch (e) {
            // Ignore errors from response reader cleanup
          }
        }

        // Cancel any pending tool calls (they won't get responses now)
        for (const [id, pending] of pendingToolCalls) {
          pending.reject(new Error("Code execution failed, cancelling pending tool calls"));
        }
        pendingToolCalls.clear();

        // Reset cancellation flag for next execution
        responseReaderCancelled = false;

        // Restore stdout/stderr so next iteration works correctly
        try {
          pyodide.runPython("sys.stdout, sys.stderr = old_stdout, old_stderr");
        } catch (e) {
          // Ignore errors during cleanup
        }

        const errorType = error.type || "Error";
        const errorMessage = (error.message || "").trim();

        // FinalOutput is a success — it's the SUBMIT signal, not an error
        if (errorType === "FinalOutput") {
          let answer = null;
          try {
            const last_exception_args = pyodide.globals.get("last_exception_args");
            const args = JSON.parse(last_exception_args());
            answer = args?.[0] ?? null;
          } catch (e) {
            // SUBMIT's built-in serializer should make this unreachable
            // (Pydantic-aware + str() fallback for unknown types). If it
            // still fires, surface the error instead of silently dropping
            // the submitted value.
            console.error(
              `[submit] Failed to capture FinalOutput args (${e}); ` +
              `answer will be null. This indicates the SUBMIT serializer ` +
              `couldn't make a value JSON-safe — please report.`,
            );
          }
          console.log(jsonrpcResult({ final: answer }, requestId));
          continue;
        }

        // Map Python error type to JSON-RPC error code
        const errorCode = JSONRPC_APP_ERRORS[errorType] ?? JSONRPC_APP_ERRORS.Unknown;
        let errorArgs = [];
        try {
          const last_exception_args = pyodide.globals.get("last_exception_args");
          errorArgs = JSON.parse(last_exception_args()) || [];
        } catch (e) {
          // Ignore errors getting exception args
        }
        console.log(jsonrpcError(
          errorCode,
          errorMessage,
          requestId,
          { type: errorType, args: errorArgs }
        ));
      }
      continue;
    }

    // Unknown method
    console.log(jsonrpcError(
      JSONRPC_PROTOCOL_ERRORS.MethodNotFound,
      `Method not found: ${method}`,
      requestId
    ));
    continue;
  }

  // --- JSON-RPC response (tool response from host, has result/error + id) ---
  if (input.id && (input.result !== undefined || input.error !== undefined)) {
    handleToolResponse(input);
    continue;
  }

  // Unknown message format
  console.log(jsonrpcError(
    JSONRPC_PROTOCOL_ERRORS.InvalidRequest,
    "Invalid Request: not a JSON-RPC 2.0 message",
    null
  ));
}
