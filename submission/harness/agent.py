"""
The agentic loop. One task is one conversation; each turn the model returns a
single JSON action:

    tool    -> run a measurement/render tool and report back
    submit  -> build the script, render it, diff it against the original
    finish  -> accept the last successful build

The opening message carries the B-rep digest and two labelled views, so the
model never has to guess a dimension. Every build returns a numeric diff, which
catches the silent no-op edit that a render cannot. If no build survives, the
unmodified input is emitted rather than nothing, since a missing output scores
a hard 0.0.
"""

import hashlib
import json
import os
import os.path as osp
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from harness import llm, prompts
from harness.cq import client as cq_client
from harness.io_adapter import TaskSpec
from harness.reporting import diff_reports, format_diff, format_report, truncate
from harness.tools import ToolContext, get_tool, run_tool


def _strip_fence(script: str) -> str:
    """Unwrap a markdown fence the model put inside the JSON string."""
    text = script.strip()
    if not text.startswith("```"):
        return script
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# settings
# ----------------------------------------------------------------------------

@dataclass
class AgentSettings:
    """Loop behaviour. Overridden per-run from the `lw_harness` config block."""

    max_steps: int = 14           # total model turns (tool calls + builds)
    max_builds: int = 6           # how many times a script may be executed

    # --- opening turn ------------------------------------------------------
    bootstrap_brep_json: bool = True
    bootstrap_views: List[str] = field(
        default_factory=lambda: ["iso_top_right", "iso_bottom_left"]
    )
    bootstrap_draw_bbox: bool = True
    bootstrap_label_faces: bool = True
    brep_json_max_chars: int = 6000
    views_resolution: int = 900
    # Off by default; see brep_render.
    views_draw_edges: bool = False

    # Dataset's own views. Empty by default - the bootstrap views supersede them.
    input_views: List[str] = field(default_factory=list)
    build_views: List[str] = field(default_factory=lambda: ["toprightiso"])
    final_check_views: List[str] = field(
        default_factory=lambda: ["toprightiso", "front", "top"]
    )
    render_size: int = 800
    keep_image_turns: int = 2     # older turns get their images stripped
    build_timeout: int = 180
    analyze_timeout: int = 120
    fallback_to_input: bool = True
    stdout_limit: int = 2500

    @classmethod
    def from_dict(cls, data: Optional[dict]):
        data = data or {}
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class AgentResult:
    status: str = "failed"        # finished | best_effort | fallback | failed
    step_path: Optional[str] = None
    script: Optional[str] = None
    used_fallback: bool = False
    builds: int = 0
    steps: int = 0
    usage: Dict[str, int] = field(default_factory=dict)
    transcript: List[dict] = field(default_factory=list)
    error: Optional[str] = None


# ----------------------------------------------------------------------------
# agent
# ----------------------------------------------------------------------------

class EditAgent:
    def __init__(self, client, model_config: dict, settings: AgentSettings):
        self.client = client
        self.model_config = model_config
        self.settings = settings

    # -- helpers -------------------------------------------------------------

    def _accumulate(self, totals: Dict[str, int], usage: Dict[str, int]):
        for key, value in (usage or {}).items():
            try:
                totals[key] = totals.get(key, 0) + int(value)
            except (TypeError, ValueError):
                continue

    def _prune_images(self, messages: List[dict]):
        """
        Drop images from all but the newest `keep_image_turns` user turns.

        Turn 0 is exempt: it holds the original part's views and is the cached
        prefix.
        """
        keep = self.settings.keep_image_turns
        turns_with_images = [
            i for i, m in enumerate(messages)
            if i > 0 and m["role"] == "user"
            and any(p["type"] == "image" for p in m["content"])
        ]
        for index in turns_with_images[:-keep] if keep > 0 else turns_with_images:
            message = messages[index]
            dropped = sum(1 for p in message["content"] if p["type"] == "image")
            message["content"] = [p for p in message["content"] if p["type"] != "image"]
            message["content"].append(
                llm.text_part(f"[{dropped} image(s) from this earlier turn omitted]")
            )

    def _visible_tools(self, steps_left: int) -> List[str]:
        names = []
        for entry in prompts.iter_tools():
            if entry.expensive and steps_left <= 2:
                continue
            names.append(entry.name)
        return names

    # -- opening context ------------------------------------------------------

    def _bootstrap_context(self, task: TaskSpec, ctx: ToolContext,
                           settings: AgentSettings, opening: List[Any]) -> bool:
        """
        Seed the first message with the B-rep digest and two opposite isometrics,
        so no step is spent discovering the part. Labels share ids with the JSON.

        Returns (attached_any_image, sent_brep_digest).
        """
        sent_brep = False
        if settings.bootstrap_brep_json:
            json_path = osp.join(ctx.work_dir, "brep_json", "input_brep.json")
            result = cq_client.step_to_json(
                step_file=task.input_step,
                json_path=json_path,
                max_chars=settings.brep_json_max_chars,
                timeout=settings.analyze_timeout,
            )
            if result.get("ok"):
                opening.append(
                    "B-rep structure of the ORIGINAL part (face/edge ids here match "
                    "the labels drawn on the views below):\n" + result["digest"]
                )
                ctx.extras["input_brep_json"] = json_path
                sent_brep = True
            else:
                print(f"    [warn] bootstrap step_to_json failed: {result.get('error')}")

        if not settings.bootstrap_views:
            return False, sent_brep

        result = cq_client.step_to_views(
            step_file=task.input_step,
            output_dir=osp.join(ctx.work_dir, "views"),
            views=settings.bootstrap_views,
            draw_bbox=settings.bootstrap_draw_bbox,
            label_faces=settings.bootstrap_label_faces,
            draw_edges=settings.views_draw_edges,
            resolution=settings.views_resolution,
            prefix="input",
            timeout=settings.build_timeout,
        )
        images = [p for p in (result.get("images") or []) if osp.exists(p)]
        if not images:
            print(f"    [warn] bootstrap step_to_views failed: {result.get('error')}")
            return False, sent_brep

        note = (f"Views of the ORIGINAL part ({', '.join(settings.bootstrap_views)}), "
                f"with the bounding box and its real corner coordinates overlaid")
        if settings.bootstrap_label_faces:
            note += ", and bbox-touching faces tagged with their B-rep ids"
        if result.get("multi_solid"):
            note += (f". This part is an assembly of {result['num_solids']} separate "
                     f"bodies, all drawn together; face ids are prefixed S<body>")
        opening.append(note + ":")
        opening.extend(llm.image_part(p) for p in images)
        return True, sent_brep

    # -- the loop ------------------------------------------------------------

    def run(self, task: TaskSpec, work_dir: str) -> AgentResult:
        os.makedirs(work_dir, exist_ok=True)
        settings = self.settings
        result = AgentResult()
        totals: Dict[str, int] = {}
        transcript: List[dict] = []

        # --- 0. measure the input before saying a word to the model ----------
        analysis = cq_client.analyze(task.input_step, timeout=settings.analyze_timeout)
        input_report = analysis.get("report") if analysis.get("ok") else None
        if input_report is None:
            print(f"    [warn] could not analyse input: {analysis.get('error')}")

        ctx = ToolContext(
            input_step=task.input_step,
            work_dir=work_dir,
            request_text=task.request_text,
            input_report=input_report,
        )
        ctx.extras["view_settings"] = {
            "resolution": settings.views_resolution,
            "draw_edges": settings.views_draw_edges,
        }

        # --- 1. opening turn: instruction + measurements + pictures ----------
        opening: List[Any] = [prompts.build_task_instruction(task.request_text)]

        bootstrap_images, sent_brep = self._bootstrap_context(task, ctx, settings, opening)

        # The digest supersedes the summary report.
        if input_report and not sent_brep:
            opening.append(format_report(input_report, title="Geometry of the ORIGINAL part"))

        opening.append(
            f"The STEP file will be at args['input_file'] "
            f"(basename: {osp.basename(task.input_step)})."
        )

        # Optional: the dataset's own pre-rendered views.
        images = task.input_images[: len(settings.input_views)]
        if images and settings.input_views:
            names = ", ".join(
                osp.splitext(osp.basename(p))[0].split("_")[-1] for p in images
            )
            opening.append(f"Additional pre-rendered views of the ORIGINAL part ({names}):")
            opening.extend(llm.image_part(p) for p in images)

        opening.append(
            prompts.FIRST_BUILD_NUDGE_WITH_CONTEXT if bootstrap_images
            else prompts.FIRST_BUILD_NUDGE
        )
        messages: List[dict] = [llm.user(*opening)]

        # --- 2. iterate -------------------------------------------------------
        best_step: Optional[str] = None
        best_script: Optional[str] = None
        finished = False
        seen_scripts: Dict[str, int] = {}
        builds = 0

        for step in range(settings.max_steps):
            result.steps = step + 1
            self._prune_images(messages)

            response = self.client.chat_with_retry(
                prompts.build_system_prompt(), messages
            )
            self._accumulate(totals, response.usage)

            if response.error:
                result.error = response.error
                print(f"    [llm error] {response.error}")
                break

            action = llm.extract_json(response.text)
            transcript.append({
                "step": step,
                "thinking": (response.thinking or "")[:4000],
                "raw": response.text[:8000],
                "usage": response.usage,
            })

            if action is None:
                messages.append(llm.assistant(response.text[:2000]))
                messages.append(llm.user(
                    "Your response was not valid JSON. Reply with exactly one JSON "
                    "object matching the response format - no prose, no extra keys."
                ))
                continue

            kind = str(action.get("action", "")).lower().strip()
            thought = str(action.get("thought", ""))[:300]

            # ---------------- finish ----------------
            if kind == "finish":
                messages.append(llm.assistant(
                    json.dumps({"action": "finish", "thought": thought})
                ))
                if best_step:
                    finished = True
                    print(f"    [step {step}] finish - accepting last good build")
                    break
                messages.append(llm.user(
                    "You cannot finish: no build has succeeded yet. Submit a "
                    "my_cad_function now."
                ))
                continue

            # ---------------- tool ----------------
            if kind == "tool":
                name = str(action.get("tool", ""))
                args = action.get("tool_args") or {}
                messages.append(llm.assistant(json.dumps(
                    {"action": "tool", "tool": name, "tool_args": args,
                     "thought": thought}, default=str
                )))

                if get_tool(name) is None:
                    messages.append(llm.user(
                        f"No tool named {name!r}. Available: "
                        f"{', '.join(self._visible_tools(settings.max_steps - step))}."
                    ))
                    continue

                print(f"    [step {step}] tool {name}({', '.join(map(str, args))})")
                tool_result = run_tool(name, ctx, args)
                parts: List[Any] = [f"Result of {name}:", tool_result.text]
                parts.extend(llm.image_part(p) for p in tool_result.images
                             if osp.exists(p))
                messages.append(llm.user(*parts))
                continue

            # ---------------- submit ----------------
            if kind == "submit":
                script = action.get("my_cad_function") or ""
                if isinstance(script, str):
                    script = _strip_fence(script)
                if not isinstance(script, str) or "def my_cad_function" not in script:
                    messages.append(llm.assistant('{"action": "submit"}'))
                    messages.append(llm.user(
                        "Your submission did not contain a `def my_cad_function(args):` "
                        "definition. Resubmit the complete function as a JSON string."
                    ))
                    continue

                if builds >= settings.max_builds:
                    messages.append(llm.assistant('{"action": "submit"}'))
                    messages.append(llm.user(
                        "Build budget exhausted. Respond {\"action\": \"finish\"} to keep "
                        "the last successful build."
                        if best_step else
                        "Build budget exhausted and nothing has built successfully."
                    ))
                    if best_step:
                        continue
                    break

                digest = hashlib.md5(script.encode("utf-8")).hexdigest()
                repeat = seen_scripts.get(digest, 0)
                seen_scripts[digest] = repeat + 1

                messages.append(llm.assistant(
                    json.dumps({"action": "submit", "thought": thought})
                    + f"\n```python\n{script}\n```"
                ))

                if repeat:
                    messages.append(llm.user(
                        "That is byte-for-byte the script you already submitted, so the "
                        "result will be identical. Change the approach or finish."
                    ))
                    continue

                builds += 1
                outcome = self._build(task, script, work_dir, builds, ctx, input_report,
                                      steps_left=settings.max_steps - step - 1)
                messages.append(llm.user(*outcome["message_parts"]))

                if outcome["step_path"]:
                    best_step = outcome["step_path"]
                    best_script = script
                    ctx.last_output_step = outcome["step_path"]
                    ctx.last_report = outcome["report"]
                    ctx.last_script = script
                continue

            # ---------------- unknown ----------------
            messages.append(llm.assistant(response.text[:1500]))
            messages.append(llm.user(
                f"Unknown action {kind!r}. Use exactly one of: tool, submit, finish."
            ))

        # --- 3. decide what to hand back -------------------------------------
        result.builds = builds
        result.usage = totals
        result.transcript = transcript

        if best_step and osp.exists(best_step):
            result.step_path = best_step
            result.script = best_script
            result.status = "finished" if finished else "best_effort"
        elif settings.fallback_to_input and osp.exists(task.input_step):
            # No build survived. Emitting the unedited part is strictly better
            # than emitting nothing: the benchmark scores a missing output 0.0.
            result.step_path = task.input_step
            result.used_fallback = True
            result.status = "fallback"
        else:
            result.status = "failed"

        return result

    # -- build one candidate --------------------------------------------------

    def _build(self, task: TaskSpec, script: str, work_dir: str, index: int,
               ctx: ToolContext, input_report: Optional[dict],
               steps_left: int) -> dict:
        """Execute, render, diff, and compose the feedback turn."""
        settings = self.settings
        build_dir = osp.join(work_dir, f"build_{index:02d}")
        os.makedirs(build_dir, exist_ok=True)

        script_path = osp.join(build_dir, "candidate.py")
        with open(script_path, "w", encoding="utf-8") as fh:
            fh.write(script)

        views = settings.build_views
        if steps_left <= 1:
            views = settings.final_check_views

        outcome = cq_client.execute(
            script_file=script_path,
            input_file=task.input_step,
            output_dir=build_dir,
            views=views,
            image_size=settings.render_size,
            timeout=settings.build_timeout,
        )

        parts: List[Any] = [f"## Build {index} of {settings.max_builds}"]

        stdout = truncate(outcome.get("stdout"), settings.stdout_limit)
        stderr = truncate(outcome.get("stderr"), settings.stdout_limit)
        if stdout:
            parts.append(f"stdout:\n{stdout}")
        if stderr:
            parts.append(f"stderr:\n{stderr}")

        step_path = outcome.get("step_path")
        report = outcome.get("report")

        if not outcome.get("ok") or not step_path:
            parts.append(f"BUILD FAILED: {outcome.get('error', 'unknown error')}")
            parts.append(prompts.BUILD_FAILED_HINT)
            if steps_left <= 2:
                parts.append(prompts.BUDGET_WARNING)
            return {"message_parts": parts, "step_path": None, "report": None}

        parts.append("BUILD OK - STEP exported.")
        if report and input_report:
            diff = diff_reports(input_report, report)
            parts.append(format_diff(diff))
            if diff.get("appears_unchanged"):
                parts.append(prompts.NO_CHANGE_HINT)

        images = [p for p in (outcome.get("images") or []) if osp.exists(p)]
        if images:
            names = ", ".join(
                osp.splitext(osp.basename(p))[0].split("_")[-1] for p in images
            )
            parts.append(f"Rendered views of YOUR BUILD ({names}):")
            parts.extend(llm.image_part(p) for p in images)

        parts.append(
            "Compare this against the original views and the request. If it is "
            'correct, respond {"action": "finish"}. If not, submit a fix.'
        )
        if steps_left <= 2:
            parts.append(prompts.BUDGET_WARNING)

        return {"message_parts": parts, "step_path": step_path, "report": report}
