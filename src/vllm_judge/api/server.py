"""
FastAPI server for vLLM Judge API.
"""
import asyncio
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import uvicorn

from vllm_judge.judge import Judge
from vllm_judge.models import EvaluationResult, JudgeConfig
from vllm_judge.metrics import BUILTIN_METRICS
from vllm_judge.exceptions import VLLMJudgeError
from vllm_judge.api.models import (
    EvaluateRequest,
    BatchEvaluateRequest,
    AsyncBatchRequest,
    EvaluationResponse,
    BatchResponse,
    AsyncBatchResponse,
    JobStatusResponse,
    MetricInfo,
    HealthResponse,
    ErrorResponse
)
from vllm_judge.templating import TemplateProcessor
from vllm_judge.models import TemplateEngine
from vllm_judge import __version__


# Global state
judge: Optional[Judge] = None
app_start_time: float = 0
total_evaluations: int = 0
active_connections: int = 0
jobs: Dict[str, Dict[str, Any]] = {}  # job_id -> job info


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global app_start_time
    app_start_time = time.time()
    yield
    # Cleanup
    if judge:
        await judge.close()


app = FastAPI(
    title="vLLM Judge API",
    description="LLM-as-a-Judge evaluation service for vLLM hosted models",
    version=__version__,
    lifespan=lifespan
)


@app.exception_handler(VLLMJudgeError)
async def vllm_judge_exception_handler(request, exc: VLLMJudgeError):
    """Handle vLLM Judge specific exceptions."""
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error=exc.__class__.__name__,
            detail=str(exc),
            code="VLLM_JUDGE_ERROR"
        ).model_dump()
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    if not judge:
        raise HTTPException(status_code=503, detail="Judge not initialized")
    
    uptime = time.time() - app_start_time
    
    return HealthResponse(
        status="healthy",
        version=__version__,
        model=judge.config.model,
        base_url=judge.config.base_url,
        uptime_seconds=uptime,
        total_evaluations=total_evaluations,
        active_connections=active_connections,
        metrics_available=len(judge.list_metrics())
    )


@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate(request: EvaluateRequest):
    """Single evaluation endpoint."""
    global total_evaluations
    
    if not judge:
        raise HTTPException(status_code=503, detail="Judge not initialized")
    
    start_time = time.time()
    
    try:
        # Convert scale list to tuple if provided
        scale = tuple(request.scale) if request.scale else None
        
        # Perform evaluation with template support
        result = await judge.evaluate(
            content=request.content,
            input=request.input,
            criteria=request.criteria,
            rubric=request.rubric,
            scale=scale,
            metric=request.metric,
            context=request.context,
            system_prompt=request.system_prompt,
            examples=request.examples,
            template_vars=request.template_vars,
            template_engine=request.template_engine
        )
        
        # Convert to response model
        duration_ms = int((time.time() - start_time) * 1000)
        total_evaluations += 1
        
        return EvaluationResponse(
            decision=result.decision,
            reasoning=result.reasoning,
            score=result.score,
            metadata=result.metadata,
            evaluation_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            duration_ms=duration_ms
        )
        
    except VLLMJudgeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/batch", response_model=BatchResponse)
async def batch_evaluate(request: BatchEvaluateRequest):
    """Synchronous batch evaluation endpoint."""
    global total_evaluations
    
    if not judge:
        raise HTTPException(status_code=503, detail="Judge not initialized")
    
    # Apply defaults if provided
    if request.default_criteria or request.default_metric:
        for item in request.data:
            if request.default_criteria and "criteria" not in item:
                item["criteria"] = request.default_criteria
            if request.default_metric and "metric" not in item:
                item["metric"] = request.default_metric
    
    try:
        # Perform batch evaluation
        batch_result = await judge.batch_evaluate(
            data=request.data,
            max_concurrent=request.max_concurrent
        )
        
        # Convert results
        results = []
        for i, r in enumerate(batch_result.results):
            if isinstance(r, EvaluationResult):
                results.append(EvaluationResponse(
                    decision=r.decision,
                    reasoning=r.reasoning,
                    score=r.score,
                    metadata=r.metadata,
                    evaluation_id=str(uuid.uuid4()),
                    timestamp=datetime.utcnow()
                ))
            else:
                # Error case
                results.append({
                    "error": str(r),
                    "index": i
                })
        
        total_evaluations += batch_result.successful
        
        return BatchResponse(
            total=batch_result.total,
            successful=batch_result.successful,
            failed=batch_result.failed,
            success_rate=batch_result.success_rate,
            duration_seconds=batch_result.duration_seconds,
            results=results
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch evaluation failed: {str(e)}")


@app.post("/batch/async", response_model=AsyncBatchResponse)
async def async_batch_evaluate(
    request: AsyncBatchRequest,
    background_tasks: BackgroundTasks
):
    """Asynchronous batch evaluation endpoint."""
    if not judge:
        raise HTTPException(status_code=503, detail="Judge not initialized")
    
    # Create job
    job_id = str(uuid.uuid4())
    job_info = {
        "id": job_id,
        "status": "pending",
        "data": request.data,
        "total": len(request.data),
        "completed": 0,
        "created_at": datetime.utcnow(),
        "callback_url": request.callback_url,
        "max_concurrent": request.max_concurrent
    }
    jobs[job_id] = job_info
    
    # Estimate duration (rough estimate: 0.5s per evaluation)
    estimated_duration = len(request.data) * 0.5 / (request.max_concurrent or judge.config.max_concurrent)
    
    # Start background task
    background_tasks.add_task(
        run_async_batch,
        job_id,
        request.data,
        request.max_concurrent,
        request.callback_url
    )
    
    return AsyncBatchResponse(
        job_id=job_id,
        status="pending",
        total_items=len(request.data),
        created_at=job_info["created_at"],
        estimated_duration_seconds=estimated_duration
    )


async def run_async_batch(
    job_id: str,
    data: List[Dict[str, Any]],
    max_concurrent: Optional[int],
    callback_url: Optional[str]
):
    """Run batch evaluation in background."""
    global total_evaluations
    
    job = jobs[job_id]
    job["status"] = "running"
    job["started_at"] = datetime.utcnow()
    
    try:
        # Progress callback
        def update_progress(completed: int, total: int):
            job["completed"] = completed
        
        # Run evaluation
        batch_result = await judge.batch_evaluate(
            data=data,
            max_concurrent=max_concurrent,
            progress_callback=update_progress
        )
        
        # Update job
        job["status"] = "completed"
        job["completed_at"] = datetime.utcnow()
        job["result"] = batch_result
        total_evaluations += batch_result.successful
        
        # Send callback if provided
        if callback_url:
            # TODO: Implement callback POST request
            pass
            
    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
        job["completed_at"] = datetime.utcnow()


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get status of async job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        progress={"completed": job.get("completed", 0), "total": job["total"]},
        created_at=job["created_at"],
        started_at=job.get("started_at"),
        completed_at=job.get("completed_at"),
        result_url=f"/jobs/{job_id}/result" if job["status"] == "completed" else None,
        error=job.get("error")
    )


@app.get("/jobs/{job_id}/result")
async def get_job_result(job_id: str):
    """Get result of completed async job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job is {job['status']}, not completed"
        )
    
    if "result" not in job:
        raise HTTPException(status_code=500, detail="Job result not found")
    
    batch_result = job["result"]
    
    # Convert to response format
    results = []
    for r in batch_result.results:
        if isinstance(r, EvaluationResult):
            results.append({
                "decision": r.decision,
                "reasoning": r.reasoning,
                "score": r.score,
                "metadata": r.metadata
            })
        else:
            results.append({"error": str(r)})
    
    return {
        "job_id": job_id,
        "total": batch_result.total,
        "successful": batch_result.successful,
        "failed": batch_result.failed,
        "success_rate": batch_result.success_rate,
        "duration_seconds": batch_result.duration_seconds,
        "results": results
    }


@app.get("/metrics", response_model=List[MetricInfo])
async def list_metrics():
    """List all available metrics."""
    if not judge:
        raise HTTPException(status_code=503, detail="Judge not initialized")
    
    metrics_info = []
    
    # Get all metrics (user-registered + built-in)
    all_metrics = {**judge.metrics, **BUILTIN_METRICS}
    
    for name, metric in all_metrics.items():
        info = MetricInfo(
            name=name,
            criteria=metric.criteria,
            has_scale=metric.scale is not None,
            scale=metric.scale,
            has_rubric=metric.rubric is not None,
            rubric_type=type(metric.rubric).__name__ if metric.rubric else None,
            has_examples=bool(metric.examples),
            example_count=len(metric.examples) if metric.examples else 0,
            has_system_prompt=metric.system_prompt is not None,
            has_template_vars=bool(metric.template_vars),
            template_vars=metric.template_vars if metric.template_vars else None,
            required_vars=metric.required_vars if hasattr(metric, 'required_vars') else None,
            template_engine=metric.template_engine.value if hasattr(metric, 'template_engine') else None
        )
        metrics_info.append(info)
    
    return metrics_info


@app.get("/metrics/{metric_name}")
async def get_metric_details(metric_name: str):
    """Get detailed information about a specific metric."""
    if not judge:
        raise HTTPException(status_code=503, detail="Judge not initialized")
    
    try:
        metric = judge.get_metric(metric_name)
    except Exception:
        raise HTTPException(status_code=404, detail=f"Metric '{metric_name}' not found")
    
    return {
        "name": metric_name,
        "criteria": metric.criteria,
        "scale": metric.scale,
        "rubric": metric.rubric,
        "examples": metric.examples,
        "system_prompt": metric.system_prompt,
        "template_vars": getattr(metric, 'template_vars', None),
        "required_vars": getattr(metric, 'required_vars', None),
        "template_engine": getattr(metric, 'template_engine', None)
    }


@app.websocket("/ws/evaluate")
async def websocket_evaluate(websocket: WebSocket):
    """WebSocket endpoint for real-time evaluations."""
    global active_connections
    
    await websocket.accept()
    active_connections += 1
    
    try:
        while True:
            # Receive evaluation request
            data = await websocket.receive_json()
            
            try:
                # Perform evaluation
                request = EvaluateRequest(**data)
                scale = tuple(request.scale) if request.scale else None
                
                result = await judge.evaluate(
                    content=request.content,
                    input=request.input,
                    criteria=request.criteria,
                    rubric=request.rubric,
                    scale=scale,
                    metric=request.metric,
                    context=request.context,
                    system_prompt=request.system_prompt,
                    examples=request.examples,
                    template_vars=request.template_vars,
                    template_engine=request.template_engine
                )
                
                # Send result
                await websocket.send_json({
                    "status": "success",
                    "result": {
                        "decision": result.decision,
                        "reasoning": result.reasoning,
                        "score": result.score,
                        "metadata": result.metadata
                    }
                })
                
            except Exception as e:
                await websocket.send_json({
                    "status": "error",
                    "error": str(e)
                })
                
    except WebSocketDisconnect:
        active_connections -= 1


@app.post("/validate/template")
async def validate_template(request: Dict[str, Any]):
    """Validate template variables for a given template."""
    template = request.get("template", "")
    template_vars = request.get("template_vars", {})
    engine = request.get("template_engine", "format")
    
    try:
        # Get required variables
        required_vars = TemplateProcessor.get_required_vars(
            template, 
            TemplateEngine(engine)
        )
        
        # Check which are missing
        provided_vars = set(template_vars.keys())
        missing_vars = required_vars - provided_vars
        
        # Try to apply template
        try:
            result = TemplateProcessor.apply_template(
                template,
                template_vars,
                TemplateEngine(engine),
                strict=True
            )
            
            return {
                "valid": True,
                "required_vars": list(required_vars),
                "provided_vars": list(provided_vars),
                "missing_vars": list(missing_vars),
                "result": result
            }
        except Exception as e:
            return {
                "valid": False,
                "required_vars": list(required_vars),
                "provided_vars": list(provided_vars),
                "missing_vars": list(missing_vars),
                "error": str(e)
            }
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Validation error: {str(e)}")


@app.post("/metrics/register")
async def register_metric(metric_data: Dict[str, Any]):
    """Register a new metric dynamically."""
    if not judge:
        raise HTTPException(status_code=503, detail="Judge not initialized")
    
    try:
        # Create metric from data
        from vllm_judge.models import Metric
        
        metric = Metric(
            name=metric_data["name"],
            criteria=metric_data["criteria"],
            rubric=metric_data.get("rubric"),
            scale=tuple(metric_data["scale"]) if metric_data.get("scale") else None,
            examples=metric_data.get("examples", []),
            system_prompt=metric_data.get("system_prompt"),
            template_vars=metric_data.get("template_vars", {}),
            required_vars=metric_data.get("required_vars", []),
            template_engine=metric_data.get("template_engine", "format")
        )
        
        # Register with judge
        judge.register_metric(metric)
        
        return {"message": f"Metric '{metric.name}' registered successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to register metric: {str(e)}")


def create_app(config: JudgeConfig) -> FastAPI:
    """Create FastAPI app with initialized Judge."""
    global judge
    judge = Judge(config)
    return app


def start_server(
    base_url: str,
    model: Optional[str] = None,
    host: str = "0.0.0.0",
    port: int = 8080,
    reload: bool = False,
    **kwargs
):
    """Start the API server."""
    global judge
    
    # Initialize judge
    config = JudgeConfig.from_url(base_url, model=model, **kwargs)
    judge = Judge(config)
    
    # Run server
    uvicorn.run(
        "vllm_judge.api.server:app",
        host=host,
        port=port,
        reload=reload
    )