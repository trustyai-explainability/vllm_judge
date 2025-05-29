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

from vllm_judge import Judge
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
        ).dict()
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
        
        # Perform evaluation
        result = await judge.evaluate(
            response=request.response,
            criteria=request.criteria,
            rubric=request.rubric,
            scale=scale,
            metric=request.metric,
            context=request.context,
            system_prompt=request.system_prompt,
            examples=request.examples
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
            has_system_prompt=metric.system_prompt is not None
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
        "system_prompt": metric.system_prompt
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
                    response=request.response,
                    criteria=request.criteria,
                    rubric=request.rubric,
                    scale=scale,
                    metric=request.metric,
                    context=request.context,
                    system_prompt=request.system_prompt,
                    examples=request.examples
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