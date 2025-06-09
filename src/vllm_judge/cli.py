"""
Command-line interface for vLLM Judge.
"""
import asyncio
import json
import sys
from typing import Optional
import click

from vllm_judge import Judge
from vllm_judge.models import JudgeConfig
from vllm_judge.api.server import start_server as start_api_server
from vllm_judge.api.client import JudgeClient
from vllm_judge.metrics import BUILTIN_METRICS


@click.group()
def cli():
    """vLLM Judge - LLM-as-a-Judge evaluation tool."""
    pass


@cli.command()
@click.option('--base-url', required=True, help='vLLM server URL')
@click.option('--model', help='Model name/path (auto-detected if not provided)')
@click.option('--host', default='0.0.0.0', help='API server host')
@click.option('--port', default=8080, help='API server port')
@click.option('--reload', is_flag=True, help='Enable auto-reload for development')
@click.option('--max-concurrent', default=50, help='Maximum concurrent requests')
@click.option('--timeout', default=30.0, help='Request timeout in seconds')
def serve(base_url: str, model: str, host: str, port: int, reload: bool, max_concurrent: int, timeout: float):
    """Start the Judge API server."""
    click.echo(f"Starting vLLM Judge API server...")
    click.echo(f"Base URL: {base_url}")
    click.echo(f"Model: {model}")
    click.echo(f"Server: http://{host}:{port}")
    
    start_api_server(
        base_url=base_url,
        model=model,
        host=host,
        port=port,
        reload=reload,
        max_concurrent=max_concurrent,
        timeout=timeout
    )


@cli.command()
@click.option('--api-url', help='Judge API URL (if using remote server)')
@click.option('--base-url', help='vLLM server URL (if using local)')
@click.option('--model', help='Model name (if using local)')
@click.option('--content', required=True, help='Text to evaluate')
@click.option('--input', help='Input/question/prompt that the content responds to')
@click.option('--criteria', help='Evaluation criteria')
@click.option('--metric', help='Pre-defined metric name')
@click.option('--scale', nargs=2, type=int, help='Numeric scale (min max)')
@click.option('--rubric', help='Evaluation rubric')
@click.option('--context', help='Additional context')
@click.option('--output', type=click.Choice(['json', 'text']), default='text', help='Output format')
def evaluate(
    api_url: Optional[str],
    base_url: Optional[str],
    model: Optional[str],
    content: str,
    input: Optional[str],
    criteria: Optional[str],
    metric: Optional[str],
    scale: Optional[tuple],
    rubric: Optional[str],
    context: Optional[str],
    output: str
):
    """Evaluate a single response."""
    async def run_evaluation():
        if api_url:
            # Use API client
            async with JudgeClient(api_url) as client:
                result = await client.evaluate(
                    content=content,
                    input=input,
                    criteria=criteria,
                    metric=metric,
                    scale=scale,
                    rubric=rubric,
                    context=context
                )
        else:
            # Use local Judge
            if not base_url:
                click.echo("Error: Either --api-url or --base-url is required", err=True)
                sys.exit(1)
            
            judge = Judge.from_url(base_url, model=model)
            async with judge:
                result = await judge.evaluate(
                    content=content,
                    input=input,
                    criteria=criteria,
                    metric=metric,
                    scale=scale,
                    rubric=rubric,
                    context=context
                )
        
        # Format output
        if output == 'json':
            click.echo(json.dumps(result.model_dump(), indent=2))
        else:
            click.echo(f"Decision: {result.decision}")
            if result.score is not None:
                click.echo(f"Score: {result.score}")
            click.echo(f"Reasoning: {result.reasoning}")
    
    asyncio.run(run_evaluation())

@cli.command()
@click.option('--api-url', help='Judge API URL (if using remote server)')
@click.option('--base-url', help='vLLM server URL (if using local)')
@click.option('--model', help='Model name (if using local)')
@click.option('--question', required=True, help='Question to evaluate answer for')
@click.option('--answer', required=True, help='Answer to evaluate')
@click.option('--criteria', default='accuracy and completeness', help='Evaluation criteria')
@click.option('--scale', nargs=2, type=int, default=[1, 10], help='Numeric scale (min max)')
@click.option('--output', type=click.Choice(['json', 'text']), default='text', help='Output format')
def qa_evaluate(
    api_url: Optional[str],
    base_url: Optional[str],
    model: Optional[str],
    question: str,
    answer: str,
    criteria: str,
    scale: tuple,
    output: str
):
    """Evaluate a QA pair (question and answer)."""
    async def run_qa_evaluation():
        if api_url:
            async with JudgeClient(api_url) as client:
                result = await client.qa_evaluate(
                    question=question,
                    answer=answer,
                    criteria=criteria,
                    scale=scale
                )
        else:
            if not base_url:
                click.echo("Error: Either --api-url or --base-url is required", err=True)
                sys.exit(1)
            
            judge = Judge.from_url(base_url, model=model)
            async with judge:
                result = await judge.qa_evaluate(
                    question=question,
                    answer=answer,
                    criteria=criteria,
                    scale=scale
                )
        
        if output == 'json':
            click.echo(json.dumps(result.model_dump(), indent=2))
        else:
            click.echo(f"Question: {question}")
            click.echo(f"Answer: {answer}")
            click.echo(f"Decision: {result.decision}")
            if result.score is not None:
                click.echo(f"Score: {result.score}")
            click.echo(f"Reasoning: {result.reasoning}")
    
    asyncio.run(run_qa_evaluation())

@cli.command()
@click.option('--api-url', help='Judge API URL (if using remote server)')
@click.option('--base-url', help='vLLM server URL (if using local)')
@click.option('--model', help='Model name (if using local)')
@click.option('--response-a', required=True, help='First response')
@click.option('--response-b', required=True, help='Second response')
@click.option('--criteria', required=True, help='Comparison criteria')
@click.option('--input', help='Input/question that both responses address')
@click.option('--output', type=click.Choice(['json', 'text']), default='text', help='Output format')
def compare(
    api_url: Optional[str],
    base_url: Optional[str],
    model: Optional[str],
    response_a: str,
    response_b: str,
    criteria: str,
    input: Optional[str],
    output: str
):
    """Compare two responses."""
    async def run_comparison():
        if api_url:
            async with JudgeClient(api_url) as client:
                result = await client.compare(
                    response_a=response_a,
                    response_b=response_b,
                    criteria=criteria,
                    input=input
                )
        else:
            if not base_url:
                click.echo("Error: Either --api-url or --base-url is required", err=True)
                sys.exit(1)
            
            judge = Judge.from_url(base_url, model=model)
            async with judge:
                result = await judge.compare(
                    response_a=response_a,
                    response_b=response_b,
                    criteria=criteria,
                    input=input
                )
        
        if output == 'json':
            click.echo(json.dumps(result.model_dump(), indent=2))
        else:
            if input:
                click.echo(f"Input: {input}")
            click.echo(f"Response A: {response_a}")
            click.echo(f"Response B: {response_b}")
            click.echo(f"Winner: {result.decision}")
            click.echo(f"Reasoning: {result.reasoning}")
    
    asyncio.run(run_comparison())


@cli.command()
@click.option('--api-url', required=True, help='Judge API URL')
def health(api_url: str):
    """Check API health status."""
    async def check_health():
        async with JudgeClient(api_url) as client:
            try:
                health_data = await client.health_check()
                click.echo(json.dumps(health_data, indent=2))
            except Exception as e:
                click.echo(f"Health check failed: {e}", err=True)
                sys.exit(1)
    
    asyncio.run(check_health())


@cli.command()
@click.option('--api-url', help='Judge API URL (if using remote server)')
@click.option('--filter', help='Filter metrics by name')
def list_metrics(api_url: Optional[str], filter: Optional[str]):
    """List available metrics."""
    async def list_all_metrics():
        if api_url:
            async with JudgeClient(api_url) as client:
                metrics = await client.list_metrics()
                for metric in metrics:
                    if filter and filter.lower() not in metric.name.lower():
                        continue
                    click.echo(f"\n{metric.name}:")
                    click.echo(f"  Criteria: {metric.criteria}")
                    if metric.has_scale:
                        click.echo(f"  Scale: {metric.scale}")
                    click.echo(f"  Has rubric: {metric.has_rubric}")
                    click.echo(f"  Examples: {metric.example_count}")
        else:
            # List built-in metrics
            for name, metric in BUILTIN_METRICS.items():
                if filter and filter.lower() not in name.lower():
                    continue
                click.echo(f"\n{name}:")
                click.echo(f"  Criteria: {metric.criteria}")
                if metric.scale:
                    click.echo(f"  Scale: {metric.scale}")
                click.echo(f"  Has rubric: {'Yes' if metric.rubric else 'No'}")
                click.echo(f"  Examples: {len(metric.examples)}")
    
    asyncio.run(list_all_metrics())


@cli.command()
@click.option('--api-url', help='Judge API URL')
@click.option('--file', required=True, type=click.File('r'), help='JSON file with batch data')
@click.option('--async', 'use_async', is_flag=True, help='Use async batch processing')
@click.option('--max-concurrent', type=int, help='Maximum concurrent requests')
@click.option('--output', type=click.File('w'), help='Output file (default: stdout)')
def batch(api_url: str, file, use_async: bool, max_concurrent: Optional[int], output):
    """Run batch evaluation from JSON file."""
    # Load batch data
    try:
        data = json.load(file)
        if not isinstance(data, list):
            click.echo("Error: Batch file must contain a JSON array", err=True)
            sys.exit(1)
    except json.JSONDecodeError as e:
        click.echo(f"Error parsing JSON: {e}", err=True)
        sys.exit(1)
    
    async def run_batch():
        async with JudgeClient(api_url) as client:
            if use_async:
                click.echo(f"Starting async batch evaluation of {len(data)} items...")
                result = await client.async_batch_evaluate(
                    data=data,
                    max_concurrent=max_concurrent
                )
            else:
                click.echo(f"Running batch evaluation of {len(data)} items...")
                result = await client.batch_evaluate(
                    data=data,
                    max_concurrent=max_concurrent
                )
            
            # Format results
            output_data = {
                "total": result.total,
                "successful": result.successful,
                "failed": result.failed,
                "success_rate": result.success_rate,
                "duration_seconds": result.duration_seconds,
                "results": []
            }
            
            for r in result.results:
                if isinstance(r, Exception):
                    output_data["results"].append({"error": str(r)})
                else:
                    output_data["results"].append({
                        "decision": r.decision,
                        "reasoning": r.reasoning,
                        "score": r.score,
                        "metadata": r.metadata
                    })
            
            # Write output
            output_file = output or sys.stdout
            json.dump(output_data, output_file, indent=2)
            if output:
                click.echo(f"Results written to {output.name}")
            
            # Summary
            click.echo(f"\nSummary:")
            click.echo(f"  Total: {result.total}")
            click.echo(f"  Successful: {result.successful}")
            click.echo(f"  Failed: {result.failed}")
            click.echo(f"  Success rate: {result.success_rate:.1%}")
            click.echo(f"  Duration: {result.duration_seconds:.1f}s")
    
    asyncio.run(run_batch())


def main():
    """Main entry point."""
    cli.help = """vLLM Judge - LLM-as-a-Judge evaluation tool.

Features:
- Single response evaluation with optional input context
- QA (Question-Answer) evaluation  
- Response comparison with optional input context
- Batch evaluation from JSON files
- API server mode
- Built-in and custom metrics with template support
"""
    cli()


if __name__ == '__main__':
    main()