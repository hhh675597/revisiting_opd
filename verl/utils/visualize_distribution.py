"""
Visualization utilities for comparing teacher and student model distributions.

This module provides tools to visualize the difference between reference (teacher)
and policy (student) model probability distributions on generated tokens.
"""

import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import json


def compute_prob_difference(
    teacher_log_probs: torch.Tensor,
    student_log_probs: torch.Tensor,
) -> torch.Tensor:
    """
    Compute absolute probability difference per token.
    
    Args:
        teacher_log_probs: Log probabilities from teacher model (batch_size, seq_len)
        student_log_probs: Log probabilities from student model (batch_size, seq_len)
    
    Returns:
        Absolute probability difference (batch_size, seq_len)
    """
    teacher_probs = torch.exp(teacher_log_probs)
    student_probs = torch.exp(student_log_probs)
    return torch.abs(teacher_probs - student_probs)


def token_to_color(
    prob_diff: float,
    max_abs_diff: float = 0.3,
) -> str:
    """
    Convert probability difference to color.
    
    Blue: Policy > Reference (student more confident)
    Red: Policy < Reference (student less confident)
    Color depth: Magnitude of difference
    
    Args:
        prob_diff: Probability difference (policy_prob - ref_prob)
                   Positive = policy more confident (blue)
                   Negative = policy less confident (red)
        max_abs_diff: Maximum absolute difference for color scaling (default: 0.3 = 30%)
    
    Returns:
        RGB color string
    """
    # Normalize to -1 to 1 range
    normalized = max(min(prob_diff / max_abs_diff, 1.0), -1.0)
    
    if normalized > 0:
        # Policy > Reference: White to Blue
        # More difference = deeper blue
        intensity = int(255 * (1 - normalized))  # 255 (white) to 0 (dark blue)
        return f"rgb({intensity}, {intensity}, 255)"
    elif normalized < 0:
        # Policy < Reference: White to Red
        # More difference = deeper red
        intensity = int(255 * (1 + normalized))  # 255 (white) to 0 (dark red)
        return f"rgb(255, {intensity}, {intensity})"
    else:
        # Equal probabilities: Light gray
        return f"rgb(240, 240, 240)"


def create_html_visualization(
    tokens: List[str],
    teacher_log_probs: List[float],
    student_log_probs: List[float],
    sample_idx: int,
    task_type: str,
    global_step: int,
    output_path: Path,
    prompt_length: Optional[int] = None,
    extra_info: Optional[Dict] = None,
    ref_top1_tokens: Optional[List[str]] = None,
) -> None:
    """
    Create an interactive HTML visualization of teacher vs student distributions.
    
    Args:
        tokens: List of token strings
        teacher_log_probs: Teacher model log probabilities per token
        student_log_probs: Student model log probabilities per token
        sample_idx: Sample index in the batch
        task_type: Task type (e.g., "math", "alfworld")
        global_step: Current training step
        output_path: Path to save HTML file
        prompt_length: Length of prompt (to distinguish from response)
        extra_info: Additional information to display (e.g., rewards, episode info)
        ref_top1_tokens: Reference model's top-1 token at each position (optional)
    """
    # Convert to numpy for easier manipulation
    teacher_log_probs = np.array(teacher_log_probs)
    student_log_probs = np.array(student_log_probs)
    
    # Compute metrics
    teacher_probs = np.exp(teacher_log_probs)
    student_probs = np.exp(student_log_probs)
    
    # Signed probability difference (student - teacher)
    # Positive = student more confident (blue)
    # Negative = student less confident (red)
    signed_prob_diff = student_probs - teacher_probs
    abs_prob_diff = np.abs(signed_prob_diff)
    
    # Determine max absolute difference for color scaling (only for response tokens)
    if prompt_length and prompt_length < len(abs_prob_diff):
        response_abs_diff = abs_prob_diff[prompt_length:]
        max_abs_diff = max(np.max(response_abs_diff), 0.1) if len(response_abs_diff) > 0 else 0.1
    else:
        max_abs_diff = max(np.max(abs_prob_diff), 0.1)  # At least 0.1 (10%) for reasonable scaling
    
    # Build token data for JavaScript
    token_data = []
    for i, token in enumerate(tokens):
        token_info = {
            "token": token,
            "teacher_log_prob": float(teacher_log_probs[i]),
            "student_log_prob": float(student_log_probs[i]),
            "teacher_prob": float(teacher_probs[i]),
            "student_prob": float(student_probs[i]),
            "signed_prob_diff": float(signed_prob_diff[i]),
            "abs_prob_diff": float(abs_prob_diff[i]),
            "color": token_to_color(signed_prob_diff[i], max_abs_diff),
            "is_prompt": i < prompt_length if prompt_length else False,
        }
        
        # Add reference model's top-1 token if available
        if ref_top1_tokens is not None and i < len(ref_top1_tokens):
            token_info["ref_top1_token"] = ref_top1_tokens[i]
        
        token_data.append(token_info)
    
    # Compute summary statistics (only for response tokens)
    if prompt_length and prompt_length < len(signed_prob_diff):
        response_signed_diff = signed_prob_diff[prompt_length:]
        response_abs_diff = abs_prob_diff[prompt_length:]
        mean_signed_diff = float(np.mean(response_signed_diff)) if len(response_signed_diff) > 0 else 0.0
        mean_abs_diff = float(np.mean(response_abs_diff)) if len(response_abs_diff) > 0 else 0.0
        max_abs_diff_val = float(np.max(response_abs_diff)) if len(response_abs_diff) > 0 else 0.0
    else:
        mean_signed_diff = float(np.mean(signed_prob_diff))
        mean_abs_diff = float(np.mean(abs_prob_diff))
        max_abs_diff_val = float(np.max(abs_prob_diff))
    
    # Create HTML
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Teacher vs Student Distribution - Step {global_step} - Sample {sample_idx}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        .header h1 {{
            margin: 0 0 10px 0;
        }}
        .header .info {{
            font-size: 14px;
            opacity: 0.9;
        }}
        .stats {{
            background-color: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stats h3 {{
            margin-top: 0;
            color: #2c3e50;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        .stat-item {{
            padding: 10px;
            background-color: #ecf0f1;
            border-radius: 4px;
        }}
        .stat-label {{
            font-size: 12px;
            color: #7f8c8d;
            text-transform: uppercase;
        }}
        .stat-value {{
            font-size: 20px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .legend {{
            background-color: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .legend h3 {{
            margin-top: 0;
            color: #2c3e50;
        }}
        .legend-item {{
            display: inline-block;
            margin-right: 20px;
            margin-bottom: 10px;
        }}
        .legend-color {{
            display: inline-block;
            width: 20px;
            height: 20px;
            vertical-align: middle;
            margin-right: 5px;
            border: 1px solid #ccc;
        }}
        .content {{
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .tokens {{
            line-height: 2;
            font-size: 16px;
            font-family: 'Courier New', monospace;
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
        .token {{
            display: inline-block;
            padding: 2px 4px;
            margin: 2px;
            border-radius: 3px;
            cursor: pointer;
            transition: transform 0.1s;
            border: 1px solid rgba(0,0,0,0.1);
        }}
        .token:hover {{
            transform: scale(1.1);
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
            z-index: 100;
        }}
        .token.prompt {{
            border: 2px solid #3498db;
            background-color: #ecf0f1 !important;
        }}
        .tooltip {{
            position: absolute; /* Changed from fixed to absolute */
            background-color: rgba(0, 0, 0, 0.9);
            color: white;
            padding: 12px;
            border-radius: 6px;
            font-size: 13px;
            z-index: 1000;
            pointer-events: none;
            max-width: 300px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }}
        .tooltip-row {{
            margin: 4px 0;
        }}
        .tooltip-label {{
            font-weight: bold;
            color: #3498db;
        }}
        .prob-bar {{
            display: inline-block;
            height: 10px;
            background-color: #3498db;
            vertical-align: middle;
            margin-left: 5px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Teacher vs Student Distribution Visualization</h1>
        <div class="info">
            <strong>Step:</strong> {global_step} | 
            <strong>Sample:</strong> {sample_idx} | 
            <strong>Task:</strong> {task_type} | 
            <strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        </div>
    </div>
    
    <div class="stats">
        <h3>Summary Statistics</h3>
        <div class="stats-grid">
            <div class="stat-item">
                <div class="stat-label">Mean Signed Diff (P-R)</div>
                <div class="stat-value" style="color: {('blue' if mean_signed_diff > 0 else 'red')}">{mean_signed_diff:+.4f}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Mean Abs Difference</div>
                <div class="stat-value">{mean_abs_diff:.4f}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Max Abs Difference</div>
                <div class="stat-value">{max_abs_diff_val:.4f}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Total Tokens</div>
                <div class="stat-value">{len(tokens)}</div>
            </div>
            {f'''<div class="stat-item">
                <div class="stat-label">Prompt Length</div>
                <div class="stat-value">{prompt_length}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Response Length</div>
                <div class="stat-value">{len(tokens) - prompt_length}</div>
            </div>''' if prompt_length else ''}
        </div>
        {f'''<div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #ddd;">
            <h4 style="margin: 5px 0;">Additional Information</h4>
            <pre style="background-color: #ecf0f1; padding: 10px; border-radius: 4px; overflow-x: auto;">{json.dumps(extra_info, indent=2)}</pre>
        </div>''' if extra_info else ''}
    </div>
    
    <div class="legend">
        <h3>Color Legend (Policy vs Reference Probability)</h3>
        <div style="margin-bottom: 15px;">
            <div class="legend-item">
                <span class="legend-color" style="background-color: rgb(200, 200, 255);"></span>
                <span>Light Blue: Policy slightly &gt; Reference (student more confident)</span>
            </div>
            <div class="legend-item">
                <span class="legend-color" style="background-color: rgb(100, 100, 255);"></span>
                <span>Medium Blue: Policy moderately &gt; Reference</span>
            </div>
            <div class="legend-item">
                <span class="legend-color" style="background-color: rgb(0, 0, 255);"></span>
                <span>Dark Blue: Policy much &gt; Reference (over-confident)</span>
            </div>
        </div>
        <div style="margin-bottom: 15px;">
            <div class="legend-item">
                <span class="legend-color" style="background-color: rgb(255, 200, 200);"></span>
                <span>Light Red: Policy slightly &lt; Reference (student less confident)</span>
            </div>
            <div class="legend-item">
                <span class="legend-color" style="background-color: rgb(255, 100, 100);"></span>
                <span>Medium Red: Policy moderately &lt; Reference</span>
            </div>
            <div class="legend-item">
                <span class="legend-color" style="background-color: rgb(255, 0, 0);"></span>
                <span>Dark Red: Policy much &lt; Reference (under-confident)</span>
            </div>
        </div>
        <div class="legend-item" style="display: block; margin-top: 10px;">
            <span class="legend-color" style="background-color: #ecf0f1; border: 2px solid #3498db;"></span>
            <span>Blue border = Prompt tokens (reference only)</span>
        </div>
    </div>
    
    <div class="content">
        <h3>Token Sequence (hover for details)</h3>
        <div class="tokens" id="tokens"></div>
    </div>
    
    <div class="tooltip" id="tooltip" style="display: none;"></div>
    
    <script>
        const tokenData = {json.dumps(token_data)};
        
        const tokensContainer = document.getElementById('tokens');
        const tooltip = document.getElementById('tooltip');
        
        // Render tokens
        tokenData.forEach((data, idx) => {{
            const span = document.createElement('span');
            span.className = 'token' + (data.is_prompt ? ' prompt' : '');
            span.textContent = data.token;
            span.style.backgroundColor = data.color;
            
            span.addEventListener('mouseenter', (e) => {{
                const maxBarWidth = 150;
                const teacherBarWidth = data.teacher_prob * maxBarWidth;
                const studentBarWidth = data.student_prob * maxBarWidth;
                const diffSign = data.signed_prob_diff >= 0 ? '+' : '';
                const diffColor = data.signed_prob_diff > 0 ? '#3498db' : (data.signed_prob_diff < 0 ? '#e74c3c' : '#95a5a6');
                const comparison = data.student_prob > data.teacher_prob ? 
                    '(Policy &gt; Reference )' : 
                    (data.student_prob < data.teacher_prob ? '(Policy &lt; Reference )' : '(Equal)');
                
                const refTop1Html = data.ref_top1_token !== undefined ? 
                    `<div class="tooltip-row" style="background-color: rgba(155, 89, 182, 0.1); padding: 4px; border-radius: 3px; margin: 4px 0;">
                        <span class="tooltip-label">Ref Top-1 Token:</span> 
                        <span style="font-family: 'Courier New', monospace; background-color: rgba(155, 89, 182, 0.2); padding: 2px 6px; border-radius: 2px;">"${{data.ref_top1_token}}"</span>
                    </div>` : '';
                
                tooltip.innerHTML = `
                    <div class="tooltip-row"><span class="tooltip-label">Token:</span> "${{data.token}}"</div>
                    <div class="tooltip-row"><span class="tooltip-label">Position:</span> ${{idx}} ${{data.is_prompt ? '(Prompt)' : '(Response)'}}</div>
                    ${{refTop1Html}}
                    <hr style="margin: 8px 0; border: none; border-top: 1px solid #555;">
                    <div class="tooltip-row">
                        <span class="tooltip-label">Reference (Teacher):</span> ${{(data.teacher_prob * 100).toFixed(2)}}%
                        <span class="prob-bar" style="width: ${{teacherBarWidth}}px; background-color: #9b59b6;"></span>
                    </div>
                    <div class="tooltip-row">
                        <span class="tooltip-label">Policy (Student):</span> ${{(data.student_prob * 100).toFixed(2)}}%
                        <span class="prob-bar" style="width: ${{studentBarWidth}}px; background-color: #e67e22;"></span>
                    </div>
                    <hr style="margin: 8px 0; border: none; border-top: 1px solid #555;">
                    <div class="tooltip-row">
                        <span class="tooltip-label">Signed Diff (P-R):</span> 
                        <span style="color: ${{diffColor}}; font-weight: bold;">${{diffSign}}${{(data.signed_prob_diff * 100).toFixed(2)}}%</span>
                        <span style="font-size: 11px; opacity: 0.8;"> ${{comparison}}</span>
                    </div>
                    <div class="tooltip-row">
                        <span class="tooltip-label">Abs Difference:</span> ${{(data.abs_prob_diff * 100).toFixed(2)}}%
                    </div>
                    <hr style="margin: 8px 0; border: none; border-top: 1px solid #555;">
                    <div class="tooltip-row" style="font-size: 11px; opacity: 0.8;">
                        <span class="tooltip-label">Reference Log Prob:</span> ${{data.teacher_log_prob.toFixed(4)}}
                    </div>
                    <div class="tooltip-row" style="font-size: 11px; opacity: 0.8;">
                        <span class="tooltip-label">Policy Log Prob:</span> ${{data.student_log_prob.toFixed(4)}}
                    </div>
                `;
                
                tooltip.style.display = 'block';
                updateTooltipPosition(e);
            }});
            
            span.addEventListener('mousemove', updateTooltipPosition);
            
            span.addEventListener('mouseleave', () => {{
                tooltip.style.display = 'none';
            }});
            
            tokensContainer.appendChild(span);
        }});
        
        function updateTooltipPosition(e) {{
            const tooltipWidth = tooltip.offsetWidth;
            const tooltipHeight = tooltip.offsetHeight;
            let left = e.pageX + 15;
            let top = e.pageY + 15;
            
            // Keep tooltip on screen
            if (left + tooltipWidth > window.innerWidth) {{
                left = e.pageX - tooltipWidth - 15;
            }}
            if (top + tooltipHeight > window.innerHeight + window.scrollY) {{
                top = e.pageY - tooltipHeight - 15;
            }}
            
            tooltip.style.left = left + 'px';
            tooltip.style.top = top + 'px';
        }}
    </script>
</body>
</html>
"""
    
    # Write HTML file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"[Visualization] Saved HTML to {output_path}")


def visualize_teacher_student_batch(
    batch,
    teacher_log_probs: torch.Tensor,
    student_log_probs: torch.Tensor,
    tokenizer,
    global_step: int,
    output_dir: str,
    num_samples: int = 2,
    task_type: Optional[str] = None,
    num_tokens: int = 1,
) -> List[Path]:
    """
    Visualize teacher vs student distributions for selected samples in a batch.
    
    Args:
        batch: DataProto containing the batch data
        teacher_log_probs: Teacher model log probabilities (batch_size, seq_len)
        student_log_probs: Student model log probabilities (batch_size, seq_len)
        tokenizer: Tokenizer to decode token IDs
        global_step: Current training step
        output_dir: Directory to save HTML files
        num_samples: Number of samples to visualize (default: 2)
        task_type: Task type for this batch (optional, extracted from batch if None)
    
    Returns:
        List of paths to generated HTML files
    """
    batch_size = len(batch)
    num_samples = min(num_samples, batch_size)
    
    # Select samples to visualize (e.g., first and last in batch)
    if batch_size <= 2:
        sample_indices = list(range(batch_size))
    else:
        sample_indices = [0, batch_size - 1][:num_samples]
    
    # Extract task type
    if task_type is None and 'task_type' in batch.non_tensor_batch:
        task_type = batch.non_tensor_batch['task_type'][sample_indices[0]]
    elif task_type is None:
        task_type = "unknown"
    
    output_paths = []
    output_base = Path(output_dir)
    
    for sample_idx in sample_indices:
        # Get full sequence (prompt + response)
        input_ids = batch.batch['input_ids'][sample_idx]
        responses = batch.batch['responses'][sample_idx]
        attention_mask = batch.batch['attention_mask'][sample_idx]
        
        # Determine prompt length
        response_length = responses.size(0)
        prompt_length = input_ids.size(0) - response_length
        
        # Get valid tokens (based on attention mask)
        valid_mask = attention_mask.bool()
        valid_input_ids = input_ids[valid_mask]
        
        # Account for padding tokens in prompt portion only (left padding)
        # Note: response may have right padding too, but we handle that separately
        prompt_attn_mask = attention_mask[:prompt_length]
        num_prompt_padding = (prompt_attn_mask == 0).sum().item()
        valid_prompt_length = prompt_length - num_prompt_padding
        
        # Decode tokens
        tokens = [tokenizer.decode([token_id]) for token_id in valid_input_ids.cpu().tolist()]
        
        # Get log probs for response tokens only (shape: response_length)
        # These are aligned to response positions, starting at index 0
        teacher_lp = teacher_log_probs[sample_idx].cpu().numpy()
        student_lp = student_log_probs[sample_idx].cpu().numpy()
        
        # Create extra info
        extra_info = {}
        if 'episode_rewards' in batch.non_tensor_batch:
            extra_info['episode_reward'] = float(batch.non_tensor_batch['episode_rewards'][sample_idx])
        if 'episode_lengths' in batch.non_tensor_batch:
            extra_info['episode_length'] = float(batch.non_tensor_batch['episode_lengths'][sample_idx])
        if 'data_source' in batch.non_tensor_batch:
            extra_info['data_source'] = str(batch.non_tensor_batch['data_source'][sample_idx])
        
        # Debug info for alignment verification
        extra_info['_debug'] = {
            'input_ids_shape': list(input_ids.shape),
            'responses_shape': list(responses.shape),
            'attention_mask_shape': list(attention_mask.shape),
            'teacher_lp_shape': list(teacher_lp.shape),
            'prompt_length': int(prompt_length),
            'response_length': int(response_length),
            'num_prompt_padding': int(num_prompt_padding),
            'valid_prompt_length': int(valid_prompt_length),
            'num_tokens': len(tokens),
            'response_start_in_tokens': int(valid_prompt_length),
            'num_valid_response_in_tokens': len(tokens) - int(valid_prompt_length),
            'teacher_lp_min': float(teacher_lp.min()),
            'teacher_lp_max': float(teacher_lp.max()),
            'teacher_lp_mean': float(teacher_lp.mean()),
            'teacher_lp_nonzero': int((teacher_lp != 0).sum()),
            'teacher_lp_first_5': [float(x) for x in teacher_lp[:5]],
            'student_lp_first_5': [float(x) for x in student_lp[:5]],
        }
        
        # Generate HTML
        filename = f"step{global_step:06d}_sample{sample_idx}_{task_type}.html"
        output_path = output_base / filename
        
        # Only create visualizations for response tokens (where we have probs)
        # Pad with zeros for prompt tokens
        full_teacher_lp = np.zeros(len(tokens))
        full_student_lp = np.zeros(len(tokens))
        
        # Fill in response token probabilities
        # teacher_lp/student_lp have shape (response_length,), aligned to response tokens starting at index 0
        # We need to place them at position valid_prompt_length in the token array
        response_start_in_tokens = int(valid_prompt_length)
        
        # Get the response portion of attention mask to identify valid response tokens
        response_attn_mask = attention_mask[prompt_length:prompt_length + response_length]
        num_valid_response_in_mask = response_attn_mask.sum().item()
        
        # Number of valid response tokens in our filtered token array
        num_valid_response_in_tokens = len(tokens) - response_start_in_tokens
        
        # The log probs array may have padding at the end (zeros for padded response tokens)
        # We copy the valid portion
        num_to_copy = min(len(teacher_lp), num_valid_response_in_tokens, int(num_valid_response_in_mask))
        
        if num_to_copy > 0:
            full_teacher_lp[response_start_in_tokens:response_start_in_tokens + num_to_copy] = teacher_lp[:num_to_copy]
            full_student_lp[response_start_in_tokens:response_start_in_tokens + num_to_copy] = student_lp[:num_to_copy]
        
        # Extract and decode ref_topk_indices (top-(num_tokens) token at each position)
        ref_top1_tokens = None
        if 'ref_topk_indices' in batch.batch:
            ref_topk_indices = batch.batch['ref_topk_indices'][sample_idx]  # Shape: (response_length, k)
            
            # Extract top-1 token (first column, index 0)
            if ref_topk_indices.dim() >= 2 and ref_topk_indices.size(1) > 0:
                ref_top1_indices = ref_topk_indices[:, 0].cpu().tolist()  # Get first column (top-1)
                
                # Decode the top-1 tokens
                ref_top1_token_strings = [tokenizer.decode([token_id]) for token_id in ref_top1_indices]
                
                # Create full array aligned with tokens (pad prompt with empty strings)
                ref_top1_tokens = [''] * len(tokens)
                
                # Fill in the response portion
                num_ref_tokens = min(len(ref_top1_token_strings), num_valid_response_in_tokens)
                for i in range(num_ref_tokens):
                    ref_top1_tokens[response_start_in_tokens + i] = ref_top1_token_strings[i]
        
        create_html_visualization(
            tokens=tokens,
            teacher_log_probs=full_teacher_lp.tolist(),
            student_log_probs=full_student_lp.tolist(),
            sample_idx=sample_idx,
            task_type=task_type,
            global_step=global_step,
            output_path=output_path,
            prompt_length=valid_prompt_length,
            extra_info=extra_info if extra_info else None,
            ref_top1_tokens=ref_top1_tokens,
        )
        
        output_paths.append(output_path)
    
    return output_paths
