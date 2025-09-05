#!/usr/bin/env python3
"""
Test button consistency using agentic browser automation

This test demonstrates:
1. Running the analysis to generate dashboard
2. Using agentic commands to test button consistency
3. Specifying LLM provider in the test
"""

import sys
import os
import asyncio
import subprocess
from pathlib import Path

# Add browser_agent to path
browser_agent_path = Path(__file__).parent.parent.parent / "browser_agent"
sys.path.append(str(browser_agent_path))

from browser_utils import setup_browser_agent, execute_custom_agentic_test

# Test Configuration
LLM_PROVIDER = "openai"  # or "anthropic"
MODEL_NAME = "gpt-4o-mini"  # or "claude-3-sonnet-20240229"
AGENTIC_COMMAND = "Open the page, take a screenshot, and confirm the buttons are now the same sizes"

async def test_button_consistency(visual_mode=False):
    """Test button consistency using agentic commands"""
    
    print(f"🧪 Testing: {AGENTIC_COMMAND}")
    print(f"🤖 Using: {LLM_PROVIDER} {MODEL_NAME}")
    
    # Step 1: Run analysis to generate dashboard
    print("📊 Running analysis to generate dashboard...")
    analysis_dir = Path(__file__).parent.parent
    original_cwd = os.getcwd()
    
    try:
        os.chdir(analysis_dir)
        result = subprocess.run([sys.executable, "run_analysis.py", "--cached-only"], 
                              capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Analysis failed: {result.stderr}")
            return False
        
        print("✅ Analysis completed")
        
        # Find the most recent output
        output_base = analysis_dir / "output"
        output_dirs = [d for d in output_base.iterdir() if d.is_dir()]
        latest_output = max(output_dirs, key=lambda d: d.stat().st_mtime)
        html_file = latest_output / "index.html"
        
        print(f"📁 Using dashboard: {html_file}")
        
    finally:
        os.chdir(original_cwd)
    
    # Step 2: Setup browser agent with specified LLM
    print(f"🤖 Setting up browser agent...")
    agent = setup_browser_agent(
        native=visual_mode, 
        llm_provider=LLM_PROVIDER, 
        model_name=MODEL_NAME
    )
    
    try:
        # Step 3: Execute agentic command
        file_url = f"file://{html_file.resolve()}"
        print(f"🎯 Executing: {AGENTIC_COMMAND}")
        
        result = await execute_custom_agentic_test(agent, AGENTIC_COMMAND, file_url)
        
        print(f"✅ Test completed!")
        print(f"Success: {result['success']}")
        
        if result['success']:
            print(f"Result: {result['result']}")
        else:
            print(f"Error: {result['error']}")
        
        return result['success']
        
    finally:
        await agent.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test button consistency')
    parser.add_argument('--visual', action='store_true', help='Run in visual mode')
    
    args = parser.parse_args()
    
    success = asyncio.run(test_button_consistency(visual_mode=args.visual))
    
    if success:
        print("✅ Button consistency test passed!")
    else:
        print("❌ Button consistency test failed!")
        sys.exit(1)

