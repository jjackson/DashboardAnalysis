#!/usr/bin/env python3
"""
Agentic browser test for Connect Location Analysis Dashboard

This test demonstrates the full workflow:
1. Run the analysis to generate dashboard
2. Load the output HTML file
3. Use agentic browser commands to test the dashboard
4. Take screenshots and perform UI validation
"""

import sys
import os
import asyncio
import subprocess
from pathlib import Path
from datetime import datetime

# Add browser_agent to path
browser_agent_path = Path(__file__).parent.parent.parent / "browser_agent"
sys.path.append(str(browser_agent_path))

from browser_utils import (
    setup_browser_agent, 
    execute_custom_agentic_test,
    test_dashboard_functionality_agentic,
    take_screenshot_agentic,
    verify_ui_elements_agentic
)

class DashboardAgenticTest:
    """Agentic test class for dashboard functionality"""
    
    def __init__(self, use_cached_data=True, visual_mode=False, llm_provider="openai", model_name="gpt-4o-mini"):
        self.use_cached_data = use_cached_data
        self.visual_mode = visual_mode
        self.llm_provider = llm_provider
        self.model_name = model_name
        self.output_dir = None
        self.agent = None
        
    async def setup(self):
        """Setup the browser agent"""
        print("🤖 Setting up browser agent...")
        self.agent = setup_browser_agent(
            native=self.visual_mode, 
            llm_provider=self.llm_provider, 
            model_name=self.model_name
        )
        print(f"✅ Browser agent ready (visual_mode: {self.visual_mode})")
        
    async def cleanup(self):
        """Cleanup browser agent"""
        if self.agent:
            try:
                await self.agent.close()
                print("✅ Browser agent closed")
            except:
                pass
    
    def run_analysis(self):
        """Run the connect location analysis to generate dashboard"""
        print("📊 Running connect location analysis...")
        
        # Change to the analysis directory
        analysis_dir = Path(__file__).parent.parent
        original_cwd = os.getcwd()
        
        try:
            os.chdir(analysis_dir)
            
            # Run analysis with appropriate flags
            cmd = [sys.executable, "run_analysis.py"]
            if self.use_cached_data:
                cmd.append("--cached-only")
            
            print(f"🔧 Executing: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"❌ Analysis failed: {result.stderr}")
                raise Exception(f"Analysis failed: {result.stderr}")
            
            print("✅ Analysis completed successfully")
            
            # Find the most recent output directory
            output_base = analysis_dir / "output"
            if not output_base.exists():
                raise Exception("Output directory not found")
            
            # Get the most recent output directory
            output_dirs = [d for d in output_base.iterdir() if d.is_dir()]
            if not output_dirs:
                raise Exception("No output directories found")
            
            self.output_dir = max(output_dirs, key=lambda d: d.stat().st_mtime)
            print(f"📁 Using output directory: {self.output_dir}")
            
            return True
            
        finally:
            os.chdir(original_cwd)
    
    async def test_dashboard_basic(self):
        """Run basic dashboard functionality test"""
        print("🧪 Running basic dashboard test...")
        
        html_file = self.output_dir / "index.html"
        if not html_file.exists():
            raise Exception(f"Dashboard HTML file not found: {html_file}")
        
        file_url = f"file://{html_file.resolve()}"
        print(f"🌐 Testing URL: {file_url}")
        
        # Run comprehensive agentic test
        result = await test_dashboard_functionality_agentic(self.agent, file_url)
        
        print(f"✅ Basic test completed. Success: {result['overall_success']}")
        return result
    
    async def test_custom_agentic_command(self, command):
        """Execute a custom agentic command on the dashboard"""
        print(f"🤖 Executing custom agentic command: {command}")
        
        html_file = self.output_dir / "index.html"
        file_url = f"file://{html_file.resolve()}"
        
        result = await execute_custom_agentic_test(self.agent, command, file_url)
        
        print(f"✅ Custom command completed. Success: {result['success']}")
        return result
    
    async def test_button_consistency(self):
        """Test that buttons have consistent styling and sizing"""
        print("🔘 Testing button consistency...")
        
        command = "Open the page, take a screenshot, and confirm the buttons are now the same sizes and have consistent styling"
        result = await self.test_custom_agentic_command(command)
        
        return result
    
    async def test_map_functionality(self):
        """Test that the map loads and displays correctly"""
        print("🗺️  Testing map functionality...")
        
        command = "Verify that the map loads correctly, displays markers, and is interactive"
        result = await self.test_custom_agentic_command(command)
        
        return result
    
    async def test_filter_functionality(self):
        """Test that filters work correctly"""
        print("🔍 Testing filter functionality...")
        
        command = "Test the filter dropdowns by selecting different options and verifying that the map and data update accordingly"
        result = await self.test_custom_agentic_command(command)
        
        return result
    
    async def run_comprehensive_test(self):
        """Run all tests in sequence"""
        print("🚀 Starting comprehensive agentic dashboard test...")
        
        results = {
            "setup": False,
            "analysis": False,
            "basic_test": None,
            "button_consistency": None,
            "map_functionality": None,
            "filter_functionality": None,
            "overall_success": False
        }
        
        try:
            # Setup
            await self.setup()
            results["setup"] = True
            
            # Run analysis
            results["analysis"] = self.run_analysis()
            
            # Run tests
            results["basic_test"] = await self.test_dashboard_basic()
            results["button_consistency"] = await self.test_button_consistency()
            results["map_functionality"] = await self.test_map_functionality()
            results["filter_functionality"] = await self.test_filter_functionality()
            
            # Determine overall success
            results["overall_success"] = all([
                results["setup"],
                results["analysis"],
                results["basic_test"]["overall_success"] if results["basic_test"] else False,
                results["button_consistency"]["success"] if results["button_consistency"] else False,
                results["map_functionality"]["success"] if results["map_functionality"] else False,
                results["filter_functionality"]["success"] if results["filter_functionality"] else False
            ])
            
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            results["error"] = str(e)
        
        finally:
            await self.cleanup()
        
        return results

async def main():
    """Main test function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run agentic dashboard tests')
    parser.add_argument('--visual', action='store_true', help='Run in visual mode (opens browser window)')
    parser.add_argument('--no-cache', action='store_true', help='Run analysis without cached data')
    parser.add_argument('--command', '-c', help='Run a single custom agentic command')
    parser.add_argument('--llm', default='openai', help='LLM provider (openai, anthropic)')
    parser.add_argument('--model', default='gpt-4o-mini', help='Model name to use')
    
    args = parser.parse_args()
    
    # Create test instance
    test = DashboardAgenticTest(
        use_cached_data=not args.no_cache,
        visual_mode=args.visual,
        llm_provider=args.llm,
        model_name=args.model
    )
    
    if args.command:
        # Run single custom command
        print(f"🎯 Running single custom command: {args.command}")
        
        await test.setup()
        test.run_analysis()
        result = await test.test_custom_agentic_command(args.command)
        await test.cleanup()
        
        print(f"✅ Command result: {result}")
        
    else:
        # Run comprehensive test
        results = await test.run_comprehensive_test()
        
        # Print summary
        print("\n" + "="*50)
        print("📋 TEST SUMMARY")
        print("="*50)
        print(f"Setup: {'✅' if results['setup'] else '❌'}")
        print(f"Analysis: {'✅' if results['analysis'] else '❌'}")
        
        if results['basic_test']:
            print(f"Basic Test: {'✅' if results['basic_test']['overall_success'] else '❌'}")
        
        if results['button_consistency']:
            print(f"Button Consistency: {'✅' if results['button_consistency']['success'] else '❌'}")
        
        if results['map_functionality']:
            print(f"Map Functionality: {'✅' if results['map_functionality']['success'] else '❌'}")
        
        if results['filter_functionality']:
            print(f"Filter Functionality: {'✅' if results['filter_functionality']['success'] else '❌'}")
        
        print(f"\nOverall Success: {'✅' if results['overall_success'] else '❌'}")
        
        if 'error' in results:
            print(f"Error: {results['error']}")

if __name__ == "__main__":
    asyncio.run(main())
