# ConnectLocationAnalysis - One-off Analysis Workspace

Template workspace for rapid creation of one-off data analysis projects with static HTML dashboards.

## Overview
- **Purpose**: Copy template folder, generate HTML dashboard
- **Workflow**: Simple copy-paste approach for rapid iteration
- **Output**: Timestamped HTML dashboards that auto-open in browser

## How to Use

### For New Analysis:
1. **Copy the template:** Copy `src/project_example/` to `src/your_project_name/`
2. **Tell AI:** "Read the INSTRUCTIONS.md file and help me build an analysis"
3. **Add data** to the `data/` folder
4. **Customize** `run_analysis.py`
5. **Run** to generate dashboard

### AI Instructions to Give:
```
Read src/project_example/INSTRUCTIONS.md and help me create a new analysis.
Copy project_example to src/[project_name]. I'll then tell you more about my data and what I want to visualize
```