# Spring FES Video - Project Instructions

## Project Context

A video processing application with Next.js frontend.

> Note: Detailed project requirements will be added to task.json as they are defined.

---

## MANDATORY: Agent Workflow

Every new agent session MUST follow this workflow:

### Step 1: Initialize Environment

```bash
./init.sh
```

This will:
- Install all dependencies
- Start the development server at http://localhost:3000

**DO NOT skip this step.** Ensure the server is running before proceeding.

### Step 2: Select Next Task

Read `task.json` and select ONE task to work on.

Selection criteria (in order of priority):
1. Choose a task where `passes: false`
2. Consider dependencies - fundamental features should be done first
3. Pick the highest-priority incomplete task

### Step 3: Implement the Task

- Read the task description and steps carefully
- Implement the functionality to satisfy all steps
- Follow existing code patterns and conventions

### Step 4: Test Thoroughly

After implementation, verify ALL steps in the task:
- Write unit tests if applicable
- Use browser testing for UI features (MCP Playwright tools)
- Run `npm run lint` and `npm run build` in hello-nextjs/
- Fix any errors before proceeding

### Step 5: Update Progress

Write your work to `progress.txt`:

```
## [Date] - Task: [task description]

### What was done:
- [specific changes made]

### Testing:
- [how it was tested]

### Notes:
- [any relevant notes for future agents]
```

### Step 6: Commit Changes

```bash
git add .
git commit -m "[task description] - completed"
```

### Step 7: Mark Task Complete

In `task.json`, change the task's `passes` from `false` to `true`.

**IMPORTANT:**
- Only mark `passes: true` after ALL steps are verified
- Never delete or modify task descriptions
- Never remove tasks from the list

---

## Project Structure

```
/
├── CLAUDE.md          # This file - workflow instructions
├── task.json          # Task definitions (source of truth)
├── progress.txt       # Progress log from each session
├── init.sh            # Initialization script
└── hello-nextjs/      # Next.js application
    ├── src/app/       # App Router pages
    ├── src/components/
    └── ...
```

## Commands

```bash
# In hello-nextjs/
npm run dev      # Start dev server
npm run build    # Production build
npm run lint     # Run linter
```

## Coding Conventions

- TypeScript strict mode
- Functional components with hooks
- Tailwind CSS for styling
- Write tests for new features

---

## Key Rules

1. **One task per session** - Focus on completing one task well
2. **Test before marking complete** - All steps must pass
3. **Document in progress.txt** - Help future agents understand your work
4. **Commit your changes** - Keep git history clean
5. **Never remove tasks** - Only flip `passes: false` to `true`
