# GitHub Issues Auto-Close Setup

This repository has been configured to automatically close issues when related PRs are merged. Here's how it works:

## 🔧 Setup Complete

### 1. Repository Setting ✅
The repository has the **"Auto-close issues with merged linked pull requests"** setting enabled in:
**Settings → General → Issues → Auto-close issues**

### 2. GitHub Action Workflow ✅
An automated workflow is configured at `.github/workflows/auto-close-issues.yml` that:
- Listens for PR open/edit events
- Extracts issue numbers from branch names
- Automatically appends `Closes #N` to PR descriptions

### 3. PR Template Updated ✅
The PR template has been updated to guide proper issue linking.

## 🤖 For Coding Agents

### Branch Naming Convention
When implementing GitHub issues, create branches using these patterns:
- `issue-N/description` (recommended)
- `N-description` 
- `issue-N/implement-feature`

**Examples:**
- `issue-5/motion-detection-worker`
- `7-face-recognition-engine`
- `issue-10/implement-storage-layer`

### Automatic Behavior
When you create a PR from a properly named branch, the GitHub Action will:
1. Detect the issue number from the branch name
2. Automatically append `Closes #N` to the PR description
3. The issue will close automatically when the PR is merged to `master`

### Manual Override
You can also manually include closing keywords in your PR description:
```
Closes #5
Fixes #7
Resolves #10
```

## 📋 Available Issues

All issues have been updated with auto-close instructions:

- **Issue #4**: Frame Capture Worker (✅ Previously implemented)
- **Issue #5**: Motion Detection Worker
- **Issue #6**: Face Detection Worker Pool  
- **Issue #7**: Face Recognition Engine
- **Issue #8**: Event Processing System
- **Issue #9**: Hardware Abstraction Layer
- **Issue #10**: Storage Layer
- **Issue #11**: Internal Notification System

## 🔍 Troubleshooting

### If an issue doesn't close automatically:

1. **Check the branch name**: Ensure it follows the pattern `issue-N/description`
2. **Check the PR description**: Verify `Closes #N` appears in the description
3. **Check the merge target**: PRs must target the `master` branch
4. **Check repository settings**: Ensure auto-close is enabled in repository settings

### Force manual closure:
If needed, you can manually add closing keywords to any PR description:
```
Closes #N
```

## 🎯 Benefits

- **Zero manual work**: Issues close automatically when PRs merge
- **Clear tracking**: Explicit connection between issues and PRs
- **Workflow automation**: Reduces administrative overhead
- **Consistent process**: All team members and coding agents follow the same pattern

---

**Note**: This setup works for both human developers and AI coding agents, providing a seamless workflow for issue management and PR tracking.