# ESLint Configuration Guide for DARF Framework

## Overview

This project uses ESLint to enforce code quality, maintain consistent style, and catch potential bugs before they make it to production. The configuration has been set up to address specific needs for both frontend (browser-based TypeScript with React) and backend (Node.js) codebases.

## Configuration Structure

We've implemented a hierarchical ESLint configuration to maximize code quality while respecting the different contexts of frontend and backend development:

1. **Root Configuration** (`.eslintrc.json`): Base rules applicable to all JavaScript/TypeScript code
2. **Backend Configuration** (`darf_webapp/.eslintrc.json`): Node.js specific rules
3. **Frontend Configuration** (`darf_frontend/.eslintrc.json`): Browser, React, and TypeScript specific rules

## Key Features

- **Extended Rule Sets**: Uses comprehensive, industry-standard rule sets from Airbnb and Airbnb TypeScript
- **Environment-Specific Rules**: Different rules for frontend and backend contexts
- **TypeScript Support**: Robust configuration for TypeScript files in the frontend
- **React Support**: Rules for React and React Hooks
- **Custom Rules**: Carefully selected rules to catch common bugs and enforce best practices

## Complete Setup Process

To set up ESLint correctly for this project, follow these steps:

```bash
# Run the setup script which installs dependencies and creates configuration files
./setup_eslint.sh
```

The setup script will:
1. Install all required ESLint dependencies
2. Create the root `.eslintrc.json` file
3. Create environment-specific configuration files
4. Create or verify the TypeScript configuration file for the frontend

No additional `npm install` is required after running the setup script, as it installs all necessary dependencies.

## TypeScript Configuration

The frontend uses TypeScript and requires a proper `tsconfig.json` file. This file is created automatically by the setup script if it doesn't exist. The ESLint configuration for the frontend references this TypeScript configuration.

## Running ESLint

To run ESLint and check your code quality, use the following npm scripts:

```bash
# Lint all files
npm run lint

# Lint and fix all fixable issues
npm run lint:fix

# Lint only frontend code
npm run lint:frontend

# Lint only backend code
npm run lint:backend
```

### Examples:

```bash
# Check all code for issues
npm run lint

# Fix issues automatically where possible
npm run lint:fix

# Check only the backend code
npm run lint:backend

# Fix issues in frontend code
npm run lint:frontend --fix
```

> **Important**: Make sure to add these scripts to your package.json if they don't exist:
>
> ```json
> "scripts": {
>   "lint": "eslint .",
>   "lint:fix": "eslint . --fix",
>   "lint:frontend": "eslint darf_frontend/",
>   "lint:backend": "eslint darf_webapp/"
> }
> ```
>
> The setup_eslint.sh script should have configured these, but if not, add them manually.

## Common Issues Detected

The current configuration will detect:

### Code Quality Issues
- Unused variables and imports
- Using `==` instead of `===` (loose equality)
- Reassigning function parameters
- Using variables before they're defined
- Variable shadowing
- Magic numbers without explanation

### Style Consistency
- Consistent indentation
- Proper spacing around operators
- Proper use of quotes (single quotes preferred)
- Consistent naming conventions (camelCase)
- Maximum line length
- Proper spacing in objects and arrays

### Potential Bugs
- Unreachable code
- Unused expressions
- Improper promises handling
- Missing return statements
- Assignment in conditionals
- Use of deprecated APIs

### TypeScript Specific
- Type inconsistencies
- Proper TypeScript syntax
- Type-aware linting rules

### Security Concerns
- Use of `eval()` or similar dangerous functions
- Improper error handling
- Unhandled promise rejections

## Frontend-Specific Checks
- Accessibility violations (jsx-a11y rules)
- React hooks rule violations
- Dangerous browser globals
- Improper event handling

## Backend-Specific Checks
- Improper use of `process.exit()`
- Consistent return statements
- Handling of Node.js specific APIs
- Proper error handling
- Parameter count limits

## Dependencies Handling

Note that the root configuration allows importing from devDependencies, while the webapp configuration restricts this. This ensures that:

- In development tools and scripts: You can import from devDependencies
- In backend production code: You cannot accidentally import development-only packages

## Extending the Configuration

To add custom rules or modify existing ones, edit the appropriate `.eslintrc.json` file. For project-wide rules, edit the root configuration; for environment-specific rules, edit the corresponding configuration file.
