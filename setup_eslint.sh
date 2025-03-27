#!/bin/bash

# Install all required ESLint dependencies
npm install --save-dev eslint eslint-config-airbnb-base eslint-plugin-import \
  @typescript-eslint/eslint-plugin @typescript-eslint/parser \
  eslint-config-airbnb-typescript eslint-plugin-jsx-a11y \
  eslint-plugin-react eslint-plugin-react-hooks

# Create main ESLint configuration file
cat > .eslintrc.json << 'EOF'
{
  "extends": [
    "eslint:recommended",
    "airbnb-base"
  ],
  "env": {
    "es2021": true,
    "browser": true,
    "node": true
  },
  "parserOptions": {
    "ecmaVersion": 12,
    "sourceType": "module"
  },
  "rules": {
    "no-unused-vars": "error",
    "no-console": "warn",
    "no-undef": "error",
    "no-var": "error",
    "prefer-const": "error",
    "eqeqeq": ["error", "always"],
    "max-len": ["warn", { "code": 100 }],
    "comma-dangle": ["error", "always-multiline"],
    "arrow-body-style": ["error", "as-needed"],
    "no-param-reassign": "error",
    "no-use-before-define": "error",
    "no-shadow": "error",
    "camelcase": "error",
    "quotes": ["error", "single", { "avoidEscape": true }],
    "object-curly-spacing": ["error", "always"],
    "no-multiple-empty-lines": ["error", { "max": 1, "maxEOF": 1 }],
    "no-trailing-spaces": "error",
    "semi": ["error", "always"],
    "import/no-extraneous-dependencies": ["error", {"devDependencies": true}]
  },
  "ignorePatterns": ["node_modules/", "build/", "dist/"]
}
EOF

# Create backend-specific ESLint configuration
cat > darf_webapp/.eslintrc.json << 'EOF'
{
  "extends": [
    "../.eslintrc.json"
  ],
  "env": {
    "node": true
  },
  "rules": {
    "no-process-exit": "error",
    "import/no-extraneous-dependencies": ["error", {"devDependencies": true}],
    "no-underscore-dangle": ["error", { "allow": ["_id"] }],
    "consistent-return": "error",
    "class-methods-use-this": "warn",
    "max-params": ["warn", 4],
    "no-throw-literal": "error",
    "prefer-promise-reject-errors": "error",
    "no-unused-expressions": ["error", { "allowShortCircuit": true, "allowTernary": true }]
  }
}
EOF

# Create frontend-specific ESLint configuration
cat > darf_frontend/.eslintrc.json << 'EOF'
{
  "extends": [
    "../.eslintrc.json",
    "airbnb-typescript/base"
  ],
  "env": {
    "browser": true
  },
  "parserOptions": {
    "project": "./tsconfig.json"
  },
  "rules": {
    "no-alert": "warn",
    "no-param-reassign": "error",
    "import/prefer-default-export": "off",
    "jsx-a11y/click-events-have-key-events": "error",
    "jsx-a11y/no-static-element-interactions": "error",
    "no-restricted-globals": ["error", "event", "name", "location"],
    "no-nested-ternary": "warn",
    "react-hooks/rules-of-hooks": "error",
    "react-hooks/exhaustive-deps": "warn"
  }
}
EOF

# Create or update tsconfig.json if it doesn't exist
if [ ! -f "darf_frontend/tsconfig.json" ]; then
  cat > darf_frontend/tsconfig.json << 'EOF'
{
  "compilerOptions": {
    "target": "es5",
    "lib": [
      "dom",
      "dom.iterable",
      "esnext"
    ],
    "allowJs": true,
    "skipLibCheck": true,
    "esModuleInterop": true,
    "allowSyntheticDefaultImports": true,
    "strict": true,
    "forceConsistentCasingInFileNames": true,
    "noFallthroughCasesInSwitch": true,
    "module": "esnext",
    "moduleResolution": "node",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "baseUrl": "src"
  },
  "include": [
    "src"
  ]
}
EOF
  echo "Created TypeScript configuration file."
else
  echo "TypeScript configuration already exists."
fi

# Add npm scripts for linting if not present
if ! grep -q '"lint"' package.json 2>/dev/null; then
  echo "Adding lint scripts to package.json"
  # This is a very simple approach - in a real scenario you might want to use jq or a more robust JSON manipulation tool
  if grep -q '"scripts"' package.json 2>/dev/null; then
    # Add to existing scripts section - basic implementation
    sed -i 's/"scripts": {/"scripts": {\n    "lint": "eslint .",\n    "lint:fix": "eslint . --fix",\n    "lint:frontend": "eslint darf_frontend\/",\n    "lint:backend": "eslint darf_webapp\/",/g' package.json
  else
    # Create scripts section if it doesn't exist - this is a simplified approach
    echo "Warning: No scripts section found in package.json. Please add manually:"
    echo '"scripts": {'
    echo '  "lint": "eslint .",'
    echo '  "lint:fix": "eslint . --fix",'
    echo '  "lint:frontend": "eslint darf_frontend/",'
    echo '  "lint:backend": "eslint darf_webapp/"'
    echo '}'
  fi
fi

echo "ESLint configuration has been set up successfully!"
echo ""
echo "To lint your code, run:"
echo "  npm run lint        # Lint all files"
echo "  npm run lint:fix    # Lint and fix all fixable issues"
echo "  npm run lint:frontend  # Lint only frontend code"
echo "  npm run lint:backend   # Lint only backend code"

# Output clear usage information 
cat << EOF

------------------------------------------------------------
DARF Framework ESLint Usage Guide
------------------------------------------------------------

To check your code quality, use these commands:

1. Check all code for issues:
   npm run lint

2. Fix issues automatically where possible:
   npm run lint:fix

3. Check only the backend code:
   npm run lint:backend

4. Check only the frontend code:
   npm run lint:frontend

For more information, see eslint-guide.md
------------------------------------------------------------

EOF
