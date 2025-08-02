#!/bin/bash
echo "Initializing Git repo and pushing to GitHub..."

git init
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git checkout -b main
git add .
git commit -m "Initial commit with full pipeline"
git push -u origin main
