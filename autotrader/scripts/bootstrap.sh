#!/usr/bin/env bash
poetry install
cp .env.example .env.local 2>/dev/null || true
echo "🚀  Project ready. Edit .env.local then run:  docker-compose -f infra/docker-compose.yml up --build"
