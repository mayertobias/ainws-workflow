#!/usr/bin/env node

const axios = require('axios');
const chalk = require('chalk');

// Service endpoints to check
const SERVICES = {
  'Frontend': 'http://localhost:3001',
  'Audio Service': 'http://localhost:8001/health',
  'Content Service': 'http://localhost:8010/health',
  'Gateway': 'http://localhost:8000/health',
  'Orchestrator': 'http://localhost:8006/health',
};

// ... existing code ... 