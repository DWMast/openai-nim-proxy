// server.js - Fast OpenAI to NVIDIA NIM Proxy
const express = require('express');
const cors = require('cors');
const axios = require('axios');
const https = require('https'); 

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json({ limit: '50mb' }));

// NVIDIA Configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// ⚡ SPEED SETTINGS (Thinking disabled for instant replies)
const SHOW_REASONING = false; 
const ENABLE_THINKING_MODE = false; 

// Keep-Alive to prevent connection drops
const httpsAgent = new https.Agent({ keepAlive: true });

const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'gpt-4': 'qwen/qwen3-coder-480b-a35b-instruct',
  'gpt-4-turbo': 'moonshotai/kimi-k2-instruct-0905',
  'gpt-4o': 'deepseek-ai/deepseek-v3.1',
  'claude-3-opus': 'openai/gpt-oss-120b',
  'claude-3-sonnet': 'openai/gpt-oss-20b',
  'gemini-pro': 'qwen/qwen3-next-80b-a3b-thinking',
  'glm-5': 'z-ai/glm5'
};

app.get('/health', (req, res) => res.json({ status: 'ok', thinking: 'OFF' }));

app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(id => ({
    id, object: 'model', created: Date.now(), owned_by: 'nvidia-proxy'
  }));
  res.json({ object: 'list', data: models });
});

// The 'async' keyword here is critical to avoid 'Unexpected identifier'
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;
    
    let nimModel = MODEL_MAPPING[model] || 'meta/llama-3.1-70b-instruct';

    console.log(`[Request] Model: ${nimModel} | Stream: ${stream}`);

    const nimRequest = {
      model: nimModel,
      messages: messages,
      temperature: temperature || 0.7,
      max_tokens: max_tokens || 4096,
      stream: stream || false
    };

    // We removed 'extra_body' here so the model doesn't waste time thinking

    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: {
        'Authorization': `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      responseType: stream ? 'stream' : 'json',
      httpsAgent: httpsAgent,
      timeout: 0 
    });

    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      response.data.pipe(res); // Direct pipe for maximum speed
    } else {
      res.json(response.data);
    }
    
  } catch (error) {
    console.error("Proxy Error:", error.response?.data || error.message);
    res.status(500).json({ error: { message: error.message } });
  }
});

app.listen(PORT, () => console.log(`🚀 Fast Proxy (No Thinking) running on ${PORT}`));
