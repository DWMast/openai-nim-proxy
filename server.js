// server.js - Optimized OpenAI to NVIDIA NIM API Proxy
const express = require('express');
const cors = require('cors');
const axios = require('axios');
const https = require('https'); 

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ limit: '50mb', extended: true }));

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// 🔥 REASONING DISPLAY CONFIG
const SHOW_REASONING = true; 
const ENABLE_THINKING_MODE = true; 

// Keep-Alive Agent to prevent "hangs" during long model pauses
const httpsAgent = new https.Agent({ 
  keepAlive: true,
  timeout: 600000 
});

// FULL MODEL MAPPING RESTORED
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'gpt-4': 'qwen/qwen3-coder-480b-a35b-instruct',
  'gpt-4-turbo': 'moonshotai/kimi-k2-instruct-0905',
  'gpt-4o': 'deepseek-ai/deepseek-v3.1',
  'claude-3-opus': 'openai/gpt-oss-120b',
  'claude-3-sonnet': 'openai/gpt-oss-20b',
  'gemini-pro': 'qwen/qwen3-next-80b-a3b-thinking',
  'glm-5': 'z-ai/glm5' // Restored!
};

app.get('/health', (req, res) => res.json({ status: 'ok', reasoning: SHOW_REASONING }));

app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(id => ({
    id, object: 'model', created: Date.now(), owned_by: 'nvidia-proxy'
  }));
  res.json({ object: 'list', data: models });
});

app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;
    
    let nimModel = MODEL_MAPPING[model];
    if (!nimModel) {
      nimModel = model.toLowerCase().includes('gpt-4') ? 'meta/llama-3.1-405b-instruct' : 'meta/llama-3.1-70b-instruct';
    }

    const nimRequest = {
      model: nimModel,
      messages: messages,
      temperature: temperature || 0.6,
      max_tokens: max_tokens || 4096,
      stream: stream || false
    };

    if (ENABLE_THINKING_MODE) {
       nimRequest.extra_body = { chat_template_kwargs: { thinking: true } };
    }

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
      let buffer = '';
      let reasoningActive = false;

      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop();

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed.startsWith('data: ')) continue;
          
          const dataStr = trimmed.slice(6);
          if (dataStr === '[DONE]') {
            res.write(`data: [DONE]\n\n`);
            continue;
          }

          try {
            const json = JSON.parse(dataStr);
            const delta = json.choices?.[0]?.delta;

            if (delta) {
              const content = delta.content || '';
              const reasoning = delta.reasoning_content;

              if (SHOW_REASONING) {
                if (reasoning) {
                  if (!reasoningActive) {
                    delta.content = '> **Thinking Process:**\n> ' + reasoning;
                    reasoningActive = true;
                  } else {
                    delta.content = reasoning;
                  }
                  delete delta.reasoning_content;
                } else if (reasoningActive && !reasoning) {
                  delta.content = '\n\n---\n\n' + content;
                  reasoningActive = false;
                }
              }

              if (delta.content) {
                  json.choices[0].delta = delta; 
                  res.write(`data: ${JSON.stringify(json)}\n\n`);
              }
            }
          } catch (e) {}
        }
      });
      
      response.data.on('end', () => res.end());
    } else {
      res.json(response.data);
    }
    
  } catch (error) {
    res.status(500).json({ error: { message: error.message } });
  }
});

app.listen(PORT, () => console.log(`🚀 Proxy running with GLM-5 on port ${PORT}`));
