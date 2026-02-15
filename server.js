// server.js - OpenAI to NVIDIA NIM API Proxy
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json({ limit: '150mb' }));
app.use(express.urlencoded({ limit: '150mb', extended: true }));

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// 🔥 REASONING DISPLAY TOGGLE
const SHOW_REASONING = true; 

// 🔥 THINKING MODE TOGGLE
const ENABLE_THINKING_MODE = true; 

// Model mapping
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

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ 
    status: 'ok', 
    service: 'OpenAI to NVIDIA NIM Proxy', 
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE
  });
});

// List models endpoint
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created: Date.now(),
    owned_by: 'nvidia-nim-proxy'
  }));
  res.json({ object: 'list', data: models });
});

// Chat completions endpoint
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;
    
    // 1. Model Selection Logic
    let nimModel = MODEL_MAPPING[model];
    if (!nimModel) {
      // Fallback logic
      const modelLower = model.toLowerCase();
      if (modelLower.includes('gpt-4') || modelLower.includes('claude-opus')) {
        nimModel = 'meta/llama-3.1-405b-instruct';
      } else {
        nimModel = 'meta/llama-3.1-70b-instruct';
      }
    }
    
    // 2. Prepare Request
    const nimRequest = {
      model: nimModel,
      messages: messages,
      temperature: temperature || 0.6,
      max_tokens: max_tokens || 4096,
      extra_body: ENABLE_THINKING_MODE ? { chat_template_kwargs: { thinking: true } } : undefined,
      stream: stream || false
    };
    
    // 3. Make Request
    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: {
        'Authorization': `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      responseType: stream ? 'stream' : 'json'
    });
    
    // 4. Handle Response
    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      
      let buffer = '';
      let reasoningStarted = false;
      
      response.data.on('data', (chunk) => {
        // Fix: Use correct split for data chunks
        buffer += chunk.toString();
        const lines = buffer.split('\n'); 
        buffer = lines.pop(); // Keep incomplete line in buffer

        for (const line of lines) {
          const trimmedLine = line.trim();
          if (!trimmedLine.startsWith('data: ')) continue;
          
          const dataStr = trimmedLine.replace('data: ', '');
          if (dataStr === '[DONE]') {
            res.write(`data: [DONE]\n\n`);
            continue;
          }

          try {
            const data = JSON.parse(dataStr);
            const delta = data.choices?.[0]?.delta;

            if (delta) {
              const content = delta.content || '';
              const reasoning = delta.reasoning_content;

              if (SHOW_REASONING) {
                // LOGIC FIX: Robust state switching
                if (reasoning) {
                  // We are receiving reasoning tokens
                  if (!reasoningStarted) {
                    delta.content = '<think>\n' + reasoning;
                    reasoningStarted = true;
                  } else {
                    delta.content = reasoning;
                  }
                  delete delta.reasoning_content; // Remove raw field so client uses content
                } else if (reasoningStarted && !reasoning) {
                  // Reasoning just stopped (or we switched to content)
                  // We MUST close the tag now, even if content is empty
                  delta.content = '</think>\n\n' + content;
                  reasoningStarted = false;
                }
              } else {
                // If reasoning is hidden, strip it completely
                if (reasoning) {
                   delta.content = ''; 
                   delete delta.reasoning_content;
                }
              }

              // Only send if we actually have something to show (avoid empty updates that confuse clients)
              if (delta.content || delta.reasoning_content === undefined) {
                 res.write(`data: ${JSON.stringify(data)}\n\n`);
              }
            }
          } catch (e) {
            // If parsing fails, ignore this line
          }
        }
      });
      
      response.data.on('end', () => res.end());
      response.data.on('error', (err) => {
        console.error('Stream error:', err);
        res.end();
      });

    } else {
      // Non-streaming fallback
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: response.data.choices.map(choice => {
            let content = choice.message?.content || '';
            if (SHOW_REASONING && choice.message?.reasoning_content) {
                content = `<think>\n${choice.message.reasoning_content}\n</think>\n\n${content}`;
            }
            return {
                index: choice.index,
                message: { role: choice.message.role, content: content },
                finish_reason: choice.finish_reason
            };
        }),
        usage: response.data.usage
      };
      res.json(openaiResponse);
    }
    
  } catch (error) {
    console.error('Proxy error:', error.message);
    res.status(500).json({ error: { message: error.message, type: 'server_error' } });
  }
});

// Catch-all
app.all('*', (req, res) => res.status(404).json({ error: 'Not found' }));

app.listen(PORT, () => {
  console.log(`OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
  console.log(`Reasoning: ${SHOW_REASONING}, Thinking Mode: ${ENABLE_THINKING_MODE}`);
});
