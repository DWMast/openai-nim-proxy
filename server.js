// server.js - OpenAI to NVIDIA NIM API Proxy (LoreBary + JanitorAI Edition)
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware - Handling large character cards from JanitorAI
app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ limit: '50mb', extended: true }));

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

const SHOW_REASONING = true; 
const ENABLE_THINKING_MODE = true; 

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

app.get('/health', (req, res) => {
  res.json({ status: 'ok', reasoning: SHOW_REASONING, thinking: ENABLE_THINKING_MODE });
});

app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model, object: 'model', created: Date.now(), owned_by: 'nvidia-nim-proxy'
  }));
  res.json({ object: 'list', data: models });
});

app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;
    
    let nimModel = MODEL_MAPPING[model] || 'meta/llama-3.1-70b-instruct';
    
    const nimRequest = {
      model: nimModel,
      messages: messages,
      temperature: temperature || 0.6,
      max_tokens: max_tokens || 4096,
      extra_body: ENABLE_THINKING_MODE ? { chat_template_kwargs: { thinking: true } } : undefined,
      stream: stream || false
    };
    
    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: { 'Authorization': `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
      responseType: stream ? 'stream' : 'json'
    });
    
    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      
      let buffer = '';
      let reasoningStarted = false;
      
      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n'); 
        buffer = lines.pop(); 

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
              // LOREBARY PATCH: We merge 'reasoning_content' into 'content'
              // because LoreBary doesn't understand the reasoning field.
              let output = "";
              if (SHOW_REASONING && delta.reasoning_content) {
                if (!reasoningStarted) {
                  output = '<think>\n' + delta.reasoning_content;
                  reasoningStarted = true;
                } else {
                  output = delta.reasoning_content;
                }
              } else if (reasoningStarted && !delta.reasoning_content) {
                output = '</think>\n\n' + (delta.content || "");
                reasoningStarted = false;
              } else {
                output = delta.content || "";
              }

              delta.content = output;
              delete delta.reasoning_content; 

              if (delta.content) {
                 res.write(`data: ${JSON.stringify(data)}\n\n`);
              }
            }
          } catch (e) {}
        }
      });
      
      response.data.on('end', () => res.end());
      response.data.on('error', () => res.end());

    } else {
      // Non-streaming logic
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
    res.status(500).json({ error: { message: error.message } });
  }
});

app.listen(PORT, () => {
  console.log(`Proxy active on port ${PORT}`);
});
