const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middlewares
app.use(cors());
app.use(express.json({ limit: '50mb' }));

// Config
const NIM_API_BASE = 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'gpt-4': 'qwen/qwen3-coder-480b-a35b-instruct',
  'gpt-4o': 'deepseek-ai/deepseek-v3.1',
  'glm-5': 'z-ai/glm5'
};

// Health Check (Railway needs this to pass)
app.get('/health', (req, res) => res.status(200).send('OK'));
app.get('/', (req, res) => res.status(200).send('Proxy is running.'));

// Chat endpoint
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, stream } = req.body;
    const nimModel = MODEL_MAPPING[model] || 'meta/llama-3.1-70b-instruct';

    console.log(`Request for model: ${model} -> ${nimModel}`);

    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, {
      ...req.body,
      model: nimModel,
      extra_body: { chat_template_kwargs: { thinking: true } }
    }, {
      headers: { 'Authorization': `Bearer ${NIM_API_KEY}` },
      responseType: stream ? 'stream' : 'json'
    });

    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      let reasoningStarted = false;

      response.data.on('data', (chunk) => {
        const lines = chunk.toString().split('\n');
        for (const line of lines) {
          if (!line.trim().startsWith('data: ')) continue;
          if (line.includes('[DONE]')) {
            res.write('data: [DONE]\n\n');
            continue;
          }

          try {
            const data = JSON.parse(line.replace('data: ', ''));
            const delta = data.choices[0].delta;

            // LoreBary Fix: Combine reasoning into the main content field
            if (delta.reasoning_content) {
              if (!reasoningStarted) {
                delta.content = '<think>\n' + delta.reasoning_content;
                reasoningStarted = true;
              } else {
                delta.content = delta.reasoning_content;
              }
              delete delta.reasoning_content;
            } else if (reasoningStarted) {
              // Close the thinking tag as soon as real content starts
              delta.content = '</think>\n\n' + (delta.content || '');
              reasoningStarted = false;
            }

            res.write(`data: ${JSON.stringify(data)}\n\n`);
          } catch (e) {
            // Skip unparseable lines
          }
        }
      });
      response.data.on('end', () => res.end());
    } else {
      res.json(response.data);
    }
  } catch (err) {
    console.error("Proxy Error:", err.message);
    res.status(500).json({ error: err.message });
  }
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`Server listening on port ${PORT}`);
});
