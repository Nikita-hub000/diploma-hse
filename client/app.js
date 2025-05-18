import { createPongAgent } from './agent.js';
import { MODELPATHS } from './const.js';

document.addEventListener('DOMContentLoaded', async () => {
  const $ = (id) => document.getElementById(id);

  const canvas = $('gameCanvas');
  const statusDiv = $('statusBanner');
  const startBtn = $('btnPlay');
  const resetBtn = $('btnReset');
  const aiScoreSpan = $('aiScore');
  const playerScoreSpan = $('playerScore');
  const touchUpBtn = $('padUp');
  const touchDownBtn = $('padDown');

  const modelTypeSwitch = $('modelTypeSwitch');
  const diffSwitch = $('difficultySwitch');

  if (
    ![
      canvas,
      statusDiv,
      startBtn,
      resetBtn,
      aiScoreSpan,
      playerScoreSpan,
      touchUpBtn,
      touchDownBtn,
      modelTypeSwitch,
      diffSwitch,
    ].every(Boolean)
  ) {
    console.error('Required DOM elements not found.');
    return;
  }

  const BASE_PADDLE_SPEED = 12.0;
  const NORM_HEIGHT = 400 / 2;
  const PADDLE_HALF_H = 60 / 2;

  const game = {
    env: null,
    agent: null,
    rafId: null,
    tsPrev: 0,
    mode: 'loading',
    aiScore: 0,
    playerScore: 0,
    playerPaddleY: 0,
    speedMultiplier: 1,
    keys: new Set(),
    touch: { up: false, down: false },
  };

  const setStatus = (txt) => (statusDiv.textContent = txt);
  const updateScore = () => {
    aiScoreSpan.textContent = game.aiScore;
    playerScoreSpan.textContent = game.playerScore;
  };
  const clamp = (v, min, max) => Math.max(min, Math.min(max, v));

  const getCurrentPath = () => {
    const type = document.querySelector(
      'input[name="modelType"]:checked'
    ).value;
    const level =
      document.querySelector('input[name="aiLevel"] :checked')?.value || // ESLint fix
      document.querySelector('input[name="aiLevel"]:checked').value;
    return MODELPATHS[type][level];
  };

  const loadAgent = async () => {
    const path = getCurrentPath();
    setStatus(`Loading model: ${path.split('/').pop()}`);
    const newAgent = createPongAgent(path);
    if (await newAgent.loadModel()) {
      game.agent = newAgent;
      setStatus('Model ready. Press Play or Space.');
      console.log(`Loaded ${path}`);
    } else {
      setStatus('Failed to load model!');
    }
  };

  const initEnv = () => {
    canvas.width = 600;
    canvas.height = 400;
    game.env = new PongEnvJs(canvas);
    resetVisuals();
  };

  const nextRound = () => {
    game.env.reset();
    game.playerPaddleY = 0;
    resetVisuals();
    game.mode = 'running';
    game.tsPrev = performance.now();
    game.rafId = requestAnimationFrame(loop);
    setStatus('Ready…');
  };

  const attachUI = () => {
    startBtn.addEventListener('click', toggleRunPause);
    resetBtn.addEventListener('click', fullReset);

    modelTypeSwitch.addEventListener('change', async () => {
      pause();
      await loadAgent();
    });
    diffSwitch.addEventListener('change', async () => {
      pause();
      await loadAgent();
    });

    document.addEventListener('keydown', ({ key, code }) => {
      if (code === 'Space') return toggleRunPause();
      if (key.toLowerCase() === 'r') return fullReset();
      if (key === 'w' || key === 's') game.keys.add(key);
    });
    document.addEventListener('keyup', ({ key }) => {
      if (key === 'w' || key === 's') game.keys.delete(key);
    });

    const tStart = (dir) => (e) => {
      e.preventDefault();
      game.touch[dir] = true;
    };
    const tEnd = (dir) => (e) => {
      e.preventDefault();
      game.touch[dir] = false;
    };
    touchUpBtn.addEventListener('touchstart', tStart('up'));
    touchUpBtn.addEventListener('touchend', tEnd('up'));
    touchDownBtn.addEventListener('touchstart', tStart('down'));
    touchDownBtn.addEventListener('touchend', tEnd('down'));
  };

  const toggleRunPause = () => (game.mode === 'running' ? pause() : start());

  const start = () => {
    if (game.mode === 'ready') {
      game.env.reset();
      game.playerPaddleY = 0;
      resetVisuals();
    }
    game.mode = 'running';
    startBtn.textContent = 'Pause';
    resetBtn.disabled = false;
    game.tsPrev = performance.now();
    game.rafId ??= requestAnimationFrame(loop);
  };

  const pause = () => {
    if (game.mode !== 'running') return;
    game.mode = 'paused';
    startBtn.textContent = 'Resume';
    cancelAnimationFrame(game.rafId);
    game.rafId = null;
  };

  const fullReset = () => {
    cancelAnimationFrame(game.rafId);
    game.rafId = null;
    Object.assign(game, {
      mode: 'ready',
      aiScore: 0,
      playerScore: 0,
      playerPaddleY: 0,
    });
    updateScore();
    game.env.reset();
    resetVisuals();
    startBtn.textContent = 'Play';
    resetBtn.disabled = true;
    setStatus('Reset.');
  };

  const resetVisuals = () => {
    if (!game.env) return;
    game.env.paddle1_y = 0;
    game.env.paddle2_y = game.playerPaddleY;
    game.env.render();
  };

  const loop = async (ts) => {
    if (game.mode !== 'running') {
      game.rafId = null;
      return;
    }

    const dt = Math.min(0.05, (ts - game.tsPrev) / 1000);
    game.tsPrev = ts;

    const dir =
      (game.keys.has('w') || game.touch.up ? -1 : 0) +
      (game.keys.has('s') || game.touch.down ? 1 : 0);
    const dy = dir * BASE_PADDLE_SPEED * 60 * dt * game.speedMultiplier;
    game.playerPaddleY = clamp(
      game.playerPaddleY + dy,
      -NORM_HEIGHT + PADDLE_HALF_H,
      NORM_HEIGHT - PADDLE_HALF_H
    );

    const state = game.env._normalize_state();
    const action = await game.agent.selectAction(state);
    const [_, reward, terminated] = game.env.step(
      action,
      game.playerPaddleY,
      dt,
      game.speedMultiplier
    );

    if (terminated) {
      reward > 0 ? game.aiScore++ : game.playerScore++;
      updateScore();
      setStatus(reward > 0 ? 'AI scores!' : 'Player scores!');
      game.mode = 'scored';
      cancelAnimationFrame(game.rafId);
      game.rafId = null;
      setTimeout(nextRound, 1000);
      return;
    }

    game.env.render();
    game.rafId = requestAnimationFrame(loop);
  };

  try {
    initEnv();
    attachUI();
    await loadAgent();
    game.mode = 'ready';
    startBtn.disabled = false;
  } catch (e) {
    console.error(e);
    setStatus('Init error.');
  }
});
