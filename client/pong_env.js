class PongEnvJs {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.width = 600;
    this.height = 400;
    this.paddle_height = 60;
    this.paddle_width = 10;
    this.ball_radius = 8;
    this.norm_width = this.width / 2.0;
    this.norm_height = this.height / 2.0;

    this.ball_x = 0.0;
    this.ball_y = 0.0;
    this.ball_vx = 0.0;
    this.ball_vy = 0.0;
    this.paddle1_y = 0.0;
    this.paddle2_y = 0.0;
    this.base_paddle_speed = 12.0;
    this.base_ball_speed_initial = 10.0;
    this.base_ball_speed_increase = 0.4;
    this.base_max_ball_speed = 24.0;
    this.current_ball_speed_factor = 1.0;

    this.state_dim = 5;
    this.action_dim = 3;

    this._internal_reset();
  }

  _internal_reset() {
    this.ball_x = 0.0;
    this.ball_y = (Math.random() - 0.5) * this.height * 0.5;
    const angle = (Math.random() - 0.5) * (Math.PI / 2);
    const direction = Math.random() < 0.5 ? 1 : -1;
    this.current_ball_speed_factor = 1.0;
    const initial_speed = this.base_ball_speed_initial;
    this.ball_vx = direction * initial_speed * Math.cos(angle);
    this.ball_vy = initial_speed * Math.sin(angle);
    this.paddle1_y = 0.0;
    this.paddle2_y = 0.0;
  }

  _normalize_state() {
    const state = [
      this.ball_x / this.norm_width,
      this.ball_y / this.norm_height,
      this.ball_vx / this.base_max_ball_speed,
      this.ball_vy / this.base_max_ball_speed,
      this.paddle1_y / this.norm_height,
    ];
    if (state.length !== this.state_dim) {
      console.error(
        `State dimension mismatch! Expected ${this.state_dim}, got ${state.length}`
      );
    }
    return new Float32Array(state);
  }

  reset() {
    this._internal_reset();
    const observation = this._normalize_state();
    const info = {};
    return [observation, info];
  }

  step(
    ai_action,
    opponent_or_player_paddle_y,
    deltaTime = 1 / 60,
    gameSpeedMultiplier = 1.0
  ) {
    const dt_factor = 60 * deltaTime * gameSpeedMultiplier;

    const paddle_move_base = this.base_paddle_speed * dt_factor;

    const current_base_speed =
      this.base_ball_speed_initial * this.current_ball_speed_factor;
    const speed_magnitude = Math.sqrt(this.ball_vx ** 2 + this.ball_vy ** 2);
    if (speed_magnitude > 1e-6) {
      const scale = current_base_speed / speed_magnitude;
      this.ball_vx *= scale;
      this.ball_vy *= scale;
    } else if (current_base_speed > 1e-6) {
      const angle = (Math.random() - 0.5) * (Math.PI / 2);
      const direction = Math.random() < 0.5 ? 1 : -1;
      this.ball_vx = direction * current_base_speed * Math.cos(angle);
      this.ball_vy = current_base_speed * Math.sin(angle);
    }
    const dt_ball_vx = this.ball_vx * dt_factor;
    const dt_ball_vy = this.ball_vy * dt_factor;

    const prev_ball_x = this.ball_x;

    if (ai_action === 1) {
      this.paddle1_y -= paddle_move_base;
    } else if (ai_action === 2) {
      this.paddle1_y += paddle_move_base;
    }
    const paddle_half_height = this.paddle_height / 2.0;
    this.paddle1_y = Math.max(
      -this.norm_height + paddle_half_height,
      Math.min(this.norm_height - paddle_half_height, this.paddle1_y)
    );

    this.paddle2_y = opponent_or_player_paddle_y;

    this.ball_x += dt_ball_vx;
    this.ball_y += dt_ball_vy;

    let reward = 0.0;
    let terminated = false;
    let paddle_hit = false;

    if (
      this.ball_y - this.ball_radius < -this.norm_height ||
      this.ball_y + this.ball_radius > this.norm_height
    ) {
      this.ball_vy *= -1;
      this.ball_y = Math.max(
        -this.norm_height + this.ball_radius,
        Math.min(this.norm_height - this.ball_radius, this.ball_y)
      );
    }

    const paddle1_front_edge = -this.norm_width + this.paddle_width;
    const paddle2_front_edge = this.norm_width - this.paddle_width;

    if (
      this.ball_vx < 0 &&
      prev_ball_x - this.ball_radius >= paddle1_front_edge &&
      this.ball_x - this.ball_radius < paddle1_front_edge &&
      Math.abs(this.ball_y - this.paddle1_y) <
        paddle_half_height + this.ball_radius
    ) {
      this.ball_vx *= -1;
      const relative_intersect_y =
        (this.paddle1_y - this.ball_y) / paddle_half_height;
      const bounce_angle = relative_intersect_y * (Math.PI / 3);
      this.current_ball_speed_factor = Math.min(
        this.current_ball_speed_factor +
          this.base_ball_speed_increase / this.base_ball_speed_initial,
        this.base_max_ball_speed / this.base_ball_speed_initial
      );
      const new_speed =
        this.base_ball_speed_initial * this.current_ball_speed_factor;
      this.ball_vx = new_speed * Math.cos(bounce_angle);
      this.ball_vy = new_speed * -Math.sin(bounce_angle);
      this.ball_x = paddle1_front_edge + this.ball_radius;
      paddle_hit = true;
    } else if (
      this.ball_vx > 0 &&
      prev_ball_x + this.ball_radius <= paddle2_front_edge &&
      this.ball_x + this.ball_radius > paddle2_front_edge &&
      Math.abs(this.ball_y - this.paddle2_y) <
        paddle_half_height + this.ball_radius
    ) {
      this.ball_vx *= -1;
      const relative_intersect_y =
        (this.paddle2_y - this.ball_y) / paddle_half_height;
      const bounce_angle = relative_intersect_y * (Math.PI / 3);
      this.current_ball_speed_factor = Math.min(
        this.current_ball_speed_factor +
          this.base_ball_speed_increase / this.base_ball_speed_initial,
        this.base_max_ball_speed / this.base_ball_speed_initial
      );
      const new_speed =
        this.base_ball_speed_initial * this.current_ball_speed_factor;
      this.ball_vx = -new_speed * Math.cos(bounce_angle);
      this.ball_vy = new_speed * -Math.sin(bounce_angle);
      this.ball_x = paddle2_front_edge - this.ball_radius;
      paddle_hit = true;
    }

    if (!paddle_hit) {
      if (this.ball_x - this.ball_radius < -this.norm_width) {
        reward = -1.0;
        terminated = true;
      } else if (this.ball_x + this.ball_radius > this.norm_width) {
        reward = 1.0;
        terminated = true;
      }
    }

    const observation = this._normalize_state();
    const truncated = false;
    const info = {};
    return [observation, reward, terminated, truncated, info];
  }

  to_screen_pos(x, y) {
    const sx = x + this.norm_width;
    const sy = y + this.norm_height;
    return [sx, sy];
  }
  render() {
    const ctx = this.ctx;
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    ctx.fillStyle = '#FFF';
    const paddle_half_height = this.paddle_height / 2.0;
    const [p1_sx, p1_sy] = this.to_screen_pos(
      -this.norm_width + this.paddle_width / 2,
      this.paddle1_y
    );
    ctx.fillRect(
      p1_sx - this.paddle_width / 2,
      p1_sy - paddle_half_height,
      this.paddle_width,
      this.paddle_height
    );
    const [p2_sx, p2_sy] = this.to_screen_pos(
      this.norm_width - this.paddle_width / 2,
      this.paddle2_y
    );
    ctx.fillRect(
      p2_sx - this.paddle_width / 2,
      p2_sy - paddle_half_height,
      this.paddle_width,
      this.paddle_height
    );
    const [ball_sx, ball_sy] = this.to_screen_pos(this.ball_x, this.ball_y);
    ctx.beginPath();
    ctx.arc(ball_sx, ball_sy, this.ball_radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#555';
    ctx.lineWidth = 2;
    ctx.setLineDash([10, 10]);
    ctx.beginPath();
    ctx.moveTo(this.width / 2, 0);
    ctx.lineTo(this.width / 2, this.height);
    ctx.stroke();
    ctx.setLineDash([]);
  }
}
